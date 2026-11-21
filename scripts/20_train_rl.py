#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_train_rl.py —— 支线 B / C：验证器驱动的策略优化

【诚实声明 · 必读】
这份实现**不是教科书 GRPO**。它就是：
    组内基线归一化的策略梯度（REINFORCE with group baseline）+ KL 正则。
原因：本项目的「等步数」约定要求「一个 rollout 批次只做一次参数更新」，
而此时采样与更新用的是同一份策略 → 重要性比 ratio ≡ 1 → PPO/GRPO 的
clip 项**永远不激活**，加了也是死代码。要让 clip 真正起作用必须引入
inner_epoch > 1，那会把参数更新次数翻倍、破坏等步数对照。
所以这里如实叫它「组基线策略梯度」，不冒充 GRPO。

【本轮相对旧版的改动】（旧版 61 步里 29 步组内奖励全同 → 优势恒为 0，
这些步只剩 KL 项，等于**主动把策略往 base 拉回**，是负贡献）
  C1  批量生成：generate_reply_batch 一次产 ng 条（实测墙钟提速 3.3x）
  C2  难度筛选：从 curated 池抽题（只含模型有时能做对的题）
  C3  loss 按总 token 数归一（旧版 logp.mean() 被 1/T 缩放，信号极弱）
  C4  优势用无偏 std
  C5  max_new_tokens 与评测对齐（192）
  C6  逐步记录 prompt_ids —— 否则奖励曲线被「抽到哪题」混淆，无法判读
  C7  KL 的 k3 估计量加裁剪（exp(d) 在大 d 时会爆）
  C9  零优势组**跳过**（不施加 KL-only 更新）
  C10 diversity 改为同题两两 1-similarity（旧版恒为 1.0，完全失效）
  C11 奖励权重重配（format/steps 已饱和恒为 1，在组内归一化下本就消去）
  C12 --init-adapter / --tag，支撑 C 臂（SFT→RL）与两臂产物隔离

用法：
  python 20_train_rl.py --probe                      # 跑 2 步，报耗时/信号率/ETA
  python 20_train_rl.py                              # 默认臂（B，从 base 起）
  python 20_train_rl.py --tag C --init-adapter <A1 的 adapter 目录>
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    banner, compute_reward, ensure_dir, human_time, lab_path, load_config, read_jsonl,
)
import labmodel as lm  # noqa: E402


def peak_mem_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def token_logprobs(model, input_ids, attention_mask, labels_mask, use_adapter=True):
    """
    返回 completion 部分每个 token 的 log 概率（tensor，长度 = 监督 token 数）。
    use_adapter=False 时关掉 LoRA，得到参考策略的 logp（用于 KL）。
    """
    import torch
    ctx = model.disable_adapter() if (not use_adapter and
                                     hasattr(model, "disable_adapter")) else _nullctx()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]                       # 预测下一个 token
    tgt = input_ids[:, 1:]
    mask = labels_mask[:, 1:].bool()
    logp = torch.log_softmax(logits.float(), dim=-1)
    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return picked[mask]


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _pairwise_diversity(texts) -> float:
    """同题多次采样之间的平均差异（1 - 相似度）。旧版的「前 80 字符去重」恒为 1.0，无意义。"""
    if len(texts) < 2:
        return 0.0
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a = "".join(texts[i].split())
            b = "".join(texts[j].split())
            sims.append(SequenceMatcher(None, a, b).ratio())
    return 1.0 - (sum(sims) / len(sims)) if sims else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--prompts-per-step", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略本臂已有的 adapter，从零（或 --init-adapter）开始")
    ap.add_argument("--init-adapter", default=None,
                    help="预热起点：把该目录的 LoRA 权重载入后再训（C 臂用 A1 的产物）")
    ap.add_argument("--tag", default="",
                    help="产物隔离后缀。空=默认臂 B；给 C 则写 adapter-C / rl_train-C.jsonl")
    args = ap.parse_args()

    cfg = load_config()
    rcfg = cfg["rl"]
    g = rcfg["grpo"]
    lab_root = lab_path(cfg, "lab_root")
    b_dir = lab_path(cfg, "branch_b")
    model_path = str(b_dir / "model")

    suffix = f"-{args.tag}" if args.tag else ""
    adapter_dir = b_dir / f"adapter{suffix}"
    log_path = b_dir / "logs" / f"rl_train{suffix}.jsonl"
    sum_path = b_dir / "logs" / f"rl_train_summary{suffix}.json"
    # 答案流 sidecar：把每条采样（解答文本 + 对错）实时落盘，供 demo 看板连续展示。
    # 与 log_path 同族、同 tag 隔离；仅追加，不影响任何训练逻辑。
    samples_path = b_dir / "logs" / f"rl_train{suffix}.samples.jsonl"
    dtype = args.dtype or cfg["base_model"].get("dtype", "float32")
    steps = args.steps if args.steps is not None else g["max_steps"]
    pps = args.prompts_per_step if args.prompts_per_step is not None else g["prompts_per_step"]
    ng = args.num_generations if args.num_generations is not None else g["num_generations"]
    max_new = args.max_new_tokens if args.max_new_tokens is not None else g["max_new_tokens"]
    kl_coef = 0.0 if args.no_kl else g["kl_coef"]
    kl_clip = float(g.get("kl_clip", 10.0))
    weights = rcfg["reward_weights"]
    sys_prompt = cfg["task"]["system_prompt"]
    ans_max = cfg["task"]["output_format"]["answer_max_chars"]
    batch_gen = bool(g.get("batch_generate", True))
    gen_chunk = int(g.get("gen_batch_size", 0) or 0)

    banner(f"支线 B · 组基线策略梯度（tag={args.tag or 'B'}）")

    # ---- 训练池：优先用分级筛查产出的 curated 池（C2 难度筛选）----
    pool_file = rcfg.get("train_pool")
    pool_path = (lab_root / pool_file) if pool_file else None
    if pool_path and pool_path.exists():
        prompts = read_jsonl(pool_path)
        print(f"  训练池 {pool_path.relative_to(lab_root)}（curated，模型有时能做对的题）")
    else:
        prompts = read_jsonl(b_dir / "data" / "prompts.jsonl")
        print(f"  ⚠ 未找到 curated 池（{pool_file}）→ 回退到原始题库")
        if pool_file:
            print(f"    建议先跑 12_screen_layers.py 生成它，否则零优势步会很多")
    if not prompts:
        print("✗ RL 题库为空，请先跑 09_gen_math_bank.py")
        return 1

    print(f"  题库 {len(prompts)} 条；每步 {pps} 题 × {ng} 采样 = {pps*ng} 次生成"
          f"（{'批量' if batch_gen else '串行'}"
          f"{f'，每批最多 {gen_chunk} 条' if gen_chunk else ''}）")
    print(f"  计划 {steps} 步，共约 {steps*pps*ng} 次生成")
    print(f"  奖励权重 {weights}")
    print(f"  KL 系数 {kl_coef}（k3，裁剪 ±{kl_clip}）   dtype {dtype}   max_new_tokens {max_new}")
    print("  ⚠ 本实现是「组基线策略梯度 + KL」，不是 GRPO："
          "单次更新下重要性比恒为 1，clip 不激活（见文件头声明）")

    import torch
    torch.set_num_threads(cfg["runtime"]["torch_threads"])

    tok = lm.load_tokenizer(model_path, cfg["base_model"]["trust_remote_code"])
    model, info = lm.load_model(model_path, dtype=dtype,
                               drop_vision=cfg["base_model"].get("text_only", True),
                               trust_remote_code=cfg["base_model"]["trust_remote_code"])
    if g.get("gradient_checkpointing", False):
        lm.enable_gradient_checkpointing(model)

    model = lm.apply_lora(model, rcfg["lora"])
    n_train = lm.mark_only_lora_trainable(model)
    print(f"  可训练参数 {n_train/1e6:.2f}M")

    if kl_coef > 0 and not hasattr(model, "disable_adapter"):
        print("  ⚠ 当前 PEFT 装配无 disable_adapter，KL 自动关闭")
        kl_coef = 0.0

    # ---- 起点：本臂续训 / 预热 / 从零 ----
    def _load_into(path: Path) -> bool:
        try:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sd = load_file(str(path / "adapter_model.safetensors"))
            set_peft_model_state_dict(model, sd)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    载入失败：{type(e).__name__}: {e}")
            return False

    started_from = "base（从零）"
    if not args.fresh and (adapter_dir / "adapter_config.json").exists():
        if _load_into(adapter_dir):
            started_from = f"本臂既有 adapter（{adapter_dir.name}，续训）"
    elif args.init_adapter:
        ip = Path(args.init_adapter)
        if (ip / "adapter_model.safetensors").exists():
            if _load_into(ip):
                started_from = f"预热自 {ip}"
        else:
            print(f"  ✗ --init-adapter 指向的目录没有 adapter_model.safetensors：{ip}")
            return 1
    print(f"  起点：{started_from}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=g["learning_rate"])
    pad_id = tok.pad_token_id or tok.eos_token_id

    if args.probe:
        steps = min(2, steps)
        # 【重要】探针绝不覆盖真实产物 —— 否则一次测速就把训练几小时的 adapter 冲掉。
        # 探针只写到一个独立的 scratch 目录，同时仍然走完整保存路径以便验证它没坏。
        adapter_dir = b_dir / f"_probe_adapter{suffix}"
        print(f"  [probe] adapter 只写到 scratch 目录 {adapter_dir.name}/，不碰真实产物")

    ensure_dir(log_path.parent)
    logf = open(log_path, "a", encoding="utf-8")
    sf = open(samples_path, "a", encoding="utf-8")

    rnd = __import__("random").Random(g["seed"])
    t_start = time.time()
    hist = []
    n_effective = 0        # 真正施加了更新的步数（其余是零优势空步）
    n_dead = 0

    for step in range(1, steps + 1):
        torch.manual_seed(g["seed"] + step)
        batch_prompts = [rnd.choice(prompts) for _ in range(pps)]

        step_rewards, step_parts, sample_rewards = [], [], []
        step_tokens = 0
        pg_terms: list = []          # adv_i * Σ_t logp_it （标量 tensor，带梯度）
        kl_terms: list = []          # 与该题总 token 数配比的 KL
        opt.zero_grad(set_to_none=True)
        t_step = time.time()

        for item in batch_prompts:
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": item["prompt"]}]

            # ---- 采样一组答案（C1：批量；支持分块以压低内存峰值）----
            model.eval()
            if batch_gen:
                texts = []
                chunk = ng if gen_chunk <= 0 else min(gen_chunk, ng)
                for off in range(0, ng, chunk):
                    k = min(chunk, ng - off)
                    part, _n, _s = lm.generate_reply_batch(
                        model, tok, msgs, num_return_sequences=k,
                        max_new_tokens=max_new, temperature=g["temperature"],
                        top_p=g["top_p"], seed=g["seed"] + step * 100 + off)
                    texts.extend(part)
            else:
                texts = []
                for k in range(ng):
                    t_, _n, _s = lm.generate_reply(
                        model, tok, msgs, max_new_tokens=max_new,
                        temperature=g["temperature"], top_p=g["top_p"],
                        seed=g["seed"] + step * 100 + k)
                    texts.append(t_)

            comps = []
            for t_ in texts:
                rew, parts = compute_reward(t_, item["answer"], weights, ans_max)
                comps.append({"text": t_, "reward": rew, "parts": parts})
                step_rewards.append(rew)
                step_parts.append(parts)
                sample_rewards.append({"id": item["id"], "reward": rew,
                                       "correct": parts["correct"] > 0,
                                       "format": parts["format"] > 0, "text": t_})
                # 答案流：每条采样即写即刷，看板可边采样边显示
                sf.write(json.dumps({
                    "step": step, "id": item["id"],
                    "prompt": item["prompt"], "gold": item["answer"],
                    "text": t_, "reward": round(rew, 4),
                    "correct": bool(parts["correct"] > 0),
                    "format": bool(parts["format"] > 0),
                }, ensure_ascii=False) + "\n")
                sf.flush()

            # ---- 组内归一化算优势（C4：无偏 std）----
            rs = torch.tensor([c["reward"] for c in comps], dtype=torch.float32)
            sd = rs.std(unbiased=True) if rs.numel() > 1 else torch.tensor(0.0)
            # C9：组内奖励全同 → 优势恒为 0。此时若继续走 loss，只剩 KL 项
            #     会把策略往 base 拉回，是**负贡献**。直接跳过该题。
            if float(sd) < 1e-6:
                continue
            adv = (rs - rs.mean()) / (sd + 1e-4)

            # ---- 策略梯度 + KL ----
            model.train()
            prefix = lm.render_generation_prompt(tok, msgs)
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]

            prompt_pg, prompt_kl, prompt_tok = [], [], 0
            for i, c in enumerate(comps):
                a_ids = tok(c["text"], add_special_tokens=False)["input_ids"]
                if not a_ids:
                    a_ids = [tok.eos_token_id]
                full = p_ids + a_ids
                lab = [-100] * len(p_ids) + a_ids
                if len(full) > 1024:                      # 安全上限
                    full, lab = full[:1024], lab[:1024]
                ids_t = torch.tensor([full], dtype=torch.long)
                mask_t = torch.ones_like(ids_t)
                lab_t = torch.tensor([lab], dtype=torch.long)

                logp = token_logprobs(model, ids_t, mask_t, lab_t, use_adapter=True)
                if logp.numel() == 0:
                    continue
                # C3：用 token 求和（而非 mean）。除以全局总 token 数统一归一，
                #     避免长度越长的回答被越弱地加权。
                prompt_pg.append(adv[i] * logp.sum())
                prompt_tok += int(logp.numel())

                if kl_coef > 0:
                    with torch.no_grad():
                        ref = token_logprobs(model, ids_t, mask_t, lab_t,
                                             use_adapter=False)
                    n = min(logp.numel(), ref.numel())
                    d = (ref[:n].detach() - logp[:n]).clamp(-kl_clip, kl_clip)  # C7
                    prompt_kl.append((torch.exp(d) - d - 1.0).sum())

            if prompt_pg:
                pg_terms.extend(prompt_pg)
                kl_terms.extend(prompt_kl)
                step_tokens += prompt_tok

        # ---- 更新（全部题目都零优势 → 本步为「空步」，不产生梯度）----
        loss_val = None
        if pg_terms and step_tokens > 0:
            pg = torch.stack(pg_terms).sum() / step_tokens
            loss = -pg
            if kl_terms:
                loss = loss + kl_coef * (torch.stack(kl_terms).sum() / step_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_val = float(loss.detach())
            n_effective += 1
        else:
            opt.zero_grad(set_to_none=True)
            n_dead += 1

        n = len(step_rewards)
        mean_r = sum(step_rewards) / max(n, 1)
        std_r = (sum((x - mean_r) ** 2 for x in step_rewards) / max(n, 1)) ** 0.5
        acc = sum(p["correct"] for p in step_parts) / max(n, 1)
        fmt_rate = sum(p["format"] for p in step_parts) / max(n, 1)
        by_id = {}
        for s in sample_rewards:
            by_id.setdefault(s["id"], []).append(s["text"])
        div = sum(_pairwise_diversity(v) for v in by_id.values()) / max(len(by_id), 1)  # C10

        rec = {
            "step": step,
            "prompt_ids": [it["id"] for it in batch_prompts],          # C6
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
            "accuracy": round(acc, 3), "format_rate": round(fmt_rate, 3),
            "n_correct": sum(1 for s in sample_rewards if s["correct"]),
            "n_samples": n,
            "diversity": round(div, 3),
            "loss": round(loss_val, 4) if loss_val is not None else None,
            "effective": bool(pg_terms),                                # 本步是否真更新
            "tokens": step_tokens,
            "peak_mem_gb": round(peak_mem_gb(), 2),
            "step_s": round(time.time() - t_step, 1),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        hist.append(rec)
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()

        el = rec["elapsed_s"]
        eta = human_time(el / step * (steps - step))
        tag_eff = "✓更新" if pg_terms else "✗空步"
        print(f"  step {step}/{steps}  R={mean_r:+.3f}(±{std_r:.3f})  "
              f"正确率 {acc:.0%}  多样 {div:.0%}  {tag_eff}  "
              f"mem {rec['peak_mem_gb']:.2f}G  {rec['step_s']:.0f}s/步  ETA {eta}",
              flush=True)

        if step % 3 == 0:
            _save(model, tok, adapter_dir)

    logf.close()
    sf.close()
    _save(model, tok, adapter_dir)

    banner("RL 训练结束")
    print(f"  用时 {human_time(time.time()-t_start)}   峰值内存 {peak_mem_gb():.2f} GB")
    print(f"  有效更新 {n_effective} / {len(hist)} 步（空步 {n_dead}，占 "
          f"{n_dead/max(len(hist),1)*100:.0f}%）")
    if len(hist) >= 4:
        k = max(1, len(hist) // 3)
        early = sum(h["mean_reward"] for h in hist[:k]) / k
        late = sum(h["mean_reward"] for h in hist[-k:]) / k
        trend = "上行 ↑" if late > early + 0.02 else (
            "持平 →" if abs(late - early) <= 0.02 else "下行 ↓")
        print(f"  奖励趋势：前段 {early:+.3f} → 后段 {late:+.3f}   {trend}")
    sum_path.write_text(
        json.dumps({"tag": args.tag, "started_from": started_from,
                    "steps": len(hist), "n_effective": n_effective,
                    "n_dead": n_dead,
                    "pool": (str(pool_path) if pool_path and pool_path.exists()
                             else "branch-B-rl/data/prompts.jsonl"),
                    "history": hist, "weights": weights, "kl_coef": kl_coef},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  adapter 已保存 {adapter_dir}")
    return 0


def _save(model, tok, adapter_dir) -> None:
    """用 labmodel.save_adapter：不产生临时文件，绕开沙箱对 unlink/rename 的拦截。"""
    lm.save_adapter(model, tok, adapter_dir)


if __name__ == "__main__":
    raise SystemExit(main())
