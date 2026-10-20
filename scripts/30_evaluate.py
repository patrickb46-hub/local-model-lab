#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_evaluate.py —— 阶段 3：四模型横向评测

对照设计（计划第 7 节）：
  base     底线：一切增量的参照。蒸馏/RL 若还不如它，这活就白干了
  A        待测：与 base 的差值 = 蒸馏的收益
  B        待测：与 base 的差值 = RL 的收益
  teacher  天花板：student 够到了多少

内存上的关键取舍：
  base / A / B 三个其实是"同一份权重 + 不同 LoRA 适配器"。所以我们只加载
  一份权重，用 PEFT 的多适配器切换来扮演三个角色 —— 而不是把 2B 模型加载三遍
  （那会直接把 16 GB 打满）。teacher 走 Ollama，不占本进程内存。

断点续跑：每条结果即时 append，中断后重跑自动跳过。

用法：
  python 30_evaluate.py --limit 5 --models base,A     # 冒烟
  python 30_evaluate.py                                # 全量四模型
  python 30_evaluate.py --models teacher               # 只跑 teacher
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    answer_equal, banner, cloud_chat, cloud_health, ensure_dir, human_time,
    lab_path, load_config, ollama_chat, parse_response, read_jsonl, write_jsonl,
)
import labmodel as lm  # noqa: E402


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    """输出相似度，用于"学得像不像 teacher"。空白归一后取最长公共子序列比。"""
    import re
    norm = lambda s: re.sub(r"\s+", "", s or "")
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def keyword_hit(text: str, keywords) -> bool:
    if not keywords:
        return None
    return any(k.lower() in (text or "").lower() for k in keywords)


# ---------------------------------------------------------------------------
# 本地模型（base / A / B 共用一份权重，切换适配器）
# ---------------------------------------------------------------------------

class LocalZoo:
    """一份权重，三个身份。"""

    def __init__(self, cfg):
        self.cfg = cfg
        lab_root = lab_path(cfg, "lab_root")
        self.a_dir = lab_path(cfg, "branch_a")
        self.b_dir = lab_path(cfg, "branch_b")
        self.model_path = str(self.a_dir / "model")
        self.model = None
        self.tok = None
        self.available = []

    def load(self, dtype: str, wanted=None):
        """wanted: 需要评测的本地角色列表（决定加载哪些 adapter）。"""
        import torch
        torch.set_num_threads(self.cfg["runtime"]["torch_threads"])
        base_p = self.a_dir / "model"
        if not (base_p / "config.json").exists():
            print("  ✗ branch-A/model 不存在，无法加载本地模型")
            return
        self.tok = lm.load_tokenizer(self.model_path,
                                     self.cfg["base_model"]["trust_remote_code"])
        model, info = lm.load_model(
            self.model_path, dtype=dtype,
            drop_vision=self.cfg["base_model"].get("text_only", True),
            trust_remote_code=self.cfg["base_model"]["trust_remote_code"])
        print(f"  权重已加载（{info['params_b']}B, dtype={info['dtype']}）")

        # 角色 → adapter 目录：读 config.eval.local_adapters（支持 A1/A2/B/C 四臂）
        mapping = dict(self.cfg["eval"].get("local_adapters", {}))
        mapping.pop("$comment", None)
        if not mapping:            # 向后兼容：没配就退回旧的固定路径
            mapping = {"A": "branch-A-distill/adapter",
                       "B": "branch-B-rl/adapter"}
        lab_root = lab_path(self.cfg, "lab_root")
        order = list(wanted) if wanted else list(mapping)
        adapters = []
        for name in order:
            rel = mapping.get(name)
            if not rel:
                continue
            d = lab_root / rel
            if (d / "adapter_model.safetensors").exists() and \
                    (d / "adapter_config.json").exists():
                adapters.append((name, str(d)))
            else:
                print(f"  ⚠ 未找到 {name} 的 adapter（{d}）—— 该角色将缺席")

        if adapters:
            from peft import PeftModel
            name0, path0 = adapters[0]
            model = PeftModel.from_pretrained(model, path0, adapter_name=name0)
            for name, path in adapters[1:]:
                try:
                    model.load_adapter(path, adapter_name=name)
                except Exception as e:  # noqa: BLE001
                    print(f"  ⚠ 加载 {name} adapter 失败：{e}")
            model.eval()
            print(f"  已挂载适配器 {[n for n, _ in adapters]}（多适配器切换，省两份权重）")

        self.model = model
        self.roles = ["base"] + [n for n, _ in adapters]
        print(f"  本地可评测角色：{self.roles}")

    def set_role(self, role: str):
        if role == "base":
            return self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else _nc()
        self.model.set_adapter(role)
        return _nc()

    def free(self):
        """
        把本地权重从内存里放掉。

        为什么必须做：本地 fp32 权重约 7.5 GB，而后面 Ollama 的 9B 还要 6.1 GB。
        两者同时驻留 = 13.6 GB，在 16 GB 机器上加上开销必爆。
        所以本地角色跑完必须立刻释放。

        PeftModel 有多层包装，容易残留循环引用，所以 gc 多跑几轮。
        """
        import gc
        self.model = None
        self.tok = None
        for _ in range(3):
            gc.collect()
        print("  本地权重已释放", flush=True)

    def chat(self, msgs, max_new, temperature, seed, enable_thinking=None):
        if self.model is None:
            return None, 0, 0.0
        return lm.generate_reply(self.model, self.tok, msgs,
                                 max_new_tokens=max_new, temperature=temperature,
                                 seed=seed, enable_thinking=enable_thinking)

    def chat_batch(self, msgs, n, max_new, temperature, seed,
                   enable_thinking=None):
        """一次批量生成 n 条（纯 CPU 吞吐随批量提升：B=1 1.28 → B=8 4.19 tok/s）。"""
        if self.model is None:
            return [], 0, 0.0
        return lm.generate_reply_batch(
            self.model, self.tok, msgs, num_return_sequences=n,
            max_new_tokens=max_new, temperature=temperature,
            seed=seed, enable_thinking=enable_thinking)


class _nc:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="逗号分隔的角色名；不传则取 config 的 eval.models（六个角色）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 题")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--no-diversity", action="store_true", help="（保留兼容，已无作用）")
    ap.add_argument("--no-probe", action="store_true", help="跳过通用能力探针")
    ap.add_argument("--dry-run", action="store_true",
                    help="只预演：打印将评测的角色、题量与预估次数，不加载模型、不生成")
    ap.add_argument("--bench", default=None,
                    help="使用分层基准（如 shared/bench_stratified.jsonl）；"
                         "默认取 config.data.bench_file，不存在则回落到 eval_file")
    ap.add_argument("--reuse-screen", action="store_true",
                    help="复用 12_screen_layers.py 的筛查采样（base/teacher 已采过），"
                         "省掉这两个角色约 5 小时")
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg["eval"]
    lab_root = lab_path(cfg, "lab_root")
    # 角色来源：命令行 > config.eval.models。
    # 默认必须取全六个角色 —— 少了 qwen2b / qwen4b，
    # 「微调后的 2B 能不能追上原版 4B」这个核心问题就直接没有答案。
    roles = ([r.strip() for r in args.models.split(",") if r.strip()]
             if args.models
             else list(cfg["eval"].get("models") or ["base", "A", "B", "teacher"]))
    # 全流程统一 float32：实测本机生成速度 fp32 1.32 tok/s vs bf16 0.70 tok/s（AVX2 无 BF16 原生指令）
    dtype = args.dtype or cfg["base_model"].get("inference_dtype", "float32")

    # 基准来源：分层基准（P3 冻结）优先，回落到原始 eval_set。
    # 为什么要分层基准：原来的 20 题里「teacher 系统性失败 且 base 有机会」
    # 只有 1 道，pass@3 上蒸馏与 teacher 都是 90%，天花板顶满、测不出差异。
    bench_rel = args.bench or cfg["data"].get("bench_file")
    bench_path = (lab_root / bench_rel) if bench_rel else None
    if bench_path and bench_path.exists():
        eval_set = read_jsonl(bench_path)
        bench_src = str(bench_path)
        n_layer = {}
        for it in eval_set:
            n_layer[it.get("layer", "?")] = n_layer.get(it.get("layer", "?"), 0) + 1
        print(f"  使用分层基准：{bench_src}")
        print(f"    分层构成 {n_layer}")
    else:
        eval_set = read_jsonl(lab_root / cfg["data"]["eval_file"])
        bench_src = cfg["data"]["eval_file"]
        print(f"  ⚠ 分层基准不存在（{bench_path}）→ 回落到 {bench_src}")
        print(f"    先跑： python 12_screen_layers.py --model report")
    # 实际评测题数：CLI --limit 优先，其次 config.data.eval_limit；基准文件本身不动。
    # 【严重坑 · 2026-10-10】原实现是 eval_set[:limit] —— 而基准文件是「按层分块写出」的
    # （L1 全部在前、L2 次之、L3 最后），于是 [:20] 恰好只截到 20 道 L1：
    # 评测集里一道 L2/L3 都没有，核心命题「L2 上 RL 臂能否越过 teacher」直接无法回答，
    # 而日志、指标、报告全都"正常"，没有任何报错 —— 沉默的错误，比崩溃危险得多。
    # 改为「分层轮转取样」：逐层轮流取，保证各层都被覆盖。
    limit = args.limit or int(cfg["data"].get("eval_limit", 0) or 0)
    if limit and limit < len(eval_set):
        buckets = {}
        for it in eval_set:
            buckets.setdefault(it.get("layer") or "_", []).append(it)
        picked, i = [], 0
        while len(picked) < limit:
            added = False
            for l in sorted(buckets):
                if len(picked) >= limit:
                    break
                if i < len(buckets[l]):
                    picked.append(buckets[l][i])
                    added = True
            if not added:
                break
            i += 1
        eval_set = picked
        got = {}
        for it in eval_set:
            k = it.get("layer") or "?"
            got[k] = got.get(k, 0) + 1
        print(f"    --limit {limit} → 分层轮转取到 {got}（原实现会全取 L1）")
    if not eval_set:
        print("✗ eval_set 为空，请先跑 09_gen_math_bank.py")
        return 1

    # 角色分三类：本地（base/A/B，一份权重切 LoRA）/ Ollama 量化原版 / 云 API 强模型
    OLLAMA = dict(cfg["eval"].get("ollama_baselines", {}))
    OLLAMA.pop("$comment", None)
    CLOUD = dict(cfg["eval"].get("cloud_baselines", {}))
    CLOUD.pop("$comment", None)
    LOCAL_KNOWN = ("base", "A", "A1", "A2", "B", "C")
    local_roles = [r for r in roles if r in LOCAL_KNOWN]
    ollama_roles = [r for r in roles if r in OLLAMA]
    cloud_roles = [r for r in roles if r in CLOUD]
    unknown = [r for r in roles
               if r not in LOCAL_KNOWN and r not in OLLAMA and r not in CLOUD]
    if unknown:
        print(f"  ⚠ 未知角色 {unknown}，将被跳过（本地：base/A/B；"
              f"Ollama：{list(OLLAMA)}；云：{list(CLOUD)}）")
        roles = local_roles + ollama_roles + cloud_roles

    raw_path = lab_path(cfg, "results") / "eval_raw.jsonl"
    ensure_dir(raw_path.parent)

    banner("阶段 3 · 多模型横向评测")
    print(f"  eval_set {len(eval_set)} 题（已锁定，训练中从未见过）")
    print(f"  参评角色 {roles}")
    print(f"  · 本地（共一份 2B 权重，切 LoRA）：{local_roles or '无'}")
    for r in ollama_roles:
        print(f"  · Ollama 官方基线：{r} → {OLLAMA[r]}")
    for r in cloud_roles:
        cc = CLOUD[r]
        print(f"  · 云 API 基线：{r} → {cc.get('model')} @ {cc.get('base_url')}")
    print(f"  dtype {dtype}（全流程统一 fp32：本机 fp32 生成比 bf16 快近一倍）")

    if args.dry_run:
        probe_all = read_jsonl(lab_root / ecfg["general_probe_file"])
        n_main = len(eval_set) * int(ecfg.get("repeats", 1))
        _do_think = "on" in (ecfg.get("think_modes") or [])
        n_think = (len(eval_set[: int(ecfg.get("think_probe_size", 6))])
                   * int(ecfg.get("think_probe_repeats", 1))) if _do_think else 0
        n_probe = 0 if args.no_probe else len(probe_all)
        per_role = n_main + n_think + n_probe
        print("\n  [dry-run] 仅预演：不加载模型、不生成")
        print(f"  每个角色：主测 {n_main} + think对照 {n_think} + 通用探针 {n_probe}"
              f" = {per_role} 次生成")
        print(f"  参评角色数 {len(roles)}  → 总生成次数 {per_role * len(roles)}")
        for r in roles:
            if r in LOCAL_KNOWN:
                kind = "本地（一份权重切 LoRA）"
            elif r in CLOUD:
                kind = f"云 API · {CLOUD[r].get('model')}"
            else:
                kind = f"Ollama · {OLLAMA.get(r)}"
            print(f"    · {r:9s} {kind}")
        if len(roles) < 6:
            print(f"\n  ⚠ 只有 {len(roles)} 个角色（期望 6）—— 检查 config 的 eval.models")
        return 0

    sys_prompt = cfg["task"]["system_prompt"]
    max_new = ecfg["max_new_tokens"]

    # 可选：复用筛查采样（base / teacher 已在同一批基准题上采过样）
    if args.reuse_screen:
        reuse_screen_samples(cfg, eval_set, raw_path,
                             roles_wanted={"base", "teacher"},
                             max_rep=int(ecfg.get("repeats", 1)))

    # 去重键：同一 (role, id, kind, rep)
    existing = {(r["role"], r["id"], r["kind"], r.get("rep", 0))
                for r in read_jsonl(raw_path)}
    print(f"  已有结果 {len(existing)} 条（断点续跑）")

    # 评测计划（提前算出来，供"要不要加载模型"判断使用）
    reps = int(ecfg.get("repeats", 1))
    stemp = float(ecfg.get("sample_temperature", 0.0))
    probe_items = read_jsonl(lab_root / ecfg["general_probe_file"])
    t_items = eval_set[: int(ecfg.get("think_probe_size", 6))]
    t_reps = int(ecfg.get("think_probe_repeats", 1))
    t_maxnew = int(ecfg.get("think_max_new_tokens", 384))
    do_think = "on" in (ecfg.get("think_modes") or [])

    def _pending(role_list) -> bool:
        """这批角色里还有没有没跑过的条目。"""
        for r in role_list:
            plan = [("main", eval_set, reps)]
            if do_think:
                plan.append(("thinkon", t_items, t_reps))
            if not args.no_probe:
                plan.append(("general", probe_items, 1))
            for kind, items, n in plan:
                for it in items:
                    for rp in range(n):
                        if (r, it["id"], kind, rp) not in existing:
                            return True
        return False

    # 本地模型只在确有未完成条目时才加载。
    # 为什么：改完配置重跑，往往只是为了用新口径重算汇总，
    # 若无条件加载就要白等 30 秒、白占 8 GB（这台机器只有 16 GB）。
    zoo = None
    if local_roles and _pending(local_roles):
        zoo = LocalZoo(cfg)
        zoo.load(dtype, local_roles)
    elif local_roles:
        print("  本地角色已全部完成 → 跳过模型加载（省约 30s 与 8 GB）", flush=True)

    t_start = time.time()
    results = []

    def run_one(role, item, kind, rep, temperature, seed,
                think=False, max_new_override=None):
        key = (role, item["id"], kind, rep)
        if key in existing:
            return None
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": item["prompt"]}]
        mn = max_new_override or max_new
        t0 = time.time()
        think_chars = None
        if role in CLOUD:
            cc = CLOUD[role]
            key_env = cc.get("api_key_env", "COPILOT_PROVIDER_API_KEY")
            api_key = os.environ.get(key_env, "")
            # 云角色用自己的 max_tokens（理由见 config 的 $max_tokens_comment）
            txt, ntok, think_chars = cloud_chat(
                cc["base_url"], cc["model"], msgs, api_key,
                max_tokens=int(cc.get("max_tokens", 1024)),
                temperature=temperature,
                think=bool(think),
                timeout=int(cc.get("timeout", 300)),
                retry=int(cc.get("retry", 3)))
        elif role in OLLAMA:
            tcfg = cfg["teacher"]
            txt = ollama_chat(tcfg["ollama_url"], OLLAMA[role], msgs,
                              options={**tcfg["options"],
                                       "temperature": temperature or 0.0,
                                       "num_predict": mn},
                              think=bool(think),
                              timeout=tcfg["request_timeout"], retry=tcfg["retry"])
            ntok = None
        else:
            with zoo.set_role(role):
                txt, ntok, _ = zoo.chat(msgs, mn, temperature, seed,
                                        enable_thinking=bool(think))
        dt = time.time() - t0
        if txt is None:
            print(f"    ✗ {role}/{item['id']} 生成失败")
            return None
        return _persist(role, item, kind, rep, think, txt,
                        think_chars, ntok, dt, temperature)

    def _persist(role, item, kind, rep, think, txt, think_chars, ntok, dt,
                 temperature):
        """构造记录并落盘 —— run_one 与 run_item_batch 共用，保证判分口径一致。"""
        key = (role, item["id"], kind, rep)
        p = parse_response(txt)
        rec = {
            "role": role, "id": item["id"], "kind": kind, "rep": rep,
            "think": bool(think),
            # 通用能力探针的题目结构不同（只有 id/prompt/keywords），
            # 没有 category / answer —— 所以这里必须用 .get 兜底。
            # 曾经的写法是 item["category"]，导致 base 角色跑完主测和 think 对照后，
            # 一到「通用能力探针」就 KeyError: 'category' 崩掉，整轮横评作废。
            "category": item.get("category", "general"),
            # 分层基准带 layer（L1/L2/L3）；非分层题（如通用探针）为 None
            "layer": item.get("layer"),
            "prompt": item["prompt"],
            "gold": item.get("answer"), "response": txt,
            "format_ok": p["format_ok"],
            "answer_norm": p["answer_norm"],
            "correct": bool(item.get("answer")) and
                       answer_equal(p["answer_norm"], item["answer"]),
            "n_steps": len(p["steps"]),
            # 云 API 直接返回独立的推理块（比解析 <think> 标签可靠），有就优先用
            "thinking_chars": (think_chars if think_chars is not None
                               else p["thinking_chars"]),
            "tokens": ntok, "latency_s": round(dt, 2),
            "temperature": temperature,
        }
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        existing.add(key)
        return rec

    def run_item_batch(role, item, kind, reps_n, temperature, seed_base,
                       think=False, max_new_override=None):
        """对一道题做 reps_n 次采样：**一次批量生成**，再逐条落盘。

        与 run_one 的唯一区别是生成方式（批量 vs 单条），判分口径完全一致（共用 _persist）。

        为什么值得单独写：本地角色一道题 3 次采样，串行约 3×76s ≈ 228s；
        批量一次约 90s。评测四个本地臂从 ~10 小时压到 ~4 小时。
        纯 CPU 解码是内存带宽受限的 —— 同一次权重读取能产出 B 个 token，
        实测 B=1 1.28 tok/s → B=8 4.19 tok/s。

        ⚠ 生成返回空**必须报错**，绝不能记成「答错」——
        同类 bug 已让 teacher 的 145 条空回答把整个分层基准静默作废一次。
        """
        pending = [rp for rp in range(reps_n)
                   if (role, item["id"], kind, rp) not in existing]
        if not pending:
            return 0
        # 非本地角色没有批量接口 → 退回逐条（它们的基准数据通常可从筛查结果复用）
        if role in CLOUD or role in OLLAMA:
            for rp in pending:
                run_one(role, item, kind, rp, temperature, seed_base + rp,
                        think=think, max_new_override=max_new_override)
            return len(pending)

        mn = max_new_override or max_new
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": item["prompt"]}]
        t0 = time.time()
        with zoo.set_role(role):
            texts, ntok, _s = zoo.chat_batch(msgs, len(pending), mn,
                                             temperature, seed_base,
                                             enable_thinking=bool(think))
        dt = time.time() - t0
        n_ok = 0
        for i, rp in enumerate(pending):
            txt = texts[i] if i < len(texts) else ""
            if not txt or not txt.strip():
                raise RuntimeError(
                    f"{role}/{item['id']} rep{rp} 本地生成返回空 —— 已中止评测。"
                    f"「生成失败」与「答错」必须区分开（参见 12_screen_layers 的同类修复）。")
            _persist(role, item, kind, rp, think, txt, None, ntok, dt,
                     temperature)
            n_ok += 1
        return n_ok

    # ============ 评测分三个阶段，本地角色跑完先释放模型 ============
    def run_role_block(role):
        """一个角色的完整评测：主测(think关) → 对照(think开) → 通用能力探针。"""
        banner(f"角色 {role} · 主测 · think=关（{len(eval_set)} 题 × {reps} 次采样，T={stemp}）")
        for i, item in enumerate(eval_set, 1):
            if role in local_roles:
                # 本地角色：一道题的 reps 次采样一次批量生成（提速 ~2.5 倍）
                t0 = time.time()
                run_item_batch(role, item, "main", reps, stemp,
                               ecfg["seed"] + 1000 * i, think=False)
                note = f"{time.time()-t0:5.0f}s"
            else:
                for rep in range(reps):
                    run_one(role, item, "main", rep, stemp,
                            ecfg["seed"] + 1000 * rep + i, think=False)
                note = "     "
            print(f"  [{i}/{len(eval_set)}] {item['id']:6s} × {reps} 次完成 {note} "
                  f"(总 {human_time(time.time()-t_start)})", flush=True)

        if do_think:
            banner(f"角色 {role} · 对照 · think=开"
                   f"（{len(t_items)} 题 × {t_reps} 次，max_new={t_maxnew}）")
            for i, item in enumerate(t_items, 1):
                for rep in range(t_reps):
                    run_one(role, item, "thinkon", rep, stemp,
                            ecfg["seed"] + 7000 * rep + i,
                            think=True, max_new_override=t_maxnew)
                print(f"  [{i}/{len(t_items)}] {item['id']:6s} 完成", flush=True)

        if not args.no_probe:
            banner(f"角色 {role} · 通用能力探针（{len(probe_items)} 题）")
            for item in probe_items:
                run_one(role, item, "general", 0, 0.0, ecfg["seed"])
            print(f"  {role} 完成", flush=True)

    # ---- 阶段一：本地角色（base / A / B 共用一份 2B 权重，切 LoRA 扮演三个身份）----
    if local_roles:
        banner(f"【本地阶段】{local_roles} —— 一份权重 + 多适配器切换")
        for role in local_roles:
            run_role_block(role)
        if zoo is not None:
            zoo.free()
            print(f"  已释放本地模型，腾出内存给 Ollama"
                  f"（已用 {human_time(time.time()-t_start)}）", flush=True)

    # ---- 阶段二：Ollama 官方基线 ----
    if ollama_roles:
        banner(f"【Ollama 阶段】{ollama_roles}")
        for role in ollama_roles:
            run_role_block(role)

    # ---- 阶段三：云 API 基线（不占本机内存，所以放最后、且无内存顾虑）----
    if cloud_roles:
        banner(f"【云 API 阶段】{cloud_roles} —— 不占本机内存")
        for role in cloud_roles:
            cc = CLOUD[role]
            key_env = cc.get("api_key_env", "COPILOT_PROVIDER_API_KEY")
            api_key = os.environ.get(key_env, "")
            err = cloud_health(cc["base_url"], cc["model"], api_key,
                               timeout=int(cc.get("timeout", 300)))
            if err:
                print(f"  ✗ 云角色 {role} 探活失败：{err}")
                print(f"    （检查环境变量 {key_env} 是否注入、base_url 是否正确）")
                continue
            print(f"  ✓ {role} 探活通过", flush=True)
            run_role_block(role)

    # ---- 汇总 ----
    rows = read_jsonl(raw_path)
    probe = read_jsonl(lab_root / ecfg["general_probe_file"])
    metrics = aggregate(rows, roles, eval_set, probe, main_kind="main")
    metrics["ollama_baselines"] = OLLAMA
    metrics["cloud_baselines"] = {r: {k: v for k, v in CLOUD[r].items()
                                      if not k.startswith("$")}
                                  for r in CLOUD}
    metrics["local_roles"] = local_roles
    metrics["bench_src"] = bench_src
    # think 开启的对照结果单独汇总
    metrics["think"] = aggregate(rows, roles, eval_set, probe,
                                 main_kind="thinkon")["models"]
    out = lab_path(cfg, "results") / "eval_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    banner("汇总 · 主测（think=关）")
    layer_cols = [l for l in ("L1", "L2", "L3")
                  if any((m.get("by_layer") or {}).get(l)
                         for m in metrics["models"].values())]
    hdr = "".join(f"{l:>8s}" for l in layer_cols)
    print(f"  {'角色':<10s} {'格式':>7s} {'正确率':>7s}{hdr}  {'95% CI':>15s} {'样例':>6s}")
    for role in roles:
        m = metrics["models"].get(role)
        if not m:
            continue
        cells = []
        for l in layer_cols:
            bl = (m.get("by_layer") or {}).get(l)
            cells.append(f"{bl['accuracy']*100:>7.1f}%" if bl else f"{'—':>8s}")
        ci = m.get("accuracy_ci") or [0.0, 0.0]
        print(f"  {role:<10s} {m['format_rate']*100:>6.1f}% {m['accuracy']*100:>6.1f}%"
              + "".join(cells)
              + f"  [{ci[0]*100:>5.1f},{ci[1]*100:>5.1f}] {m['n_main']:>6d}")

    pw = metrics.get("pairwise") or {}
    if pw:
        banner("配对显著性（McNemar 精确检验）")
        focus = ("A1_vs_B", "C_vs_A2", "B_vs_teacher", "A1_vs_teacher", "C_vs_B",
                 "A1_vs_A2")
        for k in focus:
            v = pw.get(k)
            if not v:
                continue
            mark = "★" if v["significant"] else " "
            pv = v["p_value"]
            pv_s = "<0.0001" if pv < 0.0001 else f"{pv:.4f}"
            print(f"  {mark} {k:<18s} 仅前者 {v['only_a']:>3d} / 仅后者 {v['only_b']:>3d}"
                  f"  共 {v['n_common']:>3d} 题  p={pv_s:<8s}"
                  f"{'显著' if v['significant'] else '不显著'}")

    if metrics.get("think"):
        banner("对照 · think=开（小题量探针）")
        print(f"  {'角色':<10s} {'格式遵从':>8s} {'正确率':>8s} {'推理字数':>9s} {'耗时(s)':>9s}")
        for role in roles:
            m = metrics["think"].get(role)
            if not m:
                continue
            print(f"  {role:<10s} {m['format_rate']*100:>7.1f}% {m['accuracy']*100:>7.1f}% "
                  f"{m.get('avg_thinking_chars', 0):>9.0f} {m['avg_latency_s']:>9.1f}")
    print(f"\n  明细 {out}")
    print(f"  原始 {raw_path}")
    print(f"  总用时 {human_time(time.time()-t_start)}")
    return 0


def reuse_screen_samples(cfg, eval_set, raw_path, roles_wanted, max_rep,
                         verbose=True):
    """把 12_screen_layers.py 的筛查采样（screen_raw.jsonl）导入成评测格式。

    为什么要复用：base 与 teacher 在**同一批基准题**上已经各有 K=5 次采样，
    而评测只需要 `reps` 次 —— 直接取前 reps 条即可，省掉两个角色约 5 小时。
    判分口径完全一致（都用 common.parse_response / answer_equal），
    温度也都来自 config.eval.sample_temperature，所以这不是"将就"而是等价数据。
    """
    src = lab_path(cfg, "lab_root") / "shared" / "screen_raw.jsonl"
    if not src.exists():
        return 0
    bench = {e["id"]: e for e in eval_set if e.get("layer")}
    if not bench:
        return 0
    existing = {(r["role"], r["id"], r["kind"], r.get("rep", 0))
                for r in read_jsonl(raw_path)}
    n = 0
    with open(raw_path, "a", encoding="utf-8") as f:
        for r in read_jsonl(src):
            if r["model"] not in roles_wanted:
                continue
            s = int(r["sample"])
            if s >= max_rep:
                continue
            it = bench.get(r["qid"])
            if not it:
                continue
            key = (r["model"], r["qid"], "main", s)
            if key in existing:
                continue
            if not (r.get("response") or "").strip():
                continue        # 空回答是失败样本，绝不复用
            p = parse_response(r["response"])
            rec = {
                "role": r["model"], "id": r["qid"], "kind": "main", "rep": s,
                "think": False,
                "category": it.get("category", "general"),
                "layer": it.get("layer"),
                "prompt": it["prompt"], "gold": it.get("answer"),
                "response": r["response"],
                "format_ok": bool(r["format_ok"]),
                "answer_norm": r.get("answer_norm", ""),
                "correct": bool(r["correct"]),
                "n_steps": len(p["steps"]),
                "thinking_chars": p["thinking_chars"],
                "tokens": None, "latency_s": 0.0,
                "temperature": float(cfg["eval"].get("sample_temperature", 0.7)),
                "from_screen": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing.add(key)
            n += 1
    if verbose and n:
        print(f"  已复用筛查采样 {n} 条（角色 {sorted(roles_wanted)}）")
    return n


def _bootstrap_ci(vals, n_boot=2000, seed=12345):
    """逐题得分的 bootstrap 95% 置信区间（纯标准库，不引 numpy）。

    为什么不用正态近似：题目数只有几十，逐题得分是 0~1 的有界量，
    正态近似给出的区间可能越过 [0,1]。bootstrap 没有这个毛病。
    """
    import random as _rnd
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(vals[0]), float(vals[0]))
    r = _rnd.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[r.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[max(0, int(0.025 * n_boot) - 1)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return (float(lo), float(hi))


def _mcnemar(b, c):
    """McNemar 精确检验（配对二分类）：b、c 是两个不一致格的计数。

    用途：判断「A 与 B 在同一批题上的对错差异」是不是真的。
    为什么必须配对：两个模型做的是同一套题，题目难度是共同变量；
    不配对会把「题目难」误算成「模型差」，这是横评里最常见的统计错误。
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def aggregate(rows, roles, eval_set, probe=None, main_kind="main"):
    """把原始结果汇总成指标表。main_kind 决定统计哪一批（main=think关 / thinkon=think开）。"""
    eval_ids = {e["id"] for e in eval_set}
    kw_map = {p["id"]: p.get("keywords", []) for p in (probe or [])}
    layer_of = {e["id"]: e.get("layer") for e in eval_set}
    teacher_txt = {}
    for r in rows:
        if r["role"] == "teacher" and r["kind"] == "main":
            teacher_txt[r["id"]] = r["response"]

    out = {"models": {}, "n_eval": len(eval_set), "main_kind": main_kind}
    per_item_all = {}
    for role in roles:
        main = [r for r in rows if r["role"] == role and r["kind"] == main_kind
                and r["id"] in eval_ids]
        # 通用能力探针只算一次，不重复计入 think 对照
        gen = ([r for r in rows if r["role"] == role and r["kind"] == "general"]
               if main_kind == "main" else [])

        if not main:
            continue
        n = len(main)
        fmt = sum(r["format_ok"] for r in main) / n
        acc = sum(r["correct"] for r in main) / n
        steps = sum(r["n_steps"] for r in main) / n
        lat = sum(r["latency_s"] for r in main) / n

        # 逐题均值 → 用于算标准差（衡量「多次采样取平均」到底降了多少误差）
        def _per_item_std(key):
            buckets = {}
            for r in main:
                buckets.setdefault(r["id"], []).append(float(r[key]))
            means = [sum(v) / len(v) for v in buckets.values()]
            if len(means) < 2:
                return 0.0
            m = sum(means) / len(means)
            return (sum((x - m) ** 2 for x in means) / (len(means) - 1)) ** 0.5

        acc_std = _per_item_std("correct")
        fmt_std = _per_item_std("format_ok")

        # 与 teacher 的一致性（按题配对，同题多采样取平均）
        cons = []
        if role != "teacher":
            for r in main:
                t = teacher_txt.get(r["id"])
                if t:
                    cons.append(similarity(r["response"], t))
        consistency = sum(cons) / len(cons) if cons else 1.0

        # 多样性：直接用主测的多次采样（同题内两两差异），不必再跑一轮
        by_id = {}
        for r in main:
            by_id.setdefault(r["id"], []).append(r["response"])
        ds = []
        for _, texts in by_id.items():
            if len(texts) < 2:
                continue
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    ds.append(1 - similarity(texts[i], texts[j]))
        diversity = sum(ds) / len(ds) if ds else 0.0

        # 通用能力：关键词命中率（无关键词的题不计分）
        gh = [keyword_hit(r["response"], kw_map.get(r["id"])) for r in gen]
        gh = [x for x in gh if x is not None]
        general = sum(gh) / len(gh) if gh else None

        # 按类别拆解
        by_cat = {}
        for r in main:
            d = by_cat.setdefault(r["category"], [0, 0, 0])
            d[0] += 1
            d[1] += int(r["format_ok"])
            d[2] += int(r["correct"])

        # ---- 按分层拆解（L1/L2/L3）----
        # 这是本实验的主指标：总分会掩盖关键差异 —— 蒸馏在 teacher 会做的题上领先，
        # 但在 teacher 系统性失败的题上，它从未见过正确解，就被结构性封顶。
        by_layer = {}
        for r in main:
            lay = layer_of.get(r["id"]) or "未分层"
            d = by_layer.setdefault(lay, [0, 0, 0])
            d[0] += 1
            d[1] += int(r["format_ok"])
            d[2] += int(r["correct"])

        # ---- 逐题得分（配对检验的基础）----
        buckets = {}
        for r in main:
            buckets.setdefault(r["id"], []).append(int(r["correct"]))
        per_item = {q: sum(v) / len(v) for q, v in buckets.items()}
        per_item_all[role] = per_item
        ci_lo, ci_hi = _bootstrap_ci(list(per_item.values()))

        out["models"][role] = {
            "n_main": n,
            "n_questions": len(buckets),
            "repeats": (n // max(len(buckets), 1)),
            "accuracy_ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "by_layer": {k: {"n": v[0], "format": v[1] / v[0],
                             "accuracy": v[2] / v[0]}
                         for k, v in sorted(by_layer.items())},
            "format_rate": round(fmt, 4),
            "format_std": round(fmt_std, 4),
            "accuracy": round(acc, 4),
            "accuracy_std": round(acc_std, 4),
            "consistency": round(consistency, 4),
            "diversity": round(diversity, 4),
            "avg_steps": round(steps, 2),
            "avg_latency_s": round(lat, 2),
            "avg_thinking_chars": round(
                sum(r.get("thinking_chars", 0) for r in main) / n, 1),
            "general_probe": round(general, 3) if general is not None else None,
            "by_category": {k: {"n": v[0], "format": v[1] / v[0],
                                "accuracy": v[2] / v[0]}
                            for k, v in sorted(by_cat.items())},
        }

    # ---- 配对显著性检验（McNemar 精确检验）----
    # 逐题多数投票二值化：该题 r 次采样中过半答对 → 记为「这题会」
    binz = {r: {q: (1 if v >= 0.5 else 0) for q, v in d.items()}
            for r, d in per_item_all.items()}
    pairwise = {}
    present = [r for r in roles if r in binz]
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ra, rb = present[i], present[j]
            common = set(binz[ra]) & set(binz[rb])
            only_a = sum(1 for q in common
                         if binz[ra][q] == 1 and binz[rb][q] == 0)
            only_b = sum(1 for q in common
                         if binz[ra][q] == 0 and binz[rb][q] == 1)
            pv = _mcnemar(only_a, only_b)
            pairwise[f"{ra}_vs_{rb}"] = {
                "n_common": len(common),
                "only_a": only_a,          # 只有前者会的题数
                "only_b": only_b,          # 只有后者会的题数
                "p_value": round(pv, 4),
                "significant": bool(pv < 0.05),
            }
    out["pairwise"] = pairwise
    out["per_item"] = {r: {q: round(v, 4) for q, v in sorted(d.items())}
                       for r, d in per_item_all.items()}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
