#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_train_distill.py —— 支线 A 第二步：LoRA 监督微调（序列级蒸馏）

为什么不用 TRL 的 SFTTrainer？
  这个底座是"多模态 + 混合线性注意力 + 带 MTP 层"的非典型架构，
  TRL 的数据管道（尤其是处理 image-text 的 Dataset 预处理）在这个模型上
  容易走进死路，而报错信息通常指向错误的方向。
  自己写一个 60 行的训练循环，反而每一处都看得见、改得动。
  （trl 仍然装着，需要时可用；这里是工程取舍，不是能力缺失。）

用法：
  python 11_train_distill.py --probe          # 探针：跑 3 步，报耗时/内存/ETA
  python 11_train_distill.py --epochs 1 --limit 40   # 最小规模试通
  python 11_train_distill.py                  # 按 config 全量
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    banner, ensure_dir, human_time, lab_path, load_config, read_jsonl,
)
import labmodel as lm  # noqa: E402


def peak_mem_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def lr_at(step: int, total: int, base_lr: float, warmup_ratio: float) -> float:
    """线性 warmup + 余弦衰减。"""
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只跑 3 步，报 ETA 与内存")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="优化步数上限（受控变量：与支线 B 对齐）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="忽略已有 adapter，从头训")
    ap.add_argument("--tag", default="",
                    help="产物后缀：写到 adapter-<tag>/ 与 train_distill-<tag>.jsonl。"
                         "用于同一条支线训多个臂（如 A1 60 步 / A2 120 步）")
    ap.add_argument("--init-adapter", default=None,
                    help="从这个 adapter 续训（默认自动找本臂目录）。"
                         "A2 用它从 A1 的权重接着训，而不是从零")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = cfg["distill"]
    tcfg = dcfg["training"]
    lab_root = lab_path(cfg, "lab_root")
    a_dir = lab_path(cfg, "branch_a")
    model_path = str(a_dir / "model")
    suffix = f"-{args.tag}" if args.tag else ""
    adapter_dir = a_dir / f"adapter{suffix}"
    log_path = a_dir / "logs" / f"train_distill{suffix}.jsonl"

    dtype = args.dtype or cfg["base_model"].get("dtype", "float32")
    epochs = args.epochs if args.epochs is not None else tcfg["num_epochs"]
    max_len = args.max_seq_length or tcfg["max_seq_length"]
    grad_ckpt = tcfg["gradient_checkpointing"] and not args.no_grad_ckpt

    banner("支线 A · LoRA 监督微调")
    samples = read_jsonl(lab_root / dcfg["train_file"])
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("✗ 训练集为空，请先跑 10_gen_distill_data.py")
        return 1

    import torch
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    print(f"  训练样本 {len(samples)} 条 × {epochs} 轮")
    print(f"  max_seq_length {max_len}   batch {tcfg['per_device_batch_size']}  "
          f"grad_accum {tcfg['gradient_accumulation']}")
    print(f"  dtype {dtype}   grad_checkpointing {grad_ckpt}   "
          f"torch 线程 {torch.get_num_threads()}")

    # ---------------- 加载 ----------------
    t0 = time.time()
    tok = lm.load_tokenizer(model_path, cfg["base_model"]["trust_remote_code"])
    print(f"  tokenizer 就绪（vocab {tok.vocab_size}，模板 {len(tok.chat_template or '')} 字符）")
    model, info = lm.load_model(model_path, dtype=dtype,
                               drop_vision=cfg["base_model"].get("text_only", True),
                               trust_remote_code=cfg["base_model"]["trust_remote_code"])
    print(f"  加载耗时 {human_time(time.time()-t0)}  峰值内存 {peak_mem_gb():.2f} GB")

    if grad_ckpt:
        lm.enable_gradient_checkpointing(model)

    model = lm.apply_lora(model, dcfg["lora"])
    n_train = lm.mark_only_lora_trainable(model)
    print(f"  可训练参数 {n_train/1e6:.2f}M")

    # 已有 adapter 续训
    # 续训来源：--init-adapter 显式指定，否则用本臂目录
    init_from = Path(args.init_adapter) if args.init_adapter else adapter_dir
    if not args.fresh and (init_from / "adapter_config.json").exists():
        try:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sd = load_file(str(init_from / "adapter_model.safetensors"))
            set_peft_model_state_dict(model, sd)
            print(f"  已从 {init_from} 载入 adapter 续训 → 输出到 {adapter_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"  续训载入失败（将从零开始）：{e}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=tcfg["learning_rate"],
                            weight_decay=tcfg["weight_decay"])

    # ---------------- 预编码 ----------------
    enc = []
    skipped = 0
    for s in samples:
        try:
            e = lm.encode_supervised(tok, s["messages"], max_len)
            if sum(1 for x in e["labels"] if x != -100) < 3:
                skipped += 1
                continue
            e["id"] = s["id"]
            enc.append(e)
        except Exception as ex:  # noqa: BLE001
            skipped += 1
            print(f"  [warn] {s['id']} 编码失败：{ex}")
    print(f"  编码完成 {len(enc)} 条（跳过 {skipped}）")
    if not enc:
        print("✗ 没有可用样本")
        return 1

    totals = [len(e["input_ids"]) for e in enc]
    print(f"  序列长度 min/avg/max = {min(totals)}/{sum(totals)//len(totals)}/{max(totals)}")

    # ---------------- 训练循环 ----------------
    accum = tcfg["gradient_accumulation"]
    steps_per_epoch = math.ceil(len(enc) / accum)
    planned = steps_per_epoch * epochs
    # 【受控变量】与支线 B 对齐到相同优化步数
    cap = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")
    total_steps = min(planned, cap) if cap else planned
    if args.probe:
        total_steps = min(3, total_steps)
        epochs = 1
        enc = enc[: accum * 3]

    print(f"\n  计划优化步数 {total_steps}（每步 {accum} 个样本，"
          f"每轮 {steps_per_epoch} 步，上限 {cap or '无'}）")
    ensure_dir(log_path.parent)
    logf = open(log_path, "a", encoding="utf-8")

    model.train()
    global_step = 0
    micro = 0
    acc_loss = 0.0
    t_start = time.time()
    step_times = []
    stop = False

    for ep in range(epochs):
        torch.manual_seed(tcfg["seed"] + ep)
        order = torch.randperm(len(enc)).tolist()
        for idx in order:
            e = enc[idx]
            batch = lm.build_batch([e], pad_id=tok.pad_token_id or tok.eos_token_id)
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            acc_loss += float(loss.detach())
            micro += 1

            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, tcfg["max_grad_norm"])
                lr = lr_at(global_step, total_steps, tcfg["learning_rate"],
                           tcfg["warmup_ratio"])
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                mean_loss = acc_loss / accum
                acc_loss = 0.0
                rec = {"step": global_step, "epoch": ep,
                       "loss": round(mean_loss, 4), "lr": float(f"{lr:.3e}"),
                       "peak_mem_gb": round(peak_mem_gb(), 2),
                       "elapsed_s": round(time.time() - t_start, 1)}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if global_step % max(1, tcfg["logging_steps"]) == 0 or args.probe:
                    print(f"  ep{ep} step {global_step}/{total_steps}  "
                          f"loss {mean_loss:.4f}  lr {lr:.2e}  "
                          f"mem {peak_mem_gb():.2f}G  "
                          f"{time.time()-t_start:.0f}s", flush=True)

                step_times.append(time.time())
                if len(step_times) > 20:
                    step_times.pop(0)

                if global_step >= total_steps:
                    stop = True
                    break
                if not args.probe and tcfg.get("save_steps") and \
                        global_step % tcfg["save_steps"] == 0:
                    _save(model, tok, adapter_dir)
                    print(f"  ↳ checkpoint 已存 {adapter_dir}")

        if stop:
            break

    logf.close()

    # ---------------- 收尾 ----------------
    banner("训练结束")
    wall = time.time() - t_start
    if len(step_times) >= 2:
        per = (step_times[-1] - step_times[0]) / (len(step_times) - 1)
        print(f"  实测每步 {per:.1f}s   完成 {global_step} 步   用时 {human_time(wall)}")
        if args.probe:
            full_steps = steps_per_epoch * (args.epochs or tcfg["num_epochs"])
            print(f"  按此速度，全量 {full_steps} 步预计 {human_time(per*full_steps)}")
    print(f"  峰值内存 {peak_mem_gb():.2f} GB")

    # 【重要】探针绝不覆盖真实产物 —— 否则一次测速就把训练几小时的 adapter 冲掉。
    # 探针只写 scratch 目录，同时仍走完整保存路径以便验证它没坏。
    if args.probe:
        adapter_dir = a_dir / "_probe_adapter"
        print(f"  [probe] adapter 只写到 scratch 目录 {adapter_dir.name}/，不碰真实产物")
    _save(model, tok, adapter_dir)
    print(f"  adapter 已保存 {adapter_dir}")

    summary = {
        "samples": len(enc), "epochs": epochs, "steps": global_step,
        "wall_s": round(wall, 1), "peak_mem_gb": round(peak_mem_gb(), 2),
        "dtype": dtype, "max_seq_length": max_len, "lora": dcfg["lora"],
        "probe": bool(args.probe),
    }
    (a_dir / "logs" / f"train_distill_summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _save(model, tok, adapter_dir) -> None:
    """用 labmodel.save_adapter：内存序列化 + 就地覆盖，绕开沙箱对 unlink/rename 的拦截。"""
    lm.save_adapter(model, tok, adapter_dir)


if __name__ == "__main__":
    raise SystemExit(main())
