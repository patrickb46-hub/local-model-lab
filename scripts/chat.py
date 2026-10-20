#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat.py —— 直接和微调后的模型对话（不依赖 Ollama）

为什么需要它：
  本机是 Intel Mac，而 Ollama 0.35 在 macOS 上会把 **safetensors 格式**的模型
  交给 MLX runner（Apple Silicon 专用），于是加载时直接 panic：
      mlx runner failed: panic: mlx: [metal::Device] Unable to construct residency set.
  官方那些 GGUF 模型（qwen3.5:2b 等）不受影响，因为走的是 llama.cpp。
  所以这里直接用 transformers 加载，绕开 Ollama —— 这条路已在横评中验证可用。

用法：
  python scripts/chat.py -m A                     # 交互式对话（蒸馏模型）
  python scripts/chat.py -m B                     # 交互式对话（RL 模型）
  python scripts/chat.py -m base                  # 对照：未微调的 2B
  python scripts/chat.py -m A -p "解方程：3x+5=11，求 x"
  python scripts/chat.py -m A --think             # 打开思考模式
  python scripts/chat.py -m A --raw               # 不套用系统提示词（原始模型行为）

模型对应关系：
  base  → branch-A-distill/model            （原版 Qwen3.5-2B，未微调）
  A     → branch-A-distill/model + adapter  （蒸馏支线，横评 83.3%）
  B     → branch-B-rl/model      + adapter  （RL 支线，横评 68.3%）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import lab_path, load_config  # noqa: E402
import labmodel as lm  # noqa: E402

BANNER = "=" * 62


def main() -> int:
    ap = argparse.ArgumentParser(description="与微调后的模型对话")
    ap.add_argument("-m", "--model", default="A", choices=["base", "A", "B"],
                    help="base=未微调原版 / A=蒸馏 / B=强化学习")
    ap.add_argument("-p", "--prompt", default=None, help="单次提问后退出（不给则进交互模式）")
    ap.add_argument("--think", action="store_true", help="打开 thinking 模式")
    ap.add_argument("--raw", action="store_true", help="不套用项目的系统提示词")
    ap.add_argument("--max-new", type=int, default=400, help="最多生成多少 token")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                    help="默认用 config 的训练精度（本机 fp32 最快）")
    args = ap.parse_args()

    cfg = load_config()
    lab_root = lab_path(cfg, "lab_root")
    a_dir = lab_path(cfg, "branch_a")
    b_dir = lab_path(cfg, "branch_b")

    # ---- 组装：底座 + 可选适配器 ----
    if args.model == "B":
        model_path = str(b_dir / "model")
        adapter = b_dir / "adapter"
        label = "B · 强化学习 LoRA（横评正确率 68.3%）"
    else:
        model_path = str(a_dir / "model")
        adapter = a_dir / "adapter" if args.model == "A" else None
        label = ("A · 蒸馏 LoRA（横评正确率 83.3%）" if args.model == "A"
                 else "base · Qwen3.5-2B 原版未微调（横评正确率 58.3%）")

    dtype = args.dtype or cfg["base_model"].get("dtype", "float32")

    print(BANNER)
    print(f"  模型  {label}")
    print(f"  精度  {dtype}   thinking {'开' if args.think else '关'}   "
          f"最长生成 {args.max_new} token")
    print(BANNER)
    print("  正在加载（首次约 30 秒）…", flush=True)

    tok = lm.load_tokenizer(model_path, cfg["base_model"]["trust_remote_code"])
    model, info = lm.load_model(
        model_path, dtype=dtype,
        drop_vision=cfg["base_model"].get("text_only", True),
        trust_remote_code=cfg["base_model"]["trust_remote_code"])

    if adapter is not None and (adapter / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"  已挂载适配器 {adapter}")
    elif args.model in ("A", "B"):
        print(f"  ⚠ 未找到适配器（{adapter}）—— 将按未微调的底座回答")

    sys_prompt = cfg["task"]["system_prompt"]

    def ask(question: str) -> None:
        msgs = ([] if args.raw else [{"role": "system", "content": sys_prompt}])
        msgs.append({"role": "user", "content": question})
        print("\n  " + "-" * 58)
        print("  回答：", end="", flush=True)
        txt, ntok, sec = lm.generate_reply(
            model, tok, msgs, max_new_tokens=args.max_new,
            temperature=args.temperature, enable_thinking=args.think)
        print(txt)
        print("  " + "-" * 58)
        speed = ntok / sec if sec > 0 else 0
        print(f"  [{ntok} token / {sec:.1f}s = {speed:.2f} tok/s]\n")

    # ---- 单次提问 ----
    if args.prompt:
        ask(args.prompt)
        return 0

    # ---- 交互模式 ----
    print("\n  进入交互模式。直接输入题目，输入 :q 或 Ctrl-C 退出。")
    print("  提示：本机纯 CPU 推理约 1.3 tok/s，一条回答通常要等 30-90 秒。\n")
    while True:
        try:
            q = input("  你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见。")
            return 0
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            print("  再见。")
            return 0
        ask(q)


if __name__ == "__main__":
    raise SystemExit(main())
