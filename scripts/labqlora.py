#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labqlora.py —— 4-bit QLoRA 支持（HQQ 后端）

为什么需要它：Qwen3.5-4B 是 4.38B 参数，fp32 权重 17.5 GB > 本机 16 GB 物理内存，
而 bf16 反正在本机病态慢 200 倍。唯一的活路是把**冻结底座压到 4-bit**，
只训 fp32 的 LoRA 旁挂参数。

为什么是 HQQ 而不是 bitsandbytes：bitsandbytes 的 macOS 轮子**全是 arm64**，
Intel Mac（x86_64）一个都没有。HQQ 是纯 PyTorch 实现，免编译。

两个必须记住的工程细节：
  * **compute_dtype 必须 float32** —— 见 docs/踩坑记录.md 坑 9（bf16 反向病态）
  * **量化很慢**（实测单个 2560×9216 的层要 11 秒），所以有缓存机制；
    2B 约 10–25 分钟，4B 约 30–60 分钟，一次性的。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 不量化的层：输出头与嵌入共享权重；视觉塔我们不使用
DEFAULT_SKIP = ("lm_head", "visual", "vision", "mtp", "embed_tokens")


def replace_linear_with_hqq(module, qcfg, compute_dtype, device: str,
                            prefix: str = "", skip: Tuple[str, ...] = DEFAULT_SKIP,
                            verbose: bool = True) -> Tuple[int, List[str]]:
    """递归把 nn.Linear 换成 HQQLinear。返回 (替换数, 被跳过的模块名)。"""
    import torch
    from hqq.core.quantize import HQQLinear

    n, skipped = 0, []
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            if any(s in full.lower() for s in skip):
                skipped.append(full)
                continue
            setattr(module, name,
                    HQQLinear(child, qcfg, compute_dtype=compute_dtype, device=device))
            n += 1
            if verbose and n % 25 == 0:
                print(f"    已量化 {n} 层…（最近 {full}）", flush=True)
        else:
            sub_n, sub_sk = replace_linear_with_hqq(
                child, qcfg, compute_dtype, device, full, skip, verbose)
            n += sub_n
            skipped.extend(sub_sk)
    return n, skipped


def quantize_model(model, nbits: int = 4, group_size: int = 128,
                   compute_dtype: str = "float32", device: str = "cpu",
                   skip: Tuple[str, ...] = DEFAULT_SKIP,
                   verbose: bool = True) -> Dict[str, Any]:
    """把模型的线性层量化到 nbits 位。返回统计信息。"""
    import torch
    from hqq.core.quantize import BaseQuantizeConfig

    dt = {"float32": torch.float32, "fp32": torch.float32,
          "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
              str(compute_dtype).lower(), torch.float32)

    qcfg = BaseQuantizeConfig(nbits=int(nbits), group_size=int(group_size),
                              quant_zero=False, quant_scale=False)
    if verbose:
        print(f"  [hqq] {nbits}-bit  group_size={group_size}  compute_dtype={compute_dtype}")
        print(f"  [hqq] 开始量化（很慢，请耐心）…", flush=True)
    t0 = time.time()
    n, skipped = replace_linear_with_hqq(model, qcfg, dt, device, verbose=verbose)
    dt_s = time.time() - t0
    if verbose:
        print(f"  [hqq] 完成：量化 {n} 层，跳过 {len(skipped)} 层，用时 {dt_s:.1f}s")
    return {"quantized_layers": n, "skipped": skipped,
            "seconds": round(dt_s, 1), "nbits": nbits, "group_size": group_size}


def attach_lora(model, lora_cfg: Dict[str, Any], target_suffixes: List[str],
                verbose: bool = True) -> int:
    """用 HQQ 自带的 PEFT 集成挂 LoRA。返回可训练参数量。"""
    from hqq.core import peft as HQQPEFT

    peft_config = {
        "target_modules": list(target_suffixes),
        "r": int(lora_cfg.get("r", 16)),
        "lora_alpha": int(lora_cfg.get("alpha", 32)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
        "trainable": True,
    }
    if verbose:
        print(f"  [hqq-lora] target={target_suffixes}  r={peft_config['r']}")
    HQQPEFT.PeftUtils.add_lora(model, peft_config)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  [hqq-lora] 可训练参数 {n/1e6:.2f}M")
    return n


# ---------------------------------------------------------------------------
# 量化结果缓存（避免每次重跑都要等半小时）
# ---------------------------------------------------------------------------

def save_quantized(model, path: Path, meta: Optional[Dict] = None) -> bool:
    """把量化后的权重存盘。失败不致命，只是下次还得重量化。"""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(model.state_dict(), path)
        if meta:
            path.with_suffix(".meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [hqq] 量化结果存盘失败（不影响本次运行）：{type(e).__name__}: {e}")
        return False


def try_load_quantized(model, path: Path) -> bool:
    """尝试载入缓存。结构不匹配就返回 False，让调用方重量化。"""
    import torch
    path = Path(path)
    if not path.exists():
        return False
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # 允许少量缺失（如被跳过的层），但量化核心参数必须在
        if unexpected:
            print(f"  [hqq] 缓存有多余键 {len(unexpected)} 个，放弃使用缓存")
            return False
        print(f"  [hqq] 已从缓存载入量化权重：{path}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [hqq] 缓存载入失败（将重新量化）：{type(e).__name__}: {e}")
        return False


def prepare_qlora(model, q_cfg: Dict[str, Any], lora_cfg: Dict[str, Any],
                  verbose: bool = True) -> Tuple[Any, int, Dict[str, Any]]:
    """
    一步到位把模型变成可训练的 QLoRA 形态。**顺序至关重要**：

      1. 先在【原始】模型上探测 LoRA 目标后缀
         —— 一旦量化完成，nn.Linear 全被换成 HQQLinear，就探测不到了
      2. 再量化（跳过 lm_head / 视觉塔 等）
      3. 最后挂 LoRA

    返回 (model, 可训练参数量, 统计信息)
    """
    import labmodel as lm

    # 1. 探测（必须在量化之前）
    suffixes, full_names = lm.discover_linear_targets(model)
    if verbose:
        print(f"  [qlora] 量化前探测到 {len(full_names)} 个 Linear，后缀 {suffixes}")

    # 2. 量化
    stats = quantize_model(
        model,
        nbits=q_cfg.get("nbits", 4),
        group_size=q_cfg.get("group_size", 128),
        compute_dtype=q_cfg.get("compute_dtype", "float32"),
        device=q_cfg.get("device", "cpu"),
        verbose=verbose,
    )

    # 3. 挂 LoRA（只挂量化时没被跳过的那些后缀）
    skipped_suffix = {s.split(".")[-1] for s in stats["skipped"]}
    targets = [s for s in suffixes if s not in skipped_suffix] or suffixes
    n = attach_lora(model, lora_cfg, targets, verbose=verbose)

    stats["lora_targets"] = targets
    return model, n, stats


def save_lora(model, out_dir: Path, verbose: bool = True) -> None:
    """保存 LoRA 权重（HQQ 的格式）。"""
    from hqq.core import peft as HQQPEFT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  [hqq-lora] 保存到 {out_dir}")
    HQQPEFT.PeftUtils.save_lora_weights(model, save_dir=str(out_dir))
