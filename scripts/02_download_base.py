#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_download_base.py —— 下载 base 权重（Qwen3.5-2B）

网络实情（本机实测）：
  HuggingFace 官方直连  ✗ 超时/隧道 502
  hf-mirror.com         ✓ 0.6s，4.5 MB/s
  GitHub 直连            ✓ 通，但只有 3.5 KB/s（等于不可用）

所以脚本强制把 HF_ENDPOINT 指向 hf-mirror，并在启动前先探一次连通性，
免得傻等 8 秒超时。支持断点续传（重复执行会跳过已下好的文件）。

用法：
  python 02_download_base.py             # 下载（已存在则增量补齐）
  python 02_download_base.py --check     # 只校验完整性，不下载
  python 02_download_base.py --force     # 忽略已有，重新拉取
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    banner, ensure_dir, human_time, lab_path, load_config, write_jsonl,
)

# 权重文件必需的几个关键资产
REQUIRED = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]
VISION_HINTS = ["preprocessor_config.json", "video_preprocessor_config.json"]


def check_endpoint(url: str, timeout: int = 15) -> bool:
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except Exception:
        return False


def verify(base_dir: Path, repo_id: str) -> bool:
    """校验：关键资产齐全 + 权重单片体积合理。"""
    banner("校验 base 权重")
    ok = True
    for f in REQUIRED:
        p = base_dir / f
        if p.exists() and p.stat().st_size > 0:
            print(f"  ✓ {f:34s} {p.stat().st_size/1e6:9.2f} MB")
        else:
            print(f"  ✗ {f:34s} 缺失")
            ok = False

    shards = sorted(base_dir.glob("*.safetensors"))
    if shards:
        total = sum(s.stat().st_size for s in shards)
        for s in shards:
            print(f"  ✓ {s.name:34s} {s.stat().st_size/1e9:9.3f} GB")
        print(f"  权重合计 {total/1e9:.2f} GB（{len(shards)} 片）")
        if total < 1e9:
            print("  ✗ 权重体积明显偏小，疑似下载中断")
            ok = False
    else:
        print("  ✗ 未找到 *.safetensors")
        ok = False

    idx = base_dir / "model.safetensors.index.json"
    if idx.exists():
        if not shards:
            print("  ✗ 有 index 却无分片")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--size", choices=["2B", "4B"], default=None,
                    help="下载哪个尺寸（默认用 config 里启用的那个）")
    args = ap.parse_args()

    cfg = load_config()
    rt, bm = cfg["runtime"], cfg["base_model"]
    endpoint = rt["hf_endpoint"]

    # --size 时从 variants 里取仓库与落盘目录
    if args.size:
        v = (bm.get("variants") or {}).get(args.size)
        if not v:
            print(f"✗ config.base_model.variants 里没有 {args.size}")
            return 1
        repo_id, dirname = v["repo_id"], v["local_dir"]
    else:
        repo_id, dirname = bm["repo_id"], bm["local_dir"]

    base_dir = Path(dirname)
    if not base_dir.is_absolute():
        base_dir = lab_path(cfg, "lab_root") / dirname
    ensure_dir(base_dir)

    if args.check:
        return 0 if verify(base_dir, repo_id) else 1

    banner(f"下载 base 权重：{repo_id}")
    print(f"  镜像    {endpoint}")
    print(f"  落盘    {base_dir}")

    if not check_endpoint(endpoint):
        print(f"  ✗ {endpoint} 不可达，终止（可换 ModelScope）")
        return 1
    print("  连通性  ✓")

    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ✗ 缺少 huggingface_hub，请先跑 01_setup_env.sh")
        return 1

    import time
    t0 = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(base_dir),
        max_workers=4,
        force_download=args.force,
        # 只取训练必需的：跳过 .gitattributes 之类零碎，但保留 vision 配置
        ignore_patterns=[".gitattributes", "*.md", "*.h5", "*.msgpack", "*.onnx"],
    )
    print(f"\n  下载完成，用时 {human_time(time.time()-t0)}")
    print(f"  路径 {path}")

    ok = verify(base_dir, repo_id)

    # 记下 tokenizer / chat template 的指纹，避免"换了模型还不知道"
    manifest = lab_path(cfg, "results") / "base_manifest.json"
    ensure_dir(manifest.parent)
    info = {
        "repo_id": repo_id,
        "endpoint": endpoint,
        "files": sorted(p.name for p in base_dir.iterdir() if p.is_file()),
        "weight_gb": round(
            sum(s.stat().st_size for s in base_dir.glob("*.safetensors")) / 1e9, 3),
   }
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  指纹已写入 {manifest}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
