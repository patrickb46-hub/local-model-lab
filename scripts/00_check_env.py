#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_check_env.py —— 环境自检（阶段 0 的验收闸门）

全绿才允许进入阶段 1。任何一项红，先修，别急着开跑。
输出同时落到 shared/results/env_report.json，便于之后回溯"当时是什么环境"。

用法：
  python 00_check_env.py
  python 00_check_env.py --quick     # 跳过网络与磁盘的重活
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    banner, ensure_dir, lab_path, load_config, read_jsonl,
)

GREEN, RED, YEL, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
RESULTS: list = []


def rec(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    tag = f"{YEL}WARN{RST}" if (warn and not ok) else (f"{GREEN} OK {RST}" if ok else f"{RED}FAIL{RST}")
    print(f"  [{tag}] {name:28s} {detail}")
    RESULTS.append({"name": name, "ok": bool(ok), "warn": bool(warn and not ok),
                    "detail": detail})


def http_ok(url: str, timeout: int = 15) -> tuple:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 400, f"HTTP {r.status}  {time.time()-t0:.2f}s"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}  {time.time()-t0:.2f}s"


# ---------------------------------------------------------------------------

def check_python(cfg) -> None:
    banner("1. Python 运行时")
    v = sys.version_info
    rec("python 版本", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    want = cfg["runtime"]["python"]
    rec("解释器与 config 一致", Path(sys.executable).resolve() == Path(want).resolve(),
        f"config 期望 {want}", warn=True)
    if platform.machine() != "arm64":
        print(f"  [ 提示 ] 架构 {platform.machine()}（Intel）—— 无 MPS，纯 CPU 训练")


def check_pkgs() -> None:
    banner("2. 依赖栈")
    # 期望值与实际来源一致：torch 来自 conda-forge（PyPI 上 Intel Mac 停在 2.2.2，装不了 Qwen3.5）
    expect = {
        "torch": "2.13.0", "transformers": "5.18.0", "peft": "0.21.2",
        "trl": "1.14.1", "datasets": "5.0.1", "accelerate": "1.15.0",
        "safetensors": "0.8.0", "numpy": "2.5.3", "huggingface_hub": "1.33.0",
    }
    mods = {}
    for m in ["torch", "transformers", "peft", "trl", "datasets", "accelerate",
              "safetensors", "numpy", "huggingface_hub", "sentencepiece"]:
        try:
            mod = importlib.import_module(m)
            mods[m] = getattr(mod, "__version__", "?")
            rec(f"import {m}", True, mods[m])
        except Exception as e:  # noqa: BLE001
            rec(f"import {m}", False, str(e)[:70])
    for k, want in expect.items():
        if k in mods:
            rec(f"{k} 版本钉死", mods[k] == want or mods[k].split("+")[0] == want,
                f"实 {mods[k]}  期 {want}", warn=True)

    # 架构支持：transformers 认不认识 qwen3_5
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        import transformers
        keys = list(CONFIG_MAPPING.keys()) if hasattr(CONFIG_MAPPING, "keys") else []
        rec("transformers 支持 qwen3_5", "qwen3_5" in keys,
            f"CONFIG_MAPPING {len(keys)} 项")
    except Exception as e:  # noqa: BLE001
        rec("transformers 支持 qwen3_5", False, str(e)[:70])

    # torch / numpy 互操作
    try:
        import torch
        import numpy as np
        r = torch.tensor(np.array([1.0, 2.0])).sum().item()
        rec("torch×numpy 互操作", r == 3.0, f"sum={r}")
        rec("CPU 指令集能力", True, str(torch.backends.cpu.get_cpu_capability()))
        rec("线程数", True, str(torch.get_num_threads()))
    except Exception as e:  # noqa: BLE001
        rec("torch×numpy 互操作", False, str(e)[:70])

    # MPS：Intel Mac 上 is_available() 会误报 True，必须实跑一次才算数
    try:
        import torch
        flagged = torch.backends.mps.is_available()
        real = False
        detail = f"标志位={flagged}"
        if flagged:
            try:
                x = torch.randn(64, 64, device="mps")
                (x @ x).sum().item()
                real = True
                detail += "  实跑=可用"
            except Exception as e:  # noqa: BLE001
                detail += f"  实跑=不可用（{str(e)[:46]}）"
        rec("MPS 实测可用性", real, detail, warn=True)
        if not real:
            print("        → 纯 CPU 训练（Intel 独显无统一内存，MPS 用不了，这是预期）")
    except Exception as e:  # noqa: BLE001
        rec("MPS 实测可用性", False, str(e)[:70], warn=True)


def check_ollama(cfg) -> None:
    banner("3. Ollama teacher")
    from common import ollama_health
    url = cfg["teacher"]["ollama_url"]
    models = ollama_health(url)
    if models is None:
        rec("Ollama 服务", False, f"{url} 无响应")
        return
    rec("Ollama 服务", True, f"{url}  模型 {len(models)} 个")
    want = {cfg["teacher"]["model"], cfg["teacher"]["fallback_model"],
            "qwen3.5:2b"}
    for m in sorted(want):
        rec(f"模型 {m}", m in models, "已就位" if m in models else f"缺失（现有 {models}）")


def check_network(cfg, quick: bool) -> None:
    banner("4. 网络可达性")
    if quick:
        print("  （--quick 跳过）")
        return
    rt = cfg["runtime"]
    for name, url, must in [
        ("hf-mirror（下载源）", rt["hf_endpoint"], True),
        ("PyPI 镜像（装包）", rt["pip_index"], True),
        ("ModelScope（备用）", "https://www.modelscope.cn", False),
        ("HuggingFace 直连（预期不通）", "https://huggingface.co", False),
    ]:
        ok, detail = http_ok(url)
        if "预期不通" in name:
            rec(name, ok, detail, warn=True)
        else:
            rec(name, ok, detail)


def check_disk_mem() -> None:
    banner("5. 磁盘与内存")
    lab = Path("/Users") if Path("/Users").exists() else Path("/")
    du = shutil.disk_usage("/")
    free_gb = du.free / 1e9
    rec("可用磁盘", free_gb > 20, f"{free_gb:.1f} GB（需 ≥ 20 GB）")
    try:
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        rec("物理内存", True, f"{total/1e9:.1f} GB")
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page = 4096
        vals = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    vals[k.strip()] = int(v)
        avail = (vals.get("Pages free", 0) + vals.get("Pages inactive", 0)
                 + vals.get("Pages speculative", 0)) * page / 1e9
        rec("当前可用内存", avail > 3.0, f"{avail:.1f} GB", warn=True)
    except Exception:  # noqa: BLE001
        pass


def check_layout(cfg) -> None:
    banner("6. 目录与文件")
    root = lab_path(cfg, "lab_root")
    for key in ["base", "branch_a", "branch_b", "shared", "results"]:
        p = lab_path(cfg, key)
        rec(f"目录 {p.name}", p.exists(), str(p))
    for sub in ["model", "data", "adapter", "logs"]:
        for br in ["branch_a", "branch_b"]:
            p = lab_path(cfg, br) / sub
            rec(f"  {lab_path(cfg, br).name}/{sub}", p.exists(), "")

    # base 与两份副本
    base = lab_path(cfg, "base")
    w = sum(f.stat().st_size for f in base.glob("*.safetensors"))
    rec("base 权重", w > 1e9, f"{w/1e9:.2f} GB")
    for br, label in [("branch_a", "支线A 副本"), ("branch_b", "支线B 副本")]:
        d = lab_path(cfg, br) / "model"
        w2 = sum(f.stat().st_size for f in d.glob("*.safetensors"))
        rec(label, w2 > 1e9, f"{w2/1e9:.2f} GB")

    # 隔离性：三份必须是不同的文件（inode 不同）
    try:
        ib = (base / "config.json").stat().st_ino
        ia = (lab_path(cfg, "branch_a") / "model" / "config.json").stat().st_ino
        ib2 = (lab_path(cfg, "branch_b") / "model" / "config.json").stat().st_ino
        rec("三条线物理隔离", len({ib, ia, ib2}) == 3, f"inode {ib}/{ia}/{ib2}", warn=True)
    except Exception as e:  # noqa: BLE001
        rec("三条线物理隔离", False, str(e)[:60])


def check_data(cfg) -> None:
    banner("7. 数据与纪律")
    root = lab_path(cfg, "lab_root")
    ev = root / cfg["data"]["eval_file"]
    a_p = lab_path(cfg, "branch_a") / "data" / "prompts.jsonl"
    b_p = lab_path(cfg, "branch_b") / "data" / "prompts.jsonl"

    ev_rows = read_jsonl(ev)
    rec("eval_set 已固化", len(ev_rows) >= 10, f"{len(ev_rows)} 条  {ev.name}")

    a_rows, b_rows = read_jsonl(a_p), read_jsonl(b_p)
    rec("蒸馏题库", len(a_rows) > 0, f"{len(a_rows)} 条")
    rec("RL 题库", len(b_rows) > 0, f"{len(b_rows)} 条")

    if ev_rows and a_rows and b_rows:
        ie = {r["id"] for r in ev_rows}
        ia = {r["id"] for r in a_rows}
        ib = {r["id"] for r in b_rows}
        rec("eval ∩ 蒸馏 = ∅", not (ie & ia), f"重叠 {len(ie & ia)} 条")
        rec("eval ∩ RL = ∅", not (ie & ib), f"重叠 {len(ie & ib)} 条")
        rec("蒸馏 ∩ RL = ∅", not (ia & ib), f"重叠 {len(ia & ib)} 条")

    # 判分器自洽：标准答案必须能被自己认出
    from common import answer_equal
    bad = [r for r in ev_rows if not answer_equal(r["answer"], r["answer"])]
    rec("判分器对标准答案自洽", not bad, f"{len(ev_rows)-len(bad)}/{len(ev_rows)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    print("=" * 64)
    print("  本地模型实验室 · 环境自检")
    print("=" * 64)
    print(f"  实验室根目录 {lab_path(cfg, 'lab_root')}")
    print(f"  时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    check_python(cfg)
    check_pkgs()
    check_ollama(cfg)
    check_network(cfg, args.quick)
    check_disk_mem()
    check_layout(cfg)
    check_data(cfg)

    fails = [r for r in RESULTS if not r["ok"] and not r["warn"]]
    warns = [r for r in RESULTS if not r["ok"] and r["warn"]]

    banner("结论")
    print(f"  通过 {len(RESULTS)-len(fails)-len(warns)} / {len(RESULTS)}"
          f"   警告 {len(warns)}   失败 {len(fails)}")
    for r in fails:
        print(f"    ✗ {r['name']}: {r['detail']}")
    for r in warns:
        print(f"    ! {r['name']}: {r['detail']}")

    out = lab_path(cfg, "results") / "env_report.json"
    ensure_dir(out.parent)
    out.write_text(json.dumps({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ok": not fails, "checks": RESULTS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告 {out}")

    if fails:
        print(f"\n{RED}阶段 0 未通过，先解决上面 ✗ 的项。{RST}")
        return 1
    print(f"\n{GREEN}阶段 0 通过 —— 可以进入阶段 1（蒸馏支线）。{RST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
