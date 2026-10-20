#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# demo_run.sh —— 演示录屏的统一入口（薄封装）
#
# 真正的逻辑全在 demo_board.py 里：同一个进程既起训练子进程、又画看板。
# 这样没有僵尸进程、没有信号转发、没有「看板比训练活得久」的时序竞争。
# 这个文件只负责找到解释器、转发参数 —— 保持「敲一条命令」的观感。
#
# 用法：
#   bash scripts/demo_run.sh A                  # 真跑支线 A（演示模式，4 步）
#   bash scripts/demo_run.sh B                  # 真跑支线 B（2 步）
#   bash scripts/demo_run.sh C                  # 真跑支线 C（2 步，预热自 adapter-A1）
#   bash scripts/demo_run.sh A --dry-run        # 只看将执行的完整命令
#   bash scripts/demo_run.sh A --steps 6        # 改步数
#   bash scripts/demo_run.sh A --intro 0        # 跳过开场声明
#   bash scripts/demo_run.sh --all              # 依次跑 A→B→C（约 34 分钟）
#   bash scripts/demo_run.sh --summary A        # 只打印上次运行的摘要
#
# 产物一律落在 adapter-DEMO-*/ 与 *_DEMO-*.jsonl，绝不触碰 adapter-A1/B/C。
set -u

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LAB_ROOT

PY_SYS=/usr/bin/python3
BOARD="$LAB_ROOT/scripts/demo_board.py"

if [ ! -f "$BOARD" ]; then
    printf '✗ 找不到 %s\n' "$BOARD" >&2
    exit 2
fi

# 第一个参数是分支名 → 补 --run；是选项或为空 → 原样转发
case "${1:-}" in
    ""|-*) exec "$PY_SYS" "$BOARD" "$@" ;;
    *)     exec "$PY_SYS" "$BOARD" --run "$@" ;;
esac
