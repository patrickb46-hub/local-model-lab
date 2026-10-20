#!/usr/bin/env bash
# =============================================================================
# 03_make_copies.sh —— 从 base 复制出两份物理隔离的工作副本
#
# 为什么必须复制而不是共用？
#   两条支线（蒸馏 A / 强化学习 B）各自的 LoRA target、tokenizer 缓存、
#   生成时写入的文件都会落在 model 目录里。共用一份 base 会让两条线互相污染，
#   评测时也说不清"这个差异是谁带来的"。隔离是这份计划的地基。
#
# base/ 本身保持只读，永不改动。
# =============================================================================
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$LAB_ROOT/base"
A="$LAB_ROOT/branch-A-distill/model"
B="$LAB_ROOT/branch-B-rl/model"

echo "=================================================================="
echo "  03_make_copies.sh  ·  制作两份工作副本"
echo "=================================================================="

if [ ! -d "$BASE" ] || [ -z "$(ls -A "$BASE" 2>/dev/null)" ]; then
    echo "✗ base/ 不存在或为空，请先跑 02_download_base.py"
    exit 1
fi

WEIGHT_BYTES=$(find "$BASE" -maxdepth 1 -name "*.safetensors" -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
if [ "${WEIGHT_BYTES:-0}" -lt 1000000000 ]; then
    echo "✗ base/ 里的权重体积异常（$WEIGHT_BYTES 字节），先校验完整性"
    exit 1
fi

copy_into() {
    local dst="$1" name="$2"
    mkdir -p "$dst"
    echo "→ $name   $dst"
    # 排除下载缓存与半成品；其余原样复制
    rsync -a --delete \
        --exclude='.cache/' \
        --exclude='*.incomplete' \
        --exclude='*.lock' \
        "$BASE"/ "$dst"/
    echo "   文件 $(find "$dst" -maxdepth 1 -type f | wc -l | tr -d ' ') 个，$(
        du -sh "$dst" | awk '{print $1}')"
}

copy_into "$A" "支线 A · 蒸馏"
copy_into "$B" "支线 B · 强化学习"

# ---------------------------------------------------------------------------
# 校验：两份副本都必须能被 tokenizer 与 config 读到
# ---------------------------------------------------------------------------
PY="${MLAB_PYTHON:-$HOME/.workbuddy/binaries/python/envs/mlab/bin/python}"
if [ -x "$PY" ]; then
    echo
    echo "校验副本可加载性："
    "$PY" - "$A" "$B" <<'PYEOF'
import json, sys
from pathlib import Path
for d in sys.argv[1:]:
    p = Path(d)
    cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    n_tok = sum(1 for f in p.glob("tokenizer*.json"))
    w = sum(f.stat().st_size for f in p.glob("*.safetensors"))
    print(f"  {p.parent.name:18s} arch={cfg['architectures'][0]:38s} "
          f"权重={w/1e9:.2f}GB tokenizer文件={n_tok}")
PYEOF
else
    echo "（跳过加载校验：mlab 解释器不存在）"
fi

# ---------------------------------------------------------------------------
# 隔离确认：两份副本的 inode 不同，改动互不影响
# ---------------------------------------------------------------------------
echo
echo "隔离确认："
echo "  A/config.json inode = $(stat -f%i "$A/config.json")"
echo "  B/config.json inode = $(stat -f%i "$B/config.json")"
echo
echo "完成。base/ 保持只读，两份 model/ 可各自独立改动。"
