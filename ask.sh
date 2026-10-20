#!/usr/bin/env bash
# =============================================================================
# ask.sh —— 一句话向练好的模型提问（完整封装：预检 + 内存自适应 + 友好报错）
#
# 用法：
#   bash ask.sh A "解方程：3x + 5 = 11，求 x 的值。"
#   bash ask.sh B "..."                ← 换成强化学习模型
#   bash ask.sh base "..."             ← 换成未微调的原版（对照用）
#   bash ask.sh A                      ← 不带题目 = 进入交互模式，可连续问
#   bash ask.sh A -t "..."             ← 加 -t 打开思考模式（慢很多）
#   bash ask.sh --info                 ← 只看环境自检，不跑模型
#
# 说明：
#   * 所有路径都是绝对路径，**在任何目录下运行都可以**（不依赖 cd）。
#   * 会自动探测可用内存：不足 8 GB 时改用 bfloat16（省内存，生成略慢）。
#   * 首次加载模型约 20-30 秒，之后每条回答约 40-90 秒 —— 纯 CPU 的物理速度，非脚本问题。
# =============================================================================
set -uo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python"
CHAT="$LAB/scripts/chat.py"

MODEL="A"
THINK=""
PROMPT=""
EXTRA_DTYPE=""
INFO=0

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)     MODEL="${2:-A}"; shift 2 ;;
        -t|--think)     THINK="--think"; shift ;;
        base|Base|BASE) MODEL="base"; shift ;;
        A|a)            MODEL="A"; shift ;;
        B|b)            MODEL="B"; shift ;;
        --info|--check) INFO=1; shift ;;
        -h|--help)      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              PROMPT="$1"; shift ;;
    esac
done

# ── 可用内存（GB）───────────────────────────────────────────────────────────
mem_avail_gb() {
    vm_stat 2>/dev/null | awk '
        /Pages free/        {gsub(/\./,"",$3); f=$3}
        /Pages inactive/    {gsub(/\./,"",$3); i=$3}
        /Pages speculative/ {gsub(/\./,"",$3); s=$3}
        END {printf "%.1f", (f+i+s)*4096/1e9}'
}

fail() { echo "✗ $*" >&2; exit 1; }

# ── 预检 ───────────────────────────────────────────────────────────────────
[ -x "$PY" ]   || fail "找不到解释器：$PY
  这是训练专用的 Python 环境。若已被删除，需重建（见 README 第二节）。"
[ -f "$CHAT" ] || fail "找不到对话脚本：$CHAT"

case "$MODEL" in
    base) MDIR="$LAB/base" ;;
    A)    MDIR="$LAB/branch-A-distill/model"
          [ -d "$LAB/branch-A-distill/adapter" ] || fail "A 的适配器缺失：$LAB/branch-A-distill/adapter" ;;
    B)    MDIR="$LAB/branch-B-rl/model"
          [ -d "$LAB/branch-B-rl/adapter" ] || fail "B 的适配器缺失：$LAB/branch-B-rl/adapter" ;;
esac
[ -f "$MDIR/config.json" ] || fail "模型缺少 config.json：$MDIR"

AVAIL="$(mem_avail_gb)"
[ -n "$AVAIL" ] || AVAIL="?"

# 内存自适应：fp32 权重约 7.5 GB，而本机总共 16 GB
LOWMEM=0
if [ "$AVAIL" != "?" ] && awk -v a="$AVAIL" 'BEGIN{exit !(a < 8.0)}'; then
    LOWMEM=1
    EXTRA_DTYPE="--dtype bfloat16"
fi

if [ "$INFO" -eq 1 ]; then
    echo "════ 环境自检 ════"
    echo "  项目目录     $LAB"
    echo "  解释器       $PY"
    echo "  对话脚本     $CHAT"
    echo "  模型 base    $([ -f "$LAB/base/config.json" ] && echo '✓ 就绪' || echo '✗ 缺失')"
    echo "  模型 A       $([ -f "$LAB/branch-A-distill/adapter/adapter_config.json" ] && echo '✓ 就绪' || echo '✗ 缺失')"
    echo "  模型 B       $([ -f "$LAB/branch-B-rl/adapter/adapter_config.json" ] && echo '✓ 就绪' || echo '✗ 缺失')"
    echo "  可用内存     ${AVAIL} GB"
    echo "  使用精度     $([ "$LOWMEM" -eq 1 ] && echo 'bfloat16（内存不足 8 GB，已自动降级）' || echo 'float32')"
    echo "  Python       $("$PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
    echo "  torch        $("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
    exit 0
fi

if [ "$LOWMEM" -eq 1 ]; then
    echo "  ⚠ 可用内存仅 ${AVAIL} GB（fp32 需约 7.5 GB）→ 自动改用 bfloat16"
    echo "    生成略慢，但不会因疯狂换页卡死。关掉浏览器等占内存的程序可恢复 fp32。"
fi

if [ -n "$PROMPT" ]; then
    exec "$PY" "$CHAT" -m "$MODEL" -p "$PROMPT" $THINK $EXTRA_DTYPE
else
    exec "$PY" "$CHAT" -m "$MODEL" $THINK $EXTRA_DTYPE
fi
