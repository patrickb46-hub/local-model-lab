#!/usr/bin/env bash
# 98_eval_full.sh —— 评测补救：把评测集从「只有 L1 的 20 题」补到「全部 52 题」
#
# 背景（2026-10-10）：旧代码用 eval_set[:limit] 截题，而基准文件是「按层分块写出」的
# （L1 全部在前、L2 次之、L3 最后），于是 eval_limit=20 恰好只截到 20 道 L1 ——
# 评测集里一道 L2/L3 都没有，核心命题「L2 上 RL 臂能否越过 teacher」无法回答，
# 而日志/指标/报告全都"正常"、毫无报错。属沉默错误。
# 已修：30_evaluate.py 改为分层轮转取样；config.data.eval_limit 设为 0（全 52 题）。
#
# 本脚本只补跑 P6（评测）+ P7（报告）。
# P3~P5 的训练产物（adapter-A1 / adapter-B / adapter-C）都是好的，一个字节都不动。
#
# 省时机制（都已在评测脚本内）：
#   · 断点续跑：L1 那 20 题的 438 条记录已存在，自动跳过
#   · --reuse-screen：base/teacher 直接复用筛查的 K=5 采样（筛查覆盖全部 140 题，含 L2/L3）
# 预计新增生成：A1/B/C/dsv4pro 各 32 题 × 3 采样 ≈ 384 条，约 3~3.5 小时。
set -uo pipefail

cd "$(dirname "$0")/.."
LAB_ROOT="$PWD"
PY="${MLAB_PYTHON:-$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python}"

export PYTHONPATH="$LAB_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"

LOG="logs/eval_full52.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "════ 评测补救启动（目标 52 题：L1 20 + L2 16 + L3 16）════"

if ! pgrep -x caffeinate >/dev/null; then
    caffeinate -dimsu &
    say "已启动 caffeinate（pid $!）防空闲睡眠"
else
    say "caffeinate 已在运行"
fi

MODELS="base,A1,B,C,teacher,dsv4pro"
if [ -z "${COPILOT_PROVIDER_API_KEY:-}" ]; then
    MODELS="base,A1,B,C,teacher"
    say "⚠ 未检测到 COPILOT_PROVIDER_API_KEY —— 云基线 dsv4pro 本次跳过"
fi

say "▶ P6-eval（补跑 L2+L3；L1 走断点续跑；base/teacher 复用筛查采样）"
t0=$(date +%s)
"$PY" -u scripts/30_evaluate.py --reuse-screen --models "$MODELS" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
say "◀ P6-eval 退出码=$rc 用时=$(( $(date +%s) - t0 ))s"
if [ "$rc" -ne 0 ]; then
    say "✗ 评测失败，停止（报告不生成 —— 避免拿残缺数据出报告）"
    exit "$rc"
fi

say "▶ P7-report"
"$PY" -u scripts/31_report.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
say "◀ P7-report 退出码=$rc"
say "════ 评测补救结束 ════"
