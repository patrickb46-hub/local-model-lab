#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 97_full_rerun.sh —— P3~P7 全流程，无人值守
#
# 设计原则（都是踩过坑换来的）：
#   1. **幂等可续跑**：每步在日志里留「▶/◀ 步骤名 退出码=0」标记，重跑自动跳过已完成项
#      （否则一次中断就要重训 5 小时的 RL）
#   2. **失败即止**：任一步非 0 即中止，不做「带病继续」
#   3. **防睡眠**：全程 caffeinate（macOS 空闲睡眠会掐断长任务）
#   4. **不猜环境**：缺云 API key 就自动把该角色从评测里去掉，而不是跑一半崩掉
#
# 用法：
#   cd ~/Desktop/AI/Local-Model-Lab
#   nohup bash scripts/97_full_rerun.sh > /dev/null 2>&1 &
#   bash scripts/progress.sh          # 看进度
#
set -u

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MLAB_PYTHON:-$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python}"
LOG="$LAB_ROOT/logs/full_rerun.log"

export PYTHONPATH="$LAB_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LAB_ROOT/logs"
cd "$LAB_ROOT" || exit 1

say() {
    local m="[$(date '+%m-%d %H:%M:%S')] $*"
    echo "$m"
    echo "$m" >> "$LOG"
}

mark_done() {
    grep -qE "◀ ${1}  *退出码=0" "$LOG" 2>/dev/null
}

run_step() {
    local name="$1"; shift
    if mark_done "$name"; then
        say "◀ ${name}（已完成，跳过）"
        return 0
    fi
    say "▶ $name"
    local t0 rc
    t0=$(date +%s)
    echo "--- [$name] cmd: $* ---" >> "$LOG"
    "$@" >> "$LOG" 2>&1
    rc=$?
    say "◀ $name  退出码=$rc  用时=$(( $(date +%s) - t0 ))s"
    return $rc
}

# ---------------------------------------------------------------------------
say "════════ 全流程 P3~P7 启动 ════════"

# ---- 防睡眠 ----
if ! pgrep -x caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu >/dev/null 2>&1 &
    say "已启动 caffeinate（防空闲睡眠）"
fi

# ---- 0. 等 P2 分层筛查结束（最多等 4 小时）----
if pgrep -f "12_screen_layers.py" >/dev/null; then
    say "▶ 等待 P2 分层筛查结束"
    for _ in $(seq 1 480); do
        pgrep -f "12_screen_layers.py" >/dev/null || break
        sleep 30
    done
    say "◀ P2 已结束（筛查进程退出）"
else
    say "P2 筛查进程不在 —— 视为已完成"
fi

# ---- 0.5 快照备份（重训会覆盖旧 adapter）----
backup_snapshot() {
    local tag="pre_rerun_$(date '+%m%d_%H%M')"
    local dst="$LAB_ROOT/backup/$tag"
    mkdir -p "$dst"
    [ -d branch-A-distill/adapter ] && cp -Rp branch-A-distill/adapter "$dst/branch-A-adapter" 2>/dev/null
    [ -d branch-B-rl/adapter ] && cp -Rp branch-B-rl/adapter "$dst/branch-B-adapter" 2>/dev/null
    cp -p configs/lab.config.json "$dst/" 2>/dev/null
    cp -p shared/screen_raw.jsonl "$dst/" 2>/dev/null
    echo "  已备份到 $dst"
    du -sh "$dst" 2>/dev/null | sed 's/^/  /'
}
run_step "P0-backup" backup_snapshot

# ---- 0.8 扩大筛查范围 ----
# GATE-1 首次未通过：40 题里 L2 命中率 15%（6 题），闸门要求 ≥24。
# 外推：需要约 140 题才能产出 ~21 题 L2。已跑完的 40 题会自动跳过。
run_step "P2b-expand" "$PY" -u scripts/12_screen_layers.py --limit 140 --k 5 || exit 1

# ---- 1. P3 冻结基准（分层 → bench + RL curated 池）----
run_step "P3-freeze" "$PY" scripts/12_screen_layers.py --model report || {
    say "⚠ P3 失败 —— 可能 L2 不足或数据未就绪，中止（不要盲目重训）"
    exit 1
}

# ---- 2. P4-A1 蒸馏 60 步（从 base 起）----
run_step "P4-A1" "$PY" -u scripts/11_train_distill.py --fresh --tag A1 || exit 1

# ---- 3. P4-A2（本轮跳过）----
# A2 = 蒸馏 120 步，用来证明「蒸馏再加步数也突破不了 teacher」。
# 因 24 小时预算收紧（扩筛查 7.5h + 训练 10h + 评测 3h ≈ 20h），本轮跳过。
# 若时间有余，手工执行：
#   $PY -u scripts/11_train_distill.py --tag A2 --init-adapter .../adapter-A1
say "P4-A2 本轮跳过（预算）"

# ---- 4. P5-B 强化学习 60 步（从 base 起）----
run_step "P5-B" "$PY" -u scripts/20_train_rl.py --fresh --tag B || exit 1

# ---- 5. P5-C 强化学习 60 步（从 A1 预热 —— 等预算对照臂）----
run_step "P5-C" "$PY" -u scripts/20_train_rl.py --fresh --tag C \
    --init-adapter "$LAB_ROOT/branch-A-distill/adapter-A1" || exit 1

# ---- 6. P6 评测（复用筛查里 base/teacher 的采样）----
MODELS="base,A1,B,C,teacher,dsv4pro"
if [ -z "${COPILOT_PROVIDER_API_KEY:-}" ]; then
    MODELS="base,A1,B,C,teacher"
    say "⚠ 未检测到 COPILOT_PROVIDER_API_KEY —— 云基线 dsv4pro 本次跳过"
fi
run_step "P6-eval" "$PY" -u scripts/30_evaluate.py --reuse-screen --models "$MODELS" || exit 1

# ---- 7. P7 报告 ----
run_step "P7-report" "$PY" -u scripts/31_report.py || exit 1

say "════════ 全流程结束 ════════"
say "报告： $LAB_ROOT/shared/results/横评报告.html"
