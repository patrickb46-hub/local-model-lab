#!/usr/bin/env bash
# =============================================================================
# resume.sh —— 智能续跑：只补没跑完的步骤，绝不重训已完成的支线
#
# 为什么需要它（关键）：
#   99_night_run.sh 是"按序跑全流程"，虽然每步都声称幂等，但训练步骤为了保证
#   受控变量用的是 --fresh（强制从零开始训）。
#   所以**直接重跑它 = 重新训练 A（约 2.2 小时）+ B（约 5 小时）**，
#   白烧一晚上，而且会把已经训好的 adapter 覆盖掉。
#
#   本脚本改为按日志里「◀ 步骤名 … 退出码=0」的完成标记来判断：
#   已完成的训练直接跳过，只补没跑完的。
#
# 用法：
#   bash scripts/resume.sh              # 检查并补跑
#   bash scripts/resume.sh --dry-run    # 只打印将要执行什么，不真的跑
#
# 安全性：如果在跑之前检测到主流程仍在运行，会拒绝启动（避免两个进程抢机器）。
# =============================================================================
set -uo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$LAB_ROOT/logs/night_run.log"
PY="$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python"

export PYTHONPATH=
export CODEBUDDY_SAFE_DELETE_ENABLED=0
export TOKENIZERS_PARALLELISM=false
cd "$LAB_ROOT"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
RAN=0

# 必须同时写日志：本脚本的幂等判断（done_step）依赖日志里的
# 「◀ 步骤名 … 退出码=0」完成标记。只 echo 到终端的话，
# 跑过的步骤在日志里不留痕，下次会被当成没跑过而重来一遍。
say() {
    local msg="[$(date '+%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >>"$LOG"
}
rule() { echo "────────────────────────────────────────────────────────────"; }

# 日志里有没有「◀ …关键词… 退出码=0」
done_step() {
    [ -f "$LOG" ] && grep -qE "◀ .*$1.*退出码=0" "$LOG"
}

run() {
    local title="$1"; shift
    RAN=1
    if [ "$DRY" -eq 1 ]; then
        say "[dry-run] 需要执行：$title"
        return 0
    fi
    rule
    say "▶ $title"
    local t0=$SECONDS
    "$@" >>"$LOG" 2>&1
    local rc=$?
    say "◀ $title  退出码=$rc  用时=$((SECONDS - t0))s"
    return $rc
}

# ---- 先确认没有别的进程在跑（避免两个流程抢机器）----
# dry-run 只看计划、不动机器，所以跳过这项检查
#
# 【模式必须带 "scripts/" 前缀】否则会误匹配到看门狗 / 守夜进程 ——
# 它们的命令行里也含有 "99_night_run.sh" 这个字符串（那是它们 pgrep 的参数）。
# 实测事故：守夜进程让本脚本误判「管线仍在运行」而拒绝启动，白等一场。
if [ "$DRY" -eq 0 ]; then
    if pgrep -f "bash scripts/99_night_run.sh" >/dev/null || \
       pgrep -f "scripts/11_train_distill.py" >/dev/null || \
       pgrep -f "scripts/20_train_rl.py" >/dev/null || \
       pgrep -f "scripts/30_evaluate.py" >/dev/null; then
        echo "⚠ 检测到管线相关进程仍在运行 —— 同时跑两个会互相抢内存，已拒绝启动。"
        echo "  当前占用的进程："
        pgrep -fl "bash scripts/99_night_run.sh|scripts/11_train_distill.py|scripts/20_train_rl.py|scripts/30_evaluate.py" | sed 's/^/    /'
        echo "  如需强制接管，先执行："
        echo "    pkill -f 99_night_run.sh; pkill -f 11_train_distill.py; pkill -f 20_train_rl.py"
        echo "  然后再跑本脚本。"
        exit 1
    fi
fi

say "════ 智能续跑检查（读到 ${LOG}）════"
[ -f "$LOG" ] || say "⚠ 还没找到日志，将按「全部都要跑」处理"

# ---- 1. 支线 A ----
if done_step "支线 A 正式训练"; then
    say "跳过 · 支线 A 正式训练（日志显示已完成，adapter 保留）"
else
    run "支线 A 探针" "$PY" -u scripts/11_train_distill.py --probe
    run "支线 A 正式训练（60 步）" "$PY" -u scripts/11_train_distill.py --fresh
fi

# ---- 2. 支线 B ----
if done_step "支线 B 正式训练"; then
    say "跳过 · 支线 B 正式训练（日志显示已完成，adapter 保留）"
else
    run "支线 B 探针" "$PY" -u scripts/20_train_rl.py --probe
    run "支线 B 正式训练（60 步）" "$PY" -u scripts/20_train_rl.py --fresh
fi

# ---- 3. 评测（脚本自带断点续跑，已生成的样本会自动跳过）----
if done_step "六模型横评"; then
    say "跳过 · 六模型横评"
else
    run "六模型横评" "$PY" -u scripts/30_evaluate.py
fi

# ---- 4. 报告 ----
if done_step "生成可视化报告"; then
    say "跳过 · 生成可视化报告"
else
    run "生成可视化报告" "$PY" scripts/31_report.py
fi

# ---- 5. 导出到 Ollama（失败不影响横评结论）----
if done_step "导出 to Ollama"; then
    say "跳过 · 导出 to Ollama"
else
    run "导出 to Ollama" bash scripts/40_export_ollama.sh both
fi

rule
if [ "$RAN" -eq 0 ]; then
    say "全部步骤都已完成，无需续跑。"
    say "报告：$LAB_ROOT/shared/results/横评报告.html"
else
    say "续跑结束。报告：$LAB_ROOT/shared/results/横评报告.html"
fi
