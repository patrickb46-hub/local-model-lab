#!/usr/bin/env bash
# =============================================================================
# 99_night_run.sh —— 夜间无人值守：按序跑完整条管线
#
# 设计要点（都是踩过坑换来的）：
#   · 两条支线【严格顺序执行】，绝不同时占机器（用户明确要求）
#   · 全程 caffeinate 顶着，防止 macOS 睡眠掐断长任务
#   · 每一步的输出都追加到 logs/night_run.log，早上起来直接看这一个文件
#   · 每一步都幂等/可续跑：中途断了重新执行本脚本即可，不会白干
#   · 每步之间打印时间戳与耗时，便于判断卡在哪
#
# 用法：
#   nohup bash scripts/99_night_run.sh &          # 最简单
#   或直接:  bash scripts/99_night_run.sh          # 前台跑（可随时 Ctrl-C）
#
# 早上看结果：
#   tail -100 logs/night_run.log                  # 全过程日志
#   python scripts/31_report.py --open            # 打开可视化横评报告
# =============================================================================
set -uo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python"
LOG="$LAB_ROOT/logs/night_run.log"
export PYTHONPATH=
export CODEBUDDY_SAFE_DELETE_ENABLED=0
export TOKENIZERS_PARALLELISM=false

cd "$LAB_ROOT"
mkdir -p logs

say()  { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
rule() { echo "────────────────────────────────────────────────────────────" | tee -a "$LOG"; }

# 每一步：记录开始时间、跑、记录退出码与耗时
step() {
    local title="$1"; shift
    rule
    say "▶ $title"
    local t0=$SECONDS
    "$@" >>"$LOG" 2>&1
    local rc=$?
    say "◀ $title  退出码=$rc  用时=$((SECONDS - t0))s"
    if [ $rc -ne 0 ]; then
        say "⚠ 该步失败。可修好后重新执行本脚本续跑（各步都支持断点续跑）。"
    fi
    return $rc
}

rule
say "本地模型实验室 · 夜间批处理启动"
say "解释器 $PY"
say "电池/睡眠：" ; pmset -g batt 2>/dev/null | head -2 | tee -a "$LOG"
say "开始时间 $(date)"

# 防止系统睡眠（-d 显示器不休眠，-i 系统不休眠，-s 接交流电时不休眠，-u 用户活跃声明）
caffeinate -dimsu -w $$ &
CAF=$!
trap 'kill $CAF 2>/dev/null' EXIT

# ---------------------------------------------------------------------------
# 0. 环境自检（不通过就别往下浪费一晚）
# ---------------------------------------------------------------------------
step "环境自检 00_check_env.py --quick" \
    "$PY" scripts/00_check_env.py --quick || exit 1

# ---------------------------------------------------------------------------
# 1. 阶段一：补齐 teacher 数据 → 清洗切分（幂等，已完成的会自动跳过）
# ---------------------------------------------------------------------------
step "teacher 生成蒸馏数据（续跑）" \
    "$PY" -u scripts/10_gen_distill_data.py
step "清洗 + 验证筛选 + 切分" \
    "$PY" scripts/10_gen_distill_data.py --clean-only

# ---------------------------------------------------------------------------
# 2. 支线 A：蒸馏（先探针，再正式）
# ---------------------------------------------------------------------------
step "支线 A 探针"  "$PY" -u scripts/11_train_distill.py --probe
# --fresh：正式训练一律从零起步，忽略探针留下的 adapter。
# 这是【受控变量】的关键 —— 否则 A（探针痕迹被继承）会比 B 多训几步，
# 在结果接近时正好会污染「A vs B」的比较。
step "支线 A 正式训练（60 步）" "$PY" -u scripts/11_train_distill.py --fresh

# ---------------------------------------------------------------------------
# 3. 支线 B：强化学习（探针 → 正式）—— 与 A 严格串行
# ---------------------------------------------------------------------------
step "支线 B 探针"  "$PY" -u scripts/20_train_rl.py --probe
# 同 A：--fresh 保证 B 也是纯 60 步、从零起步（受控变量）
step "支线 B 正式训练（60 步）" "$PY" -u scripts/20_train_rl.py --fresh

# ---------------------------------------------------------------------------
# 4. 阶段三：六模型横评（think 开/关都测）+ 可视化报告
# ---------------------------------------------------------------------------
step "六模型横评"   "$PY" -u scripts/30_evaluate.py
step "生成可视化报告" "$PY" scripts/31_report.py

# ---------------------------------------------------------------------------
# 5. 阶段四：导出到 Ollama
# ---------------------------------------------------------------------------
step "导出 to Ollama" bash scripts/40_export_ollama.sh both

rule
say "全部完成。结束时间 $(date)"
say "报告：$LAB_ROOT/shared/results/横评报告.html"
say "明细：$LAB_ROOT/shared/results/eval_metrics.json"
ls -la "$LAB_ROOT/shared/results/" 2>/dev/null | tee -a "$LOG"
