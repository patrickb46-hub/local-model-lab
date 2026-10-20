#!/usr/bin/env bash
# =============================================================================
# watchdog.sh —— 管线护航进程
#
# 它与守夜进程分工不同：
#   守夜（sleep 循环 + 状态快照）：只管"管线结束了 → 唤醒人来接手"
#   护航（本脚本）：在跑的全程做实际保障，四件事 ——
#
#   1. 每 5 分钟把进度写进 logs/watchdog.log（主流程若静默死亡也留痕）
#   2. 扫描主日志的异常关键词（Traceback / OOM / killed）
#   3. ★ 支线 B 一训完就立刻快照备份两条支线的 adapter
#        —— 这是 7 小时训练的成果。万一有人误跑 99_night_run.sh（它带 --fresh）
#           就会把 adapter 整个覆盖掉，有备份才救得回来。
#   4. 巡检 Ollama 服务、磁盘、swap（评测阶段要连打 3 个 Ollama 基线）
#
# 资源占用极低（每 5 分钟一次只读操作），不与训练抢 CPU/内存。
# 主流程退出后自动做最后一次快照并结束。
#
# 用法：nohup bash scripts/watchdog.sh &
# =============================================================================
set -uo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$LAB_ROOT/logs/night_run.log"
WDLOG="$LAB_ROOT/logs/watchdog.log"
BACKUP="$LAB_ROOT/backup"
cd "$LAB_ROOT"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$WDLOG"; }

# 支线 B 的正式训练是否已完成（按主日志的完成标记判断）
b_done() { grep -qE "◀ .*支线 B 正式训练.*退出码=0" "$LOG" 2>/dev/null; }

snapshot_products() {
    local tag="$1"
    mkdir -p "$BACKUP/$tag"
    for br in branch-A-distill branch-B-rl; do
        if [ -f "$LAB_ROOT/$br/adapter/adapter_model.safetensors" ]; then
            mkdir -p "$BACKUP/$tag/$br"
            cp -Rp "$LAB_ROOT/$br/adapter/." "$BACKUP/$tag/$br/" 2>/dev/null
            cp -p "$LAB_ROOT/$br/logs/"*.jsonl "$BACKUP/$tag/$br/" 2>/dev/null
        fi
    done
    cp -p "$LAB_ROOT/configs/lab.config.json" "$BACKUP/$tag/" 2>/dev/null
    say "已快照备份 → backup/${tag}（$(du -sh "$BACKUP/$tag" 2>/dev/null | cut -f1)）"
}

say "════ 护航启动 ════"
# 启动时先给已有产物做一次初始快照 —— 此刻支线 A 已经训完了，
# 那是 2.2 小时的成果，先落到 backup/ 再继续盯着。
snapshot_products "initial"
BACKED=0
ROUND=0
ERRREPORTED=0

while true; do
    ROUND=$((ROUND + 1))

    ALIVE=0
    pgrep -f "99_night_run.sh" >/dev/null && ALIVE=1

    # --- 1. 进度留痕 ---
    LAST=$(grep -E "step [0-9]+/60|◀ " "$LOG" 2>/dev/null | tail -1 | cut -c1-105)
    say "[#$ROUND] ${LAST:-（尚无进度）}"

    # --- 2. 异常扫描（只报一次，避免刷屏）---
    if [ -f "$LOG" ] && [ "$ERRREPORTED" -eq 0 ]; then
        NERR=$(grep -ciE "Traceback|out of memory|Killed|RuntimeError" "$LOG" 2>/dev/null)
        NERR=${NERR:-0}
        if [ "$NERR" -gt 0 ]; then
            say "⚠ 主日志出现 $NERR 处异常关键词，需人工核查："
            grep -iE "Traceback|out of memory|Killed|RuntimeError" "$LOG" 2>/dev/null | tail -3 | sed 's/^/    /' | tee -a "$WDLOG"
            ERRREPORTED=1
        fi
    fi

    # --- 3. 支线 B 训完 → 立刻备份（保护 7 小时成果）---
    if [ "$BACKED" -eq 0 ] && b_done; then
        say "检测到支线 B 正式训练已完成 → 开始快照备份两条支线的 adapter"
        snapshot_products "trained"
        BACKED=1
    fi

    # --- 4. 资源与 Ollama 巡检（每 6 轮 ≈ 30 分钟一次）---
    if [ $((ROUND % 6)) -eq 0 ]; then
        DISK=$(df -h / | tail -1 | awk '{print $4}')
        SW=$(sysctl -n vm.swapusage 2>/dev/null | sed 's/.*used = \([0-9.]*M\).*/\1/')
        OL=$(curl -sS --max-time 8 http://127.0.0.1:11434/api/tags 2>/dev/null | grep -c '"name"')
        OL=${OL:-0}
        say "  巡检：磁盘可用 $DISK | swap已用 $SW | Ollama 模型 $OL 个"
        if [ "$OL" -lt 3 ]; then
            say "  ⚠ Ollama 返回的模型数不足 3 个 —— 评测的官方基线可能取不到数据"
        fi
    fi

    # --- 退出条件 ---
    if [ "$ALIVE" -eq 0 ]; then
        say "主流程已不在运行"
        [ "$BACKED" -eq 0 ] && snapshot_products "final"
        break
    fi

    sleep 300
done

say "════ 护航结束 ════"
