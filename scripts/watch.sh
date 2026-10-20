#!/usr/bin/env bash
# =============================================================================
# watch.sh —— 管线进度查看器 / Pipeline progress viewer
#
# 用法 / Usage:
#   bash scripts/watch.sh              # 中文快照 / Chinese snapshot
#   bash scripts/watch.sh --en         # 英文快照 / English snapshot
#   bash scripts/watch.sh --both       # 中英各一份（分开两段）/ both, stacked
#   bash scripts/watch.sh -w           # 每 20 秒自动刷新（Ctrl-C 退出）
#   bash scripts/watch.sh -w 5 --en    # 自定义刷新间隔 + 英文
#   bash scripts/watch.sh -l 200       # 额外附带日志尾部 200 行
#
# 渲染逻辑在 watch_render.py 里（独立成文件，避免大段代码塞进 shell 时的引号问题）。
# Rendering lives in watch_render.py — keeps this script free of quoting hazards.
#
# 只读脚本，不触碰任何数据。/ Read-only, touches no data.
# =============================================================================
set -uo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$LAB_ROOT/logs/night_run.log"
RENDER="$LAB_ROOT/scripts/watch_render.py"
PY="${WATCH_PYTHON:-/usr/bin/python3}"

WATCH=0
INTERVAL=20
TAILN=0
LANG_OPT="zh"

while [ $# -gt 0 ]; do
    case "$1" in
        -w|--watch)
            WATCH=1; shift
            if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then INTERVAL="$1"; shift; fi
            ;;
        -l|--log)
            shift
            if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then TAILN="$1"; shift; fi
            ;;
        --en|--english) LANG_OPT="en"; shift ;;
        --zh|--chinese) LANG_OPT="zh"; shift ;;
        --both)         LANG_OPT="both"; shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) shift ;;
    esac
done

snapshot() {
    LAB_ROOT="$LAB_ROOT" LOG="$LOG" TAILN="$TAILN" WATCH_LANG="$1" \
        "$PY" "$RENDER"
}

# 陈旧日志警告 —— 本查看器解析的是「夜间批处理」日志（logs/night_run.log）。
# 如果它已经很久没更新，那屏幕上的「全部完成」是**历史记录**，
# 而不是当前进度。这种误导比没有进度条更糟，所以必须显式提醒。
staleness_warning() {
    [ -f "$LOG" ] || return 0
    local mtime age days
    mtime="$(stat -f %m "$LOG" 2>/dev/null || echo 0)"
    age=$(( $(date +%s) - mtime ))
    [ "$age" -gt 86400 ] || return 0
    days=$(( age / 86400 ))
    printf '\n'
    printf '  ⚠ 注意：%s 已经 %d 天没更新。\n' "$(basename "$LOG")" "$days"
    printf '    它记录的是**历史**流程（上一次夜间批处理）—— 屏幕上的「完成」\n'
    printf '    不代表当前正在跑的进度。\n'
    printf '    查看分层筛查（P2）的实时进度，请用：\n'
    printf '      bash scripts/screen_status.sh\n\n'
}

render() {
    staleness_warning
    if [ "$LANG_OPT" = "both" ]; then
        snapshot zh
        printf '\n'
        snapshot en
    else
        snapshot "$LANG_OPT"
    fi
}

if [ "$WATCH" -eq 1 ]; then
    trap 'printf "\n"; exit 0' INT TERM
    while true; do
        clear
        render
        case "$LANG_OPT" in
            zh)   printf "\n  每 %ss 自动刷新 · Ctrl-C 退出\n" "$INTERVAL" ;;
            en)   printf "\n  refreshing every %ss · Ctrl-C to quit\n" "$INTERVAL" ;;
            both) printf "\n  每 %ss 自动刷新 / refreshing every %ss · Ctrl-C 退出/quit\n" \
                        "$INTERVAL" "$INTERVAL" ;;
        esac
        sleep "$INTERVAL"
    done
else
    render
fi
