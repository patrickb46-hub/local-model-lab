#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# labctl.sh —— 实验室主控：暂停 / 恢复 / 状态
#
# 用途：你要用电脑干别的事时，把实验停掉（释放约 12GB 内存），用完再接上。
#
# 为什么敢随便停：
#   * **评测**：每条样本落盘即视为完成，重启时按键（角色/题号/kind/采样序号）自动跳过。
#     最坏情况只丢「正在生成的那一题」，恢复时会自动重跑它 —— 无损。
#   * **训练**：不是无损的。SFT 每 5 步、RL 每 5 步存一次 checkpoint，
#     暂停会丢掉自上个 checkpoint 以来的步数，且续接需要额外处理。
#     所以：**训练期间暂停要加 --force，并且知道自己在丢什么。**
#
# 用法：
#   bash scripts/labctl.sh status            # 现在在跑什么、到哪了
#   bash scripts/labctl.sh pause             # 暂停（评测阶段可随意停）
#   bash scripts/labctl.sh pause --force     # 暂停（训练也照停，会丢若干步）
#   bash scripts/labctl.sh resume            # 恢复（接着上次跑）
#
# 【重要】恢复请在**你自己的终端**里执行（不是从这里代跑）——
# 这样启动的进程属于你的登录会话，比工具托管的更不容易被回收。
set -uo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"

STATE="$LAB_ROOT/logs/lab_pause_state.env"
LOG="$LAB_ROOT/logs/labctl.log"
PY="${MLAB_PYTHON:-$HOME/.workbuddy/binaries/python/envs/mlab-conda/bin/python}"

export PYTHONPATH="$LAB_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 模式都带 scripts/ 前缀：避免 pgrep/pkill 误匹配到守护脚本自己的命令行参数
WORK_PY="scripts/30_evaluate.py|scripts/11_train_distill.py|scripts/20_train_rl.py|scripts/12_screen_layers.py|scripts/31_report.py"
TRAIN_PY="scripts/11_train_distill.py|scripts/20_train_rl.py"
CTRL_SH="97_full_rerun.sh|98_eval_full.sh"

say()  { echo "$*"; echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
rule() { printf '%s\n' "────────────────────────────────────────────────────────────"; }

running() { pgrep -f "$1" >/dev/null 2>&1; }

# 某模式下所有进程的常驻内存合计（KB）。
# 读不到时返回空串 —— 沙箱里 `ps` 会被拒（operation not permitted），
# 那种情况下宁可不显示数字，也不要显示一个误导性的 0。
mem_kb() {
    local total=0 pid rss got=0
    for pid in $(pgrep -f "$1" 2>/dev/null); do
        rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')"
        if [ -n "$rss" ]; then
            total=$((total + rss))
            got=1
        fi
    done
    [ "$got" -eq 1 ] && echo "$total" || echo ""
}

detect_main() {
    if running "98_eval_full.sh"; then echo "98_eval_full.sh"; return; fi
    if running "97_full_rerun.sh"; then echo "97_full_rerun.sh"; return; fi
    echo ""
}

# 各角色在评测里的完成度（按层），用来在暂停前留个快照
progress_line() {
    "$PY" - <<'PY' 2>/dev/null
import json, collections, os
try:
    ev = [json.loads(l) for l in open("shared/results/eval_raw.jsonl", encoding="utf-8")]
    bench = [json.loads(l) for l in open("shared/bench_stratified.jsonl", encoding="utf-8")]
except Exception:
    raise SystemExit(0)
need = collections.Counter(b.get("layer") for b in bench)
cnt = collections.Counter()
for r in ev:
    if r.get("kind") == "main":
        cnt[(r["role"], r.get("layer") or "?")] += 1
parts = []
for role in ("base", "A1", "B", "C", "teacher", "dsv4pro"):
    g = sum(cnt.get((role, l), 0) for l in ("L1", "L2", "L3"))
    t = sum(need.get(l, 0) for l in ("L1", "L2", "L3")) * 3
    parts.append(f"{role} {g}/{t}")
print("   " + " | ".join(parts))
PY
}

# ============================================================ pause
cmd_pause() {
    local force="${1:-0}"
    rule
    if [ -f "$STATE" ] && ! running "$CTRL_SH" && ! running "$WORK_PY"; then
        say "⏸  当前已处于暂停状态（记录于 ${STATE}）"
        say "   恢复：bash scripts/labctl.sh resume"
        return 0
    fi

    if running "$TRAIN_PY" && [ "$force" != "1" ]; then
        say "⚠ 检测到正在**训练** —— 暂停不是无损的："
        say "   · SFT 每 5 步存一次 checkpoint，RL 同理；会丢掉自上个 checkpoint 以来的步数"
        say "   · 且训练步骤不幂等，恢复时要额外处理（不能简单地重跑容器脚本）"
        say "   评测阶段可以随便停；训练阶段建议等它跑完再停。"
        say "   确实要停，请加 --force：bash scripts/labctl.sh pause --force"
        return 1
    fi

    local main; main="$(detect_main)"
    say "⏸  暂停实验室"
    say "   主流程：${main:-（无容器脚本，但仍有工作进程）}"
    say "   暂停前进度："
    progress_line
    rule

    # 数据保险：评测原始记录留底（几十 KB，成本可忽略）
    if [ -f "$LAB_ROOT/shared/results/eval_raw.jsonl" ]; then
        cp -p "$LAB_ROOT/shared/results/eval_raw.jsonl" \
              "$LAB_ROOT/logs/eval_raw_before_pause.jsonl" 2>/dev/null \
            && say "   已留底评测记录 → logs/eval_raw_before_pause.jsonl"
    fi

    # 记录状态（key=value 格式，resume 时直接 source）
    local resume_cmd=""
    case "$main" in
        98_eval_full.sh) resume_cmd="bash scripts/98_eval_full.sh" ;;
        97_full_rerun.sh) resume_cmd="bash scripts/97_full_rerun.sh" ;;
    esac
    mkdir -p "$LAB_ROOT/logs"
    {
        printf 'PAUSED_AT=%q\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        printf 'MAIN_SCRIPT=%q\n' "$main"
        printf 'RESUME_CMD=%q\n' "$resume_cmd"
    } > "$STATE"

    # 显式暂停标记。自动巡检每 2 小时醒一次，看到主流程不在会"帮忙续跑" ——
    # 那会把用户刚暂停的东西重新拉起来。这个文件就是给巡检看的"别动"信号。
    touch "$LAB_ROOT/logs/PAUSED"

    local mem_before; mem_before="$(mem_kb "$WORK_PY")"

    # ---- 1. 先停工作进程：让容器脚本收到非 0 退出码后自行收尾（它不会继续下一步）----
    for pat in 30_evaluate.py 11_train_distill.py 20_train_rl.py 12_screen_layers.py 31_report.py; do
        if pkill -TERM -f "scripts/${pat}" 2>/dev/null; then
            say "   → 已请求停止 ${pat}（SIGTERM）"
        fi
    done

    # ---- 2. 等它把当前这一条落盘并退出（最多 25s）----
    local i
    for i in $(seq 1 25); do
        running "$WORK_PY" || break
        sleep 1
    done
    if running "$WORK_PY"; then
        pkill -KILL -f "$WORK_PY" 2>/dev/null
        say "   → 工作进程未响应 SIGTERM，已强制结束（最多丢当前这一条）"
    fi

    # ---- 3. 停容器脚本（正常情况下它们已因工作进程退出而自行结束）----
    for pat in "97_full_rerun.sh" "98_eval_full.sh"; do
        if running "$pat"; then
            pkill -TERM -f "$pat" 2>/dev/null
            sleep 1
            running "$pat" && pkill -KILL -f "$pat" 2>/dev/null
            say "   → 已停止容器脚本 ${pat}"
        fi
    done

    # ---- 4. 停护航与防睡眠 ----
    if running "guard.sh"; then
        pkill -TERM -f "guard.sh" 2>/dev/null
        say "   → 已停止护航 guard.sh"
    fi
    if pgrep -x caffeinate >/dev/null 2>&1; then
        pkill -x caffeinate 2>/dev/null
        say "   → 已停止 caffeinate（不再强撑不休眠）"
    fi

    sleep 2
    local mem_after; mem_after="$(mem_kb "$WORK_PY")"
    rule
    say "✅ 已暂停"
    if [ -n "$mem_before" ] && [ -n "$mem_after" ]; then
        say "   释放约 $(( (mem_before - mem_after) / 1024 / 1024 )) GB 内存（实验进程常驻内存已归零）"
    fi
    say "   机器现在归你用了。"
    rule
    say "   恢复：bash scripts/labctl.sh resume"
    say "   看进度：bash scripts/progress.sh"
}

# ============================================================ resume
cmd_resume() {
    rule
    if running "$CTRL_SH"; then
        say "▶  主流程已在运行，无需恢复"
        say "   看进度：bash scripts/progress.sh"
        return 0
    fi
    if running "$WORK_PY"; then
        say "⚠ 有工作进程在跑、但没有容器脚本在管它 —— 状态不干净。"
        say "   建议先：bash scripts/labctl.sh pause"
        return 1
    fi

    local resume_cmd=""
    if [ -f "$STATE" ]; then
        # shellcheck disable=SC1090
        . "$STATE" 2>/dev/null || true
        resume_cmd="${RESUME_CMD:-}"
        say "   读到暂停记录：${PAUSED_AT:-未知}"
    else
        say "   没有暂停记录 —— 按当前阶段默认恢复评测补跑"
    fi
    [ -z "$resume_cmd" ] && resume_cmd="bash scripts/98_eval_full.sh"

    # Ollama 是 teacher 侧的依赖，先探活
    if ! curl -sS --max-time 6 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        say "   Ollama 不可达 —— 尝试拉起"
        open -a Ollama >/dev/null 2>&1
        sleep 20
        if curl -sS --max-time 6 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            say "   ✓ Ollama 已就绪"
        else
            say "   ✗ Ollama 仍不可达 —— 恢复后 teacher 侧会缺数据，运行时会自行中止而不是静默出错"
        fi
    else
        say "   ✓ Ollama 已就绪"
    fi

    rule
    say "▶  恢复：$resume_cmd"
    nohup bash -c "$resume_cmd" >> "$LAB_ROOT/logs/lab_resume.log" 2>&1 &
    disown
    sleep 3

    if ! running "98_eval_full.sh" && ! running "97_full_rerun.sh"; then
        say "⚠ 启动后没看到主流程进程 —— 可能被会话回收了。"
        say "   请在你的终端里直接跑：$resume_cmd"
        return 1
    fi

    if ! running "guard.sh"; then
        nohup bash scripts/guard.sh >> "$LAB_ROOT/logs/guard_stdout.log" 2>&1 &
        disown
        say "   ✓ 护航已重启（Ollama 保活 + 定时备份）"
    fi

    # 恢复成功就把暂停标记清掉，避免自动巡检继续把我们当"已暂停"而不管
    rm -f "$LAB_ROOT/logs/PAUSED"
    [ -f "$STATE" ] && mv "$STATE" "$STATE.done" 2>/dev/null

    rule
    say "✅ 已恢复运行"
    say "   看进度：bash scripts/progress.sh"
}

# ============================================================ status
cmd_status() {
    rule
    echo "  本地模型实验室 · 状态"
    rule

    if [ -f "$LAB_ROOT/logs/PAUSED" ] && ! running "$CTRL_SH" && ! running "$WORK_PY"; then
        # shellcheck disable=SC1090
        . "$STATE" 2>/dev/null || true
        echo "  ⏸ 已暂停（${PAUSED_AT:-?}）"
        echo "     恢复：bash scripts/labctl.sh resume"
    elif running "$CTRL_SH" || running "$WORK_PY"; then
        echo "  ▶ 运行中"
    else
        echo "  · 无暂停记录，当前也没有实验进程"
    fi

    echo
    echo "  【进程】"
    ANY=0
    check() { if running "$1"; then echo "    ✅ $2"; ANY=1; fi; }
    check "98_eval_full.sh"          "评测补跑（98）"
    check "97_full_rerun.sh"         "全流程（97）"
    check "scripts/30_evaluate.py"   "评测生成"
    check "scripts/11_train_distill.py" "蒸馏训练"
    check "scripts/20_train_rl.py"   "RL 训练"
    check "guard.sh"                 "护航"
    # caffeinate 用精确进程名匹配：-f 会误匹配到命令行里含这个词的其他 shell
    if pgrep -x caffeinate >/dev/null 2>&1; then echo "    ✅ 防睡眠"; ANY=1; fi
    [ "$ANY" -eq 0 ] && echo "    （没有实验进程在跑）"

    local mb; mb="$(mem_kb "$WORK_PY")"
    echo
    if [ -n "$mb" ]; then
        echo "  【占用】实验进程常驻内存约 $(( mb / 1024 )) MB（暂停后应降到接近 0）"
    else
        echo "  【占用】此环境读不到进程内存（沙箱限制 ps）；在你自己的终端里跑可看到数字"
    fi
    echo
    echo "  【进度快照】"
    progress_line
    rule
}

# ============================================================ 入口
CMD="${1:-status}"
ARG2="${2:-}"
case "$CMD" in
    pause|stop)   FORCE=0; [ "$ARG2" = "--force" ] && FORCE=1; cmd_pause "$FORCE" ;;
    resume|start|go) cmd_resume ;;
    status|st)    cmd_status ;;
    -h|--help|help)
        sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        echo "未知命令：$CMD"
        echo "用法：bash scripts/labctl.sh {status|pause [--force]|resume}"
        exit 2
        ;;
esac
