#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# guard.sh —— 长任务护航（巡检 + Ollama 保活 + 定期备份）
#
# 为什么需要它（每一条都对应一次真实事故）：
#   * Ollama 中途退出 → 145 次调用静默失败 → 分层基准作废  ⇒ 必须保活
#   * 训练 5 小时的 adapter 被后续步骤覆盖 / 误删         ⇒ 定期备份
#   * 内存耗尽疯狂换页把长任务拖慢数倍                    ⇒ 记录 swap/内存
#   * 长任务「静默停滞」比崩溃更隐蔽                      ⇒ 检测日志活动
#
# 它只做四件事，且只读+重启，不修改任何实验产物。
set -u

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$LAB_ROOT/logs/guard.log"
mkdir -p "$LAB_ROOT/logs"

say() {
    local m="[$(date '+%m-%d %H:%M:%S')] $*"
    echo "$m" >> "$LOG"
}

say "════ 护航启动 ════"

n=0
while true; do
    sleep 300
    n=$((n + 1))

    # ---- 0. 主流程还在吗 ----
    # 注意：这里必须同时认「全流程脚本」和「评测补跑脚本」。
    # 曾经只认 97_full_rerun.sh —— 于是 97 一结束（哪怕紧接着启动了 98 补跑），
    # 护航就自行收工，补跑期间完全没有保护。判据要覆盖当前真正在跑的主流程。
    if ! pgrep -f "97_full_rerun.sh" >/dev/null 2>&1 &&
       ! pgrep -f "98_eval_full.sh" >/dev/null 2>&1; then
        say "主流程脚本已退出（第 ${n} 次巡检）—— 护航收工"
        break
    fi

    # ---- 1. Ollama 保活（teacher 侧 / Ollama 角色依赖它）----
    if ! curl -sS --max-time 6 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        say "⚠ Ollama 不可达 —— 尝试拉起"
        open -a Ollama >/dev/null 2>&1
        sleep 25
        if curl -sS --max-time 6 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            say "✓ Ollama 已恢复"
        else
            say "✗ Ollama 仍不可达 —— 若正在跑 teacher 侧，该步骤会自行中止（不会静默出错）"
        fi
    fi

    # ---- 2. 关键进程快照 ----
    procs=""
    for p in 12_screen_layers 11_train_distill 20_train_rl 30_evaluate; do
        pgrep -f "${p}.py" >/dev/null 2>&1 && procs="${procs} ${p}"
    done
    [ -z "$procs" ] && procs=" (无)"

    # ---- 3. 内存 / swap 快照 ----
    swap="$(sysctl -n vm.swapusage 2>/dev/null | tr -s ' ')"
    freem="$(/usr/bin/python3 -c "
import subprocess
o=subprocess.check_output(['vm_stat'],text=True); v={}
for l in o.splitlines():
    if ':' in l:
        k,x=l.split(':',1); x=x.strip().rstrip('.')
        if x.isdigit(): v[k.strip()]=int(x)
print(f\\\"{(v.get('Pages free',0)+v.get('Pages inactive',0)+v.get('Pages speculative',0))*4096/1e9:.1f}\\\")
" 2>/dev/null || echo "?")"

    # ---- 4. 定期备份 adapter（每 6 次巡检 ≈ 30 分钟）----
    bk=""
    if [ $((n % 6)) -eq 0 ]; then
        dst="$LAB_ROOT/backup/rolling"
        mkdir -p "$dst"
        for arm in branch-A-distill/adapter-A1 branch-A-distill/adapter-A2 \
                   branch-B-rl/adapter-B branch-B-rl/adapter-C; do
            if [ -f "$LAB_ROOT/${arm}/adapter_model.safetensors" ]; then
                name="$(basename "$(dirname "$arm")")-$(basename "$arm")"
                cp -Rp "$LAB_ROOT/${arm}/." "$dst/${name}/" 2>/dev/null
            fi
        done
        bk=" 已备份"
    fi

    say "#${n} 进程:${procs} | 可用内存 ${freem}G | ${swap}${bk}"
done

# ---- 唤醒时的快照（读最新在跑的那个日志，别只盯 full_rerun.log）----
say "════ 收工检查 ════"
snap_log="$LAB_ROOT/logs/full_rerun.log"
if [ -f "$LAB_ROOT/logs/eval_full52.log" ]; then
    if [ ! -f "$snap_log" ] || [ "$LAB_ROOT/logs/eval_full52.log" -nt "$snap_log" ]; then
        snap_log="$LAB_ROOT/logs/eval_full52.log"
    fi
fi
tail -25 "$snap_log" 2>/dev/null | while IFS= read -r line; do
    say "  ${line}"
done
for arm in branch-A-distill/adapter-A1 branch-A-distill/adapter-A2 \
           branch-B-rl/adapter-B branch-B-rl/adapter-C; do
    f="$LAB_ROOT/${arm}/adapter_model.safetensors"
    if [ -f "$f" ]; then
        sz=$(stat -f%z "$f" 2>/dev/null || echo 0)
        say "  产物 ${arm}: $((sz / 1000000)) MB"
    else
        say "  产物 ${arm}: 未产出"
    fi
done
say "════ 护航结束 ════"
