#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# screen_status.sh —— 分层筛查（P2）进度查看器
#
# 为什么单独写一个：
#   watch.sh 看的是「夜间批处理」的日志（logs/night_run.log），它按 ▶/◀ 步骤标记解析。
#   而分层筛查是一个**长任务**，没有那套标记 —— 拿 watch.sh 去看它，
#   屏幕上显示的会是**上一轮流程的历史完成记录**，看起来「全部完成」，
#   而实际上筛查才跑到一半。这种误导比没有进度条更糟。
#
# 用法：
#   bash scripts/screen_status.sh          # 看一次快照
#   bash scripts/screen_status.sh -w       # 每 20 秒自动刷新（Ctrl-C 退出）
#   bash scripts/screen_status.sh -w 5     # 自定义刷新间隔
#   bash scripts/screen_status.sh -l 20    # 附带最近 20 行日志

set -u

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH=0
INTERVAL=20
TAILN=0

while [ $# -gt 0 ]; do
    case "$1" in
        -w|--watch) WATCH=1; shift
                    if [ $# -gt 0 ] && [ "$1" -eq "$1" ] 2>/dev/null; then INTERVAL="$1"; shift; fi ;;
        -l|--log)   shift; TAILN="${1:-20}"; shift ;;
        -h|--help)  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          shift ;;
    esac
done

snapshot() {
    LAB_ROOT="$LAB_ROOT" TAILN="$TAILN" /usr/bin/python3 - <<'PY'
import os, re, glob, json, time, datetime, subprocess

LAB = os.environ["LAB_ROOT"]
TAILN = int(os.environ.get("TAILN", "0") or 0)

BAR = "─" * 68


def human(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=8).stdout.strip()
    except Exception:
        return ""


print(BAR)
print(f"  分层筛查（P2 · GATE-1）进度   {datetime.datetime.now():%m-%d %H:%M:%S}")
print(BAR)

# ---- 日志文件：取 logs/ 下最新的 screen_*.log ----
cands = [p for p in glob.glob(os.path.join(LAB, "logs", "screen_*.log"))
         if not p.endswith(".bak")]
log = max(cands, key=os.path.getmtime) if cands else None
running = os.popen("pgrep -f 12_screen_layers.py").read().strip() != ""

print("\n【进程与数据源】")
print(f"  筛查进程   {'✅ 在跑' if running else '⏹  已结束'}")
if log:
    age = time.time() - os.path.getmtime(log)
    print(f"  日志       {os.path.basename(log)}（{human(age)} 前更新）")
else:
    print("  日志       （未找到 screen_pilot*.log）")

# ---- 已完成的采样 ----
raw = os.path.join(LAB, "shared", "screen_raw.jsonl")
rows = []
if os.path.exists(raw):
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

by_model = {}
for r in rows:
    by_model.setdefault(r["model"], {}).setdefault(r["qid"], []).append(int(r["correct"]))

print("\n【采样进度】")
K = 5
for m in ("base", "teacher"):
    d = by_model.get(m, {})
    cnt = len([r for r in rows if r["model"] == m])
    full = [q for q, v in d.items() if len(v) >= K]
    print(f"  {m:8s} {cnt:>5d} 条 / {len(d):>3d} 题（满 {K} 采样的 {len(full):>3d} 题）")

# ---- base 侧难度分布（早期信号）----
d = by_model.get("base", {})
full = {q: sum(v) for q, v in d.items() if len(v) >= K}
if full:
    z = [q for q, s in full.items() if s == 0]
    s_ = [q for q, s in full.items() if 0 < s < K]
    a = [q for q, s in full.items() if s == K]
    tot = len(full)
    p1 = sum(full.values()) / (K * tot)
    print(f"\n【base 侧难度分布】（{tot} 题）")
    print(f"  三次全错（L3 候选）      {len(z):>3d} 题  {len(z)/tot*100:5.1f}%")
    print(f"  部分对（**L2 候选池**）  {len(s_):>3d} 题  {len(s_)/tot*100:5.1f}%   ← 关键")
    print(f"  全部做对（送分）         {len(a):>3d} 题  {len(a)/tot*100:5.1f}%")
    print(f"  base pass@1 = {p1*100:.1f}%")
    print("  说明：L2 还需 teacher **系统性失败**才能成立，等 teacher 侧跑完才能定层。")

# ---- ETA ----
if log and running and full:
    try:
        born = os.stat(log).st_birthtime
        elapsed = time.time() - born
        limit = 40
        # base 侧已完成题数（含断点续跑带来的历史完成）
        done_q = len(d)
        if done_q > 0:
            per_q = elapsed / max(1, done_q - 12)   # 减去首轮已完成的 12 题
            left = max(0, limit - done_q)
            print(f"\n【预计】")
            print(f"  base 剩余 {left} 题 × 约 {per_q/60:.1f} 分钟 ≈ {human(left*per_q)}")
            print(f"  teacher 40 题 × 5 采样 × 约 40s ≈ 约 2.2 小时")
            print(f"  → 筛查整体还需约 {human(left*per_q + 2.2*3600)}")
    except Exception:
        pass

# ---- 分层产物 ----
print("\n【分层产物】（筛查跑完后由 report 写出）")
mp = os.path.join(LAB, "shared", "results", "layer_manifest.json")
if os.path.exists(mp):
    try:
        man = json.load(open(mp, encoding="utf-8"))
        c = man.get("counts", {})
        print(f"  L1 {c.get('L1', 0)} / L2 {c.get('L2', 0)} / L3 {c.get('L3', 0)}"
              f"   闸门要求 L2 >= {man.get('gate_min_l2', 24)}"
              f"   → {'通过 ✓' if man.get('gate_pass') else '未通过'}")
    except Exception as e:
        print(f"  读取失败: {e}")
else:
    print("  （尚无 layer_manifest.json —— 还没跑完）")

# ---- 日志尾部 ----
if TAILN and log:
    print(f"\n【日志尾部 {TAILN} 行】")
    out = run(f"grep -vE 'it/s\\]|Loading|falling|^\\[transformers\\]|UserWarning|warnings' "
              f"'{log}' | tail -{TAILN}")
    for line in out.splitlines():
        print("  " + line)

print("\n" + BAR)
if not running:
    print("  筛查已结束。若 layer_manifest.json 存在 → 看上面的闸门判定。")
PY
}

if [ "$WATCH" -eq 1 ]; then
    trap 'printf "\n"; exit 0' INT TERM
    while true; do
        clear
        snapshot
        printf "\n  每 %ss 自动刷新 · Ctrl-C 退出\n" "$INTERVAL"
        sleep "$INTERVAL"
    done
else
    snapshot
fi
