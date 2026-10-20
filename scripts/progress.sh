#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# progress.sh —— 全流程进度的唯一入口（一眼看完：跑到哪、还剩多久）
#
# 为什么写它：项目里现在有多个日志（分层筛查 / 全流程重跑 / 评测补跑），
# 各自记各自的进度。谁去记「哪个日志才是当前的」就容易看错 ——
# 之前就发生过「看 watch.sh 显示全部完成，其实筛查才跑一半」。
# 这个脚本把当前真正在跑的东西 + 全流程时间线 + 各臂产物状态放在一屏里。
#
# 用法：
#   bash scripts/progress.sh          # 中文，看一次
#   bash scripts/progress.sh -w       # 每 20 秒自动刷新（Ctrl-C 退出）
#   bash scripts/progress.sh -w 5     # 自定义刷新间隔
#   bash scripts/progress.sh --en     # 英文版（也可以直接跑 progress_en.sh）
#   bash scripts/progress.sh --en -w  # 可组合
#
# 时间一律自动进位：90s → 1m30s，5400s → 1h30m，90000s → 1d1h。
set -u

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH=0
INTERVAL=20
LANG_MODE=zh

while [ $# -gt 0 ]; do
    case "$1" in
        -w|--watch) WATCH=1; shift
                    if [ $# -gt 0 ] && [ "$1" -eq "$1" ] 2>/dev/null; then INTERVAL="$1"; shift; fi ;;
        -e|--en|--english) LANG_MODE=en; shift ;;
        -z|--zh|--cn|--chinese) LANG_MODE=zh; shift ;;
        -h|--help)  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          shift ;;
    esac
done

snapshot() {
    LAB_ROOT="$LAB_ROOT" PROGRESS_LANG="$LANG_MODE" /usr/bin/python3 - <<'PY'
import os, json, datetime, subprocess, collections, unicodedata

LAB = os.environ["LAB_ROOT"]
EN = os.environ.get("PROGRESS_LANG", "zh") == "en"
BAR = "═" * 70


def t(zh, en):
    """按语言挑文案。"""
    return en if EN else zh


def pad(s, width):
    """按**显示宽度**补齐（CJK 字符占 2 列）—— 中英混排才对得齐。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


def human(sec):
    """秒 → 自动进位。90 → 1m30s；5400 → 1h30m；90000 → 1d1h。"""
    sec = int(max(0, sec))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def parse_dur(s):
    """日志里的耗时字段 → 秒。

    两种格式都要认：
      · 97 脚本写的是纯秒            '25549s'
      · 98 脚本写的是已进位的形式    '1h0m50s' / '56m27s'
    """
    s = (s or "").strip()
    if not s:
        return None
    if s.endswith("s") and s[:-1].isdigit():
        return int(s[:-1])
    total, num = 0, ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            if num:
                total += int(num) * {"d": 86400, "h": 3600, "m": 60, "s": 1}.get(ch, 0)
                num = ""
    if num.isdigit():
        total += int(num)
    return total or None


def dur_str(sec):
    return "—" if sec is None else human(sec)


def running(pat):
    try:
        return subprocess.run(f"pgrep -f '{pat}'", shell=True,
                              capture_output=True, text=True).stdout.strip() != ""
    except Exception:
        return False


def jl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


print(BAR)
print("  " + pad(t("本地模型实验室 · 全流程进度", "Local Model Lab · Pipeline Progress"), 56)
      + f"{datetime.datetime.now():%m-%d %H:%M:%S}")
print(BAR)

# ---------------- 当前活动 ----------------
print("\n" + t("【正在跑什么】", "[Running now]"))
act = False
if running("12_screen_layers.py"):
    act = True
    rows = jl(os.path.join(LAB, "shared", "screen_raw.jsonl"))
    nb = len({r["qid"] for r in rows if r["model"] == "base"})
    nt = len({r["qid"] for r in rows if r["model"] == "teacher"})
    print("  ✅ " + t(f"分层筛查 P2        base {nb} 题 / teacher {nt} 题",
                     f"Layered screening P2    base {nb} / teacher {nt} questions"))
for pat, label in (("11_train_distill.py", t("蒸馏训练", "Distillation training")),
                   ("20_train_rl.py", t("强化学习训练", "RL training")),
                   ("30_evaluate.py", t("评测", "Evaluation")),
                   ("31_report.py", t("报告生成", "Report generation")),
                   ("97_full_rerun.sh", t("全流程脚本", "Pipeline script")),
                   ("98_eval_full.sh", t("评测补跑（把评测集从 20 题 L1 补到 52 题全集）",
                                         "Eval catch-up (20 L1-only → full 52 questions)"))):
    if running(pat):
        act = True
        # 48 是按最长的英文标签（44 列）定的：短了会 pad 失效、括号紧贴到文字上
        print("  ✅ " + pad(label, 48) + f"({pat})")
if not act:
    print("  " + t("（当前没有实验进程在跑）", "(no experiment process running)"))

# ---------------- 全流程时间线 ----------------
print("\n" + t("【全流程时间线】（logs/full_rerun.log）",
               "[Pipeline timeline] (logs/full_rerun.log)"))
flog = os.path.join(LAB, "logs", "full_rerun.log")
if os.path.exists(flog):
    steps, cur = [], None
    for line in open(flog, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if "] ▶ " in line:
            cur = {"name": line.split("▶ ", 1)[1], "start": line[1:15]}
        elif "] ◀ " in line and cur and line.split("◀ ", 1)[1].startswith(cur["name"]):
            rest = line.split("◀ ", 1)[1]
            cur["done"] = True
            cur["ok"] = "退出码=0" in rest
            m = rest.split("用时=")
            cur["dur_s"] = parse_dur(m[1]) if len(m) > 1 else None
            steps.append(cur)
            cur = None
    for s in steps:
        st = "✅" if s.get("ok") else "❌"
        out = f"  {st} {pad(s['name'], 22)} {dur_str(s.get('dur_s')):>8s}"
        out += "   " + t("起于", "started") + f" {s['start']}"
        print(out)
    if cur:
        try:
            t0 = datetime.datetime.strptime(
                f"{datetime.datetime.now().year}-{cur['start']}", "%Y-%m-%d %H:%M:%S")
            el = (datetime.datetime.now() - t0).total_seconds()
            h = human(el)
            print(f"  ▶  {pad(cur['name'], 22)} " + t(f"已跑 {h}", f"running for {h}"))
        except Exception:
            print(f"  ▶  {pad(cur['name'], 22)} " + t("（进行中）", "(in progress)"))
    if not steps and not cur:
        print("  " + t("（日志还没有步骤记录）", "(no steps recorded yet)"))
else:
    print("  " + t("（尚未启动 97_full_rerun.sh）", "(97_full_rerun.sh not started yet)"))

# ---------------- 四个臂 ----------------
print("\n" + t("【四个臂的产物】（本轮 A1/B/C 各 40 步；A2 本轮跳过）",
               "[Four arms · artifacts] (this round A1/B/C = 40 steps each; A2 skipped)"))
LABELS = [("A1", t("A1 · 蒸馏 40 步", "A1 · distill 40 steps"), "branch-A-distill", "adapter-A1"),
          ("A2", t("A2 · 蒸馏 120 步（本轮跳过）", "A2 · distill 120 steps (skipped)"),
           "branch-A-distill", "adapter-A2"),
          ("B", t("B  · RL 40 步", "B  · RL 40 steps"), "branch-B-rl", "adapter-B"),
          ("C", t("C  · SFT→RL 40 步", "C  · SFT→RL 40 steps"), "branch-B-rl", "adapter-C")]
for key, label, br, ad in LABELS:
    p = os.path.join(LAB, br, ad, "adapter_model.safetensors")
    if os.path.exists(p):
        print(f"  ✅ {pad(label, 36)} {os.path.getsize(p)/1e6:6.1f} MB")
    else:
        # 与上一行的数字列对齐：占同样宽度再写文字
        print(f"  ⏳ {pad(label, 36)} {'—':>6s}     " + t("未产出", "not produced"))

# ---------------- 数据产物 ----------------
print("\n" + t("【数据产物】", "[Data artifacts]"))
for rel, zh, en in (("shared/screen_raw.jsonl", "分层筛查原始采样", "screening raw samples"),
                    ("shared/bench_stratified.jsonl", "分层基准（冻结）", "stratified bench (frozen)"),
                    ("branch-B-rl/data/rl_pool_curated.jsonl", "RL 训练池", "RL training pool"),
                    ("shared/results/eval_raw.jsonl", "评测原始结果", "eval raw results"),
                    ("shared/results/layer_manifest.json", "闸门判定", "gate verdict")):
    p = os.path.join(LAB, rel)
    if os.path.exists(p):
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  ✅ {pad(t(zh, en), 26)} {n:>5d} " + t("条/行", "rows"))
    else:
        print(f"  ⏳ {pad(t(zh, en), 26)} " + t("未产出", "not produced"))

# ---------------- 评测进度（主测按层）----------------
# 为什么要单列这一块：2026-10-10 出过一次「评测集里全是 L1」的沉默事故 ——
# 报告看上去正常，核心命题却无法回答。这里按层摊开，一眼能看出某层是不是空的。
try:
    ev = jl(os.path.join(LAB, "shared", "results", "eval_raw.jsonl"))
    bench = jl(os.path.join(LAB, "shared", "bench_stratified.jsonl"))
    reps = int(json.load(open(os.path.join(LAB, "configs", "lab.config.json"),
                              encoding="utf-8"))["eval"].get("repeats", 3))
    need = collections.Counter(b.get("layer") for b in bench)
    cnt = collections.Counter()
    for r in ev:
        if r.get("kind") == "main":
            cnt[(r["role"], r.get("layer") or "?")] += 1
    if need:
        head = t(f"【评测进度 · 主测按层】（题数×{reps} 采样；base/teacher 由筛查采样复用）",
                 f"[Eval progress · main test by layer] (questions x{reps}; "
                 f"base/teacher reuse screening samples)")
        print("\n" + head)
        print("  " + pad(t("角色", "role"), 11)
              + "".join(f"{l:>13}" for l in ("L1", "L2", "L3"))
              + "  " + pad(t("完成", "done"), 8))
        for role in ("base", "A1", "B", "C", "teacher", "dsv4pro"):
            cells, got_all, tot_all = "", 0, 0
            for l in ("L1", "L2", "L3"):
                g, tl = cnt.get((role, l), 0), need.get(l, 0) * reps
                got_all += g
                tot_all += tl
                mark = "✓" if g >= tl else " "
                cells += f"{g:>5d}/{tl:<7d}{mark}"
            pct = got_all / tot_all * 100 if tot_all else 0
            flag = "✅" if pct >= 100 else "⏳"
            print(f"  {pad(role, 11)}{cells} {flag} {pct:5.1f}%")
except Exception as e:
    print("\n" + t(f"【评测进度】读取失败：{e}", f"[Eval progress] read failed: {e}"))

# 闸门判定
mp = os.path.join(LAB, "shared", "results", "layer_manifest.json")
if os.path.exists(mp):
    try:
        man = json.load(open(mp, encoding="utf-8"))
        c = man.get("counts", {})
        verdict = (t("✅ 通过", "PASS") if man.get("gate_pass") else t("❌ 未通过", "FAIL"))
        print("\n" + t(f"【GATE-1 判定】L1 {c.get('L1',0)} / L2 {c.get('L2',0)} / L3 {c.get('L3',0)}"
                       f"   需要 L2 ≥ {man.get('gate_min_l2',24)}  → {verdict}",
                       f"[GATE-1 verdict] L1 {c.get('L1',0)} / L2 {c.get('L2',0)} / L3 {c.get('L3',0)}"
                       f"   need L2 >= {man.get('gate_min_l2',24)}  -> {verdict}"))
    except Exception:
        pass

# ---------------- 预计 ----------------
print("\n" + t("【预计】", "[Outlook]"))
print("  · " + t("训练已全部完成（A1/B/C 三个 adapter 就位）",
                 "Training complete (adapters A1/B/C in place)"))
print("  · " + t("当前：评测补跑（L2+L3 共 32 题 × 3 采样 × A1/B/C/dsv4pro），约 3~3.5 小时",
                 "Now: eval catch-up (32 q L2+L3 x 3 samples x A1/B/C/dsv4pro), ~3-3.5 h"))
print("  · " + t("之后：报告自动重生成 → 全部结束",
                 "Then: report auto-regenerated -> done"))
print(BAR)
PY
}

if [ "$WATCH" -eq 1 ]; then
    trap 'printf "\n"; exit 0' INT TERM
    while true; do
        clear
        snapshot
        if [ "$LANG_MODE" = "en" ]; then
            printf "\n  Refreshing every %ss - Ctrl-C to quit\n" "$INTERVAL"
        else
            printf "\n  每 %ss 自动刷新 · Ctrl-C 退出\n" "$INTERVAL"
        fi
        sleep "$INTERVAL"
    done
else
    snapshot
fi
