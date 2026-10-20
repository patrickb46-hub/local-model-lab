#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watch_render.py —— 管线进度渲染器（watch.sh 的渲染后端）

为什么单独成文件：渲染逻辑里有大量引号和中文，塞进 `python3 -c "..."`
会被 shell 的引号解析反复咬到。独立成文件后 watch.sh 只负责参数与循环。

环境变量入参：
  LAB_ROOT    实验室根目录
  LOG         主日志路径
  TAILN       额外附带日志尾部行数（0 = 不附带）
  WATCH_LANG  zh | en
"""

import datetime
import os
import re
import subprocess
import sys

LAB = os.environ["LAB_ROOT"]
LOG = os.environ["LOG"]
TAILN = int(os.environ.get("TAILN", "0"))
LANG = os.environ.get("WATCH_LANG", "zh")
BAR = "─" * 68

# ───────────────────────── 双语词条 / bilingual strings ─────────────────────────
T = {
    "zh": {
        "title": "本地模型实验室 · 管线进度",
        "started": "本次运行自 {t} 起，已历时 {d}",
        "now": "当前时间 {t}",
        "no_log": "✗ 还没有日志文件：",
        "no_log_hint": "  说明夜间批处理尚未启动过。启动命令：",
        "steps": "【步骤时间线】",
        "running_word": "进行中",
        "elapsed": "已跑 {d}",
        "pending": "待跑",
        "cur_step": "【当前步骤】",
        "cur_step_at": "  {n}   （开始于 {t}）",
        "no_running": "  没有正在跑的步骤 —— 可能已全部完成，或被中断。",
        "latest_out": "  最新输出：",
        "eta": "【预计完成】",
        "eta_cur": "  当前步骤剩余   约 {d}",
        "eta_rest": "  后续步骤粗估   约 {d}   （横评 {h}h + 报告 + 导出）",
        "eta_all": "  → 预计全部完成   {t} 前后（还需约 {d}）",
        "branches": "【两条支线】",
        "no_adapter": "  {n:<10s} 尚未产出 adapter",
        "adapter_ok": "  {n:<10s} {sz:.1f} MB   正式训练产出（{mt}）",
        "adapter_probe": "  {n:<10s} {sz:.1f} MB   ⚠ 目前是探针产物（{mt}），正式训练进行中",
        "sft_loss": "  SFT  loss  {a} → {b}（正式训练 {n} 步）",
        "resources": "【资源】",
        "procs": "  进程  {x}",
        "procs_none": "无（管线未运行）",
        "mem": "  内存  可用 {g:.2f} GB",
        "disk": "  磁盘  可用 {d}",
        "files": "【已产出文件】",
        "tail": "【日志尾部 {n} 行】",
        "label_A": "A · 蒸馏",
        "label_B": "B · RL",
        "p_pipeline": "批处理", "p_sft": "SFT 训练", "p_rl": "GRPO 训练",
        "p_eval": "评测", "p_report": "报告", "p_sleep": "防睡眠",
        "display": ["环境自检", "teacher 生成蒸馏数据", "清洗 + 验证筛选 + 切分",
                    "支线 A 探针", "支线 A 正式训练", "支线 B 探针", "支线 B 正式训练",
                    "六模型横评", "生成可视化报告", "导出 to Ollama"],
    },
    "en": {
        "title": "Local Model Lab · Pipeline Progress",
        "started": "Started {t} · elapsed {d}",
        "now": "Now {t}",
        "no_log": "✗ No log file yet:",
        "no_log_hint": "  The pipeline has not been started. Start it with:",
        "steps": "[Step timeline]",
        "running_word": "running",
        "elapsed": "running {d}",
        "pending": "pending",
        "cur_step": "[Current step]",
        "cur_step_at": "  {n}   (started {t})",
        "no_running": "  No step is running — either all finished, or interrupted.",
        "latest_out": "  Latest output:",
        "eta": "[Estimated completion]",
        "eta_cur": "  This step remaining   ~{d}",
        "eta_rest": "  Later steps (rough)   ~{d}   (eval {h}h + report + export)",
        "eta_all": "  → All done around {t} (about {d} left)",
        "branches": "[Two branches]",
        "no_adapter": "  {n:<16s} no adapter yet",
        "adapter_ok": "  {n:<16s} {sz:.1f} MB   from full training ({mt})",
        "adapter_probe": "  {n:<16s} {sz:.1f} MB   warning: currently probe artifact ({mt}), full training running",
        "sft_loss": "  SFT  loss  {a} -> {b}  ({n} steps of full training)",
        "resources": "[Resources]",
        "procs": "  procs  {x}",
        "procs_none": "none (pipeline not running)",
        "mem": "  mem    {g:.2f} GB free",
        "disk": "  disk   {d} free",
        "files": "[Produced files]",
        "tail": "[Last {n} log lines]",
        "label_A": "A · distillation",
        "label_B": "B · RL",
        "p_pipeline": "pipeline", "p_sft": "SFT train", "p_rl": "GRPO train",
        "p_eval": "eval", "p_report": "report", "p_sleep": "caffeinate",
        "display": ["Env self-check", "Teacher data generation", "Clean / verify / split",
                    "Branch A probe", "Branch A full training", "Branch B probe",
                    "Branch B full training", "Six-model evaluation", "Visual report",
                    "Export to Ollama"],
    },
}
t = T.get(LANG, T["zh"])

# 日志里写的是中文步骤名，用它做识别；显示时按语言取名字
CN_STEPS = ["环境自检", "teacher 生成蒸馏数据", "清洗 + 验证筛选 + 切分",
            "支线 A 探针", "支线 A 正式训练", "支线 B 探针", "支线 B 正式训练",
            "六模型横评", "生成可视化报告", "导出 to Ollama"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def fmt_dur(sec):
    if sec is None:
        return "   —"
    h, r = divmod(int(sec), 3600)
    mi, s = divmod(r, 60)
    if h:
        return f"{h}h{mi:02d}m"
    if mi:
        return f"{mi}m{s:02d}s"
    return f"{s}s"


# 日志里写的是中文原文，英文模式下做局部替换（只动展示，不动数据）
_EN_LINE = [("正确率", "acc"), ("格式", "fmt"), ("多样", "div"), ("s/步", "s/step")]


def localize_step(title):
    """把日志里的中文步骤名翻成当前语言（含「（60 步）」这类括号说明）。"""
    if not title or LANG == "zh":
        return title
    out = title
    for cn, shown in zip(CN_STEPS, t["display"]):
        if cn in out:
            out = out.replace(cn, shown)
    out = re.sub(r"（(\d+) 步）", r" (\1 steps)", out)
    return out


def localize_line(line):
    """训练进度行是日志原文，英文模式下替换其中的中文指标名。"""
    if not line or LANG == "zh":
        return line
    out = line
    for a, b in _EN_LINE:
        out = out.replace(a, b)
    return out


# ───────────────────────── 读日志 ─────────────────────────
if not os.path.exists(LOG):
    print(t["no_log"], LOG)
    print(t["no_log_hint"])
    print("    cd ~/Desktop/AI/Local-Model-Lab && bash scripts/99_night_run.sh")
    sys.exit(0)

raw = open(LOG, encoding="utf-8", errors="replace").read().splitlines()

start_idx = 0
for i, ln in enumerate(raw):
    if "夜间批处理启动" in ln:
        start_idx = i
raw = raw[start_idx:]

steps, pending = [], []
for ln in raw:
    m = re.match(r"^\[(\d\d-\d\d) (\d\d:\d\d:\d\d)\] (▶|◀|⚠) (.+)$", ln)
    if not m:
        continue
    d, tt, mark, rest = m.groups()
    ts = f"{d} {tt}"
    if mark == "▶":
        pending.append({"title": re.sub(r"\s+$", "", rest), "start": ts})
    elif mark == "◀":
        if not pending:
            continue
        s = pending.pop()
        mm = re.search(r"用时=(\d+)s", rest)
        steps.append({"title": s["title"], "start": s["start"], "end": ts,
                      "dur": int(mm.group(1)) if mm else None,
                      "ok": "退出码=0" in rest})

running = pending[-1] if pending else None

sft = rl = None
for ln in raw:
    m = re.search(r"ep\d+ step (\d+)/(\d+)\s+loss\s+([0-9.]+)", ln)
    if m:
        sft = {"cur": int(m.group(1)), "tot": int(m.group(2)),
               "loss": m.group(3), "line": ln.strip()}
    m = re.search(r"^\s*step (\d+)/(\d+)\s+R=([+-][0-9.]+)", ln)
    if m:
        rl = {"cur": int(m.group(1)), "tot": int(m.group(2)), "line": ln.strip()}

now = datetime.datetime.now()
first_ts = None
for ln in raw:
    m = re.match(r"^\[(\d\d-\d\d \d\d:\d\d:\d\d)\]", ln)
    if m:
        first_ts = m.group(1)
        break

# ───────────────────────── 头部 ─────────────────────────
print(BAR)
print(" " + t["title"])
if first_ts:
    try:
        t0 = datetime.datetime.strptime(f"{now.year}-{first_ts}", "%Y-%m-%d %H:%M:%S")
        el = int((now - t0).total_seconds())
        h, r = divmod(el, 3600)
        mi, s = divmod(r, 60)
        print(" " + t["started"].format(t=first_ts, d=f"{h}h{mi:02d}m"))
    except Exception:
        pass
print(" " + t["now"].format(t=now.strftime("%Y-%m-%d %H:%M:%S")))
print(BAR)

# ───────────────────────── 步骤时间线 ─────────────────────────
run_title = running["title"] if running else None
print("\n" + t["steps"])
for cn, shown in zip(CN_STEPS, t["display"]):
    matched = None
    for s in steps:
        if cn in s["title"]:
            matched = s
    active = run_title and cn in run_title
    if matched:
        mark = "✅" if matched["ok"] else "❌"
        print(f"  {mark} {shown:<26s} {fmt_dur(matched['dur']):>9s}   "
              f"{matched['start']} → {matched['end']}")
    elif active:
        el = t["running_word"]
        try:
            t0 = datetime.datetime.strptime(f"{now.year}-{running['start']}",
                                            "%Y-%m-%d %H:%M:%S")
            esec = int((now - t0).total_seconds())
            h, r = divmod(esec, 3600)
            mi, s = divmod(r, 60)
            el = t["elapsed"].format(d=(f"{h}h{mi:02d}m" if h else f"{mi}m{s:02d}s"))
        except Exception:
            pass
        print(f"  ▶  {shown:<26s} {el:<20s} ← {t['running_word']}")
    else:
        print(f"  ⏳ {shown:<26s} {t['pending']:>9s}")

# ───────────────────────── 当前步骤 ─────────────────────────
print("\n" + t["cur_step"])
if run_title:
    print(t["cur_step_at"].format(n=localize_step(run_title), t=running["start"]))
else:
    print(t["no_running"])

detail = None
if run_title and "支线 B" in run_title and rl:
    detail = ("GRPO", rl["cur"], rl["tot"], rl["line"])
elif run_title and "支线 A" in run_title and sft:
    detail = ("SFT", sft["cur"], sft["tot"], sft["line"])

if detail:
    tag, cur, tot, line = detail
    pct = cur / tot * 100 if tot else 0
    filled = int(pct / 100 * 40)
    print(f"\n  {tag}  [{'█' * filled}{'░' * (40 - filled)}] {cur}/{tot}  {pct:.0f}%")
    print(f"  {localize_line(line)}")
else:
    tail = [l for l in raw[-12:] if l.strip()]
    if tail:
        print("\n" + t["latest_out"])
        for l in tail[-6:]:
            print(f"    {l[:110]}")

# ───────────────────────── 预计完成 ─────────────────────────
AFTER_KEYS = [("横评", 7.0 * 3600), ("报告", 300), ("导出", 1500)]

eta_now = None
if detail:
    mm = re.search(r"ETA\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", detail[3])
    if mm:
        eta_now = (int(mm.group(1) or 0) * 3600 + int(mm.group(2) or 0) * 60
                   + int(mm.group(3) or 0))

if run_title:
    rest, hit = 0.0, False
    for key, sec in AFTER_KEYS:
        if hit:
            rest += sec
        if key in run_title:
            hit = True
    if not hit:
        rest = sum(s for _, s in AFTER_KEYS)
    left = (eta_now or 0) + rest
    finish = now + datetime.timedelta(seconds=left)
    print("\n" + t["eta"])
    if eta_now:
        print(t["eta_cur"].format(d=f"{eta_now // 3600}h{(eta_now % 3600) // 60:02d}m"))
    if rest > 0:
        print(t["eta_rest"].format(d=f"{rest / 3600:.1f}h",
                                   h=f"{AFTER_KEYS[0][1] / 3600:.1f}"))
    print(t["eta_all"].format(t=finish.strftime("%m-%d %H:%M"),
                              d=f"{int(left) // 3600}h{int(left) % 3600 // 60:02d}m"))

# ───────────────────────── 两条支线 ─────────────────────────
print("\n" + t["branches"])
finished = {s["title"] for s in steps if s["ok"]}
for label_key, name, cn_formal in (("label_A", "branch-A-distill", "支线 A 正式训练"),
                                   ("label_B", "branch-B-rl", "支线 B 正式训练")):
    label = t[label_key]
    ad = os.path.join(LAB, name, "adapter", "adapter_model.safetensors")
    if not os.path.exists(ad):
        print(t["no_adapter"].format(n=label))
        continue
    sz = os.path.getsize(ad) / 1e6
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(ad)).strftime("%H:%M:%S")
    tmpl = t["adapter_ok"] if any(cn_formal in x for x in finished) else t["adapter_probe"]
    print(tmpl.format(n=label, sz=sz, mt=mt))

sft_start = None
for i, ln in enumerate(raw):
    if "] ▶ " in ln and "支线 A 正式训练" in ln:
        sft_start = i
seg = raw[sft_start:] if sft_start is not None else []
sft_lines = [l for l in seg if re.search(r"ep\d+ step \d+/\d+\s+loss", l)]
if sft_lines:
    a = re.search(r"loss\s+([0-9.]+)", sft_lines[0])
    b = re.search(r"loss\s+([0-9.]+)", sft_lines[-1])
    if a and b:
        print(t["sft_loss"].format(a=a.group(1), b=b.group(1), n=len(sft_lines)))

# ───────────────────────── 资源 ─────────────────────────
print("\n" + t["resources"])
procs = []
for pat, key in (("99_night_run.sh", "p_pipeline"), ("11_train_distill.py", "p_sft"),
                 ("20_train_rl.py", "p_rl"), ("30_evaluate.py", "p_eval"),
                 ("31_report.py", "p_report"), ("caffeinate", "p_sleep")):
    n = sh(f"pgrep -f '{pat}' | wc -l").strip()
    if n and n != "0":
        procs.append(t[key] + " ✓")
print(t["procs"].format(x=("  ".join(procs) if procs else t["procs_none"])))

out = sh("vm_stat")
v = {}
for l in out.splitlines():
    if ":" in l:
        k, x = l.split(":", 1)
        x = x.strip().rstrip(".")
        if x.isdigit():
            v[k.strip()] = int(x)
avail = (v.get("Pages free", 0) + v.get("Pages inactive", 0)
         + v.get("Pages speculative", 0)) * 4096 / 1e9
print(t["mem"].format(g=avail) + "   |   " + sh("sysctl -n vm.swapusage"))

d = sh("df -h / | tail -1").split()
if len(d) >= 4:
    print(t["disk"].format(d=d[3]))

res = os.path.join(LAB, "shared", "results")
if os.path.isdir(res):
    items = [f for f in os.listdir(res) if not f.startswith(".")]
    if items:
        print("\n" + t["files"])
        for f in sorted(items):
            print(f"  {f}   ({os.path.getsize(os.path.join(res, f)) / 1024:.0f} KB)")

if TAILN > 0:
    print("\n" + t["tail"].format(n=TAILN))
    for l in raw[-TAILN:]:
        print("  " + l[:150])

print("\n" + BAR)
