#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_board.py —— 三栏实时看板 + 演示训练编排（单进程）

为什么是单进程而不是「bash 编排 + python 看板」：
  双进程要处理僵尸进程、信号转发、退出码捕获、看板比训练活得久……
  每一条都足以让一次 13 分钟的录制作废。这里让同一个进程既管
  subprocess 又画屏，用 proc.wait() 拿退出码，没有僵尸、没有时序竞争。

为什么不用 clear 刷新：
  clear 会让整屏先变空再重画，录屏里就是一次闪白。这里用
  \\033[H 光标归位 + 行尾 \\033[K 清残字 + 末尾 \\033[J 清多余行。

为什么所有产物都带 DEMO- 前缀：
  11_train_distill.py 的 jsonl 是 open(...,"a") 追加的。同名 tag 二次运行
  会让看板一开场就是满格 —— 整段视频都在说谎。所以启动前必须轮转旧 jsonl，
  且演示产物永远落在 adapter-DEMO-*/，绝不碰 adapter-A1/B/C。

答案流（RL 实时判定）：
  20_train_rl.py 会把每条采样（题目 + 解答文本 + 对错）写到 *.samples.jsonl。
  跑 B / C 时，看板在底部把它渲染成「答案流」：题目、标准答案、8 条采样的
  解答与 ✓/✗，加上累计正确率和迷你趋势图 —— 让「RL 进步过程」在终端连续可见。

用法：
  bash scripts/demo_run.sh A            # 演示：真跑支线 A（推荐入口）
  python3 scripts/demo_board.py --run A
  python3 scripts/demo_board.py --dry-run A     # 只看将执行的命令
  python3 scripts/demo_board.py --replay branch-B-rl/logs/rl_train-C.jsonl --leg B --steps 2
  python3 scripts/demo_board.py --summary A     # 跑完后的真实摘要

只依赖标准库；兼容系统自带的 Python 3.9。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata

# ─────────────────────────────── 路径与常量 ───────────────────────────────
LAB_ROOT = os.environ.get("LAB_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(LAB_ROOT, "logs", "demo")
PLAN_PATH = os.path.join(DEMO_DIR, "plan.json")
# 训练解释器：优先环境变量 LAB_PY，其次 config.runtime.python，最后退回当前解释器
PY = os.environ.get("LAB_PY") or \
     os.path.expanduser("~/.workbuddy/binaries/python/envs/mlab-conda/bin/python")
if not os.path.exists(PY):
    PY = sys.executable

RULE = "─"
ESC = "\033["

# tqdm / 进度条行（权重加载时用 \r 覆盖，落到文件里是垃圾行）：读日志尾部时过滤掉
_TQDM = re.compile(r"\d+%\|[^|]*\||\|\s*\d+/\d+\s*\[|\d+\.\d+\s*(?:it|s)/s")

# 整行着色：先拼好纯文本再包色，所以不影响宽度计算
C_DONE, C_RUN, C_PEND, C_FAIL, C_DIM, C_HDR, C_RST = (
    "\033[32m", "\033[36m", "\033[90m", "\033[31m", "\033[2m", "\033[1m", "\033[0m")

LEGS = [
    {
        "id": "A", "zh": "支线 A · 蒸馏", "en": "Branch A · Distill",
        "kind": "sft", "tag": "DEMO-A", "steps": 4, "full": 40,
        "est": 130.0, "load": 110.0,
        "jsonl": "branch-A-distill/logs/train_distill-DEMO-A.jsonl",
        "sumjson": "branch-A-distill/logs/train_distill_summary-DEMO-A.json",
        "samples": None,
        "adapter": "branch-A-distill/adapter-DEMO-A",
        "anchor": "branch-A-distill/adapter-A1",   # 诚实性锚点：真实产物
        "script": "scripts/11_train_distill.py",
        "note_zh": "teacher 生成 → 清洗 → LoRA SFT",
        "note_en": "teacher -> clean -> LoRA SFT",
    },
    {
        "id": "B", "zh": "支线 B · 强化学习", "en": "Branch B · RL",
        "kind": "rl", "tag": "DEMO-B", "steps": 2, "full": 40,
        "est": 344.0, "load": 110.0,
        "jsonl": "branch-B-rl/logs/rl_train-DEMO-B.jsonl",
        "sumjson": "branch-B-rl/logs/rl_train_summary-DEMO-B.json",
        "samples": "branch-B-rl/logs/rl_train-DEMO-B.samples.jsonl",
        "adapter": "branch-B-rl/adapter-DEMO-B",
        "anchor": "branch-B-rl/adapter-B",
        "script": "scripts/20_train_rl.py",
        "note_zh": "从 base 起，验证器驱动",
        "note_en": "from base, verifier-driven",
    },
    {
        "id": "C", "zh": "支线 C · SFT→RL", "en": "Branch C · SFT->RL",
        "kind": "rl", "tag": "DEMO-C", "steps": 2, "full": 40,
        "est": 224.0, "load": 110.0,
        "jsonl": "branch-B-rl/logs/rl_train-DEMO-C.jsonl",
        "sumjson": "branch-B-rl/logs/rl_train_summary-DEMO-C.json",
        "samples": "branch-B-rl/logs/rl_train-DEMO-C.samples.jsonl",
        "adapter": "branch-B-rl/adapter-DEMO-C",
        "anchor": "branch-B-rl/adapter-C",
        "script": "scripts/20_train_rl.py",
        "note_zh": "预热自 adapter-A1",
        "note_en": "warm-started from adapter-A1",
    },
]
LEG_BY_ID = dict((x["id"], x) for x in LEGS)


# 命令一个字都不能改：RL 的 --init-adapter 被 `if not args.fresh` 挡住，
# 漏了 --fresh 就会静默从上次的 adapter-DEMO-C 续训，「预热自 A1」变成谎话。
def build_cmd(leg, steps, lab_root=None):
    root = lab_root or LAB_ROOT
    lid = leg["id"]
    if lid == "A":
        return [PY, "-u", "scripts/11_train_distill.py",
                "--fresh", "--tag", "DEMO-A", "--max-steps", str(steps)]
    if lid == "B":
        return [PY, "-u", "scripts/20_train_rl.py",
                "--fresh", "--tag", "DEMO-B", "--steps", str(steps)]
    return [PY, "-u", "scripts/20_train_rl.py",
            "--fresh", "--tag", "DEMO-C", "--steps", str(steps),
            "--init-adapter", os.path.join(root, "branch-A-distill", "adapter-A1")]


# ─────────────────────────── 显示宽度（CJK 安全） ───────────────────────────
# 这几个符号在 Terminal.app 里是 1 列，但在别的终端可能被当成 2 列。
# 录屏用 Terminal.app 就保持默认；换终端时用 DEMO_AMBIG_WIDE=1 纠正。
_AMBIG = set("█░▶") if os.environ.get("DEMO_AMBIG_WIDE") == "1" else set()


def cw(ch):
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 2 if ch in _AMBIG else 1


def vlen(s):
    return sum(cw(c) for c in s)


def cut(s, w):
    """按显示宽度截断，超宽补省略号。"""
    if w <= 0:
        return ""
    if vlen(s) <= w:
        return s
    out, acc = "", 0
    for c in s:
        n = cw(c)
        if acc + n > w - 1:
            break
        out += c
        acc += n
    return out + "…"


def pad(s, w, align="<"):
    s = cut(s, w)
    d = max(0, w - vlen(s))
    if align == ">":
        return " " * d + s
    if align == "^":
        left = d // 2
        return " " * left + s + " " * (d - left)
    return s + " " * d


def join_fit(items, w, sep="   "):
    """指标按宽度预算贪心保留：从右往左删，永远保住第一项（最重要）。"""
    items = [x for x in items if x]
    while len(items) > 1 and vlen(sep.join(items)) > w:
        items = items[:-1]
    return cut(sep.join(items), w)


# ─────────────────────────────── 时间格式 ───────────────────────────────
def hms(sec):
    """90 -> 1m30s；5400 -> 1h30m；90000 -> 1d1h。"""
    sec = int(max(0, sec))
    if sec < 60:
        return "%ds" % sec
    if sec < 3600:
        return "%dm%02ds" % (sec // 60, sec % 60)
    if sec < 86400:
        return "%dh%02dm" % (sec // 3600, (sec % 3600) // 60)
    return "%dd%dh" % (sec // 86400, (sec % 86400) // 3600)


def clock(ts=None):
    return dt.datetime.fromtimestamp(ts or time.time()).strftime("%H:%M:%S")


# ─────────────────────────────── 状态词条 ───────────────────────────────
WORDS = {
    "pending": {"zh": "待跑", "en": "PENDING"},
    "running": {"zh": "正在跑", "en": "RUNNING"},
    "loading": {"zh": "载入中", "en": "LOADING"},
    "done":    {"zh": "已完成", "en": "DONE"},
    "failed":  {"zh": "失败", "en": "FAILED"},
    "stale":   {"zh": "无响应", "en": "STALE"},
}
MARKS = {"pending": "·", "running": "▶", "loading": "▶", "done": "✓",
         "failed": "✗", "stale": "?"}


def words(state, lang):
    w = WORDS.get(state, WORDS["pending"])
    # both 模式用中文：状态已经由 ✓ ▶ · 三重编码，中文词服务主体观众
    return w["en"] if lang == "en" else w["zh"]


def leg_title(leg, lang):
    if lang == "zh":
        return leg["zh"]
    if lang == "en":
        return leg["en"]
    return "%s  /  %s" % (leg["zh"], leg["en"])


# ─────────────────────────────── 数据读取 ───────────────────────────────
def read_json(path):
    """容错读 JSON：写到一半、文件不存在、格式坏了，都返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        pass


def read_records(path, cap=262144):
    """只读末尾 256 KB；半行（正在 flush）直接丢弃，绝不抛异常。"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - cap))
            blob = f.read()
    except OSError:
        return []
    text = blob.decode("utf-8", "replace")
    if size > cap and "\n" in text:
        text = text.split("\n", 1)[1]      # 丢掉被截断的首行
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def read_samples(path, cap=131072):
    """读答案流 sidecar 的末尾若干行（每行含完整解答文本，可能较长）。"""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - cap))
            blob = f.read()
    except OSError:
        return []
    text = blob.decode("utf-8", "replace")
    if size > cap and "\n" in text:
        text = text.split("\n", 1)[1]      # 丢掉被截断的首行
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def read_tail(path, n=5, maxcol=110):
    """
    取日志尾部若干行做「这是真进程」的证据。
    权重加载的 tqdm 用 \\r 覆盖，落到文件里是上千字符的垃圾行 —— 必须过滤掉，
    否则 tail 全是它，反而看不出训练在干什么。
    """
    if n <= 0 or not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            blob = f.read()
    except OSError:
        return []
    text = blob.decode("utf-8", "replace")
    parts = re.split(r"[\r\n]+", text)
    keep = []
    for raw in parts:
        s = raw.rstrip()
        if not s:
            continue
        if _TQDM.search(s):
            continue
        if len(s) > maxcol * 3:
            s = s[:maxcol * 3]
        keep.append(s.strip())
    return keep[-n:]


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def dir_size(path):
    total = 0
    try:
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    except OSError:
        return 0
    return total


def age_of(path):
    try:
        return max(0.0, time.time() - os.path.getmtime(path))
    except OSError:
        return None


# ─────────────────────────────── plan.json ───────────────────────────────
def new_plan(lang, refresh, intro, steps_map):
    legs = []
    for leg in LEGS:
        steps = steps_map.get(leg["id"], leg["steps"])
        legs.append({
            "id": leg["id"],
            "steps": steps,
            "full_steps": leg["full"],
            "est_step_s": leg["est"],
            "load_s": leg["load"],
            "cmd": build_cmd(leg, steps),
            "jsonl": leg["jsonl"],
            "samples": leg.get("samples"),
            "raw_log": os.path.join("logs", "demo", leg["id"] + ".log"),
            "exit_file": os.path.join("logs", "demo", leg["id"] + ".exit"),
            "adapter": leg["adapter"],
            "anchor": leg["anchor"],
            "status": "pending",
            "pid": None,
            "started_epoch": None,
            "exit_code": None,
            "ended_epoch": None,
            "adapter_mtime_before": None,
            "anchor_mtime_before": None,
        })
    return {
        "schema": "demo-plan/1",
        "lab_root": LAB_ROOT,
        "run_id": dt.datetime.now().strftime("%Y%m%d-%H%M"),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "lang": lang,
        "refresh_s": refresh,
        "intro_s": intro,
        "legs": legs,
    }


def load_plan():
    p = read_json(PLAN_PATH)
    if not isinstance(p, dict) or "legs" not in p:
        return None
    return p


def plan_leg(plan, lid):
    if not plan:
        return None
    for x in plan.get("legs", []):
        if x.get("id") == lid:
            return x
    return None


def update_plan(mutator):
    plan = load_plan() or new_plan("both", 1, 6, {})
    mutator(plan)
    write_json(PLAN_PATH, plan)
    return plan


def abspath(rel):
    return rel if os.path.isabs(rel) else os.path.join(LAB_ROOT, rel)


# ─────────────────────────────── 状态判定 ───────────────────────────────
def leg_view(leg, pleg, running_id):
    """
    把「文件里的事实」拼成一个渲染用的视图。
    优先级：exit 哨兵 > 进程是否在跑 > 日志是否新鲜。
    """
    steps = int((pleg or {}).get("steps") or leg["steps"])
    # jsonl 也以 plan 为准：replay / --steps 会改写它
    jsonl_rel = (pleg or {}).get("jsonl") or leg["jsonl"]
    recs = read_records(abspath(jsonl_rel))
    done = len(recs)
    jsonl_path = abspath(jsonl_rel)
    log_path = abspath((pleg or {}).get("raw_log")
                       or os.path.join("logs", "demo", leg["id"] + ".log"))
    exit_path = abspath((pleg or {}).get("exit_file")
                        or os.path.join("logs", "demo", leg["id"] + ".exit"))

    exit_code = None
    if os.path.exists(exit_path):
        try:
            with open(exit_path, "r", encoding="utf-8") as f:
                exit_code = int((f.read() or "").strip() or "0")
        except (OSError, ValueError):
            exit_code = 0

    log_age = age_of(log_path)
    is_current = (leg["id"] == running_id)

    if exit_code is not None:
        state = "done" if exit_code == 0 else "failed"
    elif is_current:
        # 当前这条：还没有任何 jsonl 行，就是还在加载模型
        state = "loading" if done == 0 else "running"
        if log_age is not None and log_age > 300:
            state = "stale"
    elif (pleg or {}).get("status") == "done":
        # exit 哨兵被清理时，仍以 plan.json 记录的状态为准
        state = "done"
    elif done > 0 and log_age is not None and log_age < 86400:
        # 之前跑过（本次会话里已完成，但 exit 文件被清了）
        state = "done"
    else:
        state = "pending"

    last = recs[-1] if recs else {}
    return {
        "leg": leg, "steps": steps, "done": min(done, steps),
        "raw_done": done, "recs": recs, "last": last,
        "state": state, "exit_code": exit_code,
        "log_path": log_path, "log_age": log_age,
        "log_bytes": file_size(log_path),
        "jsonl_path": jsonl_path,
        "samples": read_samples(abspath(leg.get("samples") or "")),
    }


def eta_seconds(leg, v):
    steps, done = v["steps"], v["done"]
    if done >= steps:
        return 0
    est = float(leg["est"])
    load = float(leg["load"])
    if done >= 2:
        if leg["kind"] == "rl":
            ss = [float(r.get("step_s") or 0) for r in v["recs"][1:]]
            ss = [x for x in ss if x > 0]
            per = sum(ss) / len(ss) if ss else est
        else:
            els = [float(r.get("elapsed_s") or 0) for r in v["recs"]]
            d = sorted([b - a for a, b in zip(els, els[1:]) if b > a])
            # 首步含 warmup，用中位数而不是均值
            per = d[len(d) // 2] if d else est
    elif done == 1:
        per = est          # 只有一步时，那一步含载入，不可外推
    else:
        return steps * est + load
    return per * (steps - done)


# ─────────────────────────────── 渲染 ───────────────────────────────
def bar(pct, w):
    w = max(6, w)
    f = int(round(pct / 100.0 * w))
    f = max(0, min(w, f))          # 浮点会越界，必须夹住
    return "[" + "█" * f + "░" * (w - f) + "]"


def metrics_text(leg, v):
    last = v["last"]
    if leg["kind"] == "sft":
        if not last:
            return []
        return [
            "loss %.4f" % float(last.get("loss") or 0),
            "lr %.1e" % float(last.get("lr") or 0),
            "mem %.2fG" % float(last.get("peak_mem_gb") or 0),
        ]
    if not last:
        return []
    eff = "✓更新" if last.get("effective") else "✗空步"
    return [
        "R %+.3f±%.3f" % (float(last.get("mean_reward") or 0),
                          float(last.get("std_reward") or 0)),
        "acc %.0f%%" % (100 * float(last.get("accuracy") or 0)),
        "div %.0f%%" % (100 * float(last.get("diversity") or 0)),
        eff,
    ]


def timing_text(leg, v, lang, beat):
    st, steps = v["state"], v["steps"]
    last = v["last"]
    zh = (lang != "en")
    if st == "loading":
        el = v["log_age"] or 0
        kb = v["log_bytes"] / 1024.0
        if zh:
            return "正在加载权重与 tokenizer %s · 已输出 %.1f KB %s" % (
                hms(el), kb, beat)
        return "loading weights + tokenizer %s · %.1f KB written %s" % (
            hms(el), kb, beat)
    if st in ("running", "stale"):
        cur = v["done"]
        # SFT 的 jsonl 只有 elapsed_s，没有 step_s —— 用相邻两步的差值算本步耗时
        if leg["kind"] == "rl":
            per = float(last.get("step_s") or 0) or float(leg["est"])
        else:
            els = [float(r.get("elapsed_s") or 0) for r in v["recs"]]
            if len(els) >= 2:
                per = els[-1] - els[-2]
            elif els:
                per = els[-1]
            else:
                per = float(leg["est"])
        el = float(last.get("elapsed_s") or 0)
        eta = eta_seconds(leg, v)
        if zh:
            s = "%d/%d 步 · 本步 %s · 累计 %s · ETA %s" % (
                cur, steps, hms(per), hms(el), hms(eta))
            if st == "stale":
                s += " · 日志已静默 %s" % hms(v["log_age"] or 0)
            return s + " " + beat
        s = "step %d/%d · %s/step · elapsed %s · ETA %s" % (
            cur, steps, hms(per), hms(el), hms(eta))
        if st == "stale":
            s += " · silent for %s" % hms(v["log_age"] or 0)
        return s + " " + beat
    if st == "done":
        els = [float(r.get("elapsed_s") or 0) for r in v["recs"]]
        wall = max(els) if els else 0.0
        avg = wall / len(els) if els else 0.0
        mb = dir_size(abspath(leg["adapter"])) / 1e6
        if zh:
            return "总计 %s · 平均 %ss/步 · 产物 %s  %.1f MB" % (
                hms(wall), ("%.0f" % avg), leg["adapter"].split("/")[-1], mb)
        return "wall %s · avg %ss/step · %s  %.1f MB" % (
            hms(wall), ("%.0f" % avg), leg["adapter"].split("/")[-1], mb)
    if st == "failed":
        if zh:
            return "退出码 %s · 训练未跑完，产物可能未保存" % v["exit_code"]
        return "exit %s · run did not finish; adapter may be unsaved" % v["exit_code"]
    # pending
    tot = steps * float(leg["est"]) + float(leg["load"])
    if zh:
        return "预计 %d × %ss + 载入 %s ≈ %s  （%s）" % (
            steps, ("%.0f" % leg["est"]), hms(leg["load"]), hms(tot),
            leg["note_zh"])
    return "est. %d x %ss + load %s ~ %s  (%s)" % (
        steps, ("%.0f" % leg["est"]), hms(leg["load"]), hms(tot),
        leg["note_en"])


def render_leg(leg, v, W, lang, beat, compact=False):
    mark = MARKS.get(v["state"], "·")
    left = "%s %s" % (mark, leg_title(leg, lang))
    right = "%s %d/%d" % (words(v["state"], lang), v["done"], v["steps"])
    if v["state"] not in ("pending",):
        pct = 100.0 * v["done"] / v["steps"] if v["steps"] else 0.0
        right += " %3.0f%%" % pct
    else:
        pct = 0.0

    gap = W - vlen(left) - vlen(right) - 2
    if gap < 2:
        left = "%s %s · %s" % (mark, leg["id"],
                               leg["en"] if lang == "en" else leg["zh"])
        gap = max(1, W - vlen(left) - vlen(right) - 1)
    head = " " + left + " " * gap + right

    bw = max(8, min(28, W - 52))
    mtxt = join_fit(metrics_text(leg, v), max(10, W - bw - 22))
    body = "   " + bar(pct, bw) + " %d/%d" % (v["done"], v["steps"])
    if mtxt:
        body += "   " + mtxt
    tail = "   " + timing_text(leg, v, lang, beat)

    lines = [head]
    if not compact:
        lines.append(body)
    lines.append(tail)

    color = {"done": C_DONE, "running": C_RUN, "loading": C_RUN,
             "pending": C_PEND, "failed": C_FAIL, "stale": C_FAIL}.get(
                 v["state"], "")
    if color:
        lines = [color + x + C_RST for x in lines]
    return lines


# ─────────────────────────── 答案流（RL 实时判定） ───────────────────────────
_ANS_RE = re.compile(r"【答案】\s*([^\n]*)")


def _collapse(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _answer_tail(text):
    m = _ANS_RE.search(text or "")
    return m.group(1).strip() if m else None


def spark(accs, w=24):
    """准确率序列 → 迷你趋势图（▁▂▃▄▅▆▇█，按绝对值 0..1 映射）。"""
    accs = list(accs)[-w:]
    if not accs:
        return "—"
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[max(0, min(7, int(round(a * 7))))] for a in accs)


def answer_stream(v, W, lang, beat):
    """RL 专用：把「本步采样 + 判定」画成实时答案流。"""
    zh = (lang != "en")
    samples = v.get("samples") or []
    by_step, latest = {}, None
    for s in samples:
        st = s.get("step")
        by_step.setdefault(st, []).append(s)
        if latest is None or st > latest:
            latest = st
    cur = by_step.get(latest, []) if latest is not None else []
    accs = [float(r.get("accuracy") or 0) for r in v.get("recs", [])]
    total_ok = sum(1 for x in samples if x.get("correct"))
    total_n = len(samples)

    if zh:
        head = " 答案流 · 实时判定   step %s · 组内 %d 条采样" % (
            latest if latest is not None else "-", len(cur))
    else:
        head = " answer stream · live verifier   step %s · %d rollouts" % (
            latest if latest is not None else "-", len(cur))
    lines = [C_HDR + pad(head, W) + C_RST]

    first = cur[0] if cur else (samples[-1] if samples else None)
    if first:
        q = _collapse(first.get("prompt", ""))
        g = _collapse(first.get("gold", ""))
        if zh:
            lines.append(pad("   题目  " + cut(q, W - 12), W))
            lines.append(pad("   标准答案  " + cut(g, W - 14), W))
        else:
            lines.append(pad("   Q    " + cut(q, W - 10), W))
            lines.append(pad("   gold " + cut(g, W - 10), W))

    if not samples:
        note = ("   正在采样第一组答案… %s" % beat) if zh else (
            "   sampling first batch... %s" % beat)
        lines.append(C_DIM + pad(note, W) + C_RST)
    else:
        for i, s in enumerate(cur):
            ok = bool(s.get("correct"))
            mark = "✓" if ok else "✗"
            color = C_DONE if ok else C_FAIL
            ans = _answer_tail(s.get("text", ""))
            body = _collapse(s.get("text", ""))
            if ans:
                idx = body.find("【答案】")
                head = (body[:idx].strip() if idx >= 0 else body) or body
                disp = cut(head, max(8, W - 36)) + "  【答案】" + cut(ans, 18)
            else:
                disp = cut(body, max(12, W - 12))
            disp = cut(disp, max(20, W - 12))
            lines.append("   #%d %s%s%s  %s" % (i + 1, color, mark, C_RST, disp))

    if cur and len(cur) < 8 and v["state"] in ("running", "loading", "stale"):
        note = ("   已到 %d 条，仍在采样… %s" % (len(cur), beat)) if zh else (
            "   %d so far, sampling... %s" % (len(cur), beat))
        lines.append(C_DIM + pad(note, W) + C_RST)

    if total_n:
        cur_ok = sum(1 for x in cur if x.get("correct"))
        rate = total_ok / total_n
        foot = " 本步 ✓ %d/%d · 累计正确率 %d%%（%d/%d）· 趋势 %s" % (
            cur_ok, len(cur), int(100 * rate + 0.5), total_ok, total_n, spark(accs))
        if not zh:
            foot = " step ✓ %d/%d · acc %d%% (%d/%d) · trend %s" % (
                cur_ok, len(cur), int(100 * rate + 0.5), total_ok, total_n, spark(accs))
        lines.append(pad(cut(foot, W - 2), W))
    return lines


def render(views, plan, W, H, lang, tick, running_id, mode=None):
    beat = "|/-\\"[tick % 4]
    compact = (mode == "compact" if mode else H < 23)
    minimal = (mode == "minimal" if mode else H < 15)

    out = []
    # 标题：宽度不够就依次退让 —— 英文短版 → 只留中文，绝不截出半个词
    l = " 本地模型实验室 · 串行演示"
    stamp = " %s [%d]" % (clock(), tick)
    avail = W - vlen(stamp) - 1
    r = ""
    for cand in ("Local Model Lab · Serial Demo", "Serial Demo", ""):
        if vlen(l) + vlen(cand) + 2 <= avail:
            r = cand
            break
    line = l
    if r:
        line += " " * (avail - vlen(l) - vlen(r)) + r
    else:
        line = pad(line, avail)
    out.append(C_HDR + line + " " + stamp + C_RST)

    if not minimal:
        steps_map = dict((x["id"], "%d/%d" % (
            next((v["steps"] for v in views if v["leg"]["id"] == x["id"]), 0),
            x["full"])) for x in LEGS)
        note = " 演示模式：仅步数缩小（A %s · B %s · C %s），每步计算量与全量一致，未加速。" % (
            steps_map["A"], steps_map["B"], steps_map["C"])
        out.append(C_DIM + pad(note, W) + C_RST)
    out.append(pad(RULE * W, W))

    if minimal:
        for v in views:
            line = " %s %s %s %d/%d %s" % (
                MARKS.get(v["state"], "·"), v["leg"]["id"],
                pad(leg_title(v["leg"], lang), 22), v["done"], v["steps"],
                words(v["state"], lang))
            out.append(pad(line, W))
    else:
        for i, v in enumerate(views):
            if i and not compact:
                out.append(pad(RULE * W, W))
            out.extend(render_leg(v["leg"], v, W, lang, beat, compact))

    # 原始日志尾部 / RL 答案流：证明这是真进程，不是画出来的图
    if not minimal:
        cur = next((v for v in views if v["leg"]["id"] == running_id), None)
        if cur:
            room = H - len(out) - 2
            if cur["leg"]["kind"] == "rl":
                # RL：答案流面板（题目 + 解答 + 对错 + 趋势），比原始日志更直观
                stream = answer_stream(cur, W, lang, beat)
                n = max(0, min(len(stream), room))
                out.extend(stream[:n])
            else:
                ag = cur["log_age"]
                rel = "%s" % (hms(ag) + " 前" if ag is not None else "未创建")
                if lang == "en":
                    rel = "%s ago" % hms(ag) if ag is not None else "not created"
                head = (" 原始输出 · %s · 更新 %s" % (
                    os.path.basename(cur["log_path"]), rel)) if lang != "en" else (
                    " raw output · %s · updated %s" % (
                        os.path.basename(cur["log_path"]), rel))
                n = 3 if compact else 5
                n = max(0, min(n, room))
                if n > 0:
                    out.append(C_DIM + pad(head, W) + C_RST)
                    tail = read_tail(cur["log_path"], n)
                    if not tail:
                        tail = ["  (等待首批输出…)" if lang != "en" else "  (waiting for first output...)"]
                    for t in tail[-n:]:
                        out.append(C_DIM + pad("   │ " + t, W) + C_RST)

    if not minimal:
        foot = " 16GB 内存只能串行：同一时刻仅一栏在动"
        if lang == "en":
            foot = " 16 GB RAM -> serial only: exactly one branch runs at a time"
        out.append(C_DIM + pad(foot, W) + C_RST)

    # 每行都补 \033[K：彩色行的 \033[0m 之后同样要清到行尾，
    # 否则上一帧更长时会留下残字。
    return [x + ESC + "K" for x in out[:max(1, H - 1)]]


# ─────────────────────────── 主循环 ───────────────────────────
def term_size():
    try:
        s = os.get_terminal_size(1)
        return max(60, s.columns), max(12, s.lines)
    except OSError:
        return 100, 30


def build_views(plan, running_id):
    views = []
    for leg in LEGS:
        views.append(leg_view(leg, plan_leg(plan, leg["id"]), running_id))
    return views


def draw(view_lines):
    sys.stdout.write(ESC + "H" + "\n".join(view_lines) + ESC + "J")
    sys.stdout.flush()


def hide_cursor():
    sys.stdout.write(ESC + "?25l")
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write(ESC + "?25h\n")
    sys.stdout.flush()


# ─────────────────────────── 演示模式声明 ───────────────────────────
def disclaimer(steps_map, lang):
    zh = [
        "",
        "  演示模式 / DEMO MODE",
        "",
        "  这是真实训练，不是模拟。为把镜头压到 10 分钟量级，",
        "  只有步数被缩小 —— 每步的计算量、精度、batch 完全未变：",
        "",
        "     A 蒸馏     %d 步（全量 %d 步，约 %s）",
        "     B RL      %d 步（全量 %d 步，约 %s）",
        "     C SFT→RL  %d 步（全量 %d 步，约 %s）",
        "",
        "  每步 130s / 344s / 224s 与全量实验实测一致，没有任何加速。",
        "  产物一律带 DEMO- 前缀，绝不触碰 adapter-A1 / adapter-B / adapter-C。",
        "",
    ]
    en = [
        "",
        "  DEMO MODE",
        "",
        "  This is a real training run, not a simulation.",
        "  Only the step count is reduced; per-step compute is unchanged:",
        "",
        "     A Distill   %d steps (full run: %d, ~%s)",
        "     B RL        %d steps (full run: %d, ~%s)",
        "     C SFT->RL   %d steps (full run: %d, ~%s)",
        "",
        "  130s / 344s / 224s per step match the full runs. Nothing is accelerated.",
        "  All artifacts carry a DEMO- prefix. adapter-A1/B/C are never touched.",
        "",
    ]
    full = {"A": 40, "B": 40, "C": 40}
    wall = {"A": "87 min", "B": "3.8 h", "C": "2.5 h"}
    def_steps = {"A": 4, "B": 2, "C": 2}
    tags = ("A", "B", "C")
    rows = []
    ti = 0                      # 三行含 %d 的声明依次对应 A / B / C
    for i in range(len(zh)):
        line = en[i] if lang == "en" else zh[i]
        if "%d" in line:
            tag = tags[ti]
            line = line % (steps_map.get(tag, def_steps[tag]),
                           full[tag], wall[tag])
            ti += 1
        rows.append(line)
    return rows


def show_intro(steps_map, lang, seconds, W, H):
    if seconds <= 0:
        return
    rows = disclaimer(steps_map, lang)
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        body = [pad(RULE * W, W)]
        body.extend(pad(r, W) for r in rows)
        body.append(pad(RULE * W, W))
        countdown = ("  开始倒计时 %ds …" % int(left + 0.99)) if lang != "en" \
            else ("  starting in %ds ..." % int(left + 0.99))
        # 声明卡片居中：上方留白，倒计时贴底
        top_pad = max(0, (H - 1 - len(body) - 2) // 2)
        body = [""] * top_pad + body
        while len(body) < H - 2:
            body.append("")
        body.append(C_DIM + pad(countdown, W) + C_RST)
        draw(body[:H - 1])
        time.sleep(0.25)


# ─────────────────────────── 运行一条腿 ───────────────────────────
def rotate_leg(leg, run_id):
    """
    训练脚本的 jsonl 是 open(...,"a") 追加的。同名 tag 二次运行会把旧行
    留在文件里，看板一开场就是满格 —— 整个演示在说谎。所以先归档。
    """
    bucket = os.path.join(DEMO_DIR, "history", run_id)
    os.makedirs(bucket, exist_ok=True)
    rels = [leg["jsonl"], leg["sumjson"]]
    if leg.get("samples"):
        rels.append(leg["samples"])
    for rel in rels:
        src = abspath(rel)
        if os.path.exists(src):
            try:
                shutil.move(src, os.path.join(
                    bucket, leg["id"] + "-" + os.path.basename(src)))
            except OSError:
                try:
                    os.remove(src)
                except OSError:
                    pass


def run_leg(lid, steps, lang, refresh, intro, tailn, dry_run=False,
            no_board=False, all_mode=False):
    leg = LEG_BY_ID[lid]
    steps_map = {"A": 4, "B": 2, "C": 2}
    steps_map[lid] = steps
    cmd = build_cmd(leg, steps)

    if dry_run:
        print("\n  leg %s  (%s / %s)" % (lid, leg["zh"], leg["en"]))
        print("  cwd  %s" % LAB_ROOT)
        pretty = " ".join(("'%s'" % c) if " " in c else c for c in cmd)
        print("  cmd  %s" % pretty)
        print("  jsonl    %s" % leg["jsonl"])
        print("  adapter  %s" % leg["adapter"])
        print("  anchor   %s  (真产物，必须不变)\n" % leg["anchor"])
        return 0

    os.makedirs(DEMO_DIR, exist_ok=True)
    plan = load_plan()
    if not plan:
        plan = new_plan(lang, refresh, intro, steps_map)
    plan["lang"], plan["refresh_s"], plan["intro_s"] = lang, refresh, intro

    rotate_leg(leg, plan["run_id"])

    exit_path = os.path.join(DEMO_DIR, lid + ".exit")
    log_path = os.path.join(DEMO_DIR, lid + ".log")
    for p in (exit_path,):
        try:
            os.remove(p)
        except OSError:
            pass

    # 更新本腿为 running，并记录真实产物的 mtime 作为诚实性锚点
    def mut(p):
        for x in p["legs"]:
            if x["id"] == lid:
                x["steps"] = steps
                x["cmd"] = cmd
                x["status"] = "running"
                x["started_epoch"] = time.time()
                x["exit_code"] = None
                x["ended_epoch"] = None
                x["adapter_mtime_before"] = None
                x["anchor_mtime_before"] = _mtime(abspath(leg["anchor"]))
    update_plan(mut)

    W, H = term_size()
    print("\n" * 2)
    if not no_board:
        show_intro(steps_map, lang, intro, W, H)

    # 输出直落文件：管道有缓冲区死锁风险，文件没有
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    t_start = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=LAB_ROOT, stdout=logf,
                                stderr=subprocess.STDOUT, env=_env())
    except OSError as e:
        logf.close()
        print("  ✗ 启动失败：%s" % e)
        return 3

    rc = None
    tick = 0
    fin_at = None          # 进程结束后的时刻，用来多停两帧让摄像机拍到终态
    if not sys.stdout.isatty() or no_board:
        rc = proc.wait()
        logf.close()
    else:
        hide_cursor()
        try:
            while True:
                if rc is None:
                    rc = proc.poll()
                W, H = term_size()
                plan = load_plan() or plan
                views = build_views(plan, lid)
                draw(render(views, plan, W, H, lang, tick, lid))
                tick += 1
                if rc is not None:
                    if fin_at is None:
                        fin_at = time.time()
                    if time.time() - fin_at >= 1.6:
                        break
                time.sleep(max(0.2, refresh))
        except KeyboardInterrupt:
            print("\n\n  收到中断，正在停止训练…")
            try:
                proc.terminate()
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.3)
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                pass
            proc.wait()
            rc = 130          # 用户中断，统一语义
        finally:
            show_cursor()
            logf.close()

    if rc is None:
        rc = proc.wait()
    # 被信号终止时 wait() 返回负数；对外统一报 130（用户中断），
    # 免得摘要里出现「exit -15」这种看不懂的东西。
    if rc < 0:
        rc = 130

    try:
        with open(exit_path, "w", encoding="utf-8") as f:
            f.write("%d\n" % rc)
    except OSError:
        pass

    def mut2(p):
        for x in p["legs"]:
            if x["id"] == lid:
                x["status"] = "done" if rc == 0 else "failed"
                x["exit_code"] = rc
                x["ended_epoch"] = time.time()
    update_plan(mut2)

    _print_summary(lid, lang)
    return rc


def _env():
    e = dict(os.environ)
    e.setdefault("PYTHONUNBUFFERED", "1")
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    e.setdefault("HF_HUB_OFFLINE", "1")
    return e


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


# ─────────────────────────── 摘要 ───────────────────────────
def _print_summary(lid, lang, width=None):
    leg = LEG_BY_ID[lid]
    plan = load_plan() or {}
    pleg = plan_leg(plan, lid) or {}
    v = leg_view(leg, pleg, None)
    W = width or min(78, term_size()[0] - 2)
    zh = (lang != "en")
    rule = "═" * W

    print("")
    print(rule)
    print((" 演示结束 / Demo finished" if zh else " Demo finished"))
    print(rule)
    title = leg_title(leg, lang)
    print(" %s %s   exit %s" % (MARKS.get(v["state"], "·"), title,
                                pleg.get("exit_code", v["exit_code"])))
    print("")
    if zh:
        print("  步数      %d/%d（演示）        真实全量 %d 步" % (
            v["done"], v["steps"], leg["full"]))
    else:
        print("  steps     %d/%d (demo)         full run %d" % (
            v["done"], v["steps"], leg["full"]))

    recs = v["recs"]
    if recs:
        els = [float(r.get("elapsed_s") or 0) for r in recs]
        wall = max(els) if els else 0
        avg = wall / len(els) if els else 0
        if zh:
            print("  墙钟      %s（含载入）  每步实测 %.0fs（全量实测 %.0fs，无加速）"
                  % (hms(wall), avg, leg["est"]))
        else:
            print("  wall      %s (incl. load)  %.0fs/step (full run %.0fs, no speedup)"
                  % (hms(wall), avg, leg["est"]))
        if leg["kind"] == "sft":
            if zh:
                print("  loss      %.4f → %.4f      峰值 %.2f GB" % (
                    float(recs[0].get("loss") or 0),
                    float(recs[-1].get("loss") or 0),
                    float(recs[-1].get("peak_mem_gb") or 0)))
            else:
                print("  loss      %.4f -> %.4f      peak %.2f GB" % (
                    float(recs[0].get("loss") or 0),
                    float(recs[-1].get("loss") or 0),
                    float(recs[-1].get("peak_mem_gb") or 0)))
        else:
            if zh:
                print("  reward    %+.3f → %+.3f    正确率 %.0f%%  峰值 %.2f GB" % (
                    float(recs[0].get("mean_reward") or 0),
                    float(recs[-1].get("mean_reward") or 0),
                    100 * float(recs[-1].get("accuracy") or 0),
                    float(recs[-1].get("peak_mem_gb") or 0)))
            else:
                print("  reward    %+.3f -> %+.3f    acc %.0f%%  peak %.2f GB" % (
                    float(recs[0].get("mean_reward") or 0),
                    float(recs[-1].get("mean_reward") or 0),
                    100 * float(recs[-1].get("accuracy") or 0),
                    float(recs[-1].get("peak_mem_gb") or 0)))

    mb = dir_size(abspath(leg["adapter"])) / 1e6
    print(("  产物      %s  %.1f MB" if zh else "  artifact  %s  %.1f MB")
          % (leg["adapter"], mb))

    # 诚实性锚点：真产物的 mtime 变没变
    anchor = abspath(leg["anchor"])
    before = pleg.get("anchor_mtime_before")
    now = _mtime(anchor)
    if before is not None and now is not None:
        same = abs(now - before) < 1e-6
        stamp = dt.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M")
        if zh:
            print("  隔离      %s mtime %s（%s）" % (
                leg["anchor"].split("/")[-1],
                "未变 ✓  本次演示未触碰真实产物" if same else "已变 ✗  请检查",
                stamp))
        else:
            print("  isolation %s mtime %s (%s)" % (
                leg["anchor"].split("/")[-1],
                "unchanged - real artifact untouched" if same else "CHANGED",
                stamp))
    print(rule)
    print("")


# ─────────────────────────── 回放调试 ───────────────────────────
def replay(jsonl_rel, lid, steps, lang, refresh):
    """
    不烧机器时间地调排版：把真实 jsonl 逐行喂给渲染器。
    用法：--replay branch-B-rl/logs/rl_train-C.jsonl --leg B --steps 2
    """
    src = abspath(jsonl_rel)
    recs = read_records(src)
    if not recs:
        print("  ✗ 回放源为空：%s" % src)
        return 3
    recs = recs[:steps]

    plan = new_plan(lang, refresh, 0, {lid: steps})
    os.makedirs(DEMO_DIR, exist_ok=True)
    tmp_jsonl = os.path.join(DEMO_DIR, "_replay-%s.jsonl" % lid)
    tmp_log = os.path.join(DEMO_DIR, "_replay-%s.log" % lid)
    plan["legs"] = [x for x in plan["legs"] if x["id"] == lid]
    plan["legs"][0]["jsonl"] = os.path.relpath(tmp_jsonl, LAB_ROOT)
    plan["legs"][0]["raw_log"] = os.path.relpath(tmp_log, LAB_ROOT)
    plan["legs"][0]["exit_file"] = os.path.join("logs", "demo", "_replay.exit")
    write_json(PLAN_PATH, plan)
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        f.write("")
    with open(tmp_log, "w", encoding="utf-8") as f:
        f.write("  (replay mode)\n")

    W, H = term_size()
    tick = 0
    hide_cursor()
    try:
        for i in range(len(recs) + 1):
            with open(tmp_jsonl, "w", encoding="utf-8") as f:
                for r in recs[:i]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(tmp_log, "a", encoding="utf-8") as f:
                f.write("  step %d/%d  replay\n" % (i, steps))
            W, H = term_size()
            views = build_views(plan, lid)
            draw(render(views, plan, W, H, lang, tick, lid))
            tick += 1
            time.sleep(max(0.3, refresh))
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
    return 0


# ─────────────────────────── 前置检查 ───────────────────────────
def sh(cmd):
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=20)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def preflight(lid, quiet=False):
    problems, warns = [], []
    rc, out = sh("df -g %s | awk 'NR==2{print $4}'" % LAB_ROOT)
    try:
        free_g = int(out.strip().split()[0])
    except (ValueError, IndexError):
        free_g = 999
    if free_g < 4:
        problems.append("磁盘只剩 %d GB（需要 ≥ 4 GB）" % free_g)
    elif free_g < 15:
        warns.append("磁盘只剩 %d GB，34 分钟 1080p 素材约需 5–10 GB" % free_g)

    rc, out = sh("vm_stat | head -5")
    if "Pages free" in out:
        try:
            free_pages = int(re.search(r"Pages free:\s+(\d+)\.", out).group(1))
            inactive = int(re.search(r"Pages inactive:\s+(\d+)\.", out).group(1))
            gb = (free_pages + inactive) * 4096 / 1e9
            if gb < 6.0:
                warns.append("可用内存 %.1f GB（单条训练峰值 11.2–11.7 GB）" % gb)
        except (AttributeError, ValueError):
            pass

    rc, out = sh("sysctl -n vm.swapusage")
    m = re.search(r"used = ([\d.]+)([MG])", out or "")
    if m:
        used = float(m.group(1)) * (1 if m.group(2) == "M" else 1024)
        if used > 2048:
            warns.append("swap 已用 %.0f MB —— 机器在换页，单步耗时会失真" % used)

    rc, out = sh("pgrep -f 'scripts/(11_train_distill|20_train_rl)\\.py'")
    pids = [x for x in out.split() if x.strip() and x.strip() != str(os.getpid())]
    if pids:
        problems.append("已有训练进程在跑（PID %s）—— 16 GB 内存不能并行"
                        % ", ".join(pids[:3]))

    rc, out = sh("pgrep -f 'ollama serve'")
    if out.strip():
        warns.append("Ollama 正在驻留（会占 2–6 GB 内存），建议先退出")

    if not os.access(PY, os.X_OK):
        problems.append("训练解释器不可执行：%s" % PY)
    for rel in ("branch-A-distill/model/config.json",
                "branch-A-distill/data/train.jsonl"):
        if not os.path.exists(os.path.join(LAB_ROOT, rel)):
            problems.append("缺少必需文件：%s" % rel)
    if lid == "C":
        anchor = os.path.join(LAB_ROOT, "branch-A-distill/adapter-A1",
                              "adapter_model.safetensors")
        if not os.path.exists(anchor):
            problems.append("C 腿需要预热源 adapter-A1，但它不存在")

    W, H = term_size()
    if W < 100 or H < 24:
        warns.append("终端 %d×%d，录屏建议 120×40，否则指标会被裁掉" % (W, H))

    if not quiet:
        for w in warns:
            print("  ⚠ %s" % w)
        for p in problems:
            print("  ✗ %s" % p)
    return problems, warns


# ─────────────────────────── CLI ───────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="三栏实时看板 + 演示训练编排",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：bash scripts/demo_run.sh A\n"
               "    python3 scripts/demo_board.py --replay branch-B-rl/logs/rl_train-C.jsonl --leg B --steps 2")
    ap.add_argument("leg", nargs="?", default=None, help="A | B | C")
    ap.add_argument("--leg", dest="leg_opt", default=None,
                    help="同位置参数 leg（给 --replay 用起来更顺）")
    ap.add_argument("--run", metavar="LEG", default=None, help="跑一条腿")
    ap.add_argument("--all", action="store_true", help="依次跑 A→B→C")
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认步数")
    ap.add_argument("--lang", default="both", choices=["zh", "en", "both"])
    ap.add_argument("--refresh", type=float, default=1.0, help="刷新间隔秒")
    ap.add_argument("--intro", type=int, default=6, help="开场声明停留秒数")
    ap.add_argument("--tail", type=int, default=5, help="原始日志尾部行数")
    ap.add_argument("--replay", metavar="JSONL", default=None, help="回放 jsonl 调排版")
    ap.add_argument("--summary", metavar="LEG", default=None, help="只打印摘要")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    ap.add_argument("--no-board", action="store_true", help="不画看板，只跑训练")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args(argv)

    lid = a.run or a.leg_opt or a.leg
    if lid:
        lid = lid.upper()
        if lid not in LEG_BY_ID:
            print("  ✗ 未知分支 %s（只支持 A / B / C）" % lid)
            return 2

    if a.summary:
        _print_summary(a.summary.upper(), a.lang)
        return 0

    if a.replay:
        return replay(a.replay, lid or "B", a.steps or 2, a.lang, a.refresh)

    if a.all:
        base = {"A": 4, "B": 2, "C": 2}
        total_rc = 0
        for i, x in enumerate(("A", "B", "C")):
            st = a.steps if (a.steps and i == 0) else base[x]
            rc = run_leg(x, st, a.lang, a.refresh, a.intro if i == 0 else 0,
                         a.tail, a.dry_run, a.no_board, True)
            total_rc = total_rc or rc
            if rc != 0:
                break
        return total_rc

    if not lid:
        ap.print_help()
        return 2

    if not a.skip_checks and not a.dry_run:
        problems, warns = preflight(lid)
        if problems:
            print("\n  前置检查未通过，已中止。修好后重试，或加 --skip-checks 跳过。\n")
            return 3
        if warns:
            print("\n  继续执行（3 秒后开始）…\n")
            time.sleep(3)

    steps = a.steps or LEG_BY_ID[lid]["steps"]
    return run_leg(lid, steps, a.lang, a.refresh, a.intro, a.tail,
                   a.dry_run, a.no_board)


if __name__ == "__main__":
    sys.exit(main())
