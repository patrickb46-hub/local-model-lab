#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_answers.py —— 答题判分器

复用 20_train_rl.py / 30_evaluate.py 用的同一套判分函数（common.compute_reward），
保证你练手判出来的分，跟模型训练/评测时用的是同一个尺度。

判分维度（与训练/评测完全一致）：
  格式三段   【思路】【步骤】【答案】齐全且顺序正确
  步骤齐全   至少 1 个编号步骤
  答案正确   归一化后与标准答案等价（含浮点容差、单位换算）
  超长惩罚   【答案】超过 64 字
总奖励 = 1.0×答案 + 0.05×格式 + 0.05×步骤 − 0.1×超长

用法：
  # 交互式：从题库选一道题，粘贴解答，即时判分
  python3 scripts/score_answers.py

  # 顺序刷题：从第 1 题开始逐题做，判完自动下一题，结束出总分报告
  python3 scripts/score_answers.py --all

  # 单题快速判（题目 + 解答，两个参数）
  python3 scripts/score_answers.py "解方程：3x+5=11" "【思路】移项。【步骤】1. 3x=6。2. x=2。【答案】2"

  # 只判某个分类
  python3 scripts/score_answers.py --cat 一次函数平移

只依赖标准库 + 项目里的 common.py，无需加载任何模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import compute_reward, load_config, parse_response, read_jsonl  # noqa: E402

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RST = "\033[0m"


def _c(code, s):
    return code + s + RST if sys.stdout.isatty() else s


def load_questions():
    """合并 RL 训练池 + 评测集，按「分类题量降序、分类内原顺序」编号。

    排序与桌面上 `本地模型实验室_测试题目*.md` 完全一致，保证题号一一对应。
    """
    items = []
    seen = set()
    for rel in ("branch-B-rl/data/rl_pool_curated.jsonl",
                "shared/eval_set.jsonl"):
        for o in read_jsonl(os.path.join(LAB_ROOT, rel)):
            if o["id"] in seen:
                continue
            seen.add(o["id"])
            items.append(o)
    # 分类分组（保持文件读取顺序）
    grouped = OrderedDict()
    for o in items:
        grouped.setdefault(o.get("category", "未分类"), []).append(o)
    # 分类按题量降序（与题目清单一致）
    grouped = OrderedDict(sorted(grouped.items(),
                                 key=lambda kv: (-len(kv[1]), kv[0])))
    # 按排序后的顺序重编全局号
    no = 1
    for c in grouped:
        for o in grouped[c]:
            o["no"] = no
            no += 1
    return grouped, items


def judge(text, gold, weights, ans_max):
    """返回 (总奖励, parts, parsed)。"""
    reward, parts = compute_reward(text, gold, weights, ans_max)
    parsed = parse_response(text)
    return reward, parts, parsed


def show_verdict(text, gold, weights, ans_max):
    reward, parts, parsed = judge(text, gold, weights, ans_max)

    correct = parts["correct"] > 0
    fmt = parts["format"] > 0
    steps = parts["steps"] > 0
    overlong = parts["overlong_penalty"] > 0

    print()
    print("  你答：%s" % (_c(DIM, (parsed["answer_raw"] or "（未提取到【答案】）"))))
    print("  标准：%s" % gold)
    print("  ─────────────────────────────────────────")
    print("  格式三段  %s   步骤齐全  %s   答案正确  %s" % (
        _c(GREEN, "✓") if fmt else _c(RED, "✗"),
        _c(GREEN, "✓") if steps else _c(RED, "✗"),
        _c(GREEN, "✓") if correct else _c(RED, "✗"),
    ))
    if overlong:
        print("  %s 答案超长（>64 字）" % _c(YELLOW, "⚠"))
    full_score = (weights.get("correct", 1.0)
                  + weights.get("format", 0.0) + weights.get("steps", 0.0))
    print("  总奖励  %+.3f%s" % (reward, _c(DIM, "  （满分为 %.3f）" % full_score)))
    print()
    return dict(correct=correct, fmt=fmt, steps=steps, overlong=overlong, reward=reward)


def read_multiline(prompt):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def run_interactive(questions, weights, ans_max, cat=None, all_mode=False):
    cfg = load_config()
    grouped, items = questions

    print(_c(DIM, "\n  本地模型实验室 · 答题判分器\n"))
    print("  判分维度：格式三段 · 步骤齐全 · 答案正确 · 超长惩罚")
    print("  总奖励 = 1.0×答案 + 0.05×格式 + 0.05×步骤 − 0.1×超长")
    print(_c(DIM, "  每题作答格式：【思路】…【步骤】1.…2.…【答案】…\n"))

    # 选择要做的题目范围
    todo = []
    if cat:
        for o in grouped.get(cat, []):
            todo.append(o)
        if not todo:
            print("  ✗ 找不到分类「%s」。可选：" % cat)
            for c in grouped:
                print("      %s（%d 题）" % (c, len(grouped[c])))
            return 1
    elif all_mode:
        for c in grouped:
            todo.extend(grouped[c])
    else:
        # 列出分类，让用户选
        cats = list(grouped.keys())
        print("  题库（%d 题）：" % len(items))
        for i, c in enumerate(cats, 1):
            print("    %2d. %s（%d 题）" % (i, c, len(grouped[c])))
        print()
        while True:
            try:
                s = input("  选分类（编号，直接回车=顺序全部，:q 退出）：").strip()
            except EOFError:
                return 0
            if s.lower() in (":q", "q"):
                return 0
            if s == "":
                for c in cats:
                    todo.extend(grouped[c])
                break
            if s.isdigit() and 1 <= int(s) <= len(cats):
                todo.extend(grouped[cats[int(s) - 1]])
                break
            print("  输入无效，重新选。")

    # 逐题判分
    stats = []
    for o in todo:
        print(_c(DIM, "  ── %s · 第 %d 题 ──" % (o["category"], o["no"])))
        print("  题目：%s" % o["prompt"])
        txt = read_multiline("  粘贴你的解答（空行结束，:q 跳过并结算）：")
        if txt.strip().lower() in (":q", "q"):
            break
        if not txt.strip():
            print("  %s（跳过）\n" % _c(DIM, "未作答"))
            continue
        stats.append(show_verdict(txt, o["answer"], weights, ans_max))

    # 结算
    if stats:
        n = len(stats)
        ok = sum(1 for s in stats if s["correct"])
        fm = sum(1 for s in stats if s["fmt"])
        st = sum(1 for s in stats if s["steps"])
        avg = sum(s["reward"] for s in stats) / n
        print(_c(DIM, "  ═════════════════════════════════════"))
        print("  结算（共 %d 题）" % n)
        print("    答案正确  %d / %d   （%.0f%%）" % (ok, n, 100 * ok / n))
        print("    格式三段  %d / %d   （%.0f%%）" % (fm, n, 100 * fm / n))
        print("    步骤齐全  %d / %d   （%.0f%%）" % (st, n, 100 * st / n))
        print("    平均奖励  %+.3f" % avg)
        print()
    return 0


def main():
    ap = argparse.ArgumentParser(description="答题判分器（复用训练/评测同一套判分）")
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("answer", nargs="?", default=None)
    ap.add_argument("--all", action="store_true", help="顺序刷完整个题库")
    ap.add_argument("--cat", default=None, help="只刷某个分类")
    a = ap.parse_args()

    cfg = load_config()
    weights = cfg["rl"]["reward_weights"]
    ans_max = cfg["task"]["output_format"]["answer_max_chars"]
    grouped, items = load_questions()

    if a.prompt is not None:
        # 单题快速判：第一个参数是题库编号（1–69）
        flat = [o for c in grouped for o in grouped[c]]
        if not a.prompt.strip().isdigit():
            print("  单题判分请给题号（1–%d），例如：" % len(flat))
            print("    python3 scripts/score_answers.py 3")
            print("    python3 scripts/score_answers.py 3 \"【思路】…【答案】2\"")
            print("  交互刷题请直接运行不带参数：python3 scripts/score_answers.py")
            return 2
        no = int(a.prompt.strip())
        if not (1 <= no <= len(flat)):
            print("  ✗ 编号越界（1–%d）" % len(flat))
            return 2
        o = flat[no - 1]
        print("  题目 #%d（%s）：%s" % (no, o["category"], o["prompt"]))
        ans = a.answer
        if ans is None:
            ans = read_multiline("  粘贴你的解答（空行结束）：")
        show_verdict(ans, o["answer"], weights, ans_max)
        return 0

    return run_interactive((grouped, items), weights, ans_max, cat=a.cat, all_mode=a.all)


if __name__ == "__main__":
    raise SystemExit(main())
