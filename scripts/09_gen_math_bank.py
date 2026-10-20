#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_gen_math_bank.py —— 程序化数学题库生成器

为什么程序化出题而不是手写题库？
  * 答案天然精确（用 Fraction 精确运算），RL 的可验证奖励才立得住
  * 可无限扩样、可固定种子复现
  * 题目与答案严格一一对应，不留"标准答案靠猜"的模糊地带

产出（全部按固定种子，可复现）：
  shared/eval_set.jsonl                固化评测集（训练前锁定，永不进训练数据）
  branch-A-distill/data/prompts.jsonl  蒸馏用题面（teacher 来解）
  branch-B-rl/data/prompts.jsonl       RL 用题面（带 ground truth，供奖励计算）
  shared/general_probe.jsonl           通用能力探针（防灾难性遗忘）

用法：
  python 09_gen_math_bank.py            # 生成全部（会覆盖已有文件，有确认提示）
  python 09_gen_math_bank.py --check    # 只校验已有题库的自洽性，不写入
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    banner, ensure_dir, lab_path, load_config, read_jsonl,
    write_jsonl, answer_equal,
)

# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def fmt_num(v) -> str:
    """Fraction/int -> 规范的答案字符串。"""
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return str(v.numerator)
        return f"{v.numerator}/{v.denominator}"
    return str(v)


def sq_free_split(n: int) -> Tuple[int, int]:
    """n = a^2 * b，返回 (a, b)，b 平方因子free。"""
    a = 1
    b = n
    d = 2
    while d * d <= b:
        while b % (d * d) == 0:
            a *= d
            b //= d * d
        d += 1
    return a, b


def poly_str(a: int, b: int, c: int) -> str:
    """拼 a x² + b x + c 的可读形式。"""
    parts = []
    if a:
        parts.append(("" if a == 1 else ("-" if a == -1 else str(a))) + "x²")
    if b:
        sign = " - " if b < 0 else " + "
        val = abs(b)
        parts.append(sign + ("" if val == 1 else str(val)) + "x")
    if c:
        sign = " - " if c < 0 else " + "
        parts.append(sign + str(abs(c)))
    s = "".join(parts).lstrip()
    if s.startswith(" + "):
        s = s[3:]
    return s or "0"


# --------------------------------------------------------------------------
# 十个题目生成器（全部产出数值或分数答案，保证可比对）
# --------------------------------------------------------------------------

def gen_poly_eval(rnd: random.Random) -> Tuple[str, str]:
    """整式运算：代入求值。"""
    a = rnd.choice([1, 2, 3, 4, -1, -2, -3])
    b = rnd.choice([-9, -7, -5, -4, -3, -2, 2, 3, 4, 5, 7, 9])
    c = rnd.choice([-9, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 9])
    x = rnd.choice([-4, -3, -2, 2, 3, 4, 5])
    p = poly_str(a, b, c)
    ans = a * x * x + b * x + c
    return f"当 x = {x} 时，求代数式 {p} 的值。", fmt_num(ans)


def gen_factor(rnd: random.Random) -> Tuple[str, str]:
    """因式分解：已知可分解为 (x+p)(x+q)，求 p²+q²。"""
    while True:
        p = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9])
        q = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9])
        # q != p：保证是二次式；p + q != 0：避免一次项系数为 0（题面会出现 "0x"）
        if q != p and (p + q) != 0:
            break
    b = -(p + q)
    c = p * q
    # 一次项系数为 ±1 时按中文惯例省略系数（写 - x 而不是 - 1x）
    if b == 1:
        b_term = "+ x"
    elif b == -1:
        b_term = "- x"
    else:
        b_term = f"+ {b}x" if b >= 0 else f"- {abs(b)}x"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    prompt = (f"已知多项式 x² {b_term} {c_str} 可以分解为 (x + p)(x + q) 的形式，"
              f"其中 p、q 为整数。求 p² + q² 的值。")
    return prompt, fmt_num(p * p + q * q)


def gen_linear_eq(rnd: random.Random) -> Tuple[str, str]:
    """一元一次方程。"""
    a = rnd.choice([2, 3, 4, 5, 6, 7, 8, 9, -2, -3, -4, -5])
    c = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    if c == a:
        c = a + 1 if a < 9 else a - 1
    b = rnd.randint(-20, 20)
    d = rnd.randint(-20, 20)
    x = Fraction(d - b, a - c)
    prompt = f"解方程：{a}x {'+' if b >= 0 else '-'} {abs(b)} = {c}x {'+' if d >= 0 else '-'} {abs(d)}，求 x 的值。"
    return prompt, fmt_num(x)


def gen_system_eq(rnd: random.Random) -> Tuple[str, str]:
    """二元一次方程组：先定整数解，再反推系数。"""
    while True:
        x0 = rnd.randint(-8, 8)
        y0 = rnd.randint(-8, 8)
        if x0 == 0 or y0 == 0 or x0 == y0:
            continue
        a1 = rnd.choice([1, 2, 3, -1, -2, -3])
        b1 = rnd.choice([1, 2, 3, -1, -2, -3])
        a2 = rnd.choice([1, 2, 3, 4, -1, -2, -3])
        b2 = rnd.choice([1, 2, 3, 4, -1, -2, -3])
        if a1 * b2 - a2 * b1 == 0:
            continue
        c1 = a1 * x0 + b1 * y0
        c2 = a2 * x0 + b2 * y0
        break

    def line(a, b, c):
        s = f"{'' if a == 1 else ('-' if a == -1 else a)}x"
        s += (" + " if b > 0 else " - ") + ("" if abs(b) == 1 else str(abs(b))) + "y"
        s += (" = " if c >= 0 else " = ") + str(c)
        return s

    prompt = (f"解方程组：\n{line(a1, b1, c1)}\n{line(a2, b2, c2)}\n"
              f"求 x - y 的值。")
    return prompt, fmt_num(x0 - y0)


def gen_inequality(rnd: random.Random) -> Tuple[str, str]:
    """一元一次不等式：求最大整数解。

    刻意让解集端点常常落在两个整数之间，避免题目退化成"抄系数减一"。
    """
    a = rnd.choice([2, 3, 4, 5, 6, 7, 8])
    b = rnd.randint(-15, 15)
    t = rnd.randint(-4, 8)
    c = a * t + b + rnd.randint(0, a - 1)   # 端点 = t + 小数部分
    bound = Fraction(c - b, a)              # x < bound
    largest = math.ceil(bound) - 1
    prompt = f"解不等式：{a}x {'+' if b >= 0 else '-'} {abs(b)} < {c}，求它的最大整数解。"
    return prompt, fmt_num(largest)


def gen_fraction_eval(rnd: random.Random) -> Tuple[str, str]:
    """分式化简求值：(x² - m²)/(x - m)。"""
    m = rnd.choice([2, 3, 4, 5, 6, 7, 8, 9])
    k = rnd.choice([-6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 10, 11])
    while k == m or k == -m:
        k = rnd.randint(-8, 12)
    prompt = (f"先化简，再求值：(x² - {m * m}) ÷ (x - {m})，其中 x = {k}。")
    return prompt, fmt_num(k + m)


def gen_radical(rnd: random.Random) -> Tuple[str, str]:
    """二次根式化简：√n = a√b，求 a+b。"""
    while True:
        a = rnd.choice([2, 3, 4, 5, 6])
        b = rnd.choice([2, 3, 5, 6, 7, 10, 11, 13, 15])
        n = a * a * b
        if n > 900:
            continue
        aa, bb = sq_free_split(n)
        if bb > 1 and aa > 1:
            break
    prompt = (f"化简二次根式 √{n}，若结果可以写成 a√b 的形式"
              f"（a、b 为正整数，b 不含平方因子），求 a + b 的值。")
    return prompt, fmt_num(aa + bb)


def gen_linear_func(rnd: random.Random) -> Tuple[str, str]:
    """一次函数：过两点，求某点函数值。"""
    k = rnd.choice([1, 2, 3, -1, -2, -3, 4, -4])
    b = rnd.randint(-9, 9)
    x1 = rnd.randint(-5, 5)
    x2 = x1 + rnd.choice([1, 2, 3, 4, -1, -2, -3])
    while x2 == x1:
        x2 = x1 + 1
    y1, y2 = k * x1 + b, k * x2 + b
    x3 = rnd.choice([-6, -4, -2, 2, 4, 6, 7, 8])
    prompt = (f"已知一次函数 y = kx + b 的图像经过点 ({x1}, {y1}) 和 ({x2}, {y2})。"
              f"求当 x = {x3} 时 y 的值。")
    return prompt, fmt_num(k * x3 + b)


def gen_geometry(rnd: random.Random) -> Tuple[str, str]:
    """几何度量：勾股 / 内角和 / 面积周长。"""
    kind = rnd.choice(["pyth", "angle", "rect", "iso"])
    if kind == "pyth":
        triples = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15),
                   (8, 15, 17), (7, 24, 25), (20, 21, 29), (12, 16, 20)]
        a, b, c = rnd.choice(triples)
        if rnd.random() < 0.5:
            return (f"在直角三角形中，两条直角边的长分别为 {a} 和 {b}，"
                    f"求斜边的长。"), fmt_num(c)
        return (f"在直角三角形中，斜边长为 {c}，一条直角边长为 {a}，"
                f"求另一条直角边的长。"), fmt_num(b)
    if kind == "angle":
        a = rnd.randint(20, 80)
        b = rnd.randint(20, 80)
        if a + b >= 175:
            b = 170 - a
        return (f"在三角形 ABC 中，∠A = {a}°，∠B = {b}°，"
                f"求 ∠C 的度数。"), fmt_num(180 - a - b)
    if kind == "rect":
        w = rnd.randint(3, 20)
        h = rnd.randint(3, 20)
        if rnd.random() < 0.5:
            return f"一个长方形的长是 {w} 厘米，宽是 {h} 厘米，求它的面积。", fmt_num(w * h)
        return f"一个长方形的长是 {w} 厘米，宽是 {h} 厘米，求它的周长。", fmt_num(2 * (w + h))
    # iso
    apex = rnd.choice([20, 30, 40, 50, 60, 80, 100, 120])
    base_angle = Fraction(180 - apex, 2)
    return (f"等腰三角形的一个顶角是 {apex}°，求它的一个底角的度数。"), fmt_num(base_angle)


def gen_ratio(rnd: random.Random) -> Tuple[str, str]:
    """比例百分数应用。

    注意：答案一律是"纯数值"（元 / 原数），绝不让答案是百分数——
    否则模型答 99 与标准答案 99% 会被判成不同答案，凭空制造假错误。
    """
    kind = rnd.choice(["discount", "increase", "reverse"])
    if kind == "discount":
        price = rnd.choice([80, 120, 150, 200, 240, 300, 400, 500, 600, 800])
        d = rnd.choice([5, 6, 7, 8, 9])
        ans = Fraction(price * d, 10)
        return (f"一件商品的原价是 {price} 元，现在打 {d} 折出售，"
                f"求现价是多少元。"), fmt_num(ans)
    if kind == "increase":
        base = rnd.choice([40, 60, 80, 100, 120, 200, 300, 400])
        r = rnd.choice([5, 10, 20, 25, 50])
        res = base * (100 + r) // 100      # 取值保证整除
        return (f"某数增加 {r}% 后等于 {res}，求这个数。"), fmt_num(base)
    # 先涨后降：结果可能是小数，判分器对分数/小数等价
    price = rnd.choice([100, 200, 400, 500, 1000])
    r = rnd.choice([10, 20, 25, 50])
    final = Fraction(price * (100 + r) * (100 - r), 10000)
    return (f"某商品的单价是 {price} 元，先上涨 {r}%，再下降 {r}%，"
            f"求最终价格是多少元。"), fmt_num(final)


GENERATORS = [
    ("整式运算", gen_poly_eval),
    ("因式分解", gen_factor),
    ("一元一次方程", gen_linear_eq),
    ("二元一次方程组", gen_system_eq),
    ("一元一次不等式", gen_inequality),
    ("分式运算", gen_fraction_eval),
    ("二次根式", gen_radical),
    ("一次函数", gen_linear_func),
    ("几何度量", gen_geometry),
    ("比例百分数", gen_ratio),
]


# --------------------------------------------------------------------------
# 高难题族（2026-10-08 新增）
#
# 为什么需要单独造难题：
#   原 10 类题在现有 20 题评测集上已经**全员饱和** —— 蒸馏与 teacher 的 pass@3
#   都是 90%，没有区分度；而「teacher 系统性做错、base 偶发能做对」（L2 层）
#   的题只有 1/20 = 5%。要做「RL 能否突破 teacher 天花板」的实验，
#   必须先有足够多的 L2 题，否则统计上不可能测出差异。
#
# 造题思路：这些题族都是「多步 / 需要中间推理」，量化 9B teacher 在长链上
#   系统性掉分，而 2B base 因为采样发散、偶发能命中 —— 正是 L2 的高产区。
#   难度靠 `depth` / 取值范围控制（参数放大 → pass@k 单调下降）。
# --------------------------------------------------------------------------

def _build_expr(rnd: random.Random, depth: int, lo: int, hi: int):
    """随机构造整数算式树（只用 + - ×，保证结果始终是整数、无除法歧义）。"""
    if depth <= 0 or (depth == 1 and rnd.random() < 0.45):
        return rnd.randint(lo, hi)
    op = rnd.choice(["+", "-", "×"])
    return (_build_expr(rnd, depth - 1, lo, hi), op,
            _build_expr(rnd, depth - 1, lo, hi))


def _expr_str(e) -> str:
    if isinstance(e, int):
        return str(e) if e >= 0 else f"({e})"
    l, op, r = e
    return f"({_expr_str(l)} {op} {_expr_str(r)})"


def _expr_val(e) -> Fraction:
    if isinstance(e, int):
        return Fraction(e)
    l, op, r = e
    a, b = _expr_val(l), _expr_val(r)
    return {"+": a + b, "-": a - b, "×": a * b}[op]


def gen_chain_arith(rnd: random.Random) -> Tuple[str, str]:
    """多步链式整数运算（带括号，3~5 层）。

    质量把关：必须**至少含一个 ×**。否则会退化成 `(6 + (12 + 10))` 这种
    纯加法嵌套 —— 既不"多步"也没有区分度，白占基准题位。
    """
    for _ in range(40):
        depth = rnd.choice([2, 3, 3, 4])
        span = rnd.choice([(1, 9), (1, 12), (2, 19)])
        expr = _build_expr(rnd, depth, *span)
        s = _expr_str(expr)
        if "×" in s and s.count(" ") >= 4:
            return (f"计算：{s}，结果用整数或最简分数表示。"), fmt_num(_expr_val(expr))
    # 兜底：手工造一个保证含乘法的
    a, b, c, d = (rnd.randint(-9, 9) for _ in range(4))
    expr = ((a, "+", b), "×", (c, "-", d))
    return (f"计算：{_expr_str(expr)}，结果用整数或最简分数表示。"), fmt_num(_expr_val(expr))


def gen_fraction_arith(rnd: random.Random) -> Tuple[str, str]:
    """分数四则运算（2~3 项，含乘除，答案最简分数或整数）。"""
    denoms = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    n_terms = rnd.choice([2, 3, 3])
    terms, disp = [], []
    for _ in range(n_terms):
        d = rnd.choice(denoms)
        n = rnd.randint(1, d * 2)
        f = Fraction(n, d)
        terms.append(f)
        disp.append(f"{n}/{d}")
    ops = [rnd.choice(["+", "-", "×", "÷"]) for _ in range(n_terms - 1)]
    acc = terms[0]
    s = disp[0]
    for i, op in enumerate(ops):
        rhs = terms[i + 1]
        if op == "×":
            acc = acc * rhs
        elif op == "÷":
            if rhs == 0:
                op = "+"
                acc = acc + rhs
            else:
                acc = acc / rhs
        elif op == "-":
            acc = acc - rhs
        else:
            acc = acc + rhs
        s += f" {op} {disp[i + 1]}"
    return (f"计算：{s}，结果用整数或最简分数表示。"), fmt_num(acc)


def gen_system_eq_comb(rnd: random.Random) -> Tuple[str, str]:
    """二元一次方程组：求组合式（比求 x-y 多一步推理）。"""
    while True:
        x0 = rnd.randint(-8, 8)
        y0 = rnd.randint(-8, 8)
        if x0 == 0 or y0 == 0 or x0 == y0:
            continue
        a1 = rnd.choice([1, 2, 3, -1, -2, -3])
        b1 = rnd.choice([1, 2, 3, -1, -2, -3])
        a2 = rnd.choice([1, 2, 3, 4, -1, -2, -3])
        b2 = rnd.choice([1, 2, 3, 4, -1, -2, -3])
        if a1 * b2 - a2 * b1 == 0:
            continue
        c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
        break

    def line(a, b, c):
        s = f"{'' if a == 1 else ('-' if a == -1 else a)}x"
        s += (" + " if b > 0 else " - ") + ("" if abs(b) == 1 else str(abs(b))) + "y"
        return s + " = " + str(c)

    ask = rnd.choice(["x + y", "xy", "2x - 3y", "x² - y²"])
    if ask == "x + y":
        ans = x0 + y0
    elif ask == "xy":
        ans = x0 * y0
    elif ask == "2x - 3y":
        ans = 2 * x0 - 3 * y0
    else:
        ans = x0 * x0 - y0 * y0
    prompt = (f"解方程组：\n{line(a1, b1, c1)}\n{line(a2, b2, c2)}\n"
              f"求 {ask} 的值。")
    return prompt, fmt_num(ans)


def gen_multi_step_eq(rnd: random.Random) -> Tuple[str, str]:
    """一元一次方程（带括号、两边都有未知数，需先去括号再移项）。"""
    a1 = rnd.choice([2, 3, -2, -3, 4, -4])
    a2 = rnd.choice([1, 2, -1, -2, 3, -3])
    p1 = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p2 = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b1 = rnd.choice([1, 2, -1, -2, 3, -3])
    b2 = rnd.choice([2, 3, -2, -3, 4])
    q1 = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    q2 = rnd.choice([-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    cL, dL = a1 + a2, a1 * p1 + a2 * p2
    cR, dR = b1 + b2, b1 * q1 + b2 * q2
    if cL == cR:
        return gen_multi_step_eq(rnd)
    x0 = Fraction(dR - dL, cL - cR)

    def term(a, p):
        inner = f"x {'+' if p >= 0 else '-'} {abs(p)}"
        return ("" if a == 1 else ("-" if a == -1 else str(a))) + f"({inner})"

    lhs = f"{term(a1, p1)} {'+' if a2 >= 0 else '-'} {term(abs(a2), p2)}"
    rhs = f"{term(b1, q1)} {'+' if b2 >= 0 else '-'} {term(abs(b2), q2)}"
    return f"解方程：{lhs} = {rhs}，求 x 的值。", fmt_num(x0)


def gen_linear_func_shift(rnd: random.Random) -> Tuple[str, str]:
    """一次函数：图像平移后求某点函数值（比原版多一步变换）。"""
    k = rnd.choice([1, 2, 3, -1, -2, -3, 4, -4])
    b = rnd.randint(-9, 9)
    m = rnd.randint(1, 6)
    dirn = rnd.choice(["右", "左", "上", "下"])
    if dirn == "右":
        kb = (k, b - k * m)
    elif dirn == "左":
        kb = (k, b + k * m)
    elif dirn == "上":
        kb = (k, b + m)
    else:
        kb = (k, b - m)
    x3 = rnd.choice([-6, -4, -2, 2, 4, 6, 7, 8])
    kk = "" if k == 1 else ("-" if k == -1 else str(k))
    prompt = (f"已知一次函数 y = {kk}x {'+' if b >= 0 else '-'} {abs(b)}。"
              f"把它的图像向{dirn}平移 {m} 个单位，求平移后当 x = {x3} 时 y 的值。")
    return prompt, fmt_num(kb[0] * x3 + kb[1])


# 高难题族：候选池按这个比例加权生成
HARD_GENERATORS = [
    ("链式运算", gen_chain_arith),
    ("分数四则", gen_fraction_arith),
    ("方程组组合式", gen_system_eq_comb),
    ("多步方程", gen_multi_step_eq),
    ("一次函数平移", gen_linear_func_shift),
]


# --------------------------------------------------------------------------
# 通用能力探针（防灾难性遗忘）
# --------------------------------------------------------------------------

GENERAL_PROBE = [
    {"id": "g01", "prompt": "用一句话介绍你自己。", "keywords": []},
    {"id": "g02", "prompt": "中国的首都是哪座城市？", "keywords": ["北京"]},
    {"id": "g03", "prompt": "一年有多少个月？", "keywords": ["12", "十二"]},
    {"id": "g04", "prompt": "请说出三种水果的名字。", "keywords": ["苹果", "香蕉", "橙", "梨", "葡萄"]},
    {"id": "g05", "prompt": "太阳从哪个方向升起？", "keywords": ["东"]},
    {"id": "g06", "prompt": "把下面这句话翻译成英文：我喜欢读书。",
     "keywords": ["I like", "I enjoy", "love reading", "like reading"]},
    {"id": "g07", "prompt": "水的化学式是什么？", "keywords": ["H2O", "H₂O"]},
    {"id": "g08", "prompt": "请用一句话说明什么是三角形。",
     "keywords": ["三条", "三个", "边"]},
    {"id": "g09", "prompt": "3 加 5 等于多少？", "keywords": ["8"]},
    {"id": "g10", "prompt": "请写一句简短的问候语给朋友。", "keywords": []},
]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def build_items(seed: int, total: int) -> List[Dict]:
    rnd = random.Random(seed)
    items: List[Dict] = []
    seen = set()
    guard = 0
    while len(items) < total and guard < total * 60:
        guard += 1
        cat, fn = GENERATORS[len(items) % len(GENERATORS)]
        try:
            prompt, ans = fn(rnd)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {cat} 生成失败: {e}")
            continue
        key = prompt.strip()
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "id": f"m{len(items):04d}",
            "category": cat,
            "prompt": prompt,
            "answer": ans,
        })
    # 打散，避免同一类别连续
    rnd.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"m{i:04d}"
    return items


def _norm_q(s: str) -> str:
    """题面归一化（折叠空白），用于跨池去重。"""
    return " ".join((s or "").split())


def build_pool(seed: int, total: int, hard_ratio: float = 0.75,
               exclude=None) -> List[Dict]:
    """
    生成「候选池」——供 12_screen_layers.py 分层筛查用的原始题海。

    与 build_items 的区别：
      * 题量更大（HARD 题族加权），供筛查后挑出基准题与训练题
      * 按 hard_ratio 加权到高难题族（teacher 更可能系统性失败 → 更容易产出 L2）
      * id 用 `p` 前缀

    ⚠ **题面级全局去重（关键纪律）**：
      前缀不同**并不能**保证题目不重 —— 实测 300 题池里就有 12 题与
      蒸馏池 / RL 池题面**完全相同**。基准题一旦与训练数据同题，
      「模型没见过」这个前提就破了，**全部结论作废**。
      所以必须把 eval / distill / RL / 探针池的题面全部作为 exclude 传进来。
    """
    rnd = random.Random(seed + 777)          # 与主题库不同种子
    items: List[Dict] = []
    seen = set(exclude or ())
    guard = 0
    while len(items) < total and guard < total * 60:
        guard += 1
        use_hard = rnd.random() < hard_ratio
        table = HARD_GENERATORS if use_hard else GENERATORS
        cat, fn = table[len(items) % len(table)]
        try:
            prompt, ans = fn(rnd)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {cat} 生成失败: {e}")
            continue
        key = _norm_q(prompt)
        if key in seen:
            continue
        seen.add(key)
        items.append({"id": f"p{len(items):04d}", "category": cat,
                      "prompt": prompt, "answer": ans})
    rnd.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"p{i:04d}"
    return items


def check(items: List[Dict]) -> None:
    """自洽性校验：答案必须能被自己的判分器认出来。"""
    bad = 0
    for it in items:
        if not answer_equal(it["answer"], it["answer"]):
            bad += 1
            print(f"  [FAIL] {it['id']} answer={it['answer']!r}")
    print(f"  自洽性校验：{len(items) - bad}/{len(items)} 通过")
    cats: Dict[str, int] = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    for k in sorted(cats):
        print(f"    {k:16s} {cats[k]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写入")
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--distill-size", type=int, default=None)
    ap.add_argument("--rl-size", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的题库")
    ap.add_argument("--pool-size", type=int, default=None,
                    help="候选池题量（供分层筛查；默认读 config data.candidate_pool_size）")
    ap.add_argument("--hard-ratio", type=float, default=None,
                    help="候选池里高难题族的占比，0~1（默认读 config data.hard_ratio）")
    ap.add_argument("--pool-only", action="store_true",
                    help="只生成候选池，不碰 eval/distill/rl 三池")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["data"]["seed"]
    eval_n = args.eval_size or cfg["data"]["eval_size"]
    dist_n = args.distill_size or cfg["data"]["distill_prompts"]
    rl_n = args.rl_size or cfg["data"]["rl_prompts"]
    pool_n = args.pool_size or cfg["data"].get("candidate_pool_size", 800)
    hard_ratio = (args.hard_ratio if args.hard_ratio is not None
                  else cfg["data"].get("hard_ratio", 0.75))

    eval_path = lab_path(cfg, "lab_root") / cfg["data"]["eval_file"]
    a_dir = lab_path(cfg, "branch_a") / "data"
    b_dir = lab_path(cfg, "branch_b") / "data"
    probe_path = lab_path(cfg, "lab_root") / cfg["eval"]["general_probe_file"]
    pool_path = lab_path(cfg, "lab_root") / cfg["data"].get(
        "candidate_pool_file", "shared/candidate_pool.jsonl")

    # ---- 候选池：独立于主三池，可单独生成 ----
    if args.pool_only:
        banner("生成候选池（供分层筛查）")

        def _prompts_of(p):
            try:
                return [_norm_q(r.get("prompt", "")) for r in read_jsonl(p)]
            except Exception:  # noqa: BLE001
                return []

        exclude = set()
        for p in (eval_path, a_dir / "prompts.jsonl",
                  b_dir / "prompts.jsonl", probe_path):
            exclude.update(_prompts_of(p))
        print(f"  种子={seed + 777}  题量={pool_n}  高难题族占比={hard_ratio:.0%}")
        print(f"  题面级排除 {len(exclude)} 条（评测集/蒸馏池/RL池/通用探针）")
        pool = build_pool(seed, pool_n, hard_ratio, exclude=exclude)
        print(f"  实得 {len(pool)} 题")
        write_jsonl(pool_path, pool)
        check(pool)
        print(f"\n  已写出 {pool_path}")
        return 0

    if args.check:
        banner("校验已有题库")
        for p in [eval_path, a_dir / "prompts.jsonl", b_dir / "prompts.jsonl"]:
            print(f"\n{p}")
            rows = read_jsonl(p)
            print(f"  条目数 {len(rows)}")
            check(rows)
        return 0

    # 已存在保护
    existing = [p for p in [eval_path, a_dir / "prompts.jsonl", b_dir / "prompts.jsonl"]
                if p.exists() and p.stat().st_size > 0]
    if existing and not args.force:
        print("以下文件已存在，未覆盖：")
        for p in existing:
            print(f"  {p}")
        print("\n加 --force 才会重新生成（会覆盖并打乱已有的题库划分）。")
        return 1

    banner("生成程序化数学题库")
    total = eval_n + dist_n + rl_n
    print(f"种子={seed}  总量={total}  "
          f"(eval {eval_n} + distill {dist_n} + rl {rl_n})")
    items = build_items(seed, total)
    print(f"实得 {len(items)} 题")

    # 严格三段切分：先 eval（锁定），再 A，再 B —— 三者互不重叠
    eval_items = items[:eval_n]
    a_items = items[eval_n:eval_n + dist_n]
    b_items = items[eval_n + dist_n:eval_n + dist_n + rl_n]

    for it in eval_items:
        it["split"] = "eval"
    for it in a_items:
        it["split"] = "distill"
    for it in b_items:
        it["split"] = "rl"
        # RL 需要 teacher 之外的参照，ground truth 已在 answer 字段
    write_jsonl(eval_path, eval_items)

    # 蒸馏：teacher 来解，所以只需要题面 + 类别 + 答案（答案用于后期质检）
    write_jsonl(a_dir / "prompts.jsonl", [
        {"id": it["id"], "category": it["category"],
         "prompt": it["prompt"], "answer": it["answer"]}
        for it in a_items
    ])
    # RL：题面 + ground truth
    write_jsonl(b_dir / "prompts.jsonl", [
        {"id": it["id"], "category": it["category"],
         "prompt": it["prompt"], "answer": it["answer"]}
        for it in b_items
    ])
    write_jsonl(probe_path, GENERAL_PROBE)

    # 划分重叠检查（纪律：eval 绝不进训练数据）
    ids_eval = {it["id"] for it in eval_items}
    ids_a = {it["id"] for it in a_items}
    ids_b = {it["id"] for it in b_items}
    assert not (ids_eval & ids_a), "eval 与 distill 重叠！"
    assert not (ids_eval & ids_b), "eval 与 rl 重叠！"
    assert not (ids_a & ids_b), "distill 与 rl 重叠！"

    banner("产出")
    print(f"  {eval_path}                 {len(eval_items)} 条（已锁定）")
    print(f"  {a_dir / 'prompts.jsonl'}   {len(a_items)} 条")
    print(f"  {b_dir / 'prompts.jsonl'}   {len(b_items)} 条")
    print(f"  {probe_path}  {len(GENERAL_PROBE)} 条")

    banner("自洽性校验")
    check(items)

    # 记录种子与指纹，供复现
    manifest = lab_path(cfg, "results") / "bank_manifest.json"
    ensure_dir(manifest.parent)
    manifest.write_text(json.dumps({
        "seed": seed,
        "counts": {"eval": len(eval_items), "distill": len(a_items), "rl": len(b_items)},
        "eval_ids": sorted(ids_eval),
        "categories": sorted({it["category"] for it in eval_items}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  指纹已写入 {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
