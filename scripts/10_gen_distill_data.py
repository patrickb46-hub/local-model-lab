#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_gen_distill_data.py —— 支线 A 第一步：让 teacher 批量生成蒸馏数据

思路（序列级蒸馏）：
  本机 Ollama 的 9B teacher 只"动嘴"——它读题、按固定格式给出分步解答。
  这些 (题面 → 解答) 对就是 2B student 的学习材料。
  不做 logit 级蒸馏：同时把 9B teacher 和 2B student 驻留内存要 23 GB，物理上不可能。

关键设计：
  * 断点续跑：每生成一条就 append 落盘，中断后重跑自动跳过已完成 id
  * teacher 质检：逐条判定"格式是否合规""答案是否正确"——teacher 自己也会翻车，
    不统计就直接拿去训练 = 在脏数据上白跑几小时
  * 只把"格式合规"的样本放进训练集（答案错误的样本进 rejected，供人工看）

用法：
  python 10_gen_distill_data.py --limit 20      # 先小量试水（强烈建议）
  python 10_gen_distill_data.py                 # 全量
  python 10_gen_distill_data.py --clean-only     # 只做清洗与切分，不再调 teacher
  python 10_gen_distill_data.py --stats          # 只看已有数据的质检统计
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    answer_equal, banner, compute_reward, ensure_dir, human_time, lab_path,
    load_config, ollama_chat, parse_response, read_jsonl, write_jsonl,
)


def build_messages(system_prompt: str, question: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate(cfg, args) -> int:
    lab_root = lab_path(cfg, "lab_root")
    a_dir = lab_path(cfg, "branch_a")
    prompts = read_jsonl(a_dir / "data" / "prompts.jsonl")
    raw_path = lab_root / cfg["distill"]["raw_file"]

    if not prompts:
        print("✗ 题库为空，请先跑 09_gen_math_bank.py")
        return 1

    tcfg = cfg["teacher"]
    model = args.model or tcfg["model"]
    sys_prompt = cfg["task"]["system_prompt"]
    weights = cfg["rl"]["reward_weights"]
    max_ans = cfg["task"]["output_format"]["answer_max_chars"]

    done = {r["id"] for r in read_jsonl(raw_path)}
    todo = [p for p in prompts if p["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    banner(f"teacher 生成蒸馏数据 · {model}")
    print(f"  题库 {len(prompts)} 条，已完成 {len(done)} 条，本轮待生成 {len(todo)} 条")
    print(f"  输出 {raw_path}")
    if not todo:
        print("  没有待办，直接进入清洗阶段")
        return 0

    ensure_dir(raw_path.parent)
    t0 = time.time()
    ok_fmt = ok_ans = 0
    lat = []

    for i, item in enumerate(todo, 1):
        st = time.time()
        resp = ollama_chat(
            tcfg["ollama_url"], model,
            build_messages(sys_prompt, item["prompt"]),
            options=tcfg["options"], think=tcfg.get("think", False),
            timeout=tcfg["request_timeout"], retry=tcfg["retry"],
        )
        dt = time.time() - st
        lat.append(dt)

        if resp is None:
            print(f"  [{i}/{len(todo)}] {item['id']}  ✗ 请求失败，跳过（可重跑续传）")
            continue

        p = parse_response(resp)
        correct = bool(p["answer_norm"]) and answer_equal(p["answer_norm"], item["answer"])
        reward, parts = compute_reward(resp, item["answer"], weights, max_ans)
        ok_fmt += int(p["format_ok"])
        ok_ans += int(correct)

        rec = {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "gold": item["answer"],
            "teacher_model": model,
            "response": resp,
            "format_ok": p["format_ok"],
            "teacher_correct": correct,
            "reward": round(reward, 4),
            "reward_parts": parts,
            "latency_s": round(dt, 2),
            "n_steps": len(p["steps"]),
        }
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        avg = sum(lat) / len(lat)
        eta = human_time(avg * (len(todo) - i))
        flag = "OK " if (p["format_ok"] and correct) else (
            "格式✗" if not p["format_ok"] else "答案✗")
        print(f"  [{i}/{len(todo)}] {item['id']:6s} {item['category']:8s} "
              f"{flag}  {dt:5.1f}s  ETA {eta}", flush=True)

    total = len(todo)
    banner("生成轮次小结")
    if total:
        print(f"  本轮 {total} 条，用时 {human_time(time.time()-t0)}，"
              f"平均 {sum(lat)/max(len(lat),1):.1f}s/条")
        print(f"  格式合规 {ok_fmt}/{total} = {ok_fmt/total*100:.1f}%")
        print(f"  答案正确 {ok_ans}/{total} = {ok_ans/total*100:.1f}%")
    return 0


def show_stats(cfg) -> int:
    raw_path = lab_path(cfg, "lab_root") / cfg["distill"]["raw_file"]
    rows = read_jsonl(raw_path)
    banner("teacher 数据质检")
    if not rows:
        print("  还没有数据")
        return 1
    n = len(rows)
    nf = sum(1 for r in rows if r["format_ok"])
    na = sum(1 for r in rows if r["teacher_correct"])
    both = sum(1 for r in rows if r["format_ok"] and r["teacher_correct"])
    print(f"  总量            {n}")
    print(f"  格式合规        {nf}  ({nf/n*100:.1f}%)")
    print(f"  答案正确        {na}  ({na/n*100:.1f}%)")
    print(f"  两者都好        {both}  ({both/n*100:.1f}%)   ← 真正能用于训练的比例")
    print(f"  平均耗时        {sum(r['latency_s'] for r in rows)/n:.1f}s/条")

    by_cat = {}
    for r in rows:
        d = by_cat.setdefault(r["category"], [0, 0, 0])
        d[0] += 1
        d[1] += int(r["format_ok"])
        d[2] += int(r["teacher_correct"])
    print(f"\n  {'类别':<12s} {'量':>4s} {'格式':>7s} {'答案':>7s}")
    for k in sorted(by_cat):
        tot, f, a = by_cat[k]
        print(f"  {k:<12s} {tot:>4d} {f/tot*100:>6.0f}% {a/tot*100:>6.0f}%")
    return 0


def clean_and_split(cfg) -> int:
    lab_root = lab_path(cfg, "lab_root")
    a_dir = lab_path(cfg, "branch_a")
    dcfg = cfg["distill"]
    raw_path = lab_root / dcfg["raw_file"]
    rows = read_jsonl(raw_path)

    banner("清洗与切分")
    if not rows:
        print("  ✗ 原始数据为空")
        return 1

    min_ratio = dcfg["min_answer_ratio"]
    use_correct = dcfg.get("filter_correct", True)
    good = [r for r in rows if r["format_ok"]]
    ratio = len(good) / len(rows)
    print(f"  原始 {len(rows)} 条；格式合规 {len(good)} 条（{ratio*100:.1f}%）")
    if ratio < min_ratio:
        print(f"  ⚠ 合规率低于阈值 {min_ratio*100:.0f}% —— 训练集质量存疑，"
              f"先看看 teacher 是不是没听懂格式要求")

    # 验证筛选（rejection sampling）：用同一个可验证的判分器把 teacher 答错的样本剔除。
    # 为什么必须做：实测官方 qwen3.5:9b 在这类算术题上错误率约 20%，
    # 不过滤的话学生等于在学错的解法。
    # 为什么这样仍然公平：支线 B 的奖励用的是同一个验证器 —— 两条线都用上了
    # 「题目可验证」这个前提，谁也没占额外便宜。
    if use_correct:
        pool = [r for r in good if r["teacher_correct"]]
        dropped = len(good) - len(pool)
        print(f"  验证筛选：剔除 teacher 答错的 {dropped} 条 → 剩余 {len(pool)} 条")
        if len(pool) == 0:
            print("  ✗ 筛选后没有可用样本，请检查 teacher 或题库")
            return 1
    else:
        pool = good
        print(f"  （未启用验证筛选：直接把 teacher 全部合规回答用于训练）")

    # 去重（同一题面保留第一条）
    seen, dedup = set(), []
    for r in pool:
        k = r["prompt"].strip()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    print(f"  去重后 {len(dedup)} 条")

    # 切成 chat 格式：system + user + assistant（assistant 即 teacher 的回答）
    sys_prompt = cfg["task"]["system_prompt"]
    samples = [{
        "id": r["id"],
        "category": r["category"],
        "gold": r["gold"],
        "teacher_correct": r["teacher_correct"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"].strip()},
        ],
    } for r in dedup]

    # 按 9:1 切；用固定规则（按 id 排序后取尾）保证可复现
    samples.sort(key=lambda s: s["id"])
    n_test = max(1, int(len(samples) * (1 - dcfg["train_ratio"])))
    test, train = samples[:n_test], samples[n_test:]

    write_jsonl(lab_root / dcfg["train_file"], train)
    write_jsonl(lab_root / dcfg["test_file"], test)

    # 被拒样本留档，便于人工复查 teacher 到底错在哪
    rejected = [r for r in rows if not r["format_ok"] or
                (use_correct and not r["teacher_correct"])]
    write_jsonl(lab_root / dcfg["raw_file"].replace(".jsonl", "_rejected.jsonl"), rejected)

    # 交叉校验：训练/测试集绝不能和 eval_set 撞题
    ev = {r["prompt"].strip() for r in read_jsonl(lab_root / cfg["data"]["eval_file"])}
    leak = [s["id"] for s in samples
            if any(m["role"] == "user" and m["content"].strip() in ev for m in s["messages"])]
    if leak:
        print(f"  ✗ 检测到 {len(leak)} 条与 eval_set 撞题，已剔除：{leak[:5]}")
        train = [s for s in train if s["id"] not in set(leak)]
        test = [s for s in test if s["id"] not in set(leak)]
        write_jsonl(lab_root / dcfg["train_file"], train)
        write_jsonl(lab_root / dcfg["test_file"], test)
    else:
        print("  ✓ 无 eval_set 泄漏")

    print(f"\n  训练集  {len(train):4d} 条  → {lab_root / dcfg['train_file']}")
    print(f"  测试集  {len(test):4d} 条  → {lab_root / dcfg['test_file']}")
    print(f"  被拒档  {len(rejected):4d} 条  （格式不合规，未进训练集）")
    for s in train[:2]:
        print(f"\n  ── 样例 {s['id']} ──")
        print(f"  问：{s['messages'][1]['content']}")
        print(f"  答：{s['messages'][2]['content'][:220]}…")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只生成前 N 条（试水用）")
    ap.add_argument("--model", default=None, help="覆盖 teacher 模型")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if args.stats:
        return show_stats(cfg)
    if args.clean_only:
        return clean_and_split(cfg)

    rc = generate(cfg, args)
    if rc != 0:
        return rc
    show_stats(cfg)
    return clean_and_split(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
