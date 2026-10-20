#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_screen_layers.py —— 分层筛查（GATE-1，整个实验的命门）

为什么需要它：
  现有 20 题评测集里「teacher 系统性做错、而 base 尚有可能做对」（L2 层）
  的题只有 1 道（5%），且 pass@3 上蒸馏与 teacher 都是 90% —— 天花板已顶满，
  **统计上不可能测出「RL 突破 teacher」**。所以必须先在大题海里把 L2 筛出来。

分层定义（只用 base 与 teacher 的测量判定，与训练方法完全无关）：
  L1  teacher pass@K > 0            —— 蒸馏的主场
  L2  teacher pass@K = 0 且 base pass@K > 0  —— RL 的战场（主端点）
  L3  两者皆 pass@K = 0             —— 谁都做不到，作为天花板证据

用 K>=5 是为了判「系统性失败」：若 teacher 只是采样噪声（如 1/3 对），
  filter_correct 会保留它答对的那次，蒸馏就不被结构性封堵、论证会塌。

产出（三份都带断点续跑，长任务随时可中断）：
  shared/screen_raw.jsonl                      每次采样的原始结果（供 P6 复用，不重复评测）
  shared/bench_stratified.jsonl                分层基准（训练前冻结）
  branch-B-rl/data/rl_pool_curated.jsonl       RL 训练池
  shared/results/layer_manifest.json           指纹与统计

用法：
  python 12_screen_layers.py --model base      # 只跑 base 侧（本机 fp32，约 7~13h）
  python 12_screen_layers.py --model teacher   # 只跑 teacher 侧（Ollama 9B，约 12~14h）
  python 12_screen_layers.py --model report    # 只汇总分层、写产物（秒级）
  python 12_screen_layers.py --model both      # 顺序跑完再汇总
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    answer_equal, banner, ensure_dir, human_time, lab_path, load_config,
    ollama_chat, parse_response, read_jsonl, write_jsonl,
)

RAW_NAME = "screen_raw.jsonl"


def _norm_q(s: str) -> str:
    """题面归一化（折叠空白），用于跨池去重。"""
    return " ".join((s or "").split())


def _key(r) -> tuple:
    return (r["qid"], r["model"], int(r["sample"]))


def screen_base(cfg, pool, raw_path, k, batch, resume: set, limit=0):
    """用本机 base（无 LoRA）对每道题采样 K 次，判对错。"""
    import torch
    import labmodel as lm

    b_dir = lab_path(cfg, "branch_b")
    model_path = str(b_dir / "model")
    dtype = cfg["base_model"].get("dtype", "float32")
    torch.set_num_threads(cfg["runtime"]["torch_threads"])

    print(f"  载入 base（{dtype}，无 LoRA）…")
    tok = lm.load_tokenizer(model_path, cfg["base_model"]["trust_remote_code"])
    model, info = lm.load_model(model_path, dtype=dtype,
                               drop_vision=cfg["base_model"].get("text_only", True),
                               trust_remote_code=cfg["base_model"]["trust_remote_code"])
    model.eval()
    print(f"  参数 {info['params_b']}B")
    sys_prompt = cfg["task"]["system_prompt"]
    temp = float(cfg["eval"].get("sample_temperature", 0.7))

    t0 = time.time()
    done = 0
    for i, item in enumerate(pool, 1):
        if limit and i > limit:
            break
        need = [s for s in range(k) if (item["id"], "base", s) not in resume]
        if not need:
            continue
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": item["prompt"]}]
        texts = []
        # 分块批量生成：**只生成 need 里缺的那几条**。
        # 注意这里必须按 len(need) 收敛、并按 need 的顺序取用 —— 若按 k 收敛再用
        # texts[s] 索引，断点续跑时（need 非 [0..k-1]）会静默取错样本。
        chunk = batch if batch > 0 else len(need)
        while len(texts) < len(need):
            n = min(chunk, len(need) - len(texts))
            if n <= 0:
                break
            part, _nt, _s = lm.generate_reply_batch(
                model, tok, msgs, num_return_sequences=n,
                max_new_tokens=int(cfg["eval"].get("max_new_tokens", 192)),
                temperature=temp, top_p=0.95,
                seed=int(cfg["data"]["seed"]) + i * 1000 + len(texts))
            if not part:
                break
            texts.extend(part)

        rows = []
        for idx, s in enumerate(need):
            txt = texts[idx] if idx < len(texts) else ""
            if not txt.strip():
                # 同理：本地生成返回空也绝不能记成「答错」
                raise RuntimeError(
                    f"{item['id']} s{s} 本地生成返回空 —— 已中止。"
                    f"「调用失败」与「答错」必须区分开。")
            p = parse_response(txt)
            rows.append({
                "qid": item["id"], "model": "base", "sample": s,
                "category": item["category"], "gold": item["answer"],
                "response": txt, "answer_norm": p["answer_norm"],
                "correct": answer_equal(p["answer_norm"], item["answer"]),
                "format_ok": p["format_ok"],
            })
        write_jsonl_append(raw_path, rows)
        for r in rows:
            resume.add(_key(r))
        done += 1
        if done % 5 == 0 or i == len(pool):
            el = time.time() - t0
            print(f"  [base {i}/{len(pool)}] {item['id']:6s}  已完成 {len(resume)} 条采样  "
                  f"用时 {human_time(el)}", flush=True)
    return 0


def _check_ollama(cfg) -> str:
    """前置探活：返回错误信息字符串；None 表示可用。

    为什么必须在开跑前探活：筛查的 teacher 侧要跑 1~2 小时。
    如果服务不可用却直接开跑，会白跑一小时**并且**产出一份看起来正常的
    错误数据（见 screen_teacher 的注释）。
    """
    from common import ollama_health
    tcfg = cfg["teacher"]
    names = ollama_health(tcfg["ollama_url"])
    if names is None:
        return f"无法连接 {tcfg['ollama_url']}（服务未启动？）"
    want = tcfg["model"]
    if want not in names and f"{want}:latest" not in names:
        return f"模型 {want} 不在本机；可用：{names}"
    return None


def screen_teacher(cfg, pool, raw_path, k, resume: set, limit=0):
    """用 Ollama 的 9B teacher 对每道题采样 K 次。

    ⚠ **调用失败绝不能记成「答错」**。
    踩过的坑（2026-10-09）：Ollama 服务中途退出，145 次调用全部返回 `None`，
    而旧代码写的是 `"correct": bool(txt) and ...` → teacher 被判成 **100% 全错**
    → 所有题被误判成 L2/L3 → **整个分层基准作废**，
    而且**日志里一行错误都没有**（只有被 502 静默吞掉的空回答）。
    现在：单条失败会重试 3 次；连续 FAIL_STREAK_LIMIT 次失败就**中止筛查**，
    且失败的题**不写盘**（留给断点续跑重试）。
    """
    tcfg = cfg["teacher"]
    model = tcfg["model"]
    temp = float(cfg["eval"].get("sample_temperature", 0.7))
    sys_prompt = cfg["task"]["system_prompt"]
    FAIL_STREAK_LIMIT = 3

    t0 = time.time()
    done = 0
    fail_streak = 0
    n_ok = n_fail = 0
    for i, item in enumerate(pool, 1):
        if limit and i > limit:
            break
        need = [s for s in range(k) if (item["id"], "teacher", s) not in resume]
        if not need:
            continue
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": item["prompt"]}]
        rows = []
        for s in need:
            txt = None
            for attempt in range(3):
                txt = ollama_chat(
                    tcfg["ollama_url"], model, msgs,
                    options={**tcfg["options"], "temperature": temp,
                             "num_predict": int(tcfg["options"].get("num_predict", 192))},
                    think=False,
                    timeout=tcfg["request_timeout"], retry=tcfg["retry"])
                if txt and txt.strip():
                    break
                time.sleep(1.0 * (attempt + 1))

            if not txt or not txt.strip():
                fail_streak += 1
                n_fail += 1
                print(f"  ✗ {item['id']} s{s} 调用失败（连续 {fail_streak} 次）", flush=True)
                if fail_streak >= FAIL_STREAK_LIMIT:
                    raise RuntimeError(
                        f"Ollama 连续 {FAIL_STREAK_LIMIT} 次调用失败 —— 已中止筛查。\n"
                        f"  常见原因：① 服务未启动或中途退出"
                        f"（`open -a Ollama`，再用 "
                        f"`curl -s http://127.0.0.1:11434/api/tags` 确认）；"
                        f"② 模型名不对。\n"
                        f"  **绝不能**把「调用失败」当成「答错」—— 那会让 teacher 被判成"
                        f" 100% 全错、所有题被误判成 L2/L3，分层基准彻底作废且不报错。")
                break

            fail_streak = 0
            n_ok += 1
            p = parse_response(txt)
            rows.append({
                "qid": item["id"], "model": "teacher", "sample": s,
                "category": item["category"], "gold": item["answer"],
                "response": txt, "answer_norm": p["answer_norm"],
                "correct": answer_equal(p["answer_norm"], item["answer"]),
                "format_ok": p["format_ok"],
            })

        if not rows:
            continue        # 整题都失败 → 不写盘，留给续跑重试
        write_jsonl_append(raw_path, rows)
        for r in rows:
            resume.add(_key(r))
        done += 1
        if done % 5 == 0 or i == len(pool):
            el = time.time() - t0
            print(f"  [teacher {i}/{len(pool)}] {item['id']:6s}  "
                  f"已完成 {len(resume)} 条采样（成功 {n_ok} / 失败 {n_fail}）  "
                  f"用时 {human_time(el)}", flush=True)
    return 0


def write_jsonl_append(path: Path, rows) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(cfg, pool, raw_path, out_dir) -> int:
    """汇总分层，写基准与训练池。"""
    rows = read_jsonl(raw_path)
    if not rows:
        print("✗ 还没有筛查数据")
        return 1
    layers_cfg = cfg.get("layers", {})
    k_need = int(layers_cfg.get("base_pass_k", 5))

    per = {}
    for r in rows:
        per.setdefault(r["qid"], {"base": [], "teacher": []})
        if r["model"] in per[r["qid"]]:
            per[r["qid"]][r["model"]].append(int(bool(r["correct"])))

    def layer_of(qid):
        d = per.get(qid, {})
        b = sum(d.get("base", []))
        t = sum(d.get("teacher", []))
        if not d.get("base") or not d.get("teacher"):
            return None, b, t
        if t > 0:
            return "L1", b, t
        if b > 0:
            return "L2", b, t
        return "L3", b, t

    classified = {}
    for it in pool:
        lay, b, t = layer_of(it["id"])
        if lay:
            classified[it["id"]] = {"layer": lay, "base_pass": b, "teacher_pass": t}

    stats = {}
    for lay in ("L1", "L2", "L3"):
        stats[lay] = sorted(q for q, v in classified.items() if v["layer"] == lay)

    banner("分层结果")
    n_scr = len(classified)
    for lay in ("L1", "L2", "L3"):
        c = len(stats[lay])
        print(f"  {lay}: {c:4d} 题" + (f"（{c/n_scr*100:.1f}%）" if n_scr else ""))
    print(f"  共筛查 {n_scr} 题（每侧 K={k_need} 次采样）")
    print(f"  ★ L2（teacher 系统性失败 且 base 有机会）= {len(stats['L2'])} 题"
          f"   闸门要求 >= {int(layers_cfg.get('min_l2', 24))}")

    # ---- 选基准题 ----
    want = {
        "L1": int(layers_cfg.get("bench_l1", 20)),
        "L2": int(layers_cfg.get("bench_l2", 40)),
        "L3": int(layers_cfg.get("bench_l3", 16)),
    }
    by_id = {it["id"]: it for it in pool}
    rnd = random.Random(cfg["data"]["seed"] + 999)
    bench = []
    for lay in ("L1", "L2", "L3"):
        cand = stats[lay][:]
        rnd.shuffle(cand)
        for qid in cand[:want[lay]]:
            it = by_id[qid]
            bench.append({"id": qid, "layer": lay, "category": it["category"],
                          "prompt": it["prompt"], "answer": it["answer"],
                          "base_pass": classified[qid]["base_pass"],
                          "teacher_pass": classified[qid]["teacher_pass"]})
    bench.sort(key=lambda r: (r["layer"], r["id"]))

    bench_path = lab_path(cfg, "lab_root") / cfg["data"].get(
        "bench_file", "shared/bench_stratified.jsonl")
    write_jsonl(bench_path, bench)

    # ---- RL 训练池：L2 为主 + 一部分 base 可解的 L1（保证奖励有方差）----
    pool_ids = stats["L2"][:]                       # L2 全部进训练池
    l1_ok = [q for q in stats["L1"] if classified[q]["base_pass"] > 0]
    rnd.shuffle(l1_ok)
    pool_ids += l1_ok[:max(0, int(layers_cfg.get("pool_l1", 60)))]
    held_out = {b["id"] for b in bench}
    pool_ids = [q for q in pool_ids if q not in held_out]   # 训练池与基准池严格隔离
    rl_pool = [{"id": by_id[q]["id"], "category": by_id[q]["category"],
                "prompt": by_id[q]["prompt"], "answer": by_id[q]["answer"],
                "layer": classified[q]["layer"]} for q in pool_ids]
    rl_path = lab_path(cfg, "branch_b") / "data" / "rl_pool_curated.jsonl"
    write_jsonl(rl_path, rl_pool)

    # ---- 隔离纪律：id 级 + 题面级，两池两两不相交 ----
    ids_bench = {b["id"] for b in bench}
    ids_rl = {r["id"] for r in rl_pool}

    def _rows_of(p):
        try:
            return read_jsonl(p)
        except Exception:  # noqa: BLE001
            return []

    eval_rows = _rows_of(lab_path(cfg, "lab_root") / cfg["data"]["eval_file"])
    dist_rows = _rows_of(lab_path(cfg, "branch_a") / "data" / "prompts.jsonl")
    ids_eval = {r["id"] for r in eval_rows}
    ids_dist = {r["id"] for r in dist_rows}

    assert not (ids_bench & ids_rl), "基准池与 RL 训练池重叠！"
    assert not (ids_bench & ids_eval), "基准池与评测集重叠！"
    assert not (ids_rl & ids_eval), "RL 训练池与评测集重叠！"

    def _ptext(rows):
        return {_norm_q(r["prompt"]) for r in rows}

    t_bench, t_rl = _ptext(bench), _ptext(rl_pool)
    t_eval, t_dist = _ptext(eval_rows), _ptext(dist_rows)
    assert not (t_bench & t_rl), "基准池与 RL 训练池存在同题（题面级）！"
    assert not (t_bench & t_eval), "基准池与评测集存在同题（题面级）！"
    assert not (t_bench & t_dist), "基准池与蒸馏池存在同题（题面级）！"
    assert not (t_rl & t_eval), "RL 训练池与评测集存在同题（题面级）！"

    manifest = {
        "screened": n_scr,
        "k": k_need,
        "counts": {lay: len(stats[lay]) for lay in ("L1", "L2", "L3")},
        "bench": {lay: sum(1 for b in bench if b["layer"] == lay) for lay in ("L1", "L2", "L3")},
        "bench_ids": sorted(ids_bench),
        "rl_pool_size": len(rl_pool),
        "gate_min_l2": int(layers_cfg.get("min_l2", 24)),
        "gate_pass": len(stats["L2"]) >= int(layers_cfg.get("min_l2", 24)),
    }
    mp = lab_path(cfg, "results") / "layer_manifest.json"
    ensure_dir(mp.parent)
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  已写出 {bench_path}（{len(bench)} 题）")
    print(f"  已写出 {rl_path}（{len(rl_pool)} 题）")
    print(f"  指纹已写入 {mp}")
    print(f"\n  🚦 GATE-1：{'通过 ✓ 可以进入重训' if manifest['gate_pass'] else '★ 未通过 —— L2 不足，应回 P1 调整题族，不要盲目投入重训'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "teacher", "both", "report"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 题（试水用）")
    ap.add_argument("--k", type=int, default=0, help="覆盖每侧采样次数")
    ap.add_argument("--batch", type=int, default=4, help="base 侧批量生成的分块大小")
    args = ap.parse_args()

    cfg = load_config()
    lab_root = lab_path(cfg, "lab_root")
    pool_path = lab_root / cfg["data"].get("candidate_pool_file",
                                           "shared/candidate_pool.jsonl")
    raw_path = lab_root / "shared" / RAW_NAME

    banner(f"分层筛查（GATE-1）· model={args.model}")
    if not pool_path.exists():
        print(f"✗ 候选池不存在：{pool_path}")
        print("  先跑： python 09_gen_math_bank.py --pool-only")
        return 1
    pool = read_jsonl(pool_path)

    # ---- 题面级过滤（双保险）----
    # 源头上 09_gen_math_bank.py 已排除训练/评测池的题面，这里再兜一次：
    # id 前缀不同**不代表**题目不同。实测就出现过 12 题与训练池题面完全一致 ——
    # 基准题一旦与训练数据同题，「模型没见过」的前提就破了，全部结论作废。
    def _prompts_of(p) -> set:
        try:
            return {_norm_q(r.get("prompt", "")) for r in read_jsonl(p)}
        except Exception:  # noqa: BLE001
            return set()

    forbid = set()
    for _p in (lab_root / cfg["data"]["eval_file"],
               lab_path(cfg, "branch_a") / "data" / "prompts.jsonl",
               lab_path(cfg, "branch_b") / "data" / "prompts.jsonl",
               lab_root / cfg["eval"]["general_probe_file"]):
        forbid |= _prompts_of(_p)
    _before = len(pool)
    pool = [it for it in pool if _norm_q(it["prompt"]) not in forbid]
    if len(pool) != _before:
        print(f"  ⚠ 题面级过滤：剔除 {_before - len(pool)} 题（与训练/评测池重复）")
    else:
        print(f"  题面级检查：与训练/评测池无重复 ✓（共比对 {len(forbid)} 条已有题面）")

    layers_cfg = cfg.get("layers", {})
    k = args.k or max(int(layers_cfg.get("base_pass_k", 5)),
                      int(layers_cfg.get("teacher_pass_k", 5)))
    print(f"  候选池 {len(pool)} 题，每侧 K={k}")
    print(f"  原始结果 {raw_path}")

    existing = {(r["qid"], r["model"], int(r["sample"])) for r in read_jsonl(raw_path)}
    print(f"  已有 {len(existing)} 条采样（断点续跑）")

    if args.model in ("base", "both"):
        banner("① base 侧（本机 fp32，无 LoRA）")
        screen_base(cfg, pool, raw_path, k, args.batch, existing, args.limit)
    if args.model in ("teacher", "both"):
        banner("② teacher 侧（Ollama qwen3.5:9b）")
        # 前置探活：宁可现在失败，也不要白跑两小时再产出错误数据
        err = _check_ollama(cfg)
        if err:
            print(f"  ✗ Ollama 不可用：{err}")
            print("    先启动服务：  open -a Ollama")
            print("    确认连通：    curl -s http://127.0.0.1:11434/api/tags")
            print("    然后重跑本脚本（已完成的部分会自动跳过）。")
            return 1
        print(f"  ✓ Ollama 就绪（模型 {cfg['teacher']['model']}）")
        screen_teacher(cfg, pool, raw_path, k, existing, args.limit)
    if args.model in ("base", "teacher", "both") and args.limit:
        banner("试点汇总（--limit 模式，仅统计已筛查部分）")
        report(cfg, pool[: args.limit] if args.model != "teacher" else pool[: args.limit],
               raw_path, lab_root)
        print("\n  ⚠ 以上是试点结果，不是最终分层。")
        print("    用「L2 命中率」外推全池：N_needed ≈ 目标 L2 数 / 命中率")
        print("    再决定池子要多大（--pool-size），去掉 --limit 跑全量。")
        return 0

    return report(cfg, pool, raw_path, lab_root)


if __name__ == "__main__":
    raise SystemExit(main())
