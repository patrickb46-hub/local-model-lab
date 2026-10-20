# Results Showcase · Distillation vs Reinforcement Learning

> **One-sentence conclusion**: on this Intel Mac with no discrete GPU, 2.2 hours of distillation + 5.5 hours of reinforcement learning
> raised Qwen3.5-2B's math accuracy from **58.3% to 83.3%** — **above the original 4B's 80.0%**.
>
> As an external reference, cloud **DeepSeek V4 Pro is 100%** (60/60 correct) — it marks the ceiling for this kind of problem.

Completed: 2026-10-05 04:36 (the head-to-head baseline was upgraded at 11:35: the quantized 2b was removed and the cloud API was brought in)
The entire pipeline ran CPU-only on this machine | the cloud baseline goes over the network and uses no local memory

---

## I. Core Results

| Model | Format compliance | **Task accuracy** | Teacher consistency | Output diversity | Avg. time | Size |
|---|---|---|---|---|---|---|
| base · Qwen3.5-2B original (not fine-tuned) | 98.3% | 58.3% | 58.7% | 40.5% | 51.0s | 4.3 GB |
| **A · Distilled LoRA** ⭐ | 100% | **83.3%** | 72.1% | 16.4% | 47.7s | 4.1 GB |
| **B · RL LoRA** | 100% | **68.3%** | 51.6% | 30.2% | 59.3s | 4.1 GB |
| **DeepSeek V4 Pro (cloud API)** | 100% | **100.0%** | 61.3% | 18.6% | **1.6s** | cloud service |
| official qwen3.5:4b (local, quantized) | 100% | **80.0%** | 67.3% | 30.5% | 11.3s | 3.4 GB |
| teacher · official qwen3.5:9b (local, quantized) | 100% | 85.0% | — | 26.7% | 20.6s | 6.6 GB |

Evaluation methodology: 20 program-generated math problems × 3 samples per problem = 60 responses per model, 438 judgments in total.
(The former "official qwen3.5:2b quantized" entry has been removed: it is the same model as base at two precisions,
so the difference reflects only quantization loss, not capability; keeping it in the table would only get in the way of reading it.)

**Four conclusions worth remembering**

1. **The distillation branch's 83.3% > official 4B's 80.0%** — the fine-tuned 2B caught up to the original 4B
2. **The two branches did diverge**: distillation's "teacher consistency" is 72.1%, while RL is only 51.6%
   — the latter is even **below the un-fine-tuned base (58.7%)**. Because RL only cares about getting the answer right and does not care at all whether it resembles the teacher
3. **The cloud baseline scored 60/60, which in turn validates the evaluation apparatus itself**: the problems are solvable, the reference answers are correct, and the grader has no false negatives.
   Previously, all 6 local models scored 0 in the "system of two linear equations in two unknowns" category, which was suspected to be a grading bug;
   the cloud model scored 100% in that category — proving it was **genuine difficulty**, not a broken apparatus
4. **An honest counter-observation**: the cloud API takes **1.6 s** per response, while the local 2B takes **47.7 s** —
   on this GPU-less machine, local inference is **30× slower** than the cloud. "Local deployment saves money" does not hold here

---

## II. How to Use These Models

### Option A · Command-line chat (recommended, works locally)

#### ① One-liner (easiest — the aliases **are already installed**)

Open a **new** terminal window and type:

```bash
ask A "解方程：3x + 5 = 11，求 x 的值。"
```

To switch models, change only the first argument:

```bash
ask base "同一个题目"     # the un-fine-tuned original 2B (for reference)
ask A    "同一个题目"     # the distilled model (best results, 83.3%)
ask B    "同一个题目"     # the reinforcement-learning model (68.3%)
```

Interactive mode (no problem given; you can ask repeatedly):

```bash
ask A
# once "你 >" appears, type the problem directly; enter :q to quit
```

If something goes wrong, run this first to check the environment:

```bash
askinfo        # equivalent to ask --info; self-check only, does not run the model
```

> The aliases have been written to `~/.bash_profile`, but they **only take effect in terminals opened afterwards**.
> In a window that is already open, run `source ~/.bash_profile` once first.

#### ② Prefer not to use the aliases — full-path version (runs from any directory)

```bash
bash ~/Desktop/AI/Local-Model-Lab/ask.sh A "解方程：3x + 5 = 11，求 x 的值。"
```

#### ③ Not even ask.sh — the most bare-bones form

```bash
<your training interpreter> \
  ~/Desktop/AI/Local-Model-Lab/scripts/chat.py -m A -p "解方程：3x + 5 = 11"
```

#### Other flags

```bash
ask A -t "题目"          # enable thinking mode (very slow locally, not recommended)
ask --help               # see all usage
```

Measured output (`ask A "解方程：3x + 5 = 11，求 x 的值。"`):

```
【思路】通过移项合并同类项求解一元一次方程。
【步骤】
1. 将常数项移至右边得 3x=6。
2. 两边同时除以系数 3 得 x=2。
【答案】2
```

> ⚠️ This machine does CPU-only inference, measured at **1.2–1.5 tokens/second**; one response usually takes **40–90 seconds**
> (the first question adds another 20–30 seconds to load the model). This is determined by the hardware (see `docs/踩坑记录.md`),
> not a problem with the script. Using interactive mode ③ avoids reloading the model every time.
>
> The entry script automatically checks available memory: **below 8 GB it switches to bfloat16** (saves memory, generation is slightly slower),
> to avoid freezing from heavy swapping.

### Option B · Ollama (temporarily **unusable** on this machine; will work after switching machines)

Both models have already been imported, but **loading them on an Intel Mac fails**:

```
$ ollama run mylab-distill
Error: mlx runner failed: panic: mlx: [metal::Device] Unable to construct residency set.
```

**Cause**: on macOS, Ollama 0.35 hands **safetensors-format** models to the **MLX runner**
(Apple's framework, **Apple Silicon only**). An Intel Mac has no unified memory, so it panics immediately.
The official `qwen3.5:2b` and others are unaffected — they are in GGUF format and go through llama.cpp.

**After switching to an M-series Mac** (the user plans to switch to an M2 Pro by the end of 2026), the two models **can be used directly**:

```bash
ollama run mylab-distill     # the distilled model
ollama run mylab-rl          # the reinforcement-learning model
```

So it is **recommended to keep them**; do not delete them. Each is **4.4 GB**.

### Option C · Calling from Python code

```python
import sys; sys.path.insert(0, "~/Desktop/AI/Local-Model-Lab/scripts")
import labmodel as lm
from peft import PeftModel

tok = lm.load_tokenizer("branch-A-distill/model")
model, _ = lm.load_model("branch-A-distill/model", dtype="float32", drop_vision=True)
model = PeftModel.from_pretrained(model, "branch-A-distill/adapter")

msgs = [{"role": "system", "content": "你是初中数学老师，按【思路】【步骤】【答案】作答。"},
        {"role": "user", "content": "解方程：3x + 5 = 11"}]
txt, n, sec = lm.generate_reply(model, tok, msgs, max_new_tokens=200, temperature=0.3)
print(txt)
```

---

## III. Reading the Report

| File | How to read it |
|---|---|
| `shared/results/横评报告.html` | **Double-click to open** (59 KB, self-contained with no external links, readable offline, printable as is) |
| `shared/results/横评报告.md` | plain-text version, good for copying elsewhere |
| `shared/results/eval_metrics.json` | full metrics, for programs to read |
| `shared/results/eval_raw.jsonl` | **438 raw responses**; every conclusion can be traced back and re-checked |

Check progress and results quickly in the terminal:

```bash
bash ~/Desktop/AI/Local-Model-Lab/scripts/watch.sh          # Chinese
bash ~/Desktop/AI/Local-Model-Lab/scripts/watch.sh --en     # English
```

---

## IV. File Map

```
Local-Model-Lab/
├── 成果展示.md                      ← this file
├── README.md                        ← quick-start and environment notes
├── docs/
│   ├── 踩坑记录.md                  ← 16 real pitfalls (environment, methodology, troubleshooting)
│   └── 原理笔记.md                  ← the basis for technical decisions
├── configs/lab.config.json          ← full config (steps, sampling, evaluation methodology)
├── scripts/                         ← 15 scripts: problem generation / training / evaluation / reporting / chat / progress
├── shared/
│   ├── eval_set.jsonl               ← evaluation set, 20 problems (locked before training, never entered training data)
│   ├── general_probe.jsonl          ← general-ability probe, 10 problems
│   └── results/                     ← reports and all raw data
├── branch-A-distill/
│   ├── model/               4.3 GB  ← working copy (the base)
│   ├── adapter/             83 MB   ← ★ the trained artifact (LoRA weights, 67.3 MB)
│   └── merged/              4.1 GB  ← the merged, standalone model
├── branch-B-rl/                     ← same as above (identical structure)
├── backup/
│   ├── initial/             167 MB  ← mid-training snapshot
│   └── trained/             199 MB  ← post-training snapshot
└── base/                    4.3 GB  ← original base (read-only, never modified)
```

**What exactly is "the trained artifact"?**
It is the **83 MB LoRA adapter** directory (of which the weights file is 67.3 MB).
It accounts for only **0.9% of the base's parameters** (16.82M / 1.882B) and must be attached to the base to be used.
Only the merged `merged/` is a complete model that can run standalone (4.1 GB).

---

## V. Environment Requirements (Copy After Switching Machines)

| Item | Value |
|---|---|
| Python | `~/.workbuddy/binaries/python/envs/mlab-conda/bin/python` (3.12.14) |
| torch | 2.13.0 (**must come from conda-forge**; on PyPI, Intel Mac stops at 2.2.2) |
| transformers | 5.18.0 (4.57.6 has no qwen3_5 architecture) |
| Precision | **float32 throughout** — this machine's CPU has no native BF16 instructions; bf16 backward is 200× slower |
| Memory | training peak 11.5–12.2 GB (a 16 GB machine can run it, but gradient checkpointing is required) |

The full pitfalls log is in `docs/踩坑记录.md` (16 pitfalls, each with reproduction and solution).
