# Local Model Lab

> **One base, two branches**: the same Qwen3.5-2B copied into two independent replicas —
> one takes distillation (learning from the local 9B teacher), one takes reinforcement learning (GRPO).
> Finally, the same set of small math problems is used for a head-to-head comparison of base / distilled / RL / teacher.

For the full design and trade-offs, see [`The Full Plan.md`](完整计划.md); this document is **quick start + current status**.

> **Don't want the technical details?** Open [`Project Overview.html`](项目全貌.html) (or [`Project Overview.md`](项目全貌.md)) —
> it's written for non-technical readers and uses analogies to explain what this project does and what the conclusions are.
> For the experimental conclusion directly: **on problems the teacher itself can't solve (L2), the RL arm reaches 56.2% while the distillation arm reaches only 16.7%**.

---

## 1. Current System Positioning (Important)

This machine is a **2018 Intel MacBook Pro (x86_64)**, and these hardware constraints shaped the entire project's technology choices.
If you plan to reproduce this on another machine, read [`docs/Pitfalls Log.md`](docs/踩坑记录.md) first — it lists six pitfalls that can stall the whole pipeline.

| Item | Current state |
|---|---|
| Architecture | x86_64 (Intel) — **no MPS, CPU-only** |
| Memory | 16 GB (the system takes 4–6 GB first) |
| Training interpreter | `envs/mlab-conda/bin/python` (conda-forge provides a modern torch) |
| base model | `Qwen/Qwen3.5-2B` (HF format, ~4.6 GB) |
| teacher | local Ollama `qwen3.5:9b` (only talks, never trained) |

### Why There Are Two Python Environments

This is an unavoidable conflict that deserves a clear, separate explanation:

```
Intel Mac's PyPI torch   → stuck at 2.2.2 (officials no longer ship macOS x86_64 wheels)
Qwen3.5 needs transformers → 5.x (there is no qwen3_5 architecture in 4.57 at all)
transformers 5.x requires  → torch >= 2.5
                          ↑ the three cannot all hold at once
```

The way out is **conda-forge**: it has built all the way up to `pytorch 2.13.0` for osx-64 (including the `cpu_mkl_py312` build).
So the final environment looks like this:

| Environment | Purpose | Contents |
|---|---|---|
| `envs/mlab-conda` | **Primary**: all scripts that need models | conda-forge python 3.12 + pytorch 2.13.0 (MKL) + transformers 5.x / peft installed via pip |
| `envs/mlab` | Fallback: scripts that don't need models | pure PyPI, torch 2.2.2 (runs, but can't handle Qwen3.5) |

Scripts point to the primary environment via `runtime.python` in `configs/lab.config.json`,
and `common.resolve_python()` automatically falls back when it doesn't exist.

---

## 2. Directory Structure

```
Local-Model-Lab/
├── 完整计划.md              ← full design (the complete plan for one base, two branches)
├── README.md                ← this file
├── configs/lab.config.json  ← global config: every size, path, and hyperparameter lives here
├── base/                    ← original base weights (read-only, never modified)
├── branch-A-distill/        ← Branch A: distillation
│   ├── model/               ←   working copy (copied from base, physically isolated)
│   ├── data/                ←   teacher-generated data + cleaned results
│   ├── adapter/             ←   LoRA training output
│   └── logs/                ←   training logs
├── branch-B-rl/             ← Branch B: reinforcement learning (same structure as above)
├── scripts/                 ← all executable scripts
├── shared/
│   ├── eval_set.jsonl       ← frozen evaluation set (locked before training, never enters training data)
│   ├── general_probe.jsonl  ← general-capability probe (guards against catastrophic forgetting)
│   └── results/             ← evaluation metrics, visualization reports
├── logs/                    ← infrastructure logs for environment setup, downloads, etc.
└── docs/                    ← How It Works, Pitfalls Log
```

**Isolation discipline**: `base/` is read-only; `branch-A-distill/` and `branch-B-rl/` never cross;
`shared/eval_set.jsonl` is frozen once generated. If any one of these three is broken, every downstream conclusion loses its meaning.

---

## 3. Unified Task Axis

Both branches run on **the same task**, so that the head-to-head comparison is valid:

> **Middle-school math problems · structured step-by-step solutions**

Fixed output format (machine-parseable):

```
【思路】State the solving direction in one sentence.
【步骤】
1. Step one…
2. Step two…
【答案】Write only the final answer.
```

The problem bank is **programmatically generated** by `09_gen_math_bank.py` (ten categories, answers computed exactly with `Fraction`).
The reasons for choosing programmatic problem generation over a hand-written bank are straightforward:

- Answers are exact by construction — only then can reinforcement learning have a "verifiable reward," with no need to train a separate reward model
- Reproducible with a fixed seed, and scalable to unlimited samples
- Problems and answers map strictly one-to-one, leaving no gray area where "the reference answer has to be guessed"

| Branch | What it learns | Reward / supervision signal |
|---|---|---|
| A · Distilled | Phrasing for step-by-step math solutions + fixed format (A2 domain + A3 format) | Mimic the teacher's answers (token-level supervision) |
| B · RL | Math problems + domain output conventions (B3 + B4) | Correct answer +1.0 / format compliance +0.5 / complete steps +0.2 / overlong answer −0.3 |

---

## 4. Running It

All scripts are run with `python scripts/xx.py`. **Follow the numbering order**:

| Step | Script | Purpose | Rough duration |
|---|---|---|---|
| 0 | `00_check_env.py` | Environment self-check; proceed only if all green | seconds |
| 0 | `01_setup_env.sh` | Set up the Python environment (idempotent, re-runnable) | 10+ minutes |
| 0 | `02_download_base.py` | Download base weights from hf-mirror | 10+ minutes |
| 0 | `03_make_copies.sh` | Copy into two isolated working replicas | 1 minute |
| 1 | `09_gen_math_bank.py` | Generate the problem bank + lock the evaluation set | seconds |
| 1 | `10_gen_distill_data.py` | teacher generates distillation data in bulk → clean and split | 1–3 hours |
| 1 | `11_train_distill.py` | LoRA supervised fine-tuning | 2–4 hours |
| 2 | `20_train_rl.py` | GRPO reinforcement learning | 1–3 hours |
| 3 | `30_evaluate.py` | Four-model head-to-head comparison | tens of minutes |
| 3 | `31_report.py` | Generate the visual head-to-head report (HTML) | seconds |
| 4 | `40_export_ollama.sh` | Merge LoRA → import into Ollama | tens of minutes |
| — | `99_night_run.sh` | **one-command overnight batch** (detailed below) | about 8–16 hours |

### One-Command Overnight Batch

If you want to save effort, one command runs everything above in order (the two branches run **strictly serially**, never occupying the machine at the same time):

```bash
cd ~/Desktop/AI/Local-Model-Lab
nohup bash scripts/99_night_run.sh > /dev/null 2>&1 &
```

It does three things that make life easier:

1. **`caffeinate` holds the whole time** — macOS sleep will cut off long tasks (we hit this while downloading models), so it has to be pinned
2. **Every step is idempotent and resumable** — if it breaks midway, re-run this script; completed steps are skipped automatically, so no work is wasted
3. **The whole run is written to `logs/night_run.log`** — in the morning you only need to look at this one file

### Check Progress Anytime ← The Most-Used Command

**Want to know "where it is now" → just look at `progress.sh`** (the single entry point, all on one screen):

```bash
cd ~/Desktop/AI/Local-Model-Lab

bash scripts/progress.sh             # Chinese, view once
bash scripts/progress.sh -w          # auto-refresh every 20 seconds (Ctrl-C to exit)
bash scripts/progress.sh -w 5        # custom refresh interval of 5 seconds

bash scripts/progress.sh --en        # English version
bash scripts/progress_en.sh          # English version (same as above, saves a flag)
bash scripts/progress_en.sh -w       # English + auto-refresh
```

It shows: **what is currently running** (which processes are alive), **the full-pipeline timeline** (the duration of each step from P3~P7),
**the artifacts of the four arms** (adapter sizes for A1/A2/B/C), **the data artifacts**,
**evaluation progress · primary run by tier** (how much each role still lacks in each tier),
and the **GATE-1 gate verdict** (whether the L2 problems number at least 24).

**Durations are always carried up automatically**, no longer a long string of seconds: `25549s` displays as `7h05m`,
`5230s` displays as `1h27m`, and beyond one day it carries up to `1d01h`.

Dedicated viewers (as needed):

| Command | What it shows |
|---|---|
| `bash scripts/progress.sh` | **the full pipeline** (recommended; this is the one above) |
| `bash scripts/screen_status.sh` | details of the tiered screening (P2): base/teacher progress, difficulty distribution, ETA |
| `bash scripts/watch.sh` | step timeline of the last "overnight batch" (**warns when the log is stale**) |

**`watch.sh`** is the viewer that shows at a glance "where it is and how much time is left" (read-only, safe to run anytime):

```bash
cd ~/Desktop/AI/Local-Model-Lab

bash scripts/watch.sh              # Chinese snapshot
bash scripts/watch.sh --en         # English snapshot (fully English UI)
bash scripts/watch.sh --both       # one of each, Chinese and English, in two separate sections
bash scripts/watch.sh -w           # auto-refresh every 20 seconds (Ctrl-C to exit)
bash scripts/watch.sh -w 5         # custom refresh interval of 5 seconds
bash scripts/watch.sh -w --en      # composable: English + auto-refresh
```

The rendering logic lives separately in `scripts/watch_render.py` (to avoid quoting problems when cramming a large block of code into shell),
and `watch.sh` only handles argument parsing and the refresh loop.

It outputs: the step timeline (which steps are done and their durations), the progress bar and ETA for the current step,
the loss / adapter status of both branches, **the estimated overall completion time**, and resource usage.

**⚠ It only recognizes the "overnight batch" log** (`logs/night_run.log`).
That log records the last complete run — if it hasn't been updated for a long time, the screen will show
"all complete," but that is a **historical record**, not the current progress. For this reason `watch.sh` now prints
a prominent reminder when the log hasn't been updated in over 24 hours, and points to the script below.

### Need the Computer for Something Else → Pause / Resume ← Use `labctl.sh`

The experiment processes hold about 12GB of memory resident. To temporarily hand the machine back to yourself:

```bash
cd ~/Desktop/AI/Local-Model-Lab

bash scripts/labctl.sh status            # what is running now and how far along (read-only, safe anytime)
bash scripts/labctl.sh pause             # pause: stop all experiment processes, free memory
bash scripts/labctl.sh resume            # resume: continue from the last progress point
```

**Why pausing is safe**: for evaluation, **each sample is considered complete the moment it hits disk**; on restart it automatically skips
by the key `(role, problem id, kind, sampling index)` — at worst you lose "the problem currently being generated,"
which is automatically re-run on resume.

> Tested in a live drill: before pausing, A1 was at 84/156; after resuming, the log showed `[21/52]…[28/52] all 0s`
> (the 8 completed problems were skipped automatically), continuing from problem 29; 654 records with **zero duplicates, zero corruption**.

**Three things you must know**:

| Case | Explanation |
|---|---|
| **Pausing during training** | **It is not lossless**. SFT / RL saves a checkpoint every 5 steps, so steps since the last checkpoint are lost. The script refuses and requires you to confirm with `pause --force` |
| **Resume from your own terminal** | That way the processes you start belong to your login session and are less likely to be reaped than tool-managed ones |
| **The auto patrol steps aside automatically** | `pause` creates a `logs/PAUSED` marker; the auto patrol that runs every 2 hours sees it and **does nothing** — otherwise it would restart what you just paused and interrupt you |

When pausing, it also backs up the raw evaluation records to `logs/eval_raw_before_pause.jsonl` (a few tens of KB),
giving you a reference in case you want to roll back.

### Watch the Running Tiered Screening ← Use `screen_status.sh`

```bash
bash scripts/screen_status.sh        # view once
bash scripts/screen_status.sh -w     # refresh every 20 seconds
bash scripts/screen_status.sh -l 20  # also show the last 20 log lines
```

It focuses on the **P2 tiered screening (GATE-1)**: process status, how many problems base/teacher have each completed,
the **difficulty distribution on the base side** (all wrong / partly right / all right — where "partly right" is the L2 candidate pool),
speed and ETA, and the post-run **gate verdict** (whether L2 has at least 24 problems).

### What If It Gets Interrupted ← Don't Just Re-run 99_night_run.sh!

**This is the easiest pitfall to fall into**: the training steps in `99_night_run.sh` use
`--fresh` (forcing training from scratch) to preserve the controlled variables. So re-running it directly will **retrain A (about 2.2 hours) + B (about 5 hours)**,
burning a whole night for nothing and overwriting the adapters you already trained.

Use **`resume.sh`** instead — it decides based on the completion marker `◀ 步骤名 … 退出码=0` in the log,
skips already-completed training, and only fills in what didn't finish:

```bash
bash scripts/resume.sh --dry-run    # first see what it plans to do (safe, doesn't touch the machine)
bash scripts/resume.sh              # run the remainder after you confirm
```

It also first checks whether other processes are running, to avoid two pipelines fighting over memory.

Checking results in the morning:

```bash
tail -100 logs/night_run.log        # the whole run and each step's duration
python scripts/31_report.py --open  # open the visual head-to-head report
```

### Run the Probe First, Then the Full Run

The most expensive way to fail on CPU is to "run for three hours and then blow up memory." So both training scripts have a probe mode:

```bash
python 11_train_distill.py --probe     # run 3 steps, report per-step time / peak memory / full-run ETA
python 20_train_rl.py --probe          # run 1 step, report the same information
```

**Recommended cadence** (also written this way in the plan): run Stage 0 first → then take Branch A through end-to-end at the smallest scale
(generate data with `--limit 20`, trial-train with `--epochs 1 --limit 40`), confirm the whole pipeline works,
then open up to full scale, and then replicate the experience to Branch B. **On the first run, get it working first, then get it good.**

### The Reality of Long Jobs

There is a lesson repeatedly learned in this machine's context: **background processes get reaped when a session ends or the mode switches**.
Therefore:

- Every long-job script supports **resumable execution** (`10_gen_distill_data.py`, `30_evaluate.py`
  write each result to disk immediately, and re-runs skip completed items automatically)
- If it breaks midway, just re-run the same command; no work is wasted
- Before running, plug in and turn off background burdens like Spotlight indexing

---

## 5. Evaluation Protocol

**Six models run the same `shared/eval_set.jsonl`**, judged along five dimensions:

| Column | Who it is | Source | Question it answers |
|---|---|---|---|
| `base` | original Qwen3.5-2B (HF weights) | local | the starting point |
| `A` | 2B + distillation LoRA | local | how much distillation gained |
| `B` | 2B + RL LoRA | local | how much RL gained |
| `dsv4pro` | DeepSeek V4 Pro | **cloud API** | **how far we still are from a truly strong model** |
| `qwen4b` | official `qwen3.5:4b` | Ollama | **can the fine-tuned 2B catch up with the original 4B** |
| `teacher` | official `qwen3.5:9b` | Ollama | the ceiling |

> The former `qwen2b` (the official `qwen3.5:2b` quantized version) **has been removed**: it and `base` are two precisions of the same model,
> and the difference reflects only quantization loss, not capability, so keeping it in the table only interferes with reading it.
> After removing it, a **strong cloud-API model** was added to fill the slot — of the first 6 roles, 5 are from the Qwen family and 3 are quantized versions,
> and they alone cannot answer "how far we still are from a truly strong model."

Each model also runs **two thinking conditions**: off (the primary run, full protocol) and on (a small-sample control probe).
With thinking on, the output grows several times longer, and at 1.32 tok/s locally it can't finish without limits, so the control is only a trend reference.
Before grading, the **reasoning block must be stripped first** — the model also writes format markers inside its reasoning, and not stripping them would count its "thinking out loud" as a formal answer.
(Local/Ollama goes through `<think>` tag parsing; the cloud API's reasoning content is a **separate content block**, selected by block type.)

⚠️ **The token budget differs by role, so the `think=on` column cannot be compared head-to-head**:
local roles use 192 / 384, cloud roles use 1024. The 192 cap was set because "one CPU-only response takes 40–90 seconds";
a cloud API takes only a few seconds, and imposing the local hardware limit on the cloud baseline would produce "artificially low scores from truncation."

Here, `base / A / B` are **the same 2B weights + different LoRA adapters**, switching between three roles via PEFT multi-adapter swaps
(saving the memory of two extra sets of weights); `qwen4b` / `teacher` go through the local Ollama; `dsv4pro` goes over the network.
**The key is written into the config only as an environment variable name (`api_key_env`); no key is ever stored in any file.**

| Metric | What it measures | Who cares most |
|---|---|---|
| Format compliance rate | all three of 【思路】/【步骤】/【答案】 present and in the right order | RL |
| Task accuracy | whether 【答案】 is equivalent to the programmatically generated ground truth | both branches |
| Agreement with teacher | similarity of the output to the 9B teacher | distillation |
| Output diversity | whether it has collapsed (outputs converging) | a priority for RL |
| General-capability retention | chit-chat / common-sense quizzes, watching for catastrophic forgetting | must-test for both |

**Protocol details**:
- The primary run uses **temperature sampling (T=0.7) × 3 repeats per problem**, reporting "mean ± standard deviation."
  **Why not greedy (T=0)**: it is deterministic, so repeated runs give exactly the same result and averaging is meaningless —
  if you want to reduce error by "averaging over multiple runs," you must use random sampling. The same batch of 3 samples is **also used** to compute output diversity,
  so no separate round is needed.
- Answer comparison is normalized: full-width to half-width, stripping LaTeX decoration (`\boxed{}`/`\frac`), stripping units,
  and equivalent comparison of fractions / decimals / mixed numbers.
- **Format compliance rate and answer accuracy are two independent metrics**: the grader has a layer of "fallback answer extraction"
  so that official models that never learned this format (writing "the answer is 2" instead of `【答案】2`) are still graded correctly —
  otherwise "not following the format" would be miscounted as "answering wrong," producing a **directionally wrong conclusion** like "the fine-tuned small model crushes the official model."
- Training and evaluation generation are unified on `float32` (on this machine fp32 beats bf16 in both training and generation;
  for the reasoning, see [`docs/Pitfalls Log.md`](docs/踩坑记录.md), Pitfall 9).

### A Measured Finding You Must Know: the teacher itself gets answers wrong

The official `qwen3.5:9b` measured an **error rate of about 20%** on this arithmetic set (verified by hand to confirm it's the model computing wrong, not the grader misjudging).

So distillation data enables **verification screening** by default (`distill.filter_correct = true`) —
using the same grader to remove wrong samples, otherwise the student is essentially learning wrong solutions.

**This keeps the two lines fair**: Branch B's reward uses the same verifier, and both branches rely on the premise that "problems are verifiable,"
so neither gets an extra advantage. To compare against "naive distillation without filtering," simply set the switch to `false` and re-run.

### Controlled Variables

Both branches are aligned to **the same number of optimization steps (60 steps)**. The reason is that the per-step cost of the two is worlds apart
(measured: one SFT step is **127.5 seconds**; one GRPO step is about **320 seconds**, since it runs 4 full generations).
If aligned by wall-clock time, in the same 5.5 hours SFT could run about 154 steps while GRPO would run only 60 —
which would become a comparison of budgets rather than of methods.

Two accompanying conventions that must be followed:
- **Always add `--fresh` to formal training**. Otherwise it inherits the adapter left by the probe, effectively becoming "66 steps vs 60 steps,"
  and the controlled variable silently fails (we've hit this: `计划优化步数 28` in the log is evidence of misalignment).
- **`num_epochs` must be large enough**, otherwise `max_steps=60` can't be reached: 110 samples ÷ accum 8 = 14 steps/epoch,
  so 2 epochs give only 28 steps before the data runs out and truncates it.

The report is rendered by `31_report.py` into **self-contained HTML** (inline SVG, no CDN dependency, viewable offline and directly printable).

---

## 6. What We Explicitly Don't Do

- ❌ **Don't train the 9B** — memory is physically insufficient, in any form
- ❌ **Don't do full-parameter fine-tuning** — only LoRA
- ❌ **Don't do logit-level distillation** — keeping teacher/student resident at once needs about 23 GB, beyond physical memory; we go sequence-level
- ❌ **Don't chase high benchmark scores** — a 2B has no discriminative power on large benchmarks; a self-built test set is more effective
- ❌ **Don't use closed-source API output as training data** — this is where the real line of legality lies
- ❌ **Don't expect "fine-tuning to turn a 2B into a 4B"** — fine-tuning doesn't add parameters. LoRA only adds 0.9% (16.8M)
  of side-attached parameters, and after merging it's still 2.21B. To obtain a 4B artifact you must base it on a 4B,
  and the 4B's fp32 weights alone need 17.5 GB > the machine's 16 GB (see `docs/踩坑记录.md` for details)
