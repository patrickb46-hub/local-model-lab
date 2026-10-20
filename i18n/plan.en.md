# Local Model Lab · The Full Plan

> **One base, two branches**: the same Qwen3.5-2B, copied into two independent copies — one takes distillation, the other takes reinforcement learning.
>
> Date drawn up: 2026-10-04 | Machine: 2018 MacBook Pro (Intel i7-8850H / 16 GB)

---

## 0. One-Sentence Goal

Take the local **Qwen3.5-2B** as the common origin and copy it into two working copies that do not interfere with each other:

- **Branch A (distillation)**: have the 2B learn from the local **Qwen3.5-9B (can be downgraded to 4B)**
- **Branch B (reinforcement learning)**: use **GRPO** to train some capability of the 2B up

Finally, compare the four models (base / distilled output / RL output / teacher) head-to-head on the **same set of problems**, to see which path actually works and where each is strong.

The legality precondition is confirmed: the entire Qwen3.5 line is Apache 2.0, which explicitly permits fine-tuning, quantization, distillation, commercial use, and closed-source derivatives. When releasing a derivative model, just attach the license and NOTICE along with the weights.

---

## 1. Hardware and Constraints (Measured)

| Item          | Measured Result                    | Impact on the Plan                           |
| ----------- | ------------------------------- | -------------------------------- |
| Architecture          | x86_64 (Intel)                   | **No PyTorch MPS acceleration**; CPU-only |
| Memory          | 16 GB                           | The OS takes 4–6 GB first, leaving about 10 GB for training          |
| CPU         | i7-8850H, 12 threads                  | Limited training compute; scale must be kept small                    |
| Free disk        | 123 GB                          | Enough, but watch for checkpoint pile-up            |
| HuggingFace | ❌ direct connection times out after 8 s                      | Switch to **hf-mirror.com**             |
| ModelScope  | ✅ reachable (302)                        | Backup model source                            |
| PyPI        | ✅ reachable                             | Dependency install works                           |
| Local models available      | qwen3.5:2b / 4b / 9b, gemma4:26b | teacher is ready to use                     |

**Three hard rules derived from the constraints:**

1. **Parameter cap: 2B.** The practical ceiling for CPU-only training on this machine is about ≤3B; 4B is an option only if you are willing to wait.
2. **No training of the 9B.** The 9B can only take part in a "talk-only" role (generating data, acting as judge); as soon as gradients must be stored, memory will blow up.
3. **No logit-level distillation.** Loading a 9B teacher and a 2B student at the same time takes about 23 GB, beyond physical memory. Use **sequence-level distillation**.

---

## 2. Architecture Overview

```
                          ┌──────────────────────────────┐
                          │  Qwen3.5-2B (original base)  │
                          │  from hf-mirror, read-only   │
                          └───────────────┬──────────────┘
                                          │ make two copies
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
      ┌──────────────────────────────┐        ┌──────────────────────────────┐
      │   Branch A · Distillation    │        │        Branch B · RL         │
      │       branch-A-distill       │        │         branch-B-rl          │
      ├──────────────────────────────┤        ├──────────────────────────────┤
      │   teacher: 9B/4B generates   │        │      verifiable reward       │
      │              ↓               │        │              ↓               │
      │  clean → train / test split  │        │ problem bank + GRPO rollout  │
      │              ↓               │        │              ↓               │
      │     LoRA trains student      │        │      LoRA trains policy      │
      └───────────────┬──────────────┘        └───────────────┬──────────────┘
                      ▼                                       ▼
                      adapter-A                              adapter-B
                      └───────────────────┬───────────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │   head-to-head (same set)    │
                          │    base / A / B / teacher    │
                          └───────────────┬──────────────┘
                                          ▼
                          ┌──────────────────────────────┐
                          │     merge adapter → GGUF     │
                          │     → import into Ollama     │
                          │        for daily use         │
                          └──────────────────────────────┘
```

**Key design point**: the two branches are **physically isolated** from start to finish — their own model copies, datasets, output directories, and logs never cross. This way, if one branch crashes, it does not contaminate the other, and there is no cross-talk during evaluation.

---

## 3. Technology Choices

| Step      | Choice                                                | Reason                           |
| ------- | ------------------------------------------------- | ---------------------------- |
| base model | `Qwen/Qwen3.5-2B` (HF format)                          | Sweet spot for local training; the GGUF in Ollama cannot be trained directly |
| teacher | local Ollama `qwen3.5:9b`                            | Highest quality; switch to `qwen3.5:4b` if too slow      |
| Training framework    | HF Transformers + PEFT + TRL                      | Ecosystem standard, runs on CPU                  |
| Fine-tuning method    | **LoRA** (fall back to QLoRA if needed)                             | Trains only 0.1% of parameters, memory-friendly              |
| Distillation type    | **sequence-level** (teacher produces data → student learns)                  | logit-level needs both models resident, not enough memory          |
| RL algorithm   | **GRPO**                                          | No value model needed; far less memory than PPO |
| Evaluation      | small self-built test set + automatic metrics                                     | CPU cannot handle a large benchmark           |
| Environment isolation    | separate venv (`~/.workbuddy/binaries/python/envs/mlab`) | Does not pollute other projects' dependencies                   |

---

## 4. Directory Structure

```
~/Desktop/AI/Local-Model-Lab/
├── 完整计划.md                 ← this file
├── README.md                   ← quick-start guide
├── configs/
│   └── lab.config.json         ← global config (sizes, paths, hyperparameters)
├── base/                       ← original base (read-only, never modified after download)
├── branch-A-distill/           ← Branch A: distillation
│   ├── model/                  ← working copy (copied from base)
│   ├── data/                   ← teacher-generated data + cleaned results
│   ├── adapter/                ← training output (LoRA weights)
│   └── logs/                   ← training logs, evaluation reports
├── branch-B-rl/                ← Branch B: reinforcement learning
│   ├── model/                  ← working copy (copied from base)
│   ├── data/                   ← problem bank + reward-related data
│   ├── adapter/                ← training output
│   └── logs/
├── scripts/                    ← all executable scripts
│   ├── 00_check_env.py         ← environment self-check
│   ├── 01_setup_env.sh         ← one-time dependency install
│   ├── 02_download_base.py     ← download base weights
│   ├── 03_make_copies.sh       ← make two working copies
│   ├── 10_gen_distill_data.py  ← generate data with the teacher
│   ├── 11_train_distill.py     ← distillation training
│   ├── 20_train_rl.py          ← GRPO training
│   ├── 30_evaluate.py          ← four-model head-to-head
│   └── 40_export_ollama.sh     ← merge and export to Ollama
├── shared/
│   ├── eval_set.jsonl          ← frozen evaluation set (locked before training, never enters training data)
│   └── results/                ← evaluation comparison table output
└── docs/                       ← technical notes, pitfalls log
```

---

## 5. Stage Plan

### Stage 0 · Environment Ready

| Item      | Content                                                                                                               |
| ------ | ---------------------------------------------------------------------------------------------------------------- |
| **Goal** | A machine that can run the pipeline                                                                                                         |
| **Tasks** | Create a separate venv; install CPU PyTorch and transformers / peft / trl / datasets / accelerate; download Qwen3.5-2B from hf-mirror; copy it into two working copies |
| **Output** | `base/` and two `model/` copies; a working Python environment                                                                            |
| **Acceptance** | `python -c "import torch, transformers, peft, trl"` with no errors; weights in both model directories complete; `00_check_env.py` all green                     |

### Stage 1 · Get the Distillation Branch Working

| Item      | Content                                                                                             |
| ------ | ---------------------------------------------------------------------------------------------- |
| **Goal** | Let the 2B learn some capability from the teacher                                                                          |
| **Tasks** | ① Decide "what to learn" (see the open question in Section 11); ② design the prompt problem bank; ③ call the Ollama teacher to generate responses in bulk; ④ clean, deduplicate, split into train/test sets; ⑤ LoRA training; ⑥ evaluation |
| **Output** | `adapter-A`; a before/after distillation comparison table                                                                            |
| **Acceptance** | training loss decreases normally (no oscillation, no blow-up); the resulting model is **no worse than base** on the target test set                                                 |

### Stage 2 · Get the Reinforcement Learning Branch Working

| Item      | Content                                                              |
| ------ | --------------------------------------------------------------- |
| **Goal** | Train some behavior of the 2B with a verifiable reward                                             |
| **Tasks** | ① Define the reward function (see the open question); ② build the prompt problem bank; ③ configure GRPO (small batch, few samples); ④ train; ⑤ evaluate |
| **Output** | `adapter-B`; reward curve + before/after comparison table                                        |
| **Acceptance** | the reward curve trends upward overall; the target metric improves over base; **output diversity does not collapse**                           |

### Stage 3 · Head-to-Head Evaluation

| Item      | Content                                              |
| ------ | ----------------------------------------------- |
| **Goal** | Answer "which path is better"                                       |
| **Tasks** | Four models (base / A / B / teacher) run the same frozen set of problems, and results are tabulated per the metrics in Section 7 |
| **Output** | One four-column comparison table + conclusion                                    |
| **Acceptance** | It is possible to state quantitatively "how much each branch improved over base, and at what cost"                  |

### Stage 4 · Export and Deployment

| Item      | Content                                             |
| ------ | ---------------------------------------------- |
| **Goal** | The trained model is usable day to day                                     |
| **Tasks** | Merge LoRA into the base → convert to GGUF → write a Modelfile → import into Ollama |
| **Output** | Two models appear in Ollama: `mylab-distill` / `mylab-rl`   |
| **Acceptance** | `ollama run mylab-rl` holds a normal conversation and behaves as trained           |

---

## 6. Key Parameter Decisions

All parameters are centralized in `configs/lab.config.json`; change one place and it takes effect globally.

| Parameter                      | Default          | Range                        | Notes                          |
| ----------------------- | ------------ | --------------------------- | --------------------------- |
| `base_size`             | `2B`         | `2B` / `4B`                 | 4B is noticeably slower; switch only if willing to wait          |
| `teacher_model`         | `qwen3.5:9b` | `qwen3.5:9b` / `qwen3.5:4b` | 9B is high quality but slow to generate; try a small batch first           |
| `finetune_method`       | `lora`       | `lora` / `qlora`            | QLoRA saves more memory and is slower               |
| `lora_rank`             | `16`         | `8`–`32`                    | Larger means more expressive capacity and more memory                |
| `max_seq_length`        | `512`        | `256`–`1024`                | The main switch for GPU/system memory                  |
| `per_device_batch_size` | `1`          | `1`–`2`                     | Basically fixed at 1 on CPU                 |
| `gradient_accumulation` | `8`          | `4`–`16`                    | Combines small batches into an equivalent large batch        |
| `learning_rate`         | `1e-4`       | `5e-5`–`2e-4`               | The usual range for LoRA                   |
| `num_epochs`            | `3`          | `1`–`5`                     | Within 5 epochs when data is small                  |
| `num_generations` (GRPO) | `4`          | `4`–`8`                     | How many answers to sample per prompt; directly determines RL runtime |
| `dataset_size`          | `200`        | `50`–`2000`                 | **Start with 200 to test the waters**; scale up once it works        |

---

## 7. Evaluation Plan

### 7.1 Control Design

| Test subject            | Role  | What to watch                      |
| --------------- | --- | ------------------------ |
| **base** (2B original) | Floor  | The reference for all deltas. If distillation ends up worse than it, the work was wasted |
| **adapter-A**   | Under test  | Difference from base = the gain from distillation       |
| **adapter-B**   | Under test  | Difference from base = the gain from RL      |
| **teacher** (9B) | Ceiling | How far the student got            |

**Discipline**: `shared/eval_set.jsonl` is locked before training begins, and **never enters the training data**. Break this and all later conclusions lose their meaning.

### 7.2 Metrics

| Metric            | What it measures       | How it is computed             | Who watches it most  |
| ------------- | --------- | ---------------- | ----- |
| Format compliance         | Whether the output can be parsed by a program | parseable ÷ total        | RL    |
| Task accuracy         | Whether the answer is correct      | correct ÷ total         | both  |
| Teacher consistency | How well it imitates     | output similarity            | distillation    |
| Output diversity         | Whether training collapsed     | difference across outputs for different prompts | RL focus |
| General-ability retention        | Whether it got dumber     | chit-chat + common-sense quiz        | both, mandatory |

### 7.3 Discipline

- **Keep the test set small**: 20–50 items are enough to see a trend; an evaluation that cannot finish on CPU is no evaluation at all
- **Fixed seed + multiple samples**: run the same prompt 3 times; look at the stable hit rate, not luck
- **Look at process metrics separately**: distillation looks at "how close it gets to the teacher", RL looks at "the reward curve"; the two have different sources, so do not compare them in the same table
- **Compare side by side only on the downstream test problems**

---

## 8. Compute Magnitude Reference

The following are **estimates**; actual values depend on the machine state at the time (background processes, temperature, and disk I/O all matter):

| Step                 | Magnitude         | Notes                   |
| ------------------ | ---------- | -------------------- |
| Downloading 2B weights           | tens of minutes       | about 4–5 GB, via hf-mirror |
| teacher generating 200 items | 1–3 hours (9B) | depends on response length; switching to 4B is noticeably faster    |
| Distillation training (200 items × 3 epochs)  | 2–4 hours     | about 10–30 s per step          |
| GRPO training (about 20 steps)    | 1–3 hours     | rollout dominates          |
| Four-model head-to-head (50 problems)        | tens of minutes       | all four models must generate per problem           |

**Practical advice**: long tasks **must be started in the background** (on this machine `nohup ... &` gets reclaimed when the session ends), and **before running, plug in power and turn off background burdens such as Spotlight indexing** — both points come from hard experience.

---

## 9. Risks and Fallbacks

| Risk        | Symptom              | Fallback                                                         |
| --------- | --------------- | ---------------------------------------------------------- |
| Out of memory      | OOM mid-training / system freeze | Lower `max_seq_length` to 256; enable gradient checkpointing; switch to QLoRA |
| Training too slow      | tens of seconds to minutes per step       | Reduce data and epochs; **watch the trend first, do not chase completion**                                |
| Worse than base | evaluation score below baseline        | This is a common outcome, not a failure — usually a data-quality or learning-rate issue; retune and rerun                             |
| Dependencies will not install     | pip errors          | Switch to the Tsinghua mirror / Aliyun mirror                                                 |
| Model download fails    | hf-mirror drops the stream    | Switch to ModelScope                                               |
| RL collapses the model  | outputs all identical          | Lower the learning rate, add a KL constraint, reduce training steps                                       |
| Catastrophic forgetting     | general ability drops          | Mix in a little general data; reduce training intensity                                            |

**The last resort**: if CPU-only truly cannot keep up, rent a cloud GPU (a 4090 is about ¥2/hour) to run only the training step, while data preparation and evaluation stay local. This is not enabled by default; it is a safety net.

---

## 10. Explicitly Out of Scope

Draw the boundary clearly to avoid drifting off course:

- ❌ **No training of the 9B** — physically not enough memory, in any form
- ❌ **No full-parameter fine-tuning** — LoRA only
- ❌ **No logit-level distillation** — the teacher/student cannot both reside in memory
- ❌ **No chasing high benchmark scores** — a 2B has no discriminative power on a large benchmark; a self-built test set is more effective
- ❌ **Do not use output from closed-source APIs as training data** — the real red line for legality is here, not on the open-source model side

---

## 11. Two Open Questions (Need Your Decision)

Every technical step in the plan is settled; only these two "what to learn" questions remain, and they determine what the data looks like:

### Open Question 1: What should the 2B learn on the distillation branch?

| Candidate          | Notes                 | Difficulty            |
| ----------- | ------------------ | ------------- |
| A1 · General conversation quality | Have the 2B broadly imitate the 9B's conversational level | Medium, large data volume        |
| A2 · Knowledge of a specific domain | e.g., aviation / some topic you know well    | Low, small data volume, quick results    |
| A3 · A specific output format | Teach the 2B to organize answers in a fixed structure   | Low, comparable with the RL branch |

### Open Question 2: What capability should the RL branch train?

| Candidate          | Notes               | Reward verifiability   |
| ----------- | ---------------- | -------- |
| B1 · Format compliance   | Strictly output JSON / a fixed structure | High, judged directly by a program  |
| B2 · Step-by-step reasoning habit | List steps before giving the conclusion         | Medium, can be checked by rules |
| B3 · Small math problems   | The most classic RL scenario       | High, just check the answer  |
| B4 · Domain output convention | Explain knowledge in a domain in a specified format    | High        |

**Suggested combination**: if you have no idea yet, **A3 + B1 is a low-risk starting plan** — both branches center on "output format", with a consistent goal and strong comparability, and it also validates whether your evaluation framework can tell the two branches apart.

---

## Appendix: Execution Order at a Glance

```
Stage 0  Environment ready ──────► Stage 1  Distillation branch ──┐
                                                                  ├─► Stage 3  Head-to-head ─► Stage 4  Export
                        Stage 2  RL branch ───────────────────────┘

(Stages 1 and 2 can run in parallel, or one can be finished before starting the other)
```

**Recommended pace**: first run Stage 0 → the minimum scale of Stage 1 (200 items, 3 epochs), confirm the whole pipeline works, then carry the lessons over to Stage 2. Do not start both branches at full scale at once — on the first run, aim for working first, good second.
