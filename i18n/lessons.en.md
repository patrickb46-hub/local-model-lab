# Key Lessons Learned

> This document is **not a list of pitfalls** (that is the job of [`踩坑记录.md`](踩坑记录.md)),
> but the **principles distilled from those 24 pitfalls that can be carried to the next project**.
>
> Project background: on a 2018 Intel MacBook Pro (no discrete GPU), a controlled experiment comparing
> distillation and reinforcement learning for Qwen3.5-2B, entirely on CPU. Two full rounds of experiments, about 40 hours of machine time.

---

## 0. Three meta-rules

The 24 pitfalls can mostly be reduced to these three. **Remembering these three is more useful than remembering 24 details.**

### Meta-rule 1 · Silent errors are far more dangerous than crashes

A crash leaves a stack trace; a silent error leaves only **garbage data that looks perfectly normal**.

The three most expensive incidents in this project are all of this kind:

| Incident | Surface symptom | Actual consequence |
|---|---|---|
| Ollama died, 145 failed calls recorded as "wrong answers" | Not a single error line in the log | The entire tiered benchmark was invalidated |
| `[:20]` truncation met tier-based chunked writes | Reports still printed, charts still drawn | Only one tier remained in the evaluation set; the core question could not be answered |
| The daemon's criterion was hard-coded | The daemon "quietly did nothing" | 3.5 hours of the catch-up run went on with no protection |

**A transferable check**:
for any long task spanning several hours, ask at design time —
**"If this step fails silently, how long until I notice? And how would I notice?"**
If the answer is "hours later, when I look at the results, and only because the numbers contradict each other," then protection is mandatory.

### Meta-rule 2 · Hard-coded assumptions break when the system grows more complex

- "The evaluation set is just those 20 problems" → after switching to a tiered benchmark, the meaning of `[:20]` changed completely
- "The main pipeline is script 97" → after adding catch-up script 98, the daemon immediately failed

**Both failures happened at the moment of "adding something new while an old piece of code held a hard-coded assumption about the old thing."**
When adding a new component, **go back and check every place that "assumes there is only one"**:
only one script, only one log, only one format, only one output directory.

### Meta-rule 3 · Tools are more reliable than memory

The same mistake (a `$VAR` immediately followed by a full-width punctuation character causing unbound) **was made 7 times**.

Attempts 1–6 all relied on "remember to be careful next time"; on the 7th, **`lint_shell.sh` caught it**.
Don't rely on willpower for something a five-line script can prevent forever.

**A transferable check**: the moment you catch yourself making the same mistake a second time, **write a check script immediately**,
rather than telling yourself "be careful next time."

---

## 1. Experimental design

### 1.1 Controlled variables must be "locked down," not "remembered"

The two branches must be aligned to the same number of optimization steps (the first 60 steps / the last 40 steps) — this constraint is written into the config
and read and displayed by `30_evaluate.py`, not left to a person to remember.

**Moreover: once a variable is adjusted for any reason, all arms must be adjusted in sync.**
When the budget forced the step count down from 60 to 40 in round 2,
**arms A1, B, and C were all lowered to 40 together** — the absolute level changed, but the **controlled relationship did not**, so the conclusions still hold.
If only one arm had been lowered, the entire comparison would be ruined.

### 1.2 Leak prevention must operate at the content level

As soon as the evaluation set shares problems with the training data, the premise that "the model has not seen them" is broken — and all conclusions are invalidated.

The check at the time deduplicated by **id**, but measurement showed that of the 300 problems in the candidate pool,
**12 had problem text exactly identical to problems in the training pool** (the generator only deduplicated within the pool; the id prefixes merely differed).

**Lesson**: **leak-prevention checks must operate on content, not on identifiers.**
The id is a number tag we assign ourselves; **the problem content is what the model actually sees.**

Ultimately three layers of protection were added: deduplicate at the source → filter again when reading the pool → upgrade the assertion checks to the problem-text level.

### 1.3 A gate must be able to "stop people," not just "shout"

The round-2 gate (requiring L2 ≥ 24 problems) was judged **not passed**,
but the script's exit code was 0, so the pipeline ran on regardless — **the gate amounted to nothing but a print statement**.

**This is a design flaw, not an execution error.** The correct approach is to make the gate exit non-zero.

**But the other half must also be acknowledged**: when the gate is already too late to stop anything (the training money has already been spent),
**the correct action is to "accept the fact + report faithfully," not to pretend it passed.**
In this round we stated it plainly — "L2 has only 16 problems, the paired test's sample size is on the small side,
and the conclusions are stated as directional evidence" — which is far more valuable than glossing over it with a pretty number.

### 1.4 Sample size is a design problem; it cannot be patched after the fact

The L2 tier has only 16 problems (24 required). This is not something "running more samples" can fix —
**repeated sampling only reduces sampling noise, whereas an insufficient sample size reduces representativeness at the problem level.**

The estimate at the time: "a 15% hit rate in a 40-problem pilot × a 288-problem pool ≈ 43 problems" —
it looked sufficient, but after expanding to 140 problems the hit rate was only 11.4%: **the extrapolation was too optimistic.**

**Lesson**: when estimating a hit rate from a small pilot, **give an interval based on the pilot's own uncertainty**,
rather than multiplying the point estimate directly.

---

## 2. Engineering implementation

### 2.1 For any "take the first N items," first ask: does the order carry meaning?

This is the positive phrasing of pitfall 21. Taking the first N items of chunk-written data (by tier, by category, by batch)
is equivalent to **silently taking just one chunk**.

**A transferable check**: when writing `[:-N]` or `[:N]`, pause for three seconds and ask:
"How is this list sorted? Does the order carry information?"
If it does, you should use **tiered round-robin sampling** (the fix finally used in this project).

### 2.2 Failures must be loud (fail loudly)

**Any default that "treats a missing result as a negative result" is a silent source of data contamination.**

Concrete practices (all implemented in the project):

- Pre-flight liveness check (ping the dependency services before starting; if unreachable, exit immediately)
- Per-item retry + immediate abort on consecutive failures
- **Failed samples are not written to disk**, left for checkpoint-resume retry rather than leaving an empty record
- For every `None` / `0` / `""` in the grader, first ask: **is this really a valid measurement?**

### 2.3 Checkpoint-resume must be designed into the **data format**

This project's checkpoint-resume is not "keeping a progress file," but **giving each sample its own unique key**
`(role, problem number, kind, sample index)`; being written to disk means it is done, and on restart it is skipped by key.

Benefit: **pause/resume, crash restart, and adding or removing problems are all automatically correct**.
Measured: after pausing for 1 minute and resuming, the 8 completed problems were skipped automatically, with 654 records showing zero duplicates and zero corruption.

**The cost is nearly zero**, and the benefit is "you can safely stop at any time." **This is the most cost-effective design decision in this project.**

### 2.4 The criterion for a daemon/monitor process must not be hard-coded

See meta-rule 2. One additional practical note: **a daemon's exit condition should be written as
"none of all the possible main pipelines are present,"** not "a specific main pipeline is not present."

---

## 3. Evaluation methodology

### 3.1 The grading standard must be single and reproducible

- All roles use the **same** `parse_response` / `answer_equal`
- Screening sampling and evaluation sampling use the **same temperature** (both take `eval.sample_temperature`),
  so screening results can be reused directly by evaluation — saving about 5 hours, and **this is not a "compromise," it is equivalent data**

### 3.2 Tiers must be split by "capability boundary," not by "difficulty"

The most valuable design in this project is the **three-tier definition of L1/L2/L3**:

| Tier | Definition | What it answers |
|---|---|---|
| L1 | teacher **can do it** | Baseline capability |
| **L2** | **teacher systematically fails, but base has a chance** | **The core proposition: can RL surpass the teacher** |
| L3 | both teacher and base **get it entirely wrong** | The limit: can RL learn a brand-new capability |

The key is L2's definition — it is not a "hard problem," but rather **"the teacher can't do it, but the student might work it out on its own."**
**Only such problems can distinguish "imitation" from "surpassing."**
Splitting by ordinary difficulty cannot make this distinction.

### 3.3 Before comparing across roles, align the budget first

The cloud API (DeepSeek V4 Pro) uses a 1024-token budget, while the local roles use 192 — this is **deliberate**:
192 is the ceiling set for "one CPU-only item takes 40–90 seconds"; a cloud API item takes a few seconds,
and applying the same ceiling would produce "spurious low scores caused by truncation."

**Consequence**: the `think=on` column **cannot be compared across roles**
(the local roles' 192 tokens were entirely consumed by thinking, so not a single character of body text was produced, recorded as 0.0%;
the cloud has 1024 and scores 100%). This column was marked in bold in the report.

**Lesson**: when you see a large cross-role gap, **first ask whether the budget/standard is unequal**,
and only then whether it is a capability gap.

---

## 4. Long tasks and pipelines

### 4.1 Run a probe first, then go full-scale

**The most expensive way to fail on CPU is "run for three hours and then blow up the memory."**
All long pipelines first run a small-scale probe (`--probe` / `--limit`) to get measured data, then scale up.

### 4.2 A long task must be able to stop, resume, and be checked

All three are indispensable:

| Capability | Implementation in this project |
|---|---|
| **Can be checked** | `progress.sh` — the single entry point, the whole picture on one screen (including a per-tier progress table) |
| **Can resume** | data carries its own unique key; written to disk means done (see 2.3) |
| **Can stop** | `labctl.sh pause/resume` — measured lossless, freeing 12 GB of memory |

**One more piece of discipline**: after pausing, ensure that **automatic inspection does not bring it back up on its own**
(this project implements it with a `logs/PAUSED` marker). Otherwise "pause" is an illusion —
the user has just freed up the machine, and two hours later it is filled up again by an automated task.

### 4.3 Choosing the wrong background method wastes the run

- `nohup ... &` **gets reclaimed by the session** in a tool-managed environment (this project hit it twice,
  one of which died only at the 4th step, wasting about 2 hours)
- The correct approach: **use the tool-managed background**; or have the user launch it in their own terminal

---

## 5. Documentation and collaboration

### 5.1 After editing, you must self-check — and actually run the tools

| What you changed | What to run |
|---|---|
| shell scripts | `lint_shell.sh` (syntax + the unbound hazard of a `$VAR` immediately followed by full-width punctuation) |
| Python scripts | `ast.parse` (**do not use `py_compile`**, which writes `__pycache__` and triggers deletion interception) |
| docs | `grep -c '^## '` to count headings |

**A hard-won lesson**: when using a section heading as an editing anchor, if you forget to write it back,
**the heading is silently swallowed** (this mistake was made 5 times).

### 5.2 If unsure, admit it — don't gloss over it

This project has two "unattractive but must-be-said" points, both written faithfully into the report:

1. **GATE-1 not passed** — L2 has only 16 problems; the sample size is insufficient
2. **Local inference is 30× slower than the cloud** — "local deployment is cheaper" does not hold on this machine

**Honestly reporting one counterexample is more valuable than one extra pretty number.**

---

## 6. If this were done again, what would change

| No. | Change | Reason |
|---|---|---|
| 1 | **Make the gate exit non-zero** | Otherwise the gate is purely nominal (pitfall 22) |
| 2 | **Tiered round-robin sampling for the evaluation set, done from the very start** | Otherwise tiers are silently dropped (pitfall 21) |
| 3 | **Write the daemon criterion as "none of the main pipelines are present"** | Otherwise it breaks the moment the script changes (pitfall 23) |
| 4 | **Make the candidate pool 2–3× larger** | L2 has only 16 problems; the paired test's power is insufficient |
| 5 | **Build pause/resume capability in round 1** | Measured, it makes "you can safely stop at any time" possible |
| 6 | **Also record the cloud baseline's token budget as a table** | Avoid cross-role misreading (3.3) |

---

## 7. The three things most worth taking away

If you could take away only three:

1. **The on-disk design where data carries its own unique key** — it makes "you can safely stop at any time" possible,
   at nearly zero cost, with benefits that cover pausing, crashes, and problem changes alike. (2.3)
2. **Tiering by "capability boundary"** — the definition of L2 (the teacher can't do it; the student might work it out on its own)
   is what makes it possible to distinguish "imitation" from "surpassing." (3.2)
3. **The "failures must be loud" discipline** — it comes directly from the most expensive incident,
   and applies to any long task. (2.2)
