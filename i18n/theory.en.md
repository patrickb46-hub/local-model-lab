# How It Works (Technical Notes)

> Notes on a few of the "why we did it this way" technical decisions in this project.
> The point is not to popularize, but to pin down the choices made at the time along with their costs, so that there is something to look back on later.

---

## 1. Why "sequence-level distillation" rather than logit-level

Distillation comes in two tiers:

| | logit-level | sequence-level (this project) |
|---|---|---|
| Supervision signal | the teacher's full probability distribution at every position | the text the teacher ultimately generates |
| Information | far greater (includes "soft labels" and inter-class similarity) | less (only the "optimal path") |
| Memory | teacher + student **both resident** | only student resident |
| Feasible on this machine | 9B + 2B ≈ **23 GB** > 16 GB ✗ | ✓ |

**Verdict**: logit-level is indeed stronger, but on this machine it requires fitting both models into 16 GB at once, which is physically impossible. So we fell back to sequence-level: **the teacher first writes the answer as text, and the student then imitates the text**.

The cost is clear — we lose the "dark knowledge" in the probability distribution. But what we get in return is a pipeline that **actually runs**. Between "it runs" and "theoretically better," 16 GB of memory left us no choice.

---

## 2. Why GRPO saves memory compared with PPO

This is the key that let the RL branch get off the ground.

**PPO** needs three things resident at once:

1. The policy model (policy)
2. The reference model (reference, used to compute KL)
3. **The value model (used to estimate advantage)**

Item 3 is a model of the **same scale** as the policy — another 2B, and memory doubles immediately.

**GRPO's approach**: drop the value model. Sample a group (G) of answers for the same problem, and **use the relative quality within the group** as the baseline:

```
A_i = (r_i − mean(r)) / (std(r) + ε)
```

Answers with high reward are reinforced and low ones suppressed; the baseline is provided by the group's own answers. Eliminating a 2B model is GRPO's only foothold on this machine.

**What about the reference model**: we originally assumed we would need to load another copy of the weights to compute KL. In fact we don't — during LoRA training, **turning off the adapter is the original policy**, so the same weights can be used by toggling the adapter state, with **zero extra memory**:

```python
with model.disable_adapter():        # turn off LoRA = back to the reference policy
    ref_logp = ...
```

KL uses the **k3 estimator** (unbiased, low variance):

```
KL ≈ E[ exp(ref_logp − logp) − (ref_logp − logp) − 1 ]
```

Its advantage is that when the policy and the reference are exactly identical, every term is exactly 0, so there is no k1-style "constant positive bias" that inflates the KL term.

---

## 3. The choice of LoRA and its cost

**Why train only LoRA**: 2.3B parameters in fp32 already means 9.2 GB of weights.
Full-parameter fine-tuning would add gradients (another 9.2 GB) and AdamW's two moment estimates (another 18.4 GB) —
37 GB, completely impossible.

LoRA turns this into: **freeze all original weights and train only two low-rank matrices**.
At `r=16`, trainable parameters are under 0.5% of the total, and the optimizer state is negligible.

**Cost**: limited expressive capacity — it cannot learn "deep representational change," only "behavior-level adjustments." That is enough for tasks like alignment / format / problem-solving patterns; it is not enough to learn a whole new language.

**What is special about this project**: Qwen3.5 uses a hybrid attention architecture (linear attention + one full attention every 4 layers), and its layer names do not follow the usual `q_proj/k_proj/v_proj` convention, so `target_modules` must be **probed at runtime** rather than copied from a tutorial:

```python
suffixes = sorted({name.split(".")[-1] for name, m in model.named_modules()
                   if isinstance(m, nn.Linear)})
```

When probing, layers related to vision and MTP must also be excluded.

---

## 4. The grader: the foundation of the entire evaluation chain

If the grader is unreliable, every number downstream is meaningless. So it deserves its own design.

### 4.1 Answer equivalence, not answer equality

The model may write `3/4`, or it may write `0.75`, `\frac{3}{4}`, `0.75元`, or full-width digits.
So normalization is required before comparison:

1. Full-width → half-width (including `１２３` → `123`, `（）` → `()`)
2. Strip LaTeX decoration: `\boxed{}`, `\text{}`, `$...$`, `\frac{a}{b}` → `a/b`
3. Cut prefixes such as "答案 / 答 / 所以 / 故"
4. Remove the trailing period and whitespace
5. Compare numeric values uniformly with `Fraction` (`3/4 == 0.75` is true); mixed numbers like `1又1/2` must also be recognized
6. Unit suffixes (元 / 厘米 / 度 …) are always stripped before comparison

**If this step is not done solidly, a flood of "false errors" appears** — the model clearly got the answer right,
but because it wrote `0.75` while the reference answer is `3/4` it is marked wrong, and you end up with a pile of wrong conclusions.

### 4.2 Format compliance is a hard verdict

Fixed format:

```
【思路】…
【步骤】
1. …
【答案】…
```

The criterion is **all three segments present and in the correct order** (`idx(思路) < idx(步骤) < idx(答案)`).
It is not "contains them" — wrong order means the model has not truly understood the structure.

### 4.3 Why all rewards are verifiable

```
answer correct     +1.0     ← compared against the program-generated ground truth
format compliant   +0.5     ← all three segments present and ordered
steps complete     +0.2     ← at least one numbered step
answer too long    −0.3     ← penalize writing the answer section as one long rambling paragraph
```

All four can be judged by a program → **no need to train a reward model** → saving another set of weights.
This is the direct dividend of the "programmatic problem generation" decision: problems are generated by a program, so the answers are naturally exact, which is what makes fully verifiable rewards possible.

### 4.4 Why diversity must be measured separately

RL has a typical failure mode: **the reward goes up, but the model learns to follow a template**.
Accuracy rises, yet the outputs are all identical — a regression from "can solve the problem" to "can fill in the format."

So diversity must be **measured separately using multiple high-temperature samples** (sample 3 times for the same problem and take the mean of the pairwise differences),
and cannot be inferred from the greedy results. Greedy is identical every time, so "diversity" looks like 0,
but that is not information — the model must be given a chance to show disagreement.

Likewise, accuracy and format-compliance rate are measured with **greedy (temperature 0)**,
because what is wanted is a "stable hit rate," not "one lucky hit."

---

## 5. Discipline: why the evaluation set must be locked down in advance

`shared/eval_set.jsonl` is generated and frozen **before training begins**, and **never enters the training data**.

This is not fastidiousness. If evaluation problems leak into the training set, the model will memorize the answers,
and the final numbers will look absurdly good — but that is memory, not capability, and the whole head-to-head comparison loses its meaning.

So hard checks were built into the code:

- When the problem bank is generated it is split strictly into three parts (eval / distillation / RL), with an assertion that the id sets of the three are pairwise disjoint
- `10_gen_distill_data.py` checks once more after cleaning whether any problem text collides with eval, and removes it if so
- `00_check_env.py` makes the three disjointness assertions part of the environment self-check

**If this rule breaks, none of the later conclusions are worth reading.** So we would rather let it "get in the way" of the pipeline.

---

## 6. Trade-offs in training hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `per_device_batch_size` | 1 | Essentially fixed on CPU; no room to move |
| `gradient_accumulation` | 8 | Accumulate from a small batch of 1 to an effective batch of 8, reducing gradient noise |
| `max_seq_length` | 512 | **The main memory switch**. Sufficient for small math problems + step-by-step solutions |
| `learning_rate` | 1e-4 (SFT) / 5e-5 (GRPO) | The usual range for LoRA; RL is more conservative to prevent training collapse |
| `lr_schedule` | linear warmup + cosine decay | warmup prevents early oscillation, cosine stabilizes the later stage |
| `gradient_checkpointing` | on | Trade time for memory. On CPU, time is cheap anyway — memory is the bottleneck |
| `lora_rank` | 16 | Larger is more expressive but overfits to format more easily; with only 200 examples, 16 is enough |
| `kl_coef` | 0.02 | Too small and it flies off; too large and it cannot learn. Validate the trend with small steps first |

**Core principle**: on this machine, **memory is a hard constraint and time is a soft constraint**.
Every trade-off leans toward "less memory, a bit slower is fine."

---

## 7. Compute magnitude (estimates, for scheduling reference)

Actual values depend on the machine's state at the time (background processes, temperature, and disk I/O all have an effect):

| Stage | Magnitude |
|---|---|
| Download 2B weights (4.6 GB, hf-mirror) | A dozen or so minutes |
| teacher (9B) generates 200 data items | 1–3 hours |
| Distillation training (200 items × 3 epochs, CPU) | 2–4 hours |
| GRPO training (~30 steps) | 1–3 hours |
| Four-model head-to-head comparison (30 problems + diversity subset) | Tens of minutes |

**The only correct way to start**: first run `--probe` to get measured numbers (time per step / peak memory / ETA),
then decide how much to scale up. Do not go full-scale based on estimates.
