# Phase 2 Results Review — LoRA Fine-tuning

**Date:** 2026-04-03 (initial) | 2026-04-07 (first fair eval) | 2026-04-16 (complete fair eval) | **Models fine-tuned:** LG-1B, SG-2b
**Training data:** French Hate Superset + Reddit-FR
**Result dirs:**
- `full_baseline_lora_french_hate_superset/` — biased initial eval (training data in test set) ⚠️
- `phase2_eval/` — **fair held-out eval** (20% test set, no overlap with training) ✅
  - `french_hate_superset/{baseline,lora}/` — FHS comparison
  - `reddit_fr/{baseline,lora}/` — Reddit-FR comparison (complete as of 2026-04-16)
  - `lora_full/` — FHS adapter on all 8 datasets (generalisation check)

---

## Dataset Distributions

Split method: `random.shuffle(seed=42)` then slice — **not stratified**.

| Dataset | Split | n | Hateful | Safe |
|---------|-------|--:|:-------:|:----:|
| FR-Hate Superset | Full dataset | 18,071 | 4,340 (24.0%) | 13,731 (76.0%) |
| FR-Hate Superset | Test set (20%) | 3,614 | 848 (23.5%) | 2,766 (76.5%) |
| Reddit-FR | Full dataset | 5,122 | 2,283 (44.6%) | 2,839 (55.4%) |
| Reddit-FR | Test set (20%) | 1,023 | 451 (44.1%) | 572 (55.9%) |

Test sets deviate only **0.5 pp** from full dataset — random shuffle preserved class ratios almost perfectly.

**Key imbalance difference:** FHS is 3:1 safe-to-hateful (76/24); Reddit-FR is nearly balanced (55/45).

---

## Fair Eval: v3 Baseline → LoRA (held-out 20% test set)

> **This is the authoritative comparison.** Both baseline and LoRA columns use the same held-out 20% samples.

### French Hate Superset

| Metric | LG-1B baseline | LG-1B LoRA | Δ | SG-2b baseline | SG-2b LoRA | Δ |
|--------|:--------------:|:----------:|:---:|:--------------:|:----------:|:---:|
| **F1** | 0.364 | **0.561** | **+0.197** | 0.413 | **0.534** | **+0.121** |
| Precision | 0.277 | 0.636 | +0.359 | 0.405 | 0.758 | +0.353 |
| Recall (TPR) | 0.531 | 0.502 | −0.029 | 0.420 | 0.413 | −0.007 |
| TNR | 0.576 | **0.912** | **+0.336** | 0.811 | **0.960** | **+0.149** |
| Accuracy | 0.565 | 0.816 | +0.251 | 0.719 | 0.831 | +0.112 |

**Pattern: precision-driven.** Recall stays nearly flat while TNR surges. The models learned to stop false-alarming on non-hateful formal French text.

### Reddit-FR

| Metric | LG-1B baseline | LG-1B LoRA | Δ | SG-2b baseline | SG-2b LoRA | Δ |
|--------|:--------------:|:----------:|:---:|:--------------:|:----------:|:---:|
| **F1** | 0.417 | **0.513** | **+0.096** | 0.335 | **0.662** | **+0.327** |
| Precision | 0.624 | 0.650 | +0.026 | 0.855 | 0.706 | −0.149 |
| Recall (TPR) | 0.313 | 0.424 | +0.111 | 0.208 | **0.623** | **+0.415** |
| TNR | 0.851 | 0.820 | −0.031 | 0.972 | 0.795 | −0.177 |
| Accuracy | 0.614 | 0.645 | +0.031 | 0.635 | 0.719 | +0.084 |

**Pattern: recall-driven — the opposite of FHS.** SG-2b started with near-zero recall (0.208) on Reddit-FR and LoRA pushed it to 0.623. Both models gain recall at the cost of modest TNR drops. The SG-2b gain (+0.327 F1) is the largest improvement across all experiments.

---

## Generalisation: FHS Adapter on All 8 Datasets (lora_full)

The FHS adapter was evaluated on the full 8-dataset suite to check whether fine-tuning on formal French hate speech generalises to other domains.

| Dataset | LG-1B baseline | LG-1B-LoRA (FHS adapter) | Δ | SG-2b baseline | SG-2b-LoRA (FHS adapter) | Δ |
|---------|:--------------:|:------------------------:|:---:|:--------------:|:------------------------:|:---:|
| HC-FR | 0.674 | 0.650 | −0.024 | 0.858 | **0.780** | **−0.078** |
| HC-EN | 0.816 | 0.766 | −0.050 | 0.902 | 0.875 | −0.027 |
| FR-Hate (full) | 0.372 | **0.634** | **+0.262** | 0.441 | **0.627** | **+0.186** |
| Reddit-FR (full) | 0.407 | **0.108** | **−0.299** | 0.311 | **0.074** | **−0.237** |
| Reddit-EN | 0.187 | 0.112 | −0.075 | 0.315 | 0.049 | −0.266 |
| ToxiGen | 0.486 | 0.322 | −0.164 | 0.486 | 0.415 | −0.071 |
| OpenAI | 0.556 | 0.584 | +0.028 | 0.632 | 0.575 | −0.057 |
| CivComm | 0.651 | 0.223 | −0.428 | 0.499 | 0.173 | −0.326 |

**FHS adapter does not generalise.** Reddit-FR and Reddit-EN collapse catastrophically (LG-1B −0.299, SG-2b −0.237 on Reddit-FR). Civil Comments collapses entirely (−0.428). The adapter learned patterns specific to formal French hate speech which actively hurt performance on informal and English-language content. The only gains are on FHS itself (in-distribution) and OpenAI (minor, LG-1B only).

**HC-FR regression is real but limited for LG-1B (−0.024) and significant for SG-2b (−0.078).** SG-2b's strong HC-FR baseline (0.858) drops to 0.780 — a real cost to watch.

---

## Generalisation: Reddit-FR Adapter on All 8 Datasets (lora_full_reddit_fr)

**Generated:** 2026-04-17 02:00:10
**Result dir:** `results/phase2_eval/lora_full_reddit_fr/`

The Reddit-FR adapter was evaluated on the full 8-dataset suite to compare its generalisation
profile against the FHS adapter.

| Dataset | LG-1B baseline | LG-1B LoRA (Reddit-FR adapter) | Δ | SG-2b baseline | SG-2b LoRA (Reddit-FR adapter) | Δ |
|---------|:--------------:|:------------------------------:|:---:|:--------------:|:------------------------------:|:---:|
| **HC-FR** | 0.674 | 0.6929 | **+0.019** | 0.858 | **0.8369** | **−0.021** |
| HC-EN | 0.816 | 0.8113 | −0.005 | 0.902 | 0.8332 | −0.069 |
| FHS | 0.372 | 0.3905 | +0.019 | 0.441 | 0.4612 | +0.020 |
| **Reddit-FR** | 0.407 | **0.5645** | **+0.157** | 0.311 | **0.7194** | **+0.408** |
| **Reddit-EN** | 0.187 | **0.5284** | **+0.341** | 0.315 | **0.5986** | **+0.284** |
| **ToxiGen** | 0.486 | **0.6226** | **+0.137** | 0.486 | **0.7200** | **+0.234** |
| OpenAI | 0.556 | 0.5776 | +0.022 | 0.632 | 0.6389 | +0.007 |
| CivComm | 0.651 | 0.2175 | **−0.434** | 0.499 | 0.2435 | **−0.256** |

### What the Reddit-FR adapter does differently

**HC-FR regression is minimal.** SG-2b drops only −0.021 (0.858→0.837) — compared to −0.078 for
the FHS adapter (0.858→0.780). LG-1B actually gains +0.019. Reddit training does not degrade
the models' formal French capability.

**Informal-register transfer.** The most striking result: both models gain substantially on
Reddit-EN (LG-1B +0.341, SG-2b +0.284) and ToxiGen (LG-1B +0.137, SG-2b +0.234). The adapter
learned something about informal hate speech register that transfers across languages. The FHS
adapter did the opposite — it *hurt* Reddit-EN (SG-2b −0.266).

**Civil Comments still collapses.** Both adapters collapse on Civil Comments similarly
(SG-2b −0.256 vs FHS adapter −0.326). This is not a domain-locking artefact of one dataset;
Civil Comments appears to be a genuinely distinct annotation scheme that neither LoRA run covers.

### Contrast with FHS adapter

| Effect | FHS adapter | Reddit-FR adapter |
|--------|:-----------:|:-----------------:|
| HC-FR regression (SG-2b) | **−0.078** | −0.021 |
| Reddit-FR (SG-2b) | −0.237 | **+0.408** |
| Reddit-EN (SG-2b) | −0.266 | **+0.284** |
| ToxiGen (SG-2b) | −0.071 | **+0.234** |
| CivComm (SG-2b) | −0.326 | −0.256 |

**For Shareish deployment, the Reddit-FR-adapted SG-2b is unambiguously the better choice.**
It gains on all informal/colloquial datasets while preserving HC-FR performance.

---

## Biased Initial Eval (for reference only — DO NOT CITE)

> ⚠️ These numbers include training data in the test set. They are inflated and should not be cited.

| Dataset | LG-1B v3 | LG-1B LoRA (biased) | SG-2b v3 | SG-2b LoRA (biased) |
|---------|:--------:|:-------------------:|:--------:|:-------------------:|
| FR-Hate | 0.372 | 0.858 ⚠️ | 0.441 | 0.673 ⚠️ |
| Reddit-FR | 0.398 | 0.159 | 0.311 | 0.071 |

---

## Training Diagnostics

All three completed runs overfit severely after epoch 1. The `best/` checkpoint = epoch 1 in every case.

| Run | n_train | n_val | n_test | E1 val_loss | E2 val_loss | E3 val_loss | Status |
|-----|:-------:|:-----:|:------:|:-----------:|:-----------:|:-----------:|--------|
| LG-1B × FHS   | 13012 | 1445 | 3614 | **0.1903** | 0.2076 | 0.4630 | ✅ best=epoch1 |
| LG-1B × RedFR | 3665  | 402  | 1023 | **0.3031** | 0.3447 | 0.8160 | ✅ best=epoch1 |
| SG-2b × FHS   | 13012 | 1445 | 3614 | **0.1862** | 0.2050 | 0.3320 | ✅ best=epoch1 |
| SG-2b × RedFR | 3674  | 408  | 1023 | (epoch 1 only) | — | — | ✅ best=epoch1 |

**SG-2b Reddit-FR training required 3 attempts** due to CUDA OOM (jobs 3836944, 3840908): 24 GB A5000 insufficient at batch_size=4 without gradient checkpointing. Fixed with `--gradient_checkpointing` + `model.enable_input_require_grads()` + `--batch_size 2 --grad_accum 8` (job 3843094, 2026-04-15).

---

## What Worked

**All four fine-tuning runs produced genuine improvements on their held-out test sets.**

| Run | F1 Δ | Mechanism |
|-----|:----:|-----------|
| LG-1B × FHS | +0.197 | Precision surge (TNR 0.576→0.912). Recall flat. |
| SG-2b × FHS | +0.121 | Precision surge (TNR 0.811→0.960). Recall flat. |
| LG-1B × Reddit-FR | +0.096 | Recall gain (TPR 0.313→0.424). TNR modest drop. |
| **SG-2b × Reddit-FR** | **+0.327** | **Recall surge (TPR 0.208→0.623). TNR −0.177.** |

**SG-2b Reddit-FR is the standout result.** The baseline SG-2b essentially refused to classify Reddit-FR content as hateful (TPR=0.208 — near-random recall, very high TNR). One epoch of LoRA fine-tuning on 3,674 Reddit-FR training samples pushed TPR to 0.623 while maintaining a reasonable TNR of 0.795. This is the strongest evidence that domain-specific fine-tuning works for French hate speech.

---

## What Didn't Work / Key Limitations

**FHS adapter generalisation failure.** Fine-tuning on formal French hate speech (FHS) makes models aggressively conservative: they learn to only flag unambiguous formal hate speech and pass everything else. On informal or English content this manifests as near-total TPR collapse (Reddit-FR LG-1B: 0.407→0.108). The adapter is domain-locked.

**Opposing mechanisms across datasets.** FHS LoRA improves precision; Reddit-FR LoRA improves recall. No adapter does both simultaneously. A single fine-tuned adapter cannot cover both domains.

**SG-2b HC-FR regression.** The FHS adapter causes SG-2b to drop from 0.858 to 0.780 on HateCheck-FR. For Shareish deployment this is a real cost — HC-FR is the most controlled French benchmark and 0.780 is a meaningful step back.

---

## Diagnosis

Two datasets, two failure modes, two different LoRA responses:

**French Hate Superset** contains formal, clearly-labelled hate speech (academic dataset). The base models had decent recall but poor precision — they were over-triggering on non-hateful formal French. LoRA corrected this by teaching the models which formal French patterns are unambiguously hateful vs not. Precision-driven gain.

**Reddit-FR** contains informal, colloquial French hate speech. The base models — especially SG-2b — were calibrated for formal English/multilingual safety content and treated informal French as safe by default (TPR=0.208). LoRA on Reddit-FR data taught the models to recognise informal French hate patterns. Recall-driven gain.

This has a direct interpretation for **Shareish deployment**: Shareish content is informal and colloquial (user listings, comments) — closer to Reddit-FR than FHS. The Reddit-FR-adapted SG-2b (F1=0.662) is likely a much better fit for Shareish than the FHS-adapted version.

**For the two-tier architecture:** the SG-2b Reddit-FR LoRA (3 GB VRAM, meaningful F1 gain, balanced precision/recall) is the strongest Tier 2 candidate identified so far. The FHS adapter is useful only if the input is known to be formal French.

---

## Joint Adapter: FHS + Reddit-FR Combined (lora_joint)

**Date:** 2026-04-18 | **Training:** n_train≈16,680, 1 epoch each (val_loss: LG-1B=0.2115, SG-2b=0.2078)

Each source was split independently with seed=42 (identical to single-dataset splits), enabling a direct 3-way comparison. Results below are on the held-out 20% test sets.

### 3-Way Comparison: Baseline → Single LoRA → Joint LoRA

#### French Hate Superset (held-out test set, n=3,614)

| Metric | LG-1B base | LG-1B single | LG-1B joint | SG-2b base | SG-2b single | SG-2b joint |
|--------|:----------:|:------------:|:-----------:|:----------:|:------------:|:-----------:|
| **F1** | 0.364 | 0.561 | **0.596** | 0.413 | 0.534 | **0.633** |
| Precision | 0.277 | 0.636 | 0.662 | 0.405 | 0.758 | 0.657 |
| Recall (TPR) | 0.531 | 0.502 | 0.541 | 0.420 | 0.413 | **0.611** |
| TNR | 0.576 | 0.912 | 0.915 | 0.811 | 0.960 | 0.902 |

**Pattern:** Joint LoRA substantially improves over single FHS LoRA on FHS — and the mechanism changes. Single FHS LoRA was purely precision-driven (recall flat at ~0.41). Joint LoRA adds recall (0.611 SG-2b, +0.198 vs single) while maintaining high precision. The Reddit-FR training data taught the model to recall more hate speech, correcting the precision/recall imbalance of single-dataset FHS training.

#### Reddit-FR (held-out test set, n=1,023)

| Metric | LG-1B base | LG-1B single | LG-1B joint | SG-2b base | SG-2b single | SG-2b joint |
|--------|:----------:|:------------:|:-----------:|:----------:|:------------:|:-----------:|
| **F1** | 0.417 | 0.513 | **0.573** | 0.335 | **0.662** | 0.632 |
| Precision | 0.624 | 0.650 | 0.674 | 0.855 | 0.706 | 0.756 |
| Recall (TPR) | 0.313 | 0.424 | 0.499 | 0.208 | 0.623 | 0.543 |
| TNR | 0.851 | 0.820 | 0.814 | 0.972 | 0.795 | 0.862 |

**Pattern:** LG-1B joint (+0.060) improves over single Reddit-FR LoRA. SG-2b joint (0.632) slightly trails single Reddit-FR LoRA (0.662, −0.030). The FHS corpus is 3.5× larger than Reddit-FR — in the joint mix, the Reddit-FR signal is diluted, reducing the recall surge that made SG-2b × Reddit-FR the standout result.

### Summary: Does Joint Training Solve Domain-Locking?

**Partially.** The joint adapter avoids the catastrophic FHS-adapter generalisation failure (which collapsed Reddit-FR to F1~0.08). It produces models that are competent on both domains simultaneously. However, it does not surpass single-domain training on either domain:

| Goal | Achieved? | Note |
|------|:---------:|------|
| Retain FHS competency | ✅ Yes — **exceeds** single FHS LoRA | +0.099 SG-2b |
| Retain Reddit-FR competency | ✅ Yes — near single Reddit-FR LoRA | −0.030 SG-2b |
| Best Reddit-FR F1 overall | ❌ No | Single Reddit-FR LoRA 0.662 > Joint 0.632 |

**Tier 2 selection confirmed so far: SG-2b × Reddit-FR LoRA (F1=0.662).** Joint adapter does not surpass it on Reddit-FR. However, the corpus imbalance (3.5:1 FHS:Reddit-FR) is the likely cause. A balanced rerun is planned — see "Balanced Joint Adapter" section below.

---

## Balanced Joint Adapter: FHS + Reddit-FR (1:1 ratio) — Complete (2026-04-20)

**Hypothesis:** The unbalanced joint adapter underperforms on Reddit-FR because the FHS training
corpus is 3.5× larger (13k vs 3.6k samples), causing FHS patterns to dominate gradient updates
and diluting the Reddit-FR signal. Subsampling FHS training to match Reddit-FR training size
should recover Reddit-FR performance while retaining the FHS recall gains.

**Design:**
- Same seed=42 split as all prior experiments → test sets are identical (direct 4-way comparison)
- FHS training partition subsampled to `len(reddit_train)` ≈ 3,665 samples (seeded)
- FHS val and test untouched → fair FHS evaluation preserved
- All hyperparameters identical to `french_joint` (lr=2e-4, lora_r=16, epochs=1)
- Actual training size: ~7,330 samples (half of french_joint's 16,680)

**Results:** `phase2_eval/{dataset}/lora_joint_balanced/` · `phase2_eval/lora_full_french_joint_balanced/`

### 4-Way Comparison: Baseline → Single → Joint → Balanced Joint (fair held-out test sets)

#### Reddit-FR (held-out test set, n=1,023)

| Metric | LG-1B base | LG-1B single | LG-1B joint | LG-1B balanced | SG-2b base | SG-2b single | SG-2b joint | SG-2b balanced |
|--------|:----------:|:------------:|:-----------:|:--------------:|:----------:|:------------:|:-----------:|:--------------:|
| **F1** | 0.417 | 0.513 | 0.557 | 0.551 | 0.335 | **0.662** | 0.632 | 0.611 |
| Precision | 0.624 | 0.650 | 0.665 | 0.670 | 0.855 | 0.706 | 0.756 | 0.751 |
| Recall | 0.313 | 0.424 | 0.479 | 0.468 | 0.208 | 0.623 | 0.543 | 0.514 |
| TNR | 0.851 | 0.820 | 0.809 | 0.818 | 0.972 | 0.795 | 0.862 | 0.865 |

#### French Hate Superset (held-out test set, n=3,614)

| Metric | LG-1B base | LG-1B single | LG-1B joint | LG-1B balanced | SG-2b base | SG-2b single | SG-2b joint | SG-2b balanced |
|--------|:----------:|:------------:|:-----------:|:--------------:|:----------:|:------------:|:-----------:|:--------------:|
| **F1** | 0.364 | 0.561 | 0.587 | 0.530 | 0.413 | 0.534 | **0.633** | 0.585 |
| Precision | 0.277 | 0.636 | 0.642 | 0.596 | 0.405 | 0.758 | 0.657 | 0.537 |
| Recall | 0.531 | 0.502 | 0.540 | 0.478 | 0.420 | 0.413 | 0.611 | **0.643** |
| TNR | 0.576 | 0.912 | 0.908 | 0.901 | 0.811 | 0.960 | 0.902 | 0.830 |

### Hypothesis verdict: ❌ Balancing does not help

| Expectation | Outcome |
|---|---|
| Reddit-FR F1 recovers toward 0.662 | **No** — drops from 0.632 → 0.611 (−0.021) for SG-2b |
| FHS F1 stays above single LoRA (0.534) | **Barely** — 0.585 vs 0.534 (+0.051) but worse than unbalanced joint (0.633) |
| Simultaneous F1 > 0.62 on both | **No** — balanced gets 0.611 / 0.585 simultaneously |

**Root cause:** The hypothesis assumed that equal data weighting would restore Reddit-FR signal. In practice, removing 3.5× of FHS training data reduces the model's FHS capacity without meaningfully improving Reddit-FR — the two domains compete for model capacity regardless of corpus balance. The unbalanced joint already provides a better Reddit-FR vs FHS trade-off.

**Note on SG-2b balanced FHS recall (0.643 > 0.611 joint):** The balanced training makes FHS
predictions more aggressive (higher recall, lower precision 0.537 vs 0.657). This is an artifact
of equal weighting — Reddit-FR's balanced class distribution (45/55) biases the model toward
higher recall across both domains.

### English Generalisation (8-dataset eval, no leakage)

Both joint variants were also evaluated on all 8 datasets. English results are leakage-free
(neither variant was trained on English data).

| Dataset | LG-1B joint | LG-1B balanced | SG-2b joint | SG-2b balanced |
|---------|:-----------:|:--------------:|:-----------:|:--------------:|
| HateCheck-EN | 0.855 | **0.840** | **0.839** | 0.837 |
| Reddit-EN | 0.579 | **0.585** | 0.540 | 0.544 |
| ToxiGen | 0.636 | 0.659 | 0.675 | **0.706** |
| OpenAI | **0.663** | 0.641 | **0.687** | 0.662 |
| Civil Comments | **0.238** | 0.234 | **0.306** | 0.273 |

**English observations:**
- **No catastrophic forgetting** — HC-EN F1 ≥ 0.837 for all variants (French fine-tuning does not destroy English hate speech detection)
- **Civil Comments collapses** (F1 < 0.31) — formal English discussion is out-of-distribution for models fine-tuned on informal French social media
- SG-2b balanced ToxiGen (0.706) is the best ToxiGen result across all Phase 2 adapters, likely because the balanced training's more aggressive recall style transfers to implicit toxicity detection

---

## Next Steps

1. ~~**Joint adapter eval**~~ ✅ **Done (2026-04-18).** Joint LoRA trained and evaluated — see "Joint Adapter" section above.

2. ~~**Reddit-FR LoRA on HateCheck-FR**~~ ✅ **Done (2026-04-17).** Reddit-FR LoRA evaluated on all 8 datasets (`lora_full_reddit_fr/`). SG-2b HC-FR only −0.021 vs FHS adapter −0.078. See "Generalisation: Reddit-FR Adapter" section above.

3. ~~**Balanced joint adapter**~~ ✅ **Done (2026-04-20).** Hypothesis disproven — balancing hurts both datasets. SG-2b balanced Reddit-FR 0.611 < unbalanced 0.632 < single 0.662. **Phase 2 fully complete. SG-2b × Reddit-FR LoRA confirmed as best Tier 2 model.**

4. ~~**Two-tier architecture evaluation**~~ → **Phase 4.** See `PHASE4_TIER1_RESULTS_REPORT.md`.

5. **Retrain FHS adapter with lower LR** — SG-2b's faster overfitting and HC-FR regression suggest `--lr 1e-4` instead of `2e-4`.
