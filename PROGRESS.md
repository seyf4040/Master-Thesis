# Thesis Progress — AI Content Moderation for Shareish

**Last updated:** 2026-04-12 | **Author:** Ural Seyfullah | *ULiège Master's Thesis*

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Baseline evaluation | ✅ Complete | 10 models × 8 datasets × 3 runs |
| Phase 2 — LoRA fine-tuning | 🔄 Partial | FHS adapters done; Reddit-FR eval + SG-2b retraining pending |
| Phase 3 — Data collection / generation | 🔜 Not started | Scraping + synthetic data scripts written, not yet run |
| Phase 4 — Two-tier architecture | 🔜 Not started | Planned: detoxify pre-filter + LoRA-tuned LG-1B |

---

## Core Thesis Argument

The thesis evaluates 10 content moderation models across 8 datasets for deployment on **Shareish**, a French-language solidarity platform run by ULiège. The central argument is:

> **Deployability — VRAM, inference speed, energy consumption — must be an explicit evaluation axis alongside accuracy.** A model requiring 14–18 GB VRAM is not viable for a small NGO with limited GPU infrastructure, regardless of its F1 score.

The deployment target is **French-language content**. HC-FR and FR-Hate F1 are the primary metrics throughout; English results are included for research comparability only.

---

## Phase 1 — Baseline Results

### F1 Scores (10 models × 8 datasets, sorted by HC-FR)

> French columns (HC-FR, FR-Hate, Red-FR) are the deployment-relevant metrics.

| Model | **HC-FR** | **FR-Hate** | **Red-FR** | HC-EN | ToxiGen | OpenAI | CivComm | Red-EN | VRAM | ms/sample |
|-------|:---------:|:-----------:|:----------:|:-----:|:-------:|:------:|:-------:|:------:|:----:|:---------:|
| ShieldGemma-9b | 0.883 | 0.442 | 0.375 | 0.913 | 0.640 | 0.712 | 0.302 | 0.269 | 18.4 GB | 41 |
| Llama-Guard-3-8B | 0.879 | 0.354 | 0.268 | **0.939** | 0.546 | 0.785 | 0.110 | 0.160 | 15.6 GB | 108 |
| ShieldGemma-2b | 0.858 | 0.441 | 0.311 | 0.902 | 0.632 | 0.499 | 0.315 | 0.227 | 5.7 GB | 25 |
| detoxify-multilingual | 0.787 | 0.292 | **0.408** | 0.803 | 0.486 | 0.688 | **0.723** | 0.332 | 1.1 GB | 6 |
| Mistral-7B | 0.783 | 0.391 | 0.319 | 0.921 | **0.669** | 0.762 | 0.295 | 0.332 | 14.0 GB | 150 |
| Llama-Guard-3-1B | 0.674 | 0.372 | 0.398 | 0.816 | 0.556 | 0.651 | 0.187 | 0.407 | 3.0 GB | 33 |
| CitizenLab | 0.644 | 0.281 | 0.318 | 0.702 | 0.430 | 0.264 | 0.293 | 0.323 | 1.1 GB | 5 |
| EthicalEye | 0.593 | 0.291 | 0.374 | 0.725 | 0.562 | 0.665 | 0.488 | 0.407 | 1.1 GB | 5 |
| detoxify-unbiased | 0.281 | 0.072 | 0.166 | 0.760 | 0.463 | 0.672 | 0.763 | 0.318 | 0.5 GB | 7 |
| KoalaAI | 0.008 | 0.040 | 0.326 | 0.694 | 0.502 | **0.938** | 0.245 | 0.299 | 0.6 GB | 14 |

⚠️ **detoxify-unbiased** and **KoalaAI** are not viable for Shareish — near-zero French F1 regardless of English scores.

All results are averages of 3 independent runs (std ≤ 0.01 everywhere — fully reproducible).

### Viable Models for Shareish (≤6 GB VRAM)

| Model | HC-FR | VRAM | ms/sample | Key limitation |
|-------|:-----:|:----:|:---------:|----------------|
| detoxify-multilingual | 0.787 | 1.1 GB | 6 ms | CPU-feasible; accuracy ceiling ~0.80 |
| Llama-Guard-3-1B | 0.674 | 3.0 GB | 33 ms | Modest accuracy; good Reddit-FR (0.398) |
| ShieldGemma-2b | 0.858 | 5.7 GB | 25 ms | Best French F1; counter-speech failure |

The three models above are the **Phase 2 fine-tuning candidates**.

### Key Findings

**1. The French/English gap is structural.** Most models perform significantly better in English. KoalaAI is the extreme case (HC-EN 0.694, HC-FR 0.008) — it is effectively an English-only model. Even Mistral-7B, a multilingual model, shows a 0.138 gap (0.921 EN → 0.783 FR), suggesting the safety fine-tuning data is predominantly English.

**2. ShieldGemma's counter-speech failure is a deployment blocker.** SG-2b flags quoted hate speech as toxic with a 94–95% error rate (`counter_quote_nh` correct-rate: 0.054 FR, 0.006 EN). On Shareish, users who quote a hateful message they received to report it would almost always be silenced. This is the primary target for Phase 2 fine-tuning.

**3. FR-Hate and Reddit-FR are universally hard.** No model exceeds F1=0.45 on FR-Hate and F1=0.41 on Reddit-FR — worse than HateCheck by a large margin. These are the primary fine-tuning targets for Phase 2.

**4. ShieldGemma-2b is the accuracy/deployability sweet spot** among viable models: highest French F1 (0.858), faster than LG-1B (25 ms vs 33 ms), fits in 5.7 GB. The v3 inference fix (token-probability scoring instead of text generation) was what unlocked it — ShieldGemma appeared broken in earlier evaluation rounds.

**5. The two-tier architecture is motivated by complementary strengths.** detoxify-multilingual handles Civil Comments and Reddit-FR well (where LLM-based models are weak), while ShieldGemma-2b / LG-1B handle the HateCheck categories well. No single model dominates all axes.

---

## Phase 2 — LoRA Fine-tuning Results

Models fine-tuned: **Llama-Guard-3-1B** and **ShieldGemma-2b**
Training data: **French Hate Superset (FHS)** and **Reddit-FR**
Method: LoRA (r=16, α=32, epoch 1 — all runs overfit after epoch 1)

### Fair Evaluation Results (held-out 20% test set)

> These are the authoritative numbers — evaluated on a held-out split with no overlap with training data.

| Dataset | Metric | LG-1B baseline | LG-1B + LoRA | Δ | SG-2b baseline | SG-2b + LoRA | Δ |
|---------|--------|:--------------:|:------------:|:---:|:--------------:|:------------:|:---:|
| **FR-Hate** | F1 | 0.371 | **0.557** | **+0.186** | 0.413 | **0.534** | **+0.121** |
| **FR-Hate** | TPR (recall) | 0.532 | 0.493 | −0.039 | 0.420 | 0.413 | −0.007 |
| **FR-Hate** | TNR (specificity) | 0.591 | **0.915** | **+0.324** | 0.811 | **0.960** | **+0.149** |
| **Reddit-FR** | F1 | 0.425 | *(not evaluated)* | — | 0.335 | *(not evaluated)* | — |

### What Worked

Both models genuinely improve on FR-Hate. The gains are **precision-driven**: fine-tuning taught the models which formal French patterns are unambiguously harmful (TNR up sharply), reducing false alarms on safe content without sacrificing much recall.

- LG-1B Precision: 0.285 → 0.640 (+0.355)
- SG-2b Precision: 0.405 → 0.758 (+0.353)

VRAM footprint is unchanged after LoRA — LG-1B stays at ~3 GB, SG-2b at ~5.7 GB.

### What Didn't Work / Open Questions

**Recall did not improve.** Both models still miss ~50% of hateful content (TPR ≈ 0.49/0.41 post-LoRA). For a content moderation system where missing hate speech is costly, this is a concern. The models learned better precision but not better generalisation to ambiguous cases.

**Implication:** the two-tier architecture becomes even more necessary — detoxify-multilingual handles the recall layer (catches obvious cases), LoRA-adapted LG-1B/SG-2b provides precision on flagged content.

### Pending Phase 2 Items

- [ ] **LG-1B Reddit-FR adapter evaluation** — adapter exists, fair eval not yet submitted
- [ ] **SG-2b Reddit-FR retraining** — training failed (SLURM timeout); resubmit with `--epochs 1`
- [ ] **Full 8-dataset eval with LoRA adapters** — HC-FR regression after LoRA is unknown
- [ ] **SG-2b retrain with lower LR** (`1e-4` instead of `2e-4`) — faster overfitting than LG-1B suggests the learning rate is too high

---

## Phase 3 — Data Collection & Synthetic Generation

**Status:** Scripts written, not yet run on cluster.

- `scrape_donnons.py` — scrapes donnons.org (French reuse/solidarity platform similar to Shareish) to build a domain-relevant corpus
- `generate_synthetic_data.py` — uses a generative LLM to produce synthetic hateful/safe examples targeting known model weaknesses (counter-speech, French slurs, obfuscation)

Goal: augment fine-tuning data to address the recall gap identified in Phase 2.

---

## Phase 4 — Two-Tier Architecture (Planned)

**Not yet implemented.** The architecture:

1. **Detoxify-multilingual** as a fast pre-filter (6 ms, CPU-feasible): clear safe → approve; clear toxic → flag
2. **LoRA-adapted LG-1B or SG-2b** for uncertain cases (~5–15% of traffic): more accurate decision + explanation

Expected benefit: ~90% of posts resolved in 6 ms; full LLM inference only on edge cases. Reduces energy cost by 4–7× compared to LLM-only.

---

## Thesis Writing Status

| Chapter | Topic | Status |
|---------|-------|--------|
| Introduction | Motivation, Shareish context, thesis plan | Draft |
| Chapter 1 | Background — NLP, hate speech, content moderation | Draft |
| Chapter 2 | Evaluation framework — models, datasets, metrics | Draft |
| Chapter 3 | Phase 1 baseline results | In progress |
| Chapter 4 | Data strategy (Phase 2 data + Phase 3) | In progress |
| Chapter 5 | LoRA fine-tuning (Phase 2) | In progress |
| Chapter 6 | Conclusions and recommendations | Not started |
| Appendix | Model cards, full result tables | Not started |

---

## Deeper Documentation

For detailed breakdowns, see:
- `code/docs/RESULTS_PHASE1_BASELINE.md` — full F1 tables, functionality heatmaps, deployability analysis
- `code/docs/PHASE1_RESULTS_REPORT.md` — visual report with figure descriptions
- `code/docs/PHASE2_LORA_RESULTS_REPORT.md` — full LoRA training diagnostics and analysis
- `code/docs/RESULTS_TRACKER.md` — live job/results status and cluster commands
