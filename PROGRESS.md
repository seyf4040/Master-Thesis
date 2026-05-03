# Thesis Progress — AI Content Moderation for Shareish

**Last updated:** 2026-05-03 | **Author:** Ural Seyfullah | *ULiège Master's Thesis*

---

> ⚠️ **Reddit-EN and Reddit-FR results should be interpreted with caution.** Both datasets were originally labeled for *rule-based moderation* — each subreddit enforces its own community rules, which do not map cleanly onto toxicity detection. Low F1 scores on these datasets partly reflect this label mismatch rather than genuine model failure; the models are being evaluated against a different task than the one the labels encode.

---

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Baseline evaluation | ✅ Complete | 10 models × 8 datasets × 3 runs |
| Phase 2 — LoRA fine-tuning | ✅ Complete | SG-2b × Reddit-FR LoRA (F1=0.662) confirmed as Tier 2 model |
| Phase 3 — Data collection / generation | 🔄 In progress | Track A (1,500 synthetic items) complete; Track B (scrapers) partial |
| Phase 4 — Two-tier architecture | 🔄 In progress | 7/8 experiments done; 2c-synthetic (epoch 2) confirmed as Tier 1 |

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

**1. The French/English gap is structural.** Most models perform significantly better in English. KoalaAI is the extreme case (HC-EN 0.694, HC-FR 0.008) — it is effectively an English-only model. Even Mistral-7B, a multilingual model, shows a 0.138 gap (0.921 EN → 0.783 FR), suggesting the training data is predominantly English.

**2. ShieldGemma's counter-speech failure is concerning.** SG-2b flags quoted hate speech as toxic with a 94–95% error rate (`counter_quote_nh` correct-rate: 0.054 FR, 0.006 EN). On Shareish, users who quote a hateful message they received to report it would almost always be silenced. 

**3. FR-Hate and Reddit-FR are universally hard.** No model exceeds F1=0.45 on FR-Hate and F1=0.41 on Reddit-FR — worse than HateCheck by a large margin. These are the primary fine-tuning targets for Phase 2.
-> For Reddit-EN and FR: it is explained by the fact that there is a mismatch between dataset purpose and evaluation. Dataset is labeled for rule based moderation (with different set of rules per dataset) while we try to evaluate for toxicity detection.

**4. ShieldGemma-2b is the accuracy/deployability sweet spot** among viable models: highest French F1 (0.858), faster than LG-1B (25 ms vs 33 ms), fits in 5.7 GB. 

**5. The two-tier architecture is motivated by complementary strengths.** detoxify-multilingual handles Civil Comments and Reddit-FR well (where LLM-based models are weak), while ShieldGemma-2b / LG-1B handle the HateCheck categories well. No single model dominates all axes.

---

## Phase 2 — LoRA Fine-tuning Results

**Status:** ✅ Complete (2026-04-20)
Models fine-tuned: **ShieldGemma-2b** (primary) and **Llama-Guard-3-1B**
Method: LoRA (r=16, α=32) — all runs overfit after epoch 1; `--epochs 1` confirmed for all adapters.

### Summary of All 4 LoRA Experiments

Baselines — LG-1B: FHS F1=0.364 / Reddit-FR F1=0.417 | SG-2b: FHS F1=0.413 / Reddit-FR F1=0.335

| ID | Adapter | Eval set | LG-1B F1 | Δ | SG-2b F1 | Δ | One-line result |
|----|---------|----------|:---------:|:-:|:---------:|:-:|-----------------|
| P2-E1 | FHS only | FHS test (n=3,614) | 0.561 | +0.197 | 0.534 | +0.121 | Precision-driven; catastrophic Reddit-FR collapse (LG-1B −0.299, SG-2b −0.237) |
| **P2-E2** | **Reddit-FR only** | Reddit-FR test (n=1,023) | 0.513 | +0.096 | **0.662** | **+0.327** | **Recall-driven; SG-2b HC-FR regression only −0.021 → Confirmed Tier 2** |
| P2-E3 | Joint (FHS + Reddit-FR) | Reddit-FR test | 0.573 | +0.156 | 0.632 | +0.297 | Beats FHS-only on both domains; trails Reddit-FR single by −0.030 (SG-2b) |
| P2-E4 | Balanced joint (1:1) | Reddit-FR test | 0.551 | +0.134 | 0.611 | +0.276 | Balancing hurts both models on both domains; hypothesis disproven |

### Key Conclusions

**SG-2b × Reddit-FR LoRA (P2-E2, F1=0.662) is the confirmed Tier 2 model.** It is stored at `lora_adapters/shieldgemma_2b/reddit_fr/best/`.

The gains are **recall-driven**: fine-tuning on Reddit-FR taught the model to catch informal/colloquial hate speech that the baseline systematically missed (+0.327 F1). The small HC-FR regression (−0.021) is an acceptable trade-off.

The FHS adapter (P2-E1) is useful for formal French test sets but actively harmful on Reddit-like informal content. Joint training dilutes the Reddit-FR signal; balanced joint dilutes both — single-domain Reddit-FR training is the Pareto-optimal choice.

---

## Phase 3 — Data Collection & Synthetic Generation

**Status:** 🔄 In progress

### Track A — Synthetic French Hate Speech (✅ Complete)

- 1,500 items generated across 5 HateCheck functionalities targeting known model weaknesses
- 3/5 functionalities usable as-is; `derog_impl_h` (implicit derogation) and `spell_leet_h` (leet-speak) need manual review before training
- Pre-processing required: strip `"1.1. "` prefix artifact — `re.sub(r"^\d+\.\d*\s*", "", text)`
- Data: `~/code/data/synthetic/{functionality}.jsonl`

### Track B — Web Scrapers (Partial)

| Scraper | Status | Collected | Target | Notes |
|---------|--------|:---------:|:------:|-------|
| donnons.org | ✅ | 3,238 | 3,400 | 17-category solidarity data; HTML entity issue in location field |
| donnerie.be | ❌ | 0 | 1,000 | Bug fixed (absolute href); resubmitted 2026-04-17, results pending |
| 2ememain.be | ❌ | 51 | 800 | Dutch filter + truncated descriptions; scraper needs fix |

**Current class imbalance: 64:1** (3,289 solidarity : 51 commercial) — the content-type classifier cannot be trained yet.

Interim fix: `generate_classifier_data.py` (not yet run on cluster) generates ~300 synthetic commercial examples to reach a minimum viable 2:1 ratio. Minimum viable set: ~1,000 solidarity + ~500 commercial.

---

## Phase 4 — Two-Tier Architecture

**Status:** 🔄 In progress (7/8 experiments complete)

**Architecture:** Fine-tuned Detoxify-multilingual (Tier 1 fast filter) → SG-2b × Reddit-FR LoRA (Tier 2 specialist for deferred cases)

**Evaluation datasets:**
- **P4-E1 to P4-E6** (F1/T1_FNR in the table below): **511-sample honest holdout from Reddit-FR** — held back from training, seed=42
- **P4-E7 generalisation eval**: intended 8 datasets, but with caveats — ToxiGen (n=0, loader bug) and OpenAI (hateful=0, loader bug) are effectively missing; Reddit-FR results are contaminated for all fine-tuned models (trained on 80% of the same file); clean signal comes from HC-FR, HC-EN, FHS, Reddit-EN, and CivComm only

### Experiment Results

| ID | Name | F1 (honest) | T1_FNR | One-line result |
|----|------|:-----------:|:------:|-----------------|
| P4-E1 | T1 pretrained baseline | 0.616 | 37.6% | Same failure as Detoxify-M baseline; same backbone → same score collapse |
| P4-E2 | T1 fine-tuned base | 0.626 | 41.7% | Honest FNR 41.7%; bimodal score collapse; 4.4× speed gain |
| P4-E3 | T1 variant 2a (10 epochs) | 0.615 | 28.5% | Overfits after epoch 2; extra epochs wasted |
| P4-E4 | T1 variant 2b (soft labels ε=0.05) | 0.619 | 28.3% | T1_FPR drops 23.8%→13.5%; partial distribution fix |
| **P4-E5** | **T1 variant 2c (synthetic data)** | **0.668** | **25.2%** | **Best: T_high=0.80, HC-FR FNR=7.0% → Currently best performing Tier 1** |
| P4-E6 | T1 variant 2d (soft + synthetic) | 0.634 | 27.1% | Effects do not stack; worse than 2c; 2c is the ceiling |
| P4-E7 | Generalisation eval (6 ckpts × 5 valid datasets) | — | 7.0% HC-FR | 2c best on HC-FR/EN; ToxiGen+OpenAI broken (loader bugs); Reddit-FR contaminated |
| P4-E8 | End-to-end simulation (2c) | ⏳ | — | Pending: `score_two_tier_finetuned.sbatch` with 2c checkpoint |

### Key Conclusions

**2c-synthetic (epoch 2) is currently the optimal Tier 1 model.** Best config: T_low=0.10, T_high=0.75, deferral rate=10.4%, combined latency ~12.7 ms/sample.

This meets the success criterion: F1=0.668 > 0.640 threshold at <30% deferral rate.

Neither Tier 1 variant exceeds Tier 2 alone in raw F1, but the 4× speed advantage is the deployability argument — ~90% of posts resolved at Tier 1 speed (~6 ms), full SG-2b inference only on the deferred ~10%.

The final pending experiment (P4-E8) runs the full end-to-end simulation with the 2c checkpoint to confirm real-world combined F1.

> 💡 **Future direction — LLM-generated evaluation dataset.** The unexpectedly strong 2c results (synthetic data alone closing HC-FR FNR from 45.8% to 7.0%) suggest that LLM-generated data is a viable signal source for both fine-tuning *and* evaluation. This opens the possibility of generating a purpose-built French hate speech dataset via LLM — one where labels directly encode toxicity rather than subreddit rules — to be used as both a training supplement and a cleaner evaluation benchmark. Phase 4 results should ultimately be re-evaluated against such a dataset: the Reddit-FR holdout used throughout is subject to the same label-mismatch caveat established for Reddit results generally.

---

## Thesis Writing Status

| Chapter | Topic | Status |
|---------|-------|--------|
| Introduction | Motivation, Shareish context, thesis plan | Draft |
| Chapter 1 | Background — NLP, hate speech, content moderation | Draft |
| Chapter 2 | Evaluation framework — models, datasets, metrics | In progress |
| Chapter 3 | Phase 1 baseline results | In progress |
| Chapter 4 | Data strategy (Phase 2 data + Phase 3) | Not started |
| Chapter 5 | LoRA fine-tuning (Phase 2) | Not started |
| Chapter 6 | Two-tier architecture (Phase 4) | Not started |
| Chapter 7 | Conclusions and recommendations | Not started |
| Appendix | Model cards, full result tables | Not started |

---

## Deeper Documentation

Phase-level detail lives in the 3-tier doc structure under `code/docs/`:

- `code/docs/phase1/index.md` + `exp_*.md` — Phase 1 baseline experiments and results
- `code/docs/phase2/index.md` + `exp_*.md` — Phase 2 LoRA fine-tuning experiments
- `code/docs/phase3/index.md` + `exp_*.md` — Phase 3 data collection and synthetic generation
- `code/docs/phase4/index.md` + `exp_*.md` — Phase 4 two-tier architecture experiments
- `code/docs/RESULTS_TRACKER.md` — Live cluster job status and commands
