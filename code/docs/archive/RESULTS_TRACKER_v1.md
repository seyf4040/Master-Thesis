# Results Tracker

Status snapshot as of **2026-04-19**.

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 — Baseline evaluation | ✅ Complete | 10 models × 8 datasets × 3 runs. See [RESULTS_PHASE1_BASELINE.md](RESULTS_PHASE1_BASELINE.md) |
| Phase 2 — LoRA fine-tuning | ✅ Complete | All 6 adapters done. Balanced joint disproves hypothesis (0.611 < 0.662). **SG-2b × Reddit-FR LoRA (F1=0.662) is the confirmed Tier 2 model.** |
| Phase 3 — Data collection | 🔄 In progress | Track A (synthetic) done. Track B (scrapers) partially collected. |
| Phase 4 — Two-tier architecture | 🔄 In progress | Generalisation eval complete (2026-04-24). **Final Tier 1: 2c** (synthetic, epoch2) F1=0.668 honest, HC-FR FNR=7.0%, T_high=0.80. ToxiGen/OpenAI loader bugs remain. Group 3 (T2 specialisation) + end-to-end 2c simulation pending. |

---

## Phase 1 — Baseline Evaluation

**Status: Complete** as of 2026-03-29.

**Experiments:** 10 pre-trained models evaluated zero-shot across 8 datasets (HateCheck EN/FR, French Hate Superset, ToxiGen, OpenAI, Civil Comments, Reddit EN/FR). 3 multi-runs for statistical stability (std ≤ 0.01). Deployability metrics (VRAM, inference time, energy) collected alongside accuracy.

**Results location:**
```
~/code/results/
├── full_baseline_v3/
│   ├── {model}/{dataset}/results.json   (80 files)
│   ├── run_1/, run_2/, run_3/           (multi-run statistics)
│   └── summary.txt / summary.json
└── hatecheck_analysis/
    ├── {model}/{hatecheck_en,fr}/       (20 files)
    └── summary.txt / summary.json
```

**Main results:**
- Best deployable model: **ShieldGemma-2b** — HC-FR F1=0.858, ~5.7 GB VRAM, 24 ms/sample
- Best overall: LG-8B / SG-9b — but >14 GB VRAM, not viable for Shareish
- French/English gap is structural (e.g. KoalaAI: 0.694 HC-EN vs 0.008 HC-FR)
- No model exceeds F1=0.45 on FR-Hate or Reddit-FR → primary fine-tuning targets

**Reports:** [PHASE1_RESULTS_REPORT.md](PHASE1_RESULTS_REPORT.md) · [RESULTS_PHASE1_BASELINE.md](RESULTS_PHASE1_BASELINE.md)

### Key reminders

- The **VRAM guard** silently skips models that don't fit — check logs for `SKIPPED` lines
- **ShieldGemma** TPR/TNR imbalance at threshold=0.5 is misleading; threshold sweep was done separately
- **CitizenLab** may fail with a torch version error (CVE-2025-32434) — needs `torch >= 2.6` or safetensors conversion

---

## Phase 2 — LoRA Fine-tuning

**Status: Complete** as of 2026-04-18.

**Experiments:** LoRA adapters trained for 2 models (LG-1B, SG-2b) × 2 datasets (French Hate Superset, Reddit-FR) = 4 single-dataset runs + 1 joint adapter (FHS+Reddit-FR combined). Fair evaluation on held-out 20% test sets (per-source for joint). Generalisation check: each adapter evaluated on all 8 datasets.

**Results location:**
```
~/code/results/phase2_eval/
├── french_hate_superset/{baseline,lora}/
├── reddit_fr/{baseline,lora}/
└── lora_full/                           (FHS adapter on all 8 datasets)

~/code/results/lora_full_reddit_fr/      (Reddit-FR adapter on all 8 datasets)
```

**Main results:**
- Best gain: **SG-2b × Reddit-FR** F1 0.335→0.662 (+0.327, recall-driven)
- FHS gains: LG-1B +0.197, SG-2b +0.121 (precision-driven)
- Reddit-FR adapter generalises: HC-FR regression only −0.021 vs −0.078 for FHS adapter
- Joint adapter (FHS+Reddit-FR): SG-2b FHS F1=0.633 (+0.099 vs single FHS LoRA, recall-driven), Reddit-FR F1=0.632 (−0.030 vs single Reddit-FR LoRA — dilution from 3.5× larger FHS corpus)
- **SG-2b × Reddit-FR LoRA (F1=0.662) remains best Tier 2 candidate — joint does not surpass it**
- FHS adapter does NOT generalise to Reddit-FR (collapses to F1 ~0.08)

**Report:** [PHASE2_LORA_RESULTS_REPORT.md](PHASE2_LORA_RESULTS_REPORT.md)

> All 3 training runs overfit after epoch 1 — future retrains should use `--epochs 1`.

### Training runs

| Run | Status | Notes |
|-----|--------|-------|
| LG-1B × FHS | ✅ Done | best=epoch1, val_loss=0.1903. Adapter: `lora_adapters/llama_guard_1b/french_hate_superset/best` |
| LG-1B × Reddit-FR | ✅ Done | best=epoch1, val_loss=0.3031. Adapter: `lora_adapters/llama_guard_1b/reddit_fr/best` |
| SG-2b × FHS | ✅ Done | best=epoch1, val_loss=0.1862. Adapter: `lora_adapters/shieldgemma_2b/french_hate_superset/best` |
| SG-2b × Reddit-FR | ✅ Done | Fixed CUDA OOM with `--gradient_checkpointing` + `enable_input_require_grads()` + `--batch_size 2 --grad_accum 8`. Job 3843094 (2026-04-15). Adapter: `lora_adapters/shieldgemma_2b/reddit_fr/best` |
| LG-1B × FHS+Reddit-FR (joint) | ✅ Done | epoch1, val_loss=0.2115. n_train=16677. Adapter: `lora_adapters/llama_guard_1b/french_joint/best` (2026-04-18) |
| SG-2b × FHS+Reddit-FR (joint) | ✅ Done | epoch1, val_loss=0.2078. n_train=16686. Adapter: `lora_adapters/shieldgemma_2b/french_joint/best` (2026-04-18) |
| LG-1B × FHS+Reddit-FR (balanced joint) | ✅ Done | Reddit-FR F1=0.551, FHS F1=0.530. Balancing hurts both vs unbalanced. Adapter: `lora_adapters/llama_guard_1b/french_joint_balanced/best` |
| SG-2b × FHS+Reddit-FR (balanced joint) | ✅ Done | Reddit-FR F1=0.611, FHS F1=0.585. Hypothesis disproven — single Reddit-FR LoRA (0.662) remains best. Adapter: `lora_adapters/shieldgemma_2b/french_joint_balanced/best` |

### Evaluation results

| Adapter | Eval scope | Key finding |
|---------|-----------|-------------|
| FHS adapter — fair eval (held-out 20%) | FHS test set | LG-1B: 0.364→0.561 (+0.197). SG-2b: 0.413→0.534 (+0.121). Precision-driven. |
| Reddit-FR adapter — fair eval (held-out 20%) | Reddit-FR test set | LG-1B: 0.417→0.513 (+0.096). **SG-2b: 0.335→0.662 (+0.327)**. Recall-driven. |
| FHS adapter — full 8-dataset eval | All datasets | HC-FR regression: LG-1B −0.024, SG-2b −0.078. Reddit-FR collapses (LG-1B −0.299, SG-2b −0.237). |
| Reddit-FR adapter — full 8-dataset eval | All datasets | SG-2b HC-FR 0.8369 (−0.021, 4× better than FHS adapter). Reddit-EN +0.284, ToxiGen +0.234 (informal-register transfer). Civil Comments collapses −0.256 (same as FHS adapter). |
| Joint adapter — fair eval (per-source test sets) | FHS + Reddit-FR test sets | SG-2b FHS: 0.413→**0.633** (+0.220, recall-driven). SG-2b Reddit-FR: 0.335→0.632 (+0.297). Joint substantially beats single FHS LoRA; slightly trails single Reddit-FR LoRA (−0.030). |

**Conclusion:** SG-2b × Reddit-FR LoRA (F1=0.662) remains the best Tier 2 candidate — joint adapter does not surpass it on Reddit-FR. Phase 2 complete.

**Full analysis:** [PHASE2_LORA_RESULTS_REPORT.md](PHASE2_LORA_RESULTS_REPORT.md)

---

## Phase 3 — Data Collection

**Status: In progress** as of 2026-04-17.

**Experiments:** Two parallel tracks — (A) synthetic French hate speech generated via Mistral-7B across 5 HateCheck-FR functionalities; (B) real-world content scraped from solidarity/commercial platforms to train a content-type classifier (solidarity_exchange vs commercial_listing) for Shareish-specific filtering.

**Results location:**
```
~/code/data/
├── synthetic/{functionality}.jsonl      (Track A — 1,500 items)
├── donnons/{category}.jsonl             (Track B — 3,238 items)
├── donnerie/donnerie.jsonl              (Track B — pending rerun)
└── 2ememain/2ememain.jsonl              (Track B — 51 items, scraper needs fix)
```

**Main results:** Track A complete (1,500 synthetic items). Track B: donnons.org done (3,238), donnerie.be at 0 (rerunning), 2ememain.be at 51/800 (scraper fix needed). No classifier trained yet.

**Report:** None yet — in progress.

### Track A — Synthetic French hate speech (`generate_synthetic_data.py`)

**Model used:** Mistral-7B-Instruct-v0.3 | **Files:** `~/code/data/synthetic/{functionality}.jsonl`

| Functionality | Label | Target | Collected | Quality |
|---------------|-------|--------|-----------|---------|
| slur_h | 1 | 300 | **300** | ✅ Good — explicit slurs, varied groups |
| spell_leet_h | 1 | 300 | **300** | ✅ Good — leet-speak obfuscation present |
| spell_char_del_h | 1 | 300 | **300** | ✅ Good — char deletion confirmed |
| derog_impl_h | 1 | 300 | **300** | ⚠️ Weak — Mistral drifts to neutral text, needs manual review |
| counter_quote_nh | 0 | 300 | **300** | ✅ Good — counter-speech framing correct |
| **Total** | | **1,500** | **1,500** | |

**Known data issue:** All examples have a `"1.1. "` prefix artifact from list parsing. Strip before training:
```python
import re
record["text"] = re.sub(r"^\d+\.\d+\.\s*", "", record["text"])
```

### Track B — Content-type classifier data (scrapers)

**Files:** `~/code/data/{donnons,donnerie,2ememain}/`

| Source | Label | Class | Target | Collected | Status |
|--------|-------|-------|--------|-----------|--------|
| donnons.org | 1 | solidarity_exchange | 3,400 | **3,238** | ✅ Complete |
| donnerie.be | 1 | solidarity_exchange | 1,000 | **0** | ❌ Rerunning — href bug fixed 2026-04-17 |
| 2ememain.be | 0 | commercial_listing | 800 | **51** | ❌ Needs scraper fix |
| Shareish (supervisor) | 1 | solidarity_exchange | TBD | TBD | ⏳ Awaiting |

**donnons.org** — 3,238 items across 17 categories. 4 categories below target (alimentation: 100, materiel-specialise: 100, animalerie: 182, vehicules: 156) — site had fewer available items. Content confirmed: real French donation posts, same register as Shareish.

Known data issue — HTML entities in `location` field (`"L&apos;Horme"`). Fix before training:
```python
import html
record["location"] = html.unescape(record["location"]) if record["location"] else None
```

**donnerie.be** — 0 items. Bug: site uses absolute hrefs (`https://donnerie.be/annonces/slug/`) but parser checked relative paths (`/annonces/slug/`). Fixed in `scrape_donnerie.py`, resubmitted. Will be highest-quality positive-class data (Belgian French, same register as Shareish).

**2ememain.be** — 51/800 items. langdetect + Dutch condition-word filter discards ~90% of listings; descriptions also truncated in static HTML JSON-LD. To fix:
- Remove or relax Dutch condition-word filter (rely on langdetect only)
- Investigate `window.__CONFIG__` parsing for full descriptions


**Full analysis:** [PHASE3_DATA_RESULTS_REPORT.md](PHASE3_DATA_RESULTS_REPORT.md)

---

## Phase 4 — Two-tier Architecture

**Status: In progress** as of 2026-04-24. Group 1 (end-to-end scoring + simulation) complete. Group 2 (improved T1 fine-tuning + 2d) complete. Generalisation eval complete (all 6 checkpoints × 8 datasets). Group 3 (Tier 2 specialisation) + end-to-end 2c simulation pending.

**Experiments:**
- **Track A:** `unitary/multilingual-toxic-xlm-roberta` pretrained — threshold analysis ✅
- **Track B:** Fine-tune Detoxify-M on Reddit-FR (MSE regression, 3 epochs) ✅
- **Group 1:** Score all 511 holdout samples with both T1 variants + SG-2b T2; simulate all thresholds ✅
- **Group 2:** Improved T1 fine-tuning — 2a (10 epochs), 2b (soft labels), 2c (synthetic data) ✅
- **Group 2d:** Combined soft labels + synthetic data — does NOT beat 2c ✅
- **Group 3:** Specialise Tier 2 on deferred distribution ⏳

**Results location:**
```
~/code/results/
├── threshold_analysis/                              (Detoxify-M Phase 1 baseline — complete)
├── tier1_detoxify_finetuned/                        (Track B fine-tuned checkpoint)
│   ├── best/                                        (epoch 3, val_loss=0.2136)
│   ├── final/
│   ├── test_set.json                                (511 held-out Reddit-FR samples)
│   └── training_meta.json
└── tier1_comparison/
    ├── unitary_multilingual-toxic-xlm-roberta/      (Track A results)
    └── _home_sural_code_results_tier1_detoxify_finetuned_best/  (Track B threshold analysis)
```

**Main results:**

### Generalisation Evaluation — All 6 Checkpoints × 8 Datasets (2026-04-24)

Results: `results/tier1_generalisation/` — raw_scores.json per checkpoint, all 8 datasets.

Key findings (English generalisation, uncontaminated):
- **2c_synthetic HC-FR FNR at T=0.5: 7.0%** (pretrained: 45.8%) — synthetic HC-FR functionality coverage generalises to near-zero missed structured hate speech
- **2c_synthetic HC-EN FNR at T=0.5: 2.9%** (pretrained: 20.2%) — cross-lingual generalisation of HC patterns
- Reddit-EN best-F1: all models similar (~0.650) — Reddit-FR fine-tuning preserves English Reddit detection
- FHS best-F1 degrades under all fine-tuning: pretrained 0.413 → 2c 0.298 (domain mismatch; Reddit-FR register ≠ formal FHS)
- Civil Comments best-F1 degrades: pretrained 0.687 → 2c 0.545 (formal English, low prevalence)
- **Reddit-FR numbers here are contaminated** (training data included) — use honest 511-sample Group 2 results (2c: 0.668)
- **ToxiGen n=0 and OpenAI hateful=0 loader bugs confirmed** — both datasets unreadable

Full analysis: [PHASE4_TIER1_RESULTS_REPORT.md](PHASE4_TIER1_RESULTS_REPORT.md) (Generalisation Evaluation section)

### Group 1 — End-to-end Two-Tier Evaluation (2026-04-19)

Honest holdout: 511 samples (216 hateful / 295 safe). Scores saved per sample for offline simulation.

| Configuration | Combined F1 | FNR | Deferral | Avg_ms | vs T2 alone |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tier 2 alone (SG-2b LoRA)** | **0.640** | 32.9% | 100% | 55.9 | — |
| Pretrained T1 + T2 (best F1) | 0.643 | 30.1% | 75.5% | 47.7 | +0.003 |
| Fine-tuned T1 + T2 (any target) | 0.626 | 44.9% | 10.4% | 12.6 | −0.014 |

- Combined F1 does not meaningfully exceed Tier 2 alone at any useful deferral rate
- **Honest T1_FNR (fine-tuned, 511-sample holdout) = 41.7%** — Track B's 25.2% was data-leakage artifact
- **Speed advantage is real:** at 10% deferral, 4× faster than Tier 2 alone (12.6 ms vs 55.9 ms)
- Bimodal score collapse confirmed: all deferral targets collapse to same operating point for fine-tuned T1
- Group 2 (soft labels, more epochs, synthetic data) is now the critical path to F1 > 0.640

**Results:** `results/two_tier_scores/{pretrained,finetuned}/`
**Report:** [PHASE4_TIER1_RESULTS_REPORT.md](PHASE4_TIER1_RESULTS_REPORT.md) (Group 1 section)

### Detoxify-M Baseline (Phase 1 — motivation for Phase 4)

Reddit-FR (Shareish proxy):

| Operating point | T_low | T_high | Deferral | **T1 FNR** | T1 FPR |
|----------------|:-----:|:------:|:--------:|:----------:|:------:|
| Low deferral   | 0.80  | 1.00   | 11.0%    | **40.9%**  | 0.0%   |
| Mid deferral   | 0.20  | 1.00   | 27.2%    | **37.0%**  | 0.0%   |
| High deferral  | 0.05  | 1.00   | 42.9%    | **34.3%**  | 0.0%   |

### Track A — unitary/multilingual-toxic-xlm-roberta (pretrained)

Single-threshold best F1 per dataset:

| Dataset | Default T=0.5 F1 | Best T | Best F1 |
|---------|:----------------:|:------:|:-------:|
| HateCheck-FR | 0.634 | 0.00 | 0.823 (all-positive) |
| FR-Hate Superset | 0.315 | 0.02 | 0.412 |
| Reddit-FR | 0.366 | 0.00 | 0.616 (all-positive) |

Reddit-FR operating points:

| Operating point | T_low | T_high | Deferral | **T1 FNR** | T1 FPR |
|----------------|:-----:|:------:|:--------:|:----------:|:------:|
| Low deferral   | 0.70  | 1.00   | 11.1%    | **41.0%**  | 0.0%   |
| Mid deferral   | 0.25  | 1.00   | 26.0%    | **37.6%**  | 0.0%   |
| High deferral  | 0.05  | 1.00   | 52.3%    | **34.7%**  | 0.0%   |

**Conclusion:** Identical failure mode to Detoxify-M. Same backbone → same score distribution collapse on informal French. T_high always 1.00, no confident-unsafe bin. Track A confirms the Phase 1 finding rather than adding signal.

### Track B — Fine-tuned Detoxify-M backbone (3 epochs, MSE regression, Reddit-FR)

Training (4148 samples, val=460, test=511, seed=42):

| Epoch | Train loss | Val loss | Time |
|-------|:----------:|:--------:|-----:|
| 1 | 0.2344 | 0.2171 | 33s |
| 2 | 0.2076 | 0.2157 | 29s |
| **3** | **0.1863** | **0.2136** | 29s |

Best checkpoint: epoch 3 (val_loss=0.2136, steady monotonic improvement — no overfitting detected).

Single-threshold best F1 per dataset:

| Dataset | Default T=0.5 F1 | Best T | Best F1 |
|---------|:----------------:|:------:|:-------:|
| HateCheck-FR | 0.691 | 0.00 | 0.722 |
| FR-Hate Superset | 0.375 | 0.00 | 0.385 |
| Reddit-FR | **0.662** | 0.00 | **0.704** |

Reddit-FR operating points:

| Operating point | T_low | T_high | Deferral | **T1 FNR** | T1 FPR |
|----------------|:-----:|:------:|:--------:|:----------:|:------:|
| All three       | 0.00  | 0.95   | 11.4%    | **25.2%**  | 15.0%  |

**Key finding:** T1_FNR drops from 37% (baseline) to 25.2% — a meaningful improvement, but the **success criterion of <15% is NOT met**. The model now produces a confident-unsafe bin (T_high=0.95 flags 88.6% of content), a qualitative shift from the pretrained model. However, all three deferral-target operating points collapse to the same configuration — indicative of a bimodal score distribution with limited tunable granularity.

**⚠️ Data leakage caveat:** Track B threshold analysis used the full `test-fr.csv` (5119 samples), 80% of which were in the fine-tuning training set. The honest T1_FNR should be re-evaluated on the held-out `test_set.json` (511 samples). The bimodal collapse (all deferral targets → same operating point) is likely a data leakage artifact — the model is overconfident on training data.

**FHS T1_FPR=67.4%:** The fine-tuned model massively overfires on FHS safe content — expected, as it was trained only on Reddit-FR informal content. Domain specificity is a cost of the fine-tuning approach.

### Cross-model comparison (Reddit-FR, ~25% deferral target)

| Model | T1 FNR | Deferral | T_low | T_high | Notes |
|-------|:------:|:--------:|:-----:|:------:|-------|
| Detoxify-M (Phase 1 baseline) | 37.0% | 27.2% | 0.20 | 1.00 | No unsafe bin |
| Track A (pretrained unitary) | 37.6% | 26.0% | 0.25 | 1.00 | Same failure |
| **Track B (fine-tuned)** | **25.2%** | **11.4%** | **0.00** | **0.95** | Leakage risk |
| **Success criterion** | **< 15%** | ~25% | — | — | Not met |

### Group 2 — Improved Tier 1 Fine-tuning (2026-04-19)

| Variant | Best val_loss | Best epoch | Reddit-FR F1 | T1_FNR (honest) | T_high | Notes |
|---------|:------------:|:----------:|:------------:|:---------------:|:------:|-------|
| 2a — 10 epochs, hard labels | 0.2083 | 2 | 0.615 | 28.5% | 0.95 | Overfits after ep2; no gain from extra epochs |
| 2b — soft labels ε=0.05 | 0.2058 | 2 | 0.619 | 28.3% | 0.95 | T1_FPR drops 23.8%→13.5%; partial distribution fix |
| **2c — synthetic data** | **0.1912** | **2** | **0.668** | **25.2%** | **0.80** | **Best: T_high breaks below 0.95; HC-FR 0.816** |
| 2d — soft + synthetic | 0.2025 | 2 | 0.634 | 27.1% | 0.80 | Soft labels redundant — synthetic data already regularises |

- **2c (synthetic, epoch 2) is the confirmed final Tier 1 model** — 2d combination does not improve on it
- All best checkpoints are epoch 2 — future runs should use `--epochs 2`
- 2c delivers HC-FR F1=0.816 (vs 2a=0.631, 2b=0.660) due to synthetic data covering HC-FR functionalities
- Honest T1_FNR = 25.2% for 2c vs 41.7% (Track B leaky estimate from Group 1)

### Next steps

1. **Fix ToxiGen and OpenAI loaders** — both confirmed broken by generalisation eval. Required before any 8-dataset generalisation table appears in the thesis.
2. **Run end-to-end simulation for 2c**: run `score_two_tier.py` with 2c as Tier 1 (`score_two_tier_finetuned.sbatch` using 2c checkpoint), then `simulate_thresholds.py`. Goal: combined F1 > 0.640 at < 30% deferral.
3. **Group 3 (Tier 2 specialisation):** Fine-tune SG-2b on deferred-distribution samples from 2c Tier 1 at T=(0.10, 0.75). Requires collecting deferred samples from Reddit-FR train set through 2c Tier 1.

**Success criterion:** Combined system F1 > 0.640 (Tier 2 alone) at deferral < 30%.

**Report:** [PHASE4_TIER1_RESULTS_REPORT.md](PHASE4_TIER1_RESULTS_REPORT.md) | [PHASE4_TWO_TIER_BRIEFING.md](PHASE4_TWO_TIER_BRIEFING.md) | [THRESHOLD_ANALYSIS_REPORT.md](THRESHOLD_ANALYSIS_REPORT.md)
