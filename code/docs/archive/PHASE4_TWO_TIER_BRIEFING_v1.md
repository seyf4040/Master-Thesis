# Phase 4 Briefing — Two-Tier Moderation Architecture

**Purpose:** Self-contained briefing for resuming Phase 4 work. Read this before touching any
Phase 4 code. Cross-references to prior-phase reports are given for deeper context.

---

## 1. Thesis Context

The thesis evaluates AI-based content moderation for **Shareish** — a small Belgian NGO running
a solidarity-exchange platform (French-language, informal register: "Canapé à donner à Jette").

**Core thesis argument:** Deployability (VRAM, inference speed, energy cost) must be an explicit
evaluation axis alongside accuracy. A model requiring 24 GB VRAM is not viable for a small NGO.

**Target content distribution:** Informal Belgian French social media text. The closest proxy
dataset available is **Reddit-FR** (French Reddit comments, balanced hate/safe). This is the
primary benchmark for all Phase 4 evaluation.

---

## 2. The Two-Tier Architecture Idea

### Motivation
Phase 1 baseline results showed a fundamental tension:

| Model | French F1 (HC-FR) | VRAM | Inference |
|-------|:-----------------:|:----:|:---------:|
| Detoxify-multilingual | 0.787 | ~0 (CPU) | ~2ms |
| ShieldGemma-2b (LoRA, Reddit-FR) | 0.858 | ~3 GB | ~30ms |

No single model is both cheap enough and accurate enough for Shareish deployment. The
two-tier architecture was proposed to get the best of both:

- **Tier 1 (fast, cheap):** Handle easy cases — content that is clearly safe or clearly unsafe.
  Runs on CPU, processes everything. Low cost per item.
- **Tier 2 (expensive, accurate):** Handle uncertain cases only.
  Runs on GPU, invoked only when Tier 1 is not confident. Higher cost but low volume.

The design goal: pass ~60% of content through Tier 1 alone (saving Tier 2 compute),
while keeping Tier 2 FNR acceptable.

### Three-zone logic
```
score < T_low   → Tier 1 passes as SAFE    (high confidence)
score > T_high  → Tier 1 flags as UNSAFE   (high confidence)
T_low ≤ score ≤ T_high → DEFER to Tier 2  (uncertain)
```

---

## 3. Phase 1: Detoxify-M as Tier 1 — Threshold Analysis Results

**Script:** `code/phase1_baseline/analyze_threshold_detoxify.py`
**Full results:** `code/docs/THRESHOLD_ANALYSIS_REPORT.md`
**Raw data:** `results/threshold_analysis/`

### What we measured
Detoxify-multilingual toxicity scores on three French datasets:
- **HateCheck-FR** (HC-FR): formal, controlled hate speech test cases (n=3,718)
- **French Hate Superset** (FHS): academic hate speech corpus (n=18,071)
- **Reddit-FR**: informal French Reddit comments, closest Shareish proxy (n=5,119)

### Key finding: structural asymmetry

| Dataset | Score distribution | T_high | T_low | Tier 1 can... |
|---------|-------------------|:------:|:-----:|----------------|
| HC-FR | Spread across [0,1] | 0.10 | **0.00** | Flag only (no safe bin) |
| FHS | Clusters near 0 | **1.00** | 0.80 | Pass only (no unsafe bin) |
| Reddit-FR | Clusters near 0 | **1.00** | 0.80 | Pass only (no unsafe bin) |

**On FHS and Reddit-FR, T_high is always 1.00** — Detoxify-M almost never assigns high
scores to any content on these datasets. There is no confident-unsafe bin. The three-zone
design degenerates into a binary gate that can only pass content, never flag it.

### Operating point costs (Reddit-FR — the Shareish proxy)

| Setting | T_low | T_high | Deferral to T2 | **T1 FNR** |
|---------|:-----:|:------:|:--------------:|:----------:|
| Aggressive | 0.05 | 1.00 | 43% | **34.3%** |
| Conservative | 0.20 | 1.00 | 27% | **37.0%** |

**T1 FNR of 34–41% is unacceptable for a safety application.** Even at 43% deferral
to Tier 2, more than 1 in 3 hateful items leaks through Tier 1 as "safe".

### Root cause
Detoxify-multilingual was trained on Jigsaw Wikipedia comment data — formal, English-dominant.
Multilingual fine-tuning covered formal text. Informal French hate speech is out-of-distribution:
the model assigns near-zero toxicity scores to content regardless of label, so the score carries
no class information on this distribution.

### What the analysis did NOT kill
The two-tier architecture concept is sound. The problem is the specific model chosen as Tier 1.
A Tier 1 model that **is** calibrated for informal French would recover the design.

---

## 4. Phase 4 Plan: Finding a Better Tier 1

Two parallel tracks:

### Track A — Test existing lightweight models as Tier 1

**Script:** `code/phase4_two_tier/analyze_threshold_tier1.py`
(generalised version of the Detoxify analysis, accepts `--model` argument)

Candidates:
| Model key | Description | Expected issue |
|-----------|-------------|----------------|
| `detoxify_multilingual` | Baseline — already tested, fails | Training data mismatch |
| `citizenlab` | XLM-RoBERTa sentiment (negative prob as toxicity proxy) | Sentiment ≠ hate speech |
| `hf_classifier --hf_model_id unitary/multilingual-toxic-xlm-roberta` | XLM-RoBERTa trained on multilingual social media toxicity — same Unitary team as Detoxify but different training corpus | Best external candidate |

Run command pattern:
```bash
python code/phase4_two_tier/analyze_threshold_tier1.py \
    --model hf_classifier \
    --hf_model_id unitary/multilingual-toxic-xlm-roberta \
    --hf_toxic_label toxic \
    --output_dir ~/code/results/tier1_comparison \
    --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv
```

Output layout: `results/tier1_comparison/{model_key}/` — one subfolder per model.

### Track B — Fine-tune Detoxify-M on Reddit-FR

**Motivation:** The XLM-RoBERTa backbone in Detoxify-M already handles multilingual text well.
The score calibration failure is a training data problem, not a model capacity problem.
Fine-tuning on Reddit-FR should directly re-calibrate output scores for informal French hate speech.

**Why this is cheap:**
- XLM-RoBERTa-base = ~270M params (10× smaller than SG-2b)
- No gradient checkpointing needed: fits comfortably on A5000
- ~5 min training per epoch on Reddit-FR (n=3,665 train samples)
- LoRA or full fine-tuning both viable

**Training objective choice:**
- Detoxify originally uses **MSE regression** (targets: 0.0 or 1.0) — preserves continuous score output ideal for threshold analysis
- Standard **cross-entropy classification** is also valid — simplifies code, still outputs probability

MSE regression is preferred for Tier 1 use because it maintains score calibration semantics.

**What success looks like:**
After fine-tuning, run `analyze_threshold_tier1.py` with the fine-tuned model. If the score
distributions show meaningful bimodal separation (safe cluster near 0, hateful cluster near 1),
T1_FNR at mid-deferral should drop from 37% to something acceptable (target: < 15%).

**Script to write:** `code/phase4_two_tier/finetune_detoxify_tier1.py`
(can reuse dataset loaders from `analyze_threshold_tier1.py`; training loop similar to
`code/phase2_lora/finetune_lora.py` but without PEFT — plain HuggingFace Trainer or manual loop)

---

## 5. Phase 2 Context (Tier 2 candidate — pending confirmation)

From Phase 2 LoRA fine-tuning results (`code/docs/PHASE2_LORA_RESULTS_REPORT.md`):

**Current best Tier 2 candidate: SG-2b Reddit-FR LoRA**

| Model | Dataset | Baseline F1 | LoRA F1 | Δ | Mechanism |
|-------|---------|:-----------:|:-------:|:---:|-----------|
| SG-2b | Reddit-FR | 0.335 | **0.662** | +0.327 | Recall surge (TPR 0.208→0.623) |

SG-2b started with near-zero recall on Reddit-FR (calibrated for formal English safety content,
treats informal French as safe by default). One epoch of LoRA on 3,674 Reddit-FR samples fixed
this. This is the strongest completed Phase 2 result.

**One experiment is now complete; one remains pending:**

1. ✅ **Reddit-FR adapter — full 8-dataset generalisation eval** (`lora_full_reddit_fr/`, completed 2026-04-17):
   SG-2b HC-FR regression only **−0.021** (0.858→0.837) — well within the < 0.05 threshold.
   Additional gains on Reddit-EN (+0.284) and ToxiGen (+0.234) confirm informal-register transfer.
   **SG-2b Reddit-FR LoRA now meets both Tier 2 selection criteria.**
   Full results: `results/phase2_eval/lora_full_reddit_fr/summary.txt`

2. ✅ **Joint adapter (FHS + Reddit-FR combined training)** — complete (2026-04-18).
   SG-2b joint: FHS F1=0.633 (+0.099 vs single FHS LoRA, recall-driven), Reddit-FR F1=0.632
   (−0.030 vs single Reddit-FR LoRA). Reddit-FR signal is diluted by the 3.5× larger FHS corpus.
   Joint adapter does **not** surpass single Reddit-FR LoRA on Reddit-FR.
   Full results: `results/phase2_eval/{french_hate_superset,reddit_fr}/lora_joint/summary.txt`

**SG-2b Reddit-FR LoRA (F1=0.662) is the confirmed Tier 2 model.** Joint adapter does not beat
it. Both Phase 2 pending experiments are resolved — proceed with Phase 4 implementation.

Current best adapter location: `~/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best/`

---

## 6. Full Two-Tier Architecture Target

```
Input text (French)
        │
        ▼
┌─────────────────────┐
│  Tier 1 (Candidate) │  ← what Phase 4 is finding
│  Fine-tuned         │  270M params, CPU, ~5ms
│  Detoxify-M /       │
│  other lightweight  │
└────────┬────────────┘
         │
  ┌──────┴──────┐
  │             │
score < T_low  T_low ≤ score ≤ T_high  score > T_high
  │             │                              │
PASS SAFE    DEFER TO TIER 2              FLAG UNSAFE
             │
             ▼
  ┌─────────────────────┐
  │  Tier 2             │  ← SG-2b Reddit-FR LoRA (confirmed)
  │  ShieldGemma-2b     │  2.6B params, GPU, ~30ms
  │  Reddit-FR LoRA     │
  └─────────────────────┘
```

**End-to-end evaluation metric:** combined system F1 / FNR / FPR on Reddit-FR, compared to
running SG-2b alone (the "no Tier 1" baseline). The two-tier design is justified if it
reduces compute cost while maintaining acceptable F1 and FNR.

---

## 7. Files Reference

| File | Purpose |
|------|---------|
| `code/phase4_two_tier/analyze_threshold_tier1.py` | Tier 1 threshold analysis, any model |
| `code/phase4_two_tier/finetune_detoxify_tier1.py` | **TO WRITE** — fine-tune Detoxify-M on Reddit-FR |
| `code/phase4_two_tier/evaluate_two_tier.py` | **TO WRITE** — end-to-end Tier 1 + Tier 2 evaluation |
| `code/phase4_two_tier/slurm/` | **TO WRITE** — SLURM jobs for all Phase 4 scripts |
| `code/docs/THRESHOLD_ANALYSIS_REPORT.md` | Full Detoxify-M threshold analysis results |
| `code/docs/PHASE2_LORA_RESULTS_REPORT.md` | Phase 2 LoRA results (Tier 2 confirmed) |
| `results/threshold_analysis/` | Detoxify-M raw scores + figures |
| `results/tier1_comparison/` | Multi-model Tier 1 comparison output (to be generated) |

---

## 8. Recommended First Steps

1. **Run Track A** — test `unitary/multilingual-toxic-xlm-roberta` via `analyze_threshold_tier1.py`.
   This takes ~1h on CPU (no GPU needed). Compare score distributions directly with Detoxify-M.

2. **Write and run `finetune_detoxify_tier1.py`** (Track B) — fine-tune Detoxify-M on Reddit-FR
   with MSE regression objective, then re-run threshold analysis on the fine-tuned checkpoint.

3. **Compare all Tier 1 candidates** — pick the one with lowest T1_FNR at 25% deferral on Reddit-FR.

4. **Write `evaluate_two_tier.py`** — pair the best Tier 1 with SG-2b Reddit-FR LoRA, run
   end-to-end evaluation, compare to SG-2b-alone baseline.
