# Results Tracker

Status snapshot as of **2026-04-07**.

**Phase 1 complete.** `full_baseline_v3/` has 10 models × 8 datasets × 3 multi-runs (std≤0.01). `hatecheck_analysis/` has all 10 models × EN + FR with v3 methods. See RESULTS_PHASE1_BASELINE.md for full analysis.

**Phase 2 fair eval complete.** `phase2_eval/` contains held-out 20% test set results for LG-1B and SG-2b on FR-Hate and Reddit-FR baselines. LoRA fine-tuning improves FR-Hate F1: LG-1B +0.186 (0.371→0.557), SG-2b +0.121 (0.413→0.534). Gains are precision-driven (TNR up sharply), not recall-driven (TPR essentially flat). See PHASE2_LORA_RESULTS_REPORT.md for full analysis.

---

## Phase 2 Status

| Item | Status | Notes |
|------|--------|-------|
| LG-1B × FHS training | ✅ done | best=epoch1, val_loss=0.1903, adapter at `lora_adapters/llama_guard_1b/french_hate_superset/best` |
| LG-1B × Reddit-FR training | ✅ done | best=epoch1, val_loss=0.3031, adapter at `lora_adapters/llama_guard_1b/reddit_fr/best` |
| SG-2b × FHS training | ✅ done | best=epoch1, val_loss=0.1862, adapter at `lora_adapters/shieldgemma_2b/french_hate_superset/best` |
| SG-2b × Reddit-FR training | ❌ failed | Only test_set.json saved, no adapter — needs resubmit with `--epochs 1` |
| Initial eval on full 8 datasets (FHS adapter) | ✅ done | `full_baseline_lora_french_hate_superset/` — FR-Hate numbers inflated ⚠️ (training data in test set) |
| **Fair eval on held-out test sets (FR-Hate)** | ✅ **done** | `phase2_eval/french_hate_superset/` — LG-1B: 0.371→0.557 (+0.186), SG-2b: 0.413→0.534 (+0.121) |
| **Fair eval baseline on Reddit-FR** | ✅ **done** | `phase2_eval/reddit_fr/baseline/` — LG-1B: 0.425, SG-2b: 0.335 |
| Reddit-FR LoRA adapter fair eval | ❌ pending | LG-1B adapter exists; SG-2b adapter missing. Need to submit separate eval job. |
| Full 8-dataset eval with LoRA adapters | ❌ pending | HC-FR regression unknown — need to run `run_full_baseline_lora.py` to confirm |

> All 3 training runs overfit after epoch 1 — future retrains should use `--epochs 1`.

---

## Submitted Jobs

| Job ID | Partition | Script | Type | Status |
|--------|-----------|--------|------|--------|
| 3788499 | a5000 | `run_full_baseline.sbatch` | Single baseline run | Submitted (PD/Priority) |
| 3788500 | a5000 | `run_hatecheck_analysis.sbatch` | HateCheck full (all models) | Submitted (PD/Priority) |
| 3788190 | all | `run_full_baseline.sbatch` | Single baseline run | Submitted (PD/Resources) |
| 3788501 | all | `run_hatecheck_light.sbatch` | HateCheck light (VRAM-filtered) | Submitted (PD/Priority) |

> **Note:** `run_full_baseline_multi.sbatch` (multi-run) was NOT submitted yet — cluster too crowded.
> Submit it once the queue clears.

---

## Current Results State

See **[RESULTS_PHASE1_BASELINE.md](RESULTS_PHASE1_BASELINE.md)** for all F1 tables, deployability numbers, and observations.

As of 2026-03-29: **Phase 1 complete.**
- `full_baseline_v3/`: 10 models × 8 datasets (flat) + 3 multi-runs (run_1, run_2, run_3) — all complete.
- `hatecheck_analysis/`: 10 models × HateCheck EN + FR — all complete with v3 inference methods.
- Primary result path: `code/results/full_baseline_v3/`

---

## Where Results Will Be

```
~/code/results/
├── full_baseline/                  ← single baseline runs
│   ├── hatecheck_fr/
│   ├── hatecheck_en/
│   ├── french_hate_superset/
│   ├── toxigen/
│   ├── openai/
│   ├── civil_comments/
│   ├── reddit_en/
│   ├── reddit_fr/
│   ├── summary.txt                 ← main results table + F1 matrix
│   └── summary.json
│
├── hatecheck_analysis/             ← functionality breakdown
│   ├── hatecheck_en/
│   ├── hatecheck_fr/
│   ├── summary.txt                 ← per-model functionality bars + heatmap
│   └── summary.json
│
└── (full_baseline/run_1/ etc.)     ← only after multi-run is submitted
```

Expected when complete: **80 files** (10 models × 8 datasets) in full_baseline, **20 files** (10 models × 2 languages) in hatecheck_analysis.

---

## When You Come Back

### 1. Check if jobs finished

```bash
ssh alan
squeue -u $USER
```

If the queue is empty, jobs are done (or failed).

### 2. Check for errors

```bash
more ~/code/logs/full_baseline/<JOBID>.err
more ~/code/logs/hatecheck_analysis/<JOBID>.err
```

Common issues:
- `CUDA out of memory` → VRAM guard should have caught this, but double-check
- `FileNotFoundError` → a dataset path doesn't exist on the cluster
- Job hit time limit → results are partial; resubmit (checkpointing resumes)
- CitizenLab torch error → needs `torch >= 2.6` or safetensors conversion

### 3. Check what actually completed

```bash
find ~/code/results/full_baseline -name "*.json" | grep -v summary | sort
find ~/code/results/hatecheck_analysis -name "*.json" | grep -v summary | sort
```

### 4. Read the summaries

```bash
cat ~/code/results/full_baseline/summary.txt
cat ~/code/results/hatecheck_analysis/summary.txt
```

### 5. Retrieve results locally

```bash
rsync -avz --progress alan:~/code/results/ ./code/results/
```

---

## What to Do Next (Phase 2)

**Phase 1 is complete. Phase 2 = LoRA fine-tuning.**

### Fine-tuning candidates (ranked)
1. **Llama-Guard-3-1B** — best VRAM/accuracy trade-off; native safety tuning format; strong EN/FR baseline (0.816/0.674).
2. **ShieldGemma-2b** — high accuracy (0.902/0.858), fast (24ms); counter-speech weakness is the target for fine-tuning.
3. **detoxify-multilingual** — if the goal is minimal VRAM and bilingual support; FR-Hate/Reddit gaps are fine-tuning targets.

### Threshold sensitivity analysis (before fine-tuning)
ShieldGemma's TPR/TNR imbalance (TPR 0.97, TNR 0.60) may be correctable with a higher threshold (0.6–0.8). Run sweep before committing to fine-tuning:
```bash
python code/run_full_baseline_v3.py \
    --output_dir ~/code/results/threshold_sweep \
    --models shieldgemma_2b,shieldgemma_9b \
    --datasets hatecheck_en,hatecheck_fr \
    --threshold 0.6  # repeat for 0.7, 0.8
```

### Dataset inspection (FR-Hate / Reddit underperformance)
All models score poorly on FR-Hate and Reddit (best F1 ≈ 0.4). Before fine-tuning on these, check whether the issue is label noise or domain shift:
```bash
python code/experiments/dataset_experiments/explore_datasets.py
```

### Shareish two-tier architecture evaluation
Planned: detoxify-multilingual as fast pre-filter, Llama-Guard-3-1B for edge cases. Needs end-to-end evaluation script.

---

## Key Reminders

- Results are **saved incrementally** — a partial run is never wasted
- The **VRAM guard** skips models silently on weak GPUs → check for `SKIPPED` lines in logs
- The **a5000 jobs** cover large models; the **all-partition jobs** cover light models on any free node
- **CitizenLab** may fail with a torch version error (CVE-2025-32434) — needs `torch >= 2.6` or the model needs to be converted to safetensors format
- **ShieldGemma** results at default threshold=0.5 are misleading — needs threshold sweep before drawing conclusions
