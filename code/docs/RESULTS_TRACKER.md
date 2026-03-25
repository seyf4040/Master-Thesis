# Results Tracker

Status snapshot as of **2026-03-25**. `run_1/` is now mostly complete (9 models × 7 datasets). See RESULTS_SUMMARY.md for full analysis.

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

See **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** for all F1 tables, deployability numbers, and observations.

As of 2026-03-25: `run_1/` has 9 models × 7 datasets (all except reddit_fr). Mistral and ShieldGemma missing reddit_en. CitizenLab missing everywhere (torch CVE). reddit_fr missing everywhere.

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

## What to Do Next (After Results Are In)

### If multi-run was not submitted yet
Submit it once the cluster is less crowded — needed for mean ± std statistics:
```bash
sbatch code/slurm_jobs/run_full_baseline_multi.sbatch  # default N_RUNS=3
```

### If some (model, dataset) pairs are missing
Check which ones are missing and resubmit — checkpointing skips already-done pairs:
```bash
sbatch code/slurm_jobs/run_full_baseline.sbatch
```

### If large models (Llama-8B, ShieldGemma-9B, Mistral-7B) are missing
They were likely skipped by the VRAM guard on a 1080Ti node.
The a5000 jobs (3788499, 3788500) should cover them — check their logs.
If they also failed, resubmit specifically:
```bash
python ~/code/run_full_baseline.py \
    --output_dir ~/code/results/full_baseline \
    --models llama_guard_8b,shieldgemma_9b,mistral_7b \
    --datasets all \
    ... (other paths)
```

### Analysis to do once results are complete

1. **Compare full baseline summary** (`summary.txt`) → identify top models per dataset
2. **Read HateCheck report** → note which functionalities each model struggles with
3. **Apply deployability filter** → eliminate models requiring >11GB VRAM for production
4. **Investigate ShieldGemma** → run threshold sweep (0.1–0.9) before concluding it's broken
5. **Submit multi-run** → get mean ± std to confirm stability of top candidates
6. **Aggregate multi-run** → `python code/aggregate_runs.py --results_dir ~/code/results/full_baseline`
7. **Select models for fine-tuning** → based on above analysis

---

## Key Reminders

- Results are **saved incrementally** — a partial run is never wasted
- The **VRAM guard** skips models silently on weak GPUs → check for `SKIPPED` lines in logs
- The **a5000 jobs** cover large models; the **all-partition jobs** cover light models on any free node
- **CitizenLab** may fail with a torch version error (CVE-2025-32434) — needs `torch >= 2.6` or the model needs to be converted to safetensors format
- **ShieldGemma** results at default threshold=0.5 are misleading — needs threshold sweep before drawing conclusions
