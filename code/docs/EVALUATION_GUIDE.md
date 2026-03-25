# Evaluation Guide

Scripts for running the full baseline evaluation and HateCheck functionality analysis.

---

## Overview

| Script | Purpose |
|--------|---------|
| `run_full_baseline.py` | All 10 models × all 8 datasets |
| `run_hatecheck_analysis.py` | All 10 models × HateCheck EN + FR with per-functionality breakdown |
| `aggregate_runs.py` | Computes mean ± std across multiple baseline runs |

| SLURM job | Purpose |
|-----------|---------|
| `slurm_jobs/run_full_baseline.sbatch` | Single baseline run |
| `slurm_jobs/run_full_baseline_multi.sbatch` | Multiple baseline runs (for statistics) |
| `slurm_jobs/run_hatecheck_analysis.sbatch` | HateCheck analysis |

---

## Prerequisites

On the cluster, before submitting any job:

```bash
# Create all required log directories
mkdir -p ~/code/logs/full_baseline
mkdir -p ~/code/logs/hatecheck_analysis
```

---

## 1. Single Baseline Run

Tests all 10 models on all 8 datasets once.
Results saved to `~/code/results/full_baseline/`.

### Submit

```bash
sbatch code/slurm_jobs/run_full_baseline.sbatch
```

### Monitor

```bash
squeue -u $USER
tail -f ~/code/logs/full_baseline/<JOBID>.out
```

### Output structure

```
results/full_baseline/
├── hatecheck_fr/
│   ├── detoxify-multilingual.json
│   ├── KoalaAI-Text-Moderation.json
│   └── ...
├── hatecheck_en/
├── openai/
├── toxigen/
├── ...
└── summary.txt          ← main results table + F1 matrix
└── summary.json         ← machine-readable full results
```

### Notes

- **Checkpointing**: if a job is interrupted and resubmitted, already-completed (model, dataset) pairs are skipped automatically.
- **VRAM guard**: models requiring more VRAM than available are skipped with a warning instead of freezing.
- **Partition**: always ensure `#SBATCH --partition=a5000` is set. The 1080Ti nodes (compute-01 to 04) will freeze on large models.

---

## 2. Multiple Baseline Runs (Statistical Reliability)

Runs the full evaluation N times, each in its own subdirectory (`run_1/`, `run_2/`, ...).
Automatically aggregates results at the end to produce mean ± std per metric.

### Configure number of runs

Edit the `N_RUNS` variable at the top of the sbatch file:

```bash
nano code/slurm_jobs/run_full_baseline_multi.sbatch
# Change: N_RUNS=3
```

### Submit

```bash
sbatch code/slurm_jobs/run_full_baseline_multi.sbatch
```

### Output structure

```
results/full_baseline/
├── run_1/
│   ├── hatecheck_fr/
│   ├── openai/
│   └── ...
├── run_2/
├── run_3/
├── aggregate_summary.txt    ← mean ± std tables
└── aggregate_summary.json
```

### Aggregate manually (if the job was interrupted before aggregation)

```bash
python code/aggregate_runs.py \
    --results_dir ~/code/results/full_baseline

# Specific models or datasets only:
python code/aggregate_runs.py \
    --results_dir ~/code/results/full_baseline \
    --models      detoxify_multilingual,koalaai \
    --datasets    openai,hatecheck_en
```

### What the aggregate report contains

- **Per-dataset table**: mean ± std for accuracy, F1, TPR, FPR per model
- **Stability ranking**: models sorted by F1 std (lower = more consistent)
- **F1 mean matrix**: models × datasets
- **F1 std matrix**: variance across runs per (model, dataset) pair

---

## 3. HateCheck Functionality Analysis

Evaluates all models using HateCheck's built-in test categories, exposing specific
strengths and weaknesses per model (e.g. "misses implicit hate", "incorrectly flags counter-speech").

Runs on both HateCheck English (`Paul/hatecheck`) and French (`Paul/hatecheck-french`).

### Submit

```bash
sbatch code/slurm_jobs/run_hatecheck_analysis.sbatch
```

### Output structure

```
results/hatecheck_analysis/
├── hatecheck_en/
│   ├── detoxify-multilingual.json
│   ├── KoalaAI-Text-Moderation.json
│   └── ...
├── hatecheck_fr/
│   └── ...
├── summary.txt     ← per-model functionality breakdown + cross-model heatmap
└── summary.json    ← full results + heatmap matrix (for plotting)
```

### Reading the report

The text report has two sections per model:

```
MODEL: KoalaAI-Text-Moderation  (Acc=0.91, F1=0.84)

  HATEFUL functionalities (detection rate ↑ better):
  derog_neg_emote_h         24  100.0%  ██████████   ← catches explicit hate
  derog_impl                24   79.2%  ████████░░   ← misses some implicit hate
  threat_dir                12  100.0%  ██████████

  NON-HATEFUL functionalities (pass rate ↑ better):
  counter_qa                24  100.0%  ██████████   ← never flags counter-speech
  reclaimed_slur            12   83.3%  ████████░░   ← occasionally flags reclaimed slurs
```

- **Hateful functionalities** → detection rate (TPR). Low = the model misses this type of hate.
- **Non-hateful functionalities** → pass rate (TNR). Low = the model incorrectly flags this type of benign content.

The report also includes a **target group breakdown** (women, immigrants, LGBTQ+, etc.)
showing whether performance varies across protected groups.

At the end of the report there is a **cross-model heatmap matrix** (model × functionality)
useful for generating figures in the thesis.

### Run on fast models only (quick test, ~30 min on a5000)

```bash
python code/run_hatecheck_analysis.py \
    --output_dir ~/code/results/hatecheck_analysis \
    --cache_dir  ~/datasets/cache \
    --models     detoxify_multilingual,detoxify_unbiased,koalaai,ethicaleye,citizenlab \
    --datasets   both
```

### Run on one language only

```bash
# French only
python code/run_hatecheck_analysis.py \
    --output_dir ~/code/results/hatecheck_analysis \
    --datasets   fr

# English only
python code/run_hatecheck_analysis.py \
    --output_dir ~/code/results/hatecheck_analysis \
    --datasets   en
```

---

## 4. Retrieve Results Locally

After jobs complete, sync results from the cluster:

```bash
rsync -avz --progress alan:~/code/results/ ./code/results/
```

---

## 5. Quick Reference — CLI Arguments

### `run_full_baseline.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | required | Where to save results |
| `--datasets` | `all` | Comma-separated or `all`. Keys: `hatecheck_fr`, `hatecheck_en`, `french_hate_superset`, `toxigen`, `openai`, `civil_comments`, `reddit_en`, `reddit_fr` |
| `--models` | `all` | Comma-separated or `all`. Keys: `detoxify_multilingual`, `detoxify_unbiased`, `koalaai`, `ethicaleye`, `citizenlab`, `llama_guard_1b`, `llama_guard_8b`, `shieldgemma_2b`, `shieldgemma_9b`, `mistral_7b` |
| `--run_id` | none | Run number for multi-run mode (saves to `output_dir/run_N/`) |
| `--max_samples_toxigen` | 5000 | Max samples from ToxiGen |
| `--max_samples_civil` | 5000 | Max samples from Civil Comments |
| `--threshold` | 0.5 | Toxicity probability threshold |
| `--no_skip` | false | Rerun even if result already exists |

### `run_hatecheck_analysis.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | required | Where to save results |
| `--models` | `all` | Same model keys as above |
| `--datasets` | `both` | `both`, `en`, or `fr` |
| `--threshold` | 0.5 | Toxicity probability threshold |
| `--no_skip` | false | Rerun even if result already exists |

### `aggregate_runs.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | required | Directory containing `run_1/`, `run_2/`, ... |
| `--models` | all | Filter to specific models |
| `--datasets` | all | Filter to specific datasets |
