#!/bin/bash
#SBATCH --job-name=eval-tier1-gen
#SBATCH --output=/home/sural/code/logs/finetune_tier1/%j.out
#SBATCH --error=/home/sural/code/logs/finetune_tier1/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a5000
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Phase 4 — Tier 1 Generalisation Evaluation (all 8 Phase 1 datasets)
#
# Evaluates all fine-tuned Tier 1 checkpoints on all 8 datasets from Phase 1:
#   French (primary): HateCheck-FR, French Hate Superset (capped 5k), Reddit-FR (full 5119)
#   English (generalisation): HateCheck-EN, Reddit-EN
#   Other (generalisation): ToxiGen, OpenAI moderation, Civil Comments
#
# FHS is capped at 5,000 samples (--max_samples) to control runtime on CPU.
# Full FHS evaluation is already done in Phase 1 baseline and Phase 2 reports.
#
# Checkpoints evaluated:
#   - Pretrained: unitary/multilingual-toxic-xlm-roberta (raw_scores.json already exists)
#   - Track B: tier1_detoxify_finetuned/best    (3 epochs, baseline fine-tune)
#   - 2a:      tier1_finetuned_e10/best         (10 epochs, hard labels)
#   - 2b:      tier1_finetuned_soft/best        (5 epochs, soft labels)
#   - 2c:      tier1_finetuned_synthetic/best   (5 epochs, synthetic data) ← primary
#   - 2d:      tier1_finetuned_2d/best          (2 epochs, soft + synthetic)
#
# Resumption: pretrained raw_scores.json already saved — inference skipped via --scores_json.
# Output: ~/code/results/tier1_generalisation/{checkpoint_name}/
#
# Author: Ural Seyfullah

echo "=========================================="
echo "Phase 4 — Tier 1 Generalisation Evaluation"
echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "=========================================="

source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

SCRIPT="$HOME/code/phase4_two_tier/analyze_threshold_tier1.py"
CACHE_DIR="$HOME/datasets/cache"
REDDIT_FR="$HOME/datasets/reddit/balanced/data-fr/test-fr.csv"
REDDIT_EN="$HOME/datasets/reddit/balanced/data-en/test-en.csv"
TOXIGEN="$HOME/datasets/toxigen/toxigen_train.jsonl"
OPENAI="$HOME/datasets/openai/samples-1680.jsonl"
OUTPUT_BASE="$HOME/code/results/tier1_generalisation"

mkdir -p "$HOME/code/logs/finetune_tier1"

# Common dataset flags for all runs.
# FHS capped at 5000 to keep CPU inference under ~6 min/checkpoint.
DATASET_FLAGS="
    --cache_dir             $CACHE_DIR
    --reddit_fr_path        $REDDIT_FR
    --reddit_en_path        $REDDIT_EN
    --toxigen_path          $TOXIGEN
    --openai_path           $OPENAI
    --include_hatecheck_en
    --include_civil_comments
    --max_samples_fhs       5000
    --max_samples_toxigen   5000
    --max_samples_civil     2000
    --t_step                0.05
    --device                cpu
"

run_eval() {
    local label="$1"
    local model_id="$2"
    local scores_flag="${3:-}"  # optional: "--scores_json path" to skip inference

    echo ""
    echo "=========================================="
    echo "Evaluating: $label"
    echo "=========================================="
    python "$SCRIPT" \
        --model          hf_classifier \
        --hf_model_id    "$model_id"   \
        --hf_toxic_label toxic          \
        --output_dir     "$OUTPUT_BASE/$label" \
        $DATASET_FLAGS \
        $scores_flag
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "$label complete: $(date)"
    else
        echo "WARNING: $label exited with code $rc — continuing"
    fi
}

# ── Pretrained baseline — inference already done, skip to analysis ────────────
PRETRAINED_SCORES="$OUTPUT_BASE/pretrained/unitary_multilingual-toxic-xlm-roberta/raw_scores.json"
if [ -f "$PRETRAINED_SCORES" ]; then
    echo "Pretrained raw_scores.json found — skipping inference, running analysis only."
    run_eval "pretrained" \
        "unitary/multilingual-toxic-xlm-roberta" \
        "--scores_json $PRETRAINED_SCORES"
else
    run_eval "pretrained" "unitary/multilingual-toxic-xlm-roberta"
fi

# ── Track B (original 3-epoch fine-tune) ─────────────────────────────────────
run_eval "track_b_3ep" \
    "$HOME/code/results/tier1_detoxify_finetuned/best"

# ── 2a (10 epochs, hard labels) ──────────────────────────────────────────────
run_eval "2a_e10" \
    "$HOME/code/results/tier1_finetuned_e10/best"

# ── 2b (5 epochs, soft labels) ───────────────────────────────────────────────
run_eval "2b_soft" \
    "$HOME/code/results/tier1_finetuned_soft/best"

# ── 2c (5 epochs, synthetic data) ────────────────────────────────────────────
run_eval "2c_synthetic" \
    "$HOME/code/results/tier1_finetuned_synthetic/best"

# ── 2d (2 epochs, soft + synthetic) ─────────────────────────────────────────
run_eval "2d_combined" \
    "$HOME/code/results/tier1_finetuned_2d/best"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Generalisation evaluation complete: $(date)"
echo ""
echo "F1 summary across all checkpoints (best single-threshold):"
python3 -c "
import json, glob
from pathlib import Path

checkpoints = ['pretrained', 'track_b_3ep', '2a_e10', '2b_soft', '2c_synthetic', '2d_combined']
datasets    = ['hatecheck_fr', 'hatecheck_en', 'french_hate_superset',
               'reddit_fr', 'reddit_en', 'toxigen', 'openai', 'civil_comments']

print(f'  {\"Checkpoint\":<20} ' + '  '.join(f'{d[:10]:>10}' for d in datasets))
print('  ' + '-' * (20 + 12 * len(datasets)))
for ckpt in checkpoints:
    paths = glob.glob(f'$OUTPUT_BASE/{ckpt}/*/threshold_analysis.json')
    if not paths:
        print(f'  {ckpt:<20}  (no results)')
        continue
    d = json.load(open(paths[0]))
    st = d.get('single_threshold', {})
    row = f'  {ckpt:<20}'
    for ds in datasets:
        b = st.get(ds, {}).get('best', {})
        f1 = b.get('f1', None)
        row += f'  {f1:.3f}' if f1 is not None else '       ?'
    print(row)
"
echo "=========================================="
