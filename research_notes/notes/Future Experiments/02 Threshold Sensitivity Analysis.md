# Threshold Sensitivity Analysis

**Status:** not started | **Priority:** high — needed before thesis conclusions

---

## Problem

All Phase 1 and Phase 2 evaluations use a fixed threshold of 0.5. This is a significant
simplification. The optimal threshold for Shareish depends on the moderation cost:
- A high FPR means moderators waste time reviewing safe content
- A high FNR means harmful content reaches users

These costs are not symmetric, and 0.5 is unlikely to be the right operating point.

## Experiment

For each model × dataset pair, sweep threshold in `[0.1, 0.9]` (step 0.05) and compute:
- Precision, Recall, F1 at each threshold
- ROC curve (TPR vs. FPR)
- Precision-Recall curve (more informative on imbalanced datasets)
- AUC-ROC and AUC-PR as threshold-independent summary metrics

## Blocker

**Per-sample probabilities are not currently saved** — only binary predictions are stored
in the result JSON files. Re-running inference is required, or the evaluation loop must
be modified to dump raw scores.

Recommended fix: add a `scores` list to the `Result` dataclass and save it alongside the
binary predictions. One-time change to `run_full_baseline_v3.py` and
`run_full_baseline_lora.py`.

## Why This Matters for the Thesis

The threshold analysis directly supports the deployability argument:
- A model with lower F1 at 0.5 may outperform at a Shareish-optimal threshold
- The AUC-PR is the right metric when positive class (unsafe) is rare in production
- Presenting ROC curves shows the full operating range, not a single point

## References
- Phase 1 results: `results/full_baseline_v3/summary.json`
