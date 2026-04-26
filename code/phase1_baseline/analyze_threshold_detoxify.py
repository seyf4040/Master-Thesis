#!/usr/bin/env python3
"""
analyze_threshold_detoxify.py — Two-tier threshold analysis for detoxify-multilingual

Collects raw toxicity scores (not binarized) across French datasets, then:

  1. Score distribution plots  — safe vs unsafe histograms per dataset
  2. Single-threshold sweep    — F1/precision/recall at every T in [0,1]
                                 → optimal single threshold per dataset
  3. Two-threshold sweep       — for every (T_low, T_high) pair:
       - coverage        = fraction Tier 1 handles (score < T_low OR score > T_high)
       - tier1_fnr       = P(hateful | score < T_low)   ← hateful leaking through as safe
       - tier1_fpr       = P(safe    | score > T_high)  ← safe wrongly flagged as unsafe
       - deferral_rate   = 1 - coverage
     Outputs contour plots and a recommended-operating-points table.

Usage:
    python code/phase1_baseline/analyze_threshold_detoxify.py \\
        --output_dir ~/code/results/threshold_analysis \\
        --cache_dir   ~/datasets/cache \\
        --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv

    # Skip model inference if scores already saved:
    python ... --scores_json ~/code/results/threshold_analysis/raw_scores.json

Author: Ural Seyfullah
"""

import gc
import json
import argparse
import random
import time
from pathlib import Path
from typing import List, Dict

import numpy as np


# ── Dataset loaders (copied from run_full_baseline_v3.py) ─────────────────────

def load_hatecheck_fr(cache_dir: str) -> List[Dict]:
    from datasets import load_dataset as _ld
    ds = _ld("Paul/hatecheck-french", cache_dir=cache_dir)
    return [
        {'text': item['test_case'], 'label': 1 if item['label_gold'] == 'hateful' else 0}
        for item in ds['test']
    ]


def load_french_hate_superset(cache_dir: str) -> List[Dict]:
    from datasets import load_dataset as _ld
    ds = _ld("manueltonneau/french-hate-speech-superset", cache_dir=cache_dir)
    safe_strings = {'none', 'normal', 'non-hateful', 'non_hateful', '0', 'nohate'}
    samples = []
    for item in ds['train']:
        text = item.get('text', '') or ''
        labels_raw = item.get('labels')
        if not text.strip() or labels_raw is None:
            continue
        if isinstance(labels_raw, list):
            label = 1 if any(str(l).strip().lower() not in safe_strings for l in labels_raw) else 0
        else:
            label = 0 if str(labels_raw).strip().lower() in safe_strings else 1
        samples.append({'text': text, 'label': label})
    return samples


def load_reddit_fr(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]


# ── Inference ─────────────────────────────────────────────────────────────────

def collect_scores(samples: List[Dict], device: str) -> List[Dict]:
    """Run detoxify-multilingual and return list of {score, label} dicts."""
    from detoxify import Detoxify
    from tqdm import tqdm
    model = Detoxify('multilingual', device=device)
    results = []
    for s in tqdm(samples, desc="scoring", unit="sample", leave=False):
        try:
            score = float(model.predict(s['text'])['toxicity'])
            results.append({'score': score, 'label': s['label']})
        except Exception as e:
            print(f"  Warning: skipped sample — {e}")
    del model
    gc.collect()
    return results


# ── Analysis helpers ──────────────────────────────────────────────────────────

def single_threshold_sweep(scored: List[Dict], thresholds: np.ndarray) -> Dict:
    """For each threshold T, compute classification metrics."""
    labels  = np.array([s['label'] for s in scored])
    scores  = np.array([s['score'] for s in scored])
    n_pos   = labels.sum()
    n_neg   = len(labels) - n_pos

    rows = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        tpr  = tp / n_pos if n_pos > 0 else 0.0
        tnr  = tn / n_neg if n_neg > 0 else 0.0
        rows.append({'t': float(t), 'f1': f1, 'precision': prec, 'recall': rec,
                     'tpr': tpr, 'tnr': tnr, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})

    best = max(rows, key=lambda r: r['f1'])
    return {'sweep': rows, 'best': best, 'n_pos': int(n_pos), 'n_neg': int(n_neg)}


def two_threshold_sweep(scored: List[Dict],
                        t_lows: np.ndarray,
                        t_highs: np.ndarray) -> Dict:
    """
    For each (T_low, T_high) pair compute:
      coverage      — fraction of samples Tier 1 handles (no deferral)
      deferral_rate — fraction sent to Tier 2
      tier1_fnr     — among samples Tier 1 passes as safe (score < T_low),
                       what fraction are actually hateful?
      tier1_fpr     — among samples Tier 1 flags as unsafe (score > T_high),
                       what fraction are actually safe?
    """
    labels = np.array([s['label'] for s in scored])
    scores = np.array([s['score'] for s in scored])
    n = len(labels)

    grid = []
    for t_low in t_lows:
        for t_high in t_highs:
            if t_low >= t_high:
                continue
            safe_mask   = scores < t_low
            unsafe_mask = scores > t_high
            defer_mask  = ~safe_mask & ~unsafe_mask

            n_safe   = safe_mask.sum()
            n_unsafe = unsafe_mask.sum()
            n_defer  = defer_mask.sum()

            # FNR of the "pass as safe" bin: hateful that slip through
            hateful_in_safe = (safe_mask & (labels == 1)).sum()
            tier1_fnr = float(hateful_in_safe / n_safe) if n_safe > 0 else 0.0

            # FPR of the "flag as unsafe" bin: safe wrongly flagged
            safe_in_unsafe = (unsafe_mask & (labels == 0)).sum()
            tier1_fpr = float(safe_in_unsafe / n_unsafe) if n_unsafe > 0 else 0.0

            coverage     = float((n_safe + n_unsafe) / n)
            deferral     = float(n_defer / n)

            grid.append({
                't_low':         float(t_low),
                't_high':        float(t_high),
                'coverage':      coverage,
                'deferral_rate': deferral,
                'tier1_fnr':     tier1_fnr,
                'tier1_fpr':     tier1_fpr,
                'n_safe':        int(n_safe),
                'n_unsafe':      int(n_unsafe),
                'n_defer':       int(n_defer),
            })

    return grid


def find_operating_points(grid: List[Dict]) -> List[Dict]:
    """
    Select representative (T_low, T_high) operating points:
      - Low deferral  (~10%): Tier 1 handles almost everything
      - Mid deferral  (~25%): balanced trade-off
      - High deferral (~50%): conservative Tier 1, Tier 2 handles half
    At each deferral target, pick the pair minimising tier1_fnr + tier1_fpr.
    """
    targets = [('low_deferral',  0.10),
               ('mid_deferral',  0.25),
               ('high_deferral', 0.50)]
    points = []
    for label, target in targets:
        candidates = sorted(grid,
                            key=lambda r: abs(r['deferral_rate'] - target))[:20]
        best = min(candidates, key=lambda r: r['tier1_fnr'] + r['tier1_fpr'])
        points.append({'label': label, 'target_deferral': target, **best})
    return points


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_score_distributions(scored_by_dataset: Dict, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, len(scored_by_dataset), figsize=(5 * len(scored_by_dataset), 4),
                             sharey=False)
    if len(scored_by_dataset) == 1:
        axes = [axes]

    for ax, (name, scored) in zip(axes, scored_by_dataset.items()):
        safe_scores    = [s['score'] for s in scored if s['label'] == 0]
        unsafe_scores  = [s['score'] for s in scored if s['label'] == 1]
        bins = np.linspace(0, 1, 51)
        ax.hist(safe_scores,   bins=bins, alpha=0.6, label=f'safe (n={len(safe_scores)})',
                color='steelblue', density=True)
        ax.hist(unsafe_scores, bins=bins, alpha=0.6, label=f'hateful (n={len(unsafe_scores)})',
                color='tomato', density=True)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='T=0.5 (default)')
        ax.set_title(name.replace('_', '\n'), fontsize=10)
        ax.set_xlabel('Detoxify toxicity score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    fig.suptitle('Detoxify-multilingual: score distributions by class', fontsize=12)
    fig.tight_layout()
    path = output_dir / 'fig_score_distributions.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_threshold_sweep(sweep_by_dataset: Dict, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, len(sweep_by_dataset),
                             figsize=(5 * len(sweep_by_dataset), 4), sharey=False)
    if len(sweep_by_dataset) == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, sweep_by_dataset.items()):
        rows = result['sweep']
        ts   = [r['t'] for r in rows]
        ax.plot(ts, [r['f1'] for r in rows],        label='F1',        color='purple')
        ax.plot(ts, [r['precision'] for r in rows], label='Precision',  color='green',    linestyle='--')
        ax.plot(ts, [r['recall'] for r in rows],    label='Recall',     color='orange',   linestyle='--')
        ax.plot(ts, [r['tnr'] for r in rows],       label='TNR',        color='steelblue',linestyle=':')
        best_t = result['best']['t']
        best_f1 = result['best']['f1']
        ax.axvline(best_t, color='black', linestyle='--', linewidth=1,
                   label=f'best T={best_t:.2f} (F1={best_f1:.3f})')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, label='T=0.5 (default)')
        ax.set_title(name.replace('_', '\n'), fontsize=10)
        ax.set_xlabel('Threshold T')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    fig.suptitle('Detoxify-multilingual: single-threshold sweep', fontsize=12)
    fig.tight_layout()
    path = output_dir / 'fig_threshold_sweep.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_two_threshold_heatmaps(grid: List[Dict], dataset_name: str, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from scipy.interpolate import griddata

    t_lows  = sorted(set(r['t_low']  for r in grid))
    t_highs = sorted(set(r['t_high'] for r in grid))

    def to_matrix(key):
        mat = np.full((len(t_highs), len(t_lows)), np.nan)
        idx_low  = {v: i for i, v in enumerate(t_lows)}
        idx_high = {v: i for i, v in enumerate(t_highs)}
        for r in grid:
            mat[idx_high[r['t_high']], idx_low[r['t_low']]] = r[key]
        return mat

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ('deferral_rate', 'Deferral rate\n(fraction sent to Tier 2)', 'YlOrRd'),
        ('tier1_fnr',     'Tier 1 FNR\n(hateful slipping through as safe)', 'Reds'),
        ('tier1_fpr',     'Tier 1 FPR\n(safe wrongly flagged)', 'Blues'),
    ]
    for ax, (key, title, cmap) in zip(axes, metrics):
        mat = to_matrix(key)
        im = ax.imshow(mat, origin='lower', aspect='auto',
                       extent=[min(t_lows), max(t_lows), min(t_highs), max(t_highs)],
                       cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('T_low (safe threshold)')
        ax.set_ylabel('T_high (unsafe threshold)')
        ax.set_title(title, fontsize=10)

    fig.suptitle(f'Detoxify-multilingual two-threshold analysis — {dataset_name}', fontsize=12)
    fig.tight_layout()
    path = output_dir / f'fig_two_threshold_{dataset_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(dataset_name: str, single: Dict, operating_points: List[Dict]):
    print(f"\n{'='*65}")
    print(f"  {dataset_name}")
    print(f"{'='*65}")
    print(f"  Samples: {single['n_pos'] + single['n_neg']}  "
          f"(hateful={single['n_pos']}, safe={single['n_neg']})")
    b = single['best']
    print(f"\n  Best single threshold: T={b['t']:.2f}  "
          f"F1={b['f1']:.3f}  P={b['precision']:.3f}  R={b['recall']:.3f}  "
          f"TPR={b['tpr']:.3f}  TNR={b['tnr']:.3f}")
    print(f"  (vs default T=0.5: F1={next(r for r in single['sweep'] if abs(r['t']-0.5)<0.01)['f1']:.3f})")

    print(f"\n  Two-threshold operating points:")
    print(f"  {'Label':<18} {'T_low':>6} {'T_high':>7} {'Deferral':>9} "
          f"{'T1 FNR':>8} {'T1 FPR':>8} {'Coverage':>9}")
    print(f"  {'-'*70}")
    for p in operating_points:
        print(f"  {p['label']:<18} {p['t_low']:>6.2f} {p['t_high']:>7.2f} "
              f"{p['deferral_rate']:>9.1%} {p['tier1_fnr']:>8.1%} "
              f"{p['tier1_fpr']:>8.1%} {p['coverage']:>9.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Two-tier threshold analysis for detoxify-multilingual')
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--cache_dir',    default=str(Path.home() / 'datasets/cache'))
    parser.add_argument('--reddit_fr_path',
                        default=str(Path.home() / 'datasets/reddit/balanced/data-fr/test-fr.csv'))
    parser.add_argument('--scores_json', default=None,
                        help='Path to previously saved raw_scores.json — skip model inference')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--t_step', type=float, default=0.05,
                        help='Step size for threshold grid (default 0.05)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference or load cached scores ──────────────────────────────────────
    scores_path = output_dir / 'raw_scores.json'

    if args.scores_json:
        print(f"Loading cached scores from {args.scores_json} ...")
        with open(args.scores_json) as f:
            scored_by_dataset = json.load(f)
    else:
        import torch
        device = args.device if (args.device == 'cpu' or
                                  (args.device == 'cuda' and torch.cuda.is_available())) else 'cpu'
        print(f"Device: {device}")

        datasets = {
            'hatecheck_fr':          load_hatecheck_fr(args.cache_dir),
            'french_hate_superset':  load_french_hate_superset(args.cache_dir),
            'reddit_fr':             load_reddit_fr(args.reddit_fr_path),
        }

        scored_by_dataset = {}
        for name, samples in datasets.items():
            print(f"\nScoring {name} ({len(samples)} samples)...")
            scored_by_dataset[name] = collect_scores(samples, device)

        with open(scores_path, 'w') as f:
            json.dump(scored_by_dataset, f)
        print(f"\nRaw scores saved → {scores_path}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    thresholds = np.arange(0.0, 1.01, 0.01)
    t_grid     = np.arange(0.0, 1.0 + args.t_step, args.t_step)

    single_results   = {}
    two_thresh_grids = {}
    all_ops          = {}

    for name, scored in scored_by_dataset.items():
        print(f"\nAnalysing {name}...")
        single   = single_threshold_sweep(scored, thresholds)
        grid     = two_threshold_sweep(scored, t_grid, t_grid)
        ops      = find_operating_points(grid)
        single_results[name]   = single
        two_thresh_grids[name] = grid
        all_ops[name]          = ops
        print_summary(name, single, ops)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    try:
        plot_score_distributions(scored_by_dataset, output_dir)
        plot_threshold_sweep(single_results, output_dir)
        for name, grid in two_thresh_grids.items():
            plot_two_threshold_heatmaps(grid, name, output_dir)
    except ImportError as e:
        print(f"  Warning: plotting skipped — {e} (install matplotlib/scipy)")

    # ── Save full results JSON ─────────────────────────────────────────────────
    results = {
        'single_threshold': single_results,
        'operating_points': all_ops,
    }
    results_path = output_dir / 'threshold_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {results_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Summary — recommended operating points across datasets")
    print(f"{'='*65}")
    for name, ops in all_ops.items():
        mid = next(p for p in ops if p['label'] == 'mid_deferral')
        print(f"  {name:<28}  T_low={mid['t_low']:.2f}  T_high={mid['t_high']:.2f}"
              f"  defer={mid['deferral_rate']:.0%}"
              f"  T1_FNR={mid['tier1_fnr']:.1%}  T1_FPR={mid['tier1_fpr']:.1%}")


if __name__ == '__main__':
    main()
