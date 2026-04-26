#!/usr/bin/env python3
"""
simulate_thresholds.py — Offline threshold simulation for the two-tier system

Loads raw_scores.json produced by score_two_tier.py and sweeps all (T_low, T_high)
combinations without reloading any model. For each configuration computes:
  - Combined F1, FNR, FPR  (system-level classification performance)
  - Deferral rate           (fraction of samples sent to Tier 2)
  - Avg inference ms        (Tier 1 always + Tier 2 only for deferred fraction)

Baselines computed from the same score file:
  - Tier 2 alone  : use t2_pred for all samples (ignoring Tier 1)
  - Tier 1 alone  : find best single-threshold F1 on t1_score

Output:
  {output_dir}/
    simulation_results.json
    summary.txt                  — operating points + baseline comparison table
    fig_combined_f1.png          — heatmap: combined F1 over (T_low, T_high) grid
    fig_fnr.png                  — heatmap: combined FNR
    fig_fpr.png                  — heatmap: combined FPR
    fig_deferral_rate.png        — heatmap: deferral rate
    fig_avg_ms.png               — heatmap: avg inference ms per sample

Usage:
    python code/phase4_two_tier/simulate_thresholds.py \\
        --scores_json ~/code/results/two_tier_scores/finetuned/raw_scores.json \\
        --output_dir  ~/code/results/two_tier_scores/finetuned/simulation

Author: Ural Seyfullah
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _f1_fnr_fpr(labels: np.ndarray, preds: np.ndarray):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fnr  = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr  = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return f1, fnr, fpr, prec, rec, tp, fp, tn, fn


# ── Simulation core ───────────────────────────────────────────────────────────

def simulate(samples: List[Dict], t_lows: np.ndarray,
             t_highs: np.ndarray) -> List[Dict]:
    """
    Sweep all (T_low, T_high) pairs and compute combined-system metrics.
    Three-zone logic:
      score < T_low  → Tier 1 passes as SAFE  (pred=0)
      score > T_high → Tier 1 flags as UNSAFE (pred=1)
      otherwise      → defer to Tier 2 (use t2_pred)
    """
    labels    = np.array([s['label']   for s in samples])
    t1_scores = np.array([s['t1_score'] for s in samples])
    t2_preds  = np.array([s['t2_pred'] for s in samples])
    t1_ms_arr = np.array([s['t1_ms']   for s in samples])
    t2_ms_arr = np.array([s['t2_ms']   for s in samples])
    n         = len(samples)

    avg_t1_ms = float(t1_ms_arr.mean())
    avg_t2_ms = float(t2_ms_arr.mean())

    grid = []
    for t_low in t_lows:
        for t_high in t_highs:
            if t_low >= t_high:
                continue

            safe_mask   = t1_scores < t_low
            unsafe_mask = t1_scores > t_high
            defer_mask  = ~safe_mask & ~unsafe_mask

            n_safe   = int(safe_mask.sum())
            n_unsafe = int(unsafe_mask.sum())
            n_defer  = int(defer_mask.sum())
            deferral = n_defer / n

            # Combined predictions
            combined = np.zeros(n, dtype=int)
            combined[unsafe_mask] = 1
            combined[defer_mask]  = t2_preds[defer_mask]
            # safe_mask stays 0

            f1, fnr, fpr, prec, rec, tp, fp, tn, fn = _f1_fnr_fpr(labels, combined)

            # Avg inference ms: Tier 1 runs on everything; Tier 2 only on deferred
            avg_ms = avg_t1_ms + deferral * avg_t2_ms

            # Tier 1 standalone FNR (hateful in safe bin without deferral)
            hateful_in_safe = int((safe_mask & (labels == 1)).sum())
            n_hateful = int(labels.sum())
            t1_fnr = hateful_in_safe / n_hateful if n_hateful > 0 else 0.0

            grid.append({
                't_low':        float(t_low),
                't_high':       float(t_high),
                'deferral_rate': deferral,
                'combined_f1':  f1,
                'combined_fnr': fnr,
                'combined_fpr': fpr,
                'combined_prec': prec,
                'combined_rec': rec,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'tier1_fnr':    t1_fnr,
                'avg_ms':       avg_ms,
                'n_safe':       n_safe,
                'n_unsafe':     n_unsafe,
                'n_defer':      n_defer,
            })
    return grid


def find_operating_points(grid: List[Dict]) -> List[Dict]:
    """For each deferral target, find the point maximising combined F1."""
    targets = [('low_deferral', 0.10), ('mid_deferral', 0.25), ('high_deferral', 0.50)]
    ops = []
    for label, target in targets:
        candidates = sorted(grid, key=lambda r: abs(r['deferral_rate'] - target))[:30]
        best = max(candidates, key=lambda r: r['combined_f1'])
        ops.append({'label': label, 'target_deferral': target, **best})
    return ops


def tier1_single_threshold_best(samples: List[Dict]) -> Dict:
    """Find the single threshold on t1_score that maximises F1."""
    labels    = np.array([s['label']   for s in samples])
    t1_scores = np.array([s['t1_score'] for s in samples])
    thresholds = np.arange(0.0, 1.01, 0.01)
    best = {'t': None, 'f1': -1}
    for t in thresholds:
        preds = (t1_scores >= t).astype(int)
        f1, *_ = _f1_fnr_fpr(labels, preds)
        if f1 > best['f1']:
            best = {'t': float(t), 'f1': f1}
    return best


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(grid: List[Dict], key: str, title: str, cmap: str,
                 output_dir: Path, vmin=0.0, vmax=1.0,
                 t2_alone_value: float = None, t1_best_value: float = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_lows  = sorted(set(r['t_low']  for r in grid))
    t_highs = sorted(set(r['t_high'] for r in grid))
    idx_low  = {v: i for i, v in enumerate(t_lows)}
    idx_high = {v: i for i, v in enumerate(t_highs)}

    mat = np.full((len(t_highs), len(t_lows)), np.nan)
    for r in grid:
        mat[idx_high[r['t_high']], idx_low[r['t_low']]] = r[key]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, origin='lower', aspect='auto',
                   extent=[min(t_lows), max(t_lows), min(t_highs), max(t_highs)],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('T_low (safe threshold)')
    ax.set_ylabel('T_high (unsafe threshold)')
    ax.set_title(title, fontsize=11)

    # Annotate baselines as horizontal reference lines on colorbar
    if t2_alone_value is not None:
        ax.set_title(f"{title}\n(Tier-2-alone: {t2_alone_value:.3f})", fontsize=10)

    fig.tight_layout()
    fname = f"fig_{key}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / fname}")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_and_save_summary(ops: List[Dict], t2_alone: Dict, t1_best: Dict,
                            avg_t1_ms: float, avg_t2_ms: float,
                            output_dir: Path):
    lines = []
    lines.append("Two-Tier System — Threshold Simulation Results")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Baselines:")
    lines.append(f"  Tier 2 alone (SG-2b LoRA):  F1={t2_alone['f1']:.3f}  "
                 f"FNR={t2_alone['fnr']:.1%}  avg_ms={avg_t2_ms:.1f}")
    lines.append(f"  Tier 1 alone (best T):       F1={t1_best['f1']:.3f}  "
                 f"T={t1_best['t']:.2f}  avg_ms={avg_t1_ms:.1f}")
    lines.append("")
    lines.append("Operating points (combined system):")
    header = (f"  {'Label':<18} {'T_low':>6} {'T_high':>7} {'Deferral':>9} "
              f"{'F1':>7} {'FNR':>7} {'FPR':>7} {'Avg_ms':>8}")
    lines.append(header)
    lines.append("  " + "-" * 68)
    for op in ops:
        lines.append(
            f"  {op['label']:<18} {op['t_low']:>6.2f} {op['t_high']:>7.2f} "
            f"{op['deferral_rate']:>9.1%} "
            f"{op['combined_f1']:>7.3f} {op['combined_fnr']:>7.1%} "
            f"{op['combined_fpr']:>7.1%} {op['avg_ms']:>8.1f}"
        )
    lines.append("")
    lines.append(f"  Tier-2-alone F1 = {t2_alone['f1']:.3f}  "
                 f"(reference: combined F1 should exceed this to justify Tier 1 cost)")

    text = "\n".join(lines)
    print("\n" + text)
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(text + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Offline two-tier threshold simulation from saved scores')
    parser.add_argument('--scores_json', required=True,
                        help='raw_scores.json from score_two_tier.py')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--t_step', type=float, default=0.05,
                        help='Threshold grid step size (default: 0.05)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading scores from {args.scores_json} ...")
    with open(args.scores_json) as f:
        data = json.load(f)

    samples   = data['samples']
    n         = data['n_samples']
    avg_t1_ms = data['avg_t1_ms']
    avg_t2_ms = data['avg_t2_ms']

    print(f"  {n} samples  (hateful={data['n_hateful']}, safe={data['n_safe']})")
    print(f"  Tier 1: {data['tier1_model_id']}")
    print(f"  Tier 2: {data['tier2_base_model']} + {data['tier2_adapter_dir']}")
    print(f"  Avg inference: T1={avg_t1_ms:.1f} ms/sample, T2={avg_t2_ms:.1f} ms/sample")

    # ── Baselines ─────────────────────────────────────────────────────────────
    labels   = np.array([s['label']  for s in samples])
    t2_preds = np.array([s['t2_pred'] for s in samples])

    t2_f1, t2_fnr, t2_fpr, t2_prec, t2_rec, *_ = _f1_fnr_fpr(labels, t2_preds)
    t2_alone = {'f1': t2_f1, 'fnr': t2_fnr, 'fpr': t2_fpr,
                 'precision': t2_prec, 'recall': t2_rec}
    t1_best  = tier1_single_threshold_best(samples)

    print(f"\n  Tier 2 alone — F1={t2_f1:.3f}  FNR={t2_fnr:.1%}  FPR={t2_fpr:.1%}")
    print(f"  Tier 1 best  — F1={t1_best['f1']:.3f}  T={t1_best['t']:.2f}")

    # ── Grid simulation ───────────────────────────────────────────────────────
    t_grid = np.arange(0.0, 1.0 + args.t_step, args.t_step)
    print(f"\nSimulating {len(t_grid)**2 // 2} (T_low, T_high) configurations ...")
    grid = simulate(samples, t_grid, t_grid)

    ops = find_operating_points(grid)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_and_save_summary(ops, t2_alone, t1_best, avg_t1_ms, avg_t2_ms, output_dir)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    print("\nGenerating heatmaps ...")
    try:
        vmax_ms = max(r['avg_ms'] for r in grid)
        plot_heatmap(grid, 'combined_f1',    'Combined F1',               'RdYlGn',
                     output_dir, vmin=0, vmax=1, t2_alone_value=t2_f1)
        plot_heatmap(grid, 'combined_fnr',   'Combined FNR (hateful missed)', 'Reds',
                     output_dir, vmin=0, vmax=1)
        plot_heatmap(grid, 'combined_fpr',   'Combined FPR (safe wrongly flagged)', 'Blues',
                     output_dir, vmin=0, vmax=1)
        plot_heatmap(grid, 'deferral_rate',  'Deferral rate (→ Tier 2)',  'YlOrRd',
                     output_dir, vmin=0, vmax=1)
        plot_heatmap(grid, 'avg_ms',         'Avg inference ms/sample',   'YlOrRd',
                     output_dir, vmin=0, vmax=vmax_ms)
    except ImportError as e:
        print(f"  Plotting skipped — {e}")

    # ── Save full results ─────────────────────────────────────────────────────
    results = {
        'tier1_model_id':   data['tier1_model_id'],
        'tier2_adapter_dir': data['tier2_adapter_dir'],
        't_step':           args.t_step,
        'n_samples':        n,
        'baselines': {
            'tier2_alone': t2_alone,
            'tier1_best':  t1_best,
        },
        'operating_points': ops,
        'grid':             grid,
    }
    results_path = output_dir / 'simulation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results → {results_path}")
    print(f"  Done.")


if __name__ == '__main__':
    main()
