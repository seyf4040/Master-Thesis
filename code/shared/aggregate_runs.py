#!/usr/bin/env python3
"""
aggregate_runs.py — Statistical aggregation across multiple baseline runs.

Scans for run_* subdirectories produced by run_full_baseline.py --run_id N,
then computes mean ± std for every metric across all runs.

Usage:
    python aggregate_runs.py --results_dir ~/code/results/full_baseline

    # Only aggregate specific models or datasets:
    python aggregate_runs.py \
        --results_dir ~/code/results/full_baseline \
        --models      detoxify_multilingual,koalaai \
        --datasets    openai,hatecheck_en
"""

import json
import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# ── Metrics we aggregate ──────────────────────────────────────────────────────

FLOAT_METRICS = [
    'accuracy', 'precision', 'recall', 'f1',
    'true_positive_rate', 'false_positive_rate',
    'true_negative_rate', 'false_negative_rate',
    'avg_inference_ms', 'energy_kwh', 'co2_kg',
]

INT_METRICS = ['tp', 'fp', 'tn', 'fn', 'errors']


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _fmt(m: float, s: float) -> str:
    return f"{m:.4f} ± {s:.4f}"


def _bar(rate: float, width: int = 10) -> str:
    filled = round(max(0.0, min(1.0, rate)) * width)
    return '█' * filled + '░' * (width - filled)


# ── Loading ───────────────────────────────────────────────────────────────────

def find_run_dirs(results_dir: Path) -> List[Path]:
    """Find all run_N subdirectories, sorted by run number."""
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
        key=lambda d: int(d.name.split('_')[1]),
    )
    return runs


def load_run(run_dir: Path) -> List[Dict]:
    """Load all result JSON files from a single run directory."""
    results = []
    for p in sorted(run_dir.rglob("*.json")):
        if p.name.startswith("summary"):
            continue
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: could not load {p}: {e}")
    return results


def load_all_runs(results_dir: Path,
                  model_filter: Optional[List[str]] = None,
                  dataset_filter: Optional[List[str]] = None):
    """
    Returns:
        run_dirs: List[Path]
        data: dict[(model, dataset)] → List[dict]  (one dict per run)
    """
    run_dirs = find_run_dirs(results_dir)
    if not run_dirs:
        print(f"No run_* directories found in {results_dir}")
        return [], {}

    data: Dict = defaultdict(list)

    for run_dir in run_dirs:
        for result in load_run(run_dir):
            model   = result.get('model', '')
            dataset = result.get('dataset', '')
            if model_filter   and model   not in model_filter:   continue
            if dataset_filter and dataset not in dataset_filter: continue
            data[(model, dataset)].append(result)

    return run_dirs, data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(runs_data: List[Dict]) -> Dict:
    """Aggregate a list of result dicts (one per run) into mean ± std."""
    agg = {
        'num_runs':    len(runs_data),
        'num_samples': runs_data[0].get('num_samples', 0),
        'model':       runs_data[0].get('model', ''),
        'dataset':     runs_data[0].get('dataset', ''),
    }
    for metric in FLOAT_METRICS:
        values = [r[metric] for r in runs_data if metric in r]
        agg[metric] = {
            'mean':   mean(values),
            'std':    std(values),
            'min':    min(values) if values else 0.0,
            'max':    max(values) if values else 0.0,
            'values': values,
        }
    for metric in INT_METRICS:
        values = [r[metric] for r in runs_data if metric in r]
        agg[metric] = {
            'mean':   mean(values),
            'std':    std(values),
            'values': values,
        }
    return agg


# ── Reporting ─────────────────────────────────────────────────────────────────

def generate_aggregate_report(aggregated: Dict, run_dirs: List[Path],
                               output_dir: Path):
    """aggregated: dict[(model, dataset)] → aggregate dict"""

    pairs    = sorted(aggregated.keys())
    datasets = sorted(set(ds for _, ds in pairs))
    models   = sorted(set(m  for m, _ in pairs))

    txt_path  = output_dir / "aggregate_summary.txt"
    json_path = output_dir / "aggregate_summary.json"

    with open(txt_path, 'w') as f:
        f.write("AGGREGATE BASELINE RESULTS — MEAN ± STD ACROSS RUNS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runs found: {len(run_dirs)}  ({', '.join(d.name for d in run_dirs)})\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Models ({len(models)}):   {', '.join(models)}\n")
        f.write(f"Datasets ({len(datasets)}): {', '.join(datasets)}\n\n")

        # ── Per-dataset table ──────────────────────────────────────────────────
        for ds in datasets:
            ds_agg = {m: aggregated[(m, ds)] for m in models if (m, ds) in aggregated}
            if not ds_agg:
                continue

            f.write(f"\n{'='*120}\n")
            f.write(f"DATASET: {ds.upper()}\n")
            f.write(f"{'='*120}\n")
            f.write(f"{'Model':<35} {'Runs':>5}  "
                    f"{'Accuracy':>18}  {'F1':>18}  {'TPR':>18}  {'FPR':>18}\n")
            f.write("-" * 120 + "\n")
            for m in sorted(ds_agg, key=lambda x: -ds_agg[x]['f1']['mean']):
                agg = ds_agg[m]
                n   = agg['num_runs']
                f.write(
                    f"{m:<35} {n:>5}  "
                    f"{_fmt(agg['accuracy']['mean'], agg['accuracy']['std']):>18}  "
                    f"{_fmt(agg['f1']['mean'],       agg['f1']['std']):>18}  "
                    f"{_fmt(agg['true_positive_rate']['mean'], agg['true_positive_rate']['std']):>18}  "
                    f"{_fmt(agg['false_positive_rate']['mean'], agg['false_positive_rate']['std']):>18}\n"
                )

        # ── Stability ranking (models by std of F1 across datasets) ───────────
        f.write(f"\n\n{'='*120}\n")
        f.write("STABILITY RANKING  (lower std = more consistent across runs)\n")
        f.write(f"{'='*120}\n")
        f.write(f"{'Model':<35} {'Dataset':<30} {'F1 mean':>10} {'F1 std':>10}  Consistency\n")
        f.write("-" * 100 + "\n")

        stability_rows = []
        for (m, ds), agg in aggregated.items():
            f1_std  = agg['f1']['std']
            f1_mean = agg['f1']['mean']
            stability_rows.append((f1_std, m, ds, f1_mean))

        for f1_std, m, ds, f1_mean in sorted(stability_rows):
            bar = _bar(1.0 - min(f1_std * 10, 1.0))  # invert: low std = full bar
            f.write(f"{m:<35} {ds:<30} {f1_mean:>10.4f} {f1_std:>10.4f}  {bar}\n")

        # ── F1 mean matrix ─────────────────────────────────────────────────────
        col_w = 16
        f.write(f"\n\n{'='*120}\n")
        f.write("F1 MEAN MATRIX\n")
        f.write(f"{'='*120}\n")
        header = f"{'Model':<35}" + "".join(f"{d[:col_w-1]:>{col_w}}" for d in datasets)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for m in sorted(models, key=lambda x: -mean(
            [aggregated[(x, ds)]['f1']['mean'] for ds in datasets if (x, ds) in aggregated]
        )):
            row = f"{m:<35}"
            for ds in datasets:
                if (m, ds) not in aggregated:
                    row += f"{'N/A':>{col_w}}"
                else:
                    agg = aggregated[(m, ds)]
                    row += f"{agg['f1']['mean']:>{col_w}.4f}"
            f.write(row + "\n")

        # ── F1 std matrix ──────────────────────────────────────────────────────
        f.write(f"\n\nF1 STD MATRIX  (variance across runs)\n")
        f.write("-" * len(header) + "\n")
        for m in sorted(models, key=lambda x: -mean(
            [aggregated[(x, ds)]['f1']['mean'] for ds in datasets if (x, ds) in aggregated]
        )):
            row = f"{m:<35}"
            for ds in datasets:
                if (m, ds) not in aggregated:
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{aggregated[(m, ds)]['f1']['std']:>{col_w}.4f}"
            f.write(row + "\n")

    # ── JSON output ───────────────────────────────────────────────────────────
    json_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_runs": len(run_dirs),
        "run_directories": [str(d) for d in run_dirs],
        "models": models,
        "datasets": datasets,
        "aggregated": {
            f"{m}__{ds}": agg
            for (m, ds), agg in aggregated.items()
        },
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\nAggregate report written:\n  {txt_path}\n  {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multiple baseline runs into mean ± std statistics'
    )
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing run_1/, run_2/, ... subdirectories')
    parser.add_argument('--models',   default=None,
                        help='Comma-separated model names to include (default: all)')
    parser.add_argument('--datasets', default=None,
                        help='Comma-separated dataset names to include (default: all)')
    args = parser.parse_args()

    results_dir    = Path(args.results_dir)
    model_filter   = [m.strip() for m in args.models.split(',')]   if args.models   else None
    dataset_filter = [d.strip() for d in args.datasets.split(',')] if args.datasets else None

    print(f"Scanning {results_dir} for run_* directories...")
    run_dirs, data = load_all_runs(results_dir, model_filter, dataset_filter)

    if not data:
        print("No results found.")
        return

    print(f"Found {len(run_dirs)} runs: {[d.name for d in run_dirs]}")
    print(f"Found {len(data)} (model, dataset) pairs\n")

    # Aggregate
    aggregated = {key: aggregate(runs) for key, runs in data.items()}

    # Report
    generate_aggregate_report(aggregated, run_dirs, results_dir)

    # Print quick console summary
    print("\nQUICK SUMMARY — F1 mean ± std")
    print("-" * 70)
    for (m, ds), agg in sorted(aggregated.items(),
                                key=lambda x: -x[1]['f1']['mean']):
        f1    = agg['f1']
        n_runs = agg['num_runs']
        print(f"  {m:<35} {ds:<25} {f1['mean']:.4f} ± {f1['std']:.4f}  (n={n_runs})")


if __name__ == '__main__':
    main()
