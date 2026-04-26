#!/usr/bin/env python3
"""
analyze_threshold_tier1.py — Tier 1 candidate comparison for two-tier moderation

Runs the same two-threshold analysis as analyze_threshold_detoxify.py but supports
multiple lightweight models as Tier 1 candidates, enabling direct comparison:

  Supported --model values:
    detoxify_multilingual  — original Detoxify-M baseline (Jigsaw/Wikipedia training)
    citizenlab             — CitizenLab XLM-RoBERTa sentiment (negative prob as proxy)
    hf_classifier          — any HuggingFace text-classification model via --hf_model_id

  Recommended new candidate (pass via --hf_model_id):
    unitary/multilingual-toxic-xlm-roberta
        XLM-RoBERTa fine-tuned on multilingual social media toxicity.
        Same Unitary team as Detoxify but different architecture/training data.
        More social-media-oriented than Detoxify's Wikipedia/formal-text corpus.

Analysis steps (identical to Detoxify analysis):
  1. Score distribution plots  — safe vs hateful score histograms per dataset
  2. Single-threshold sweep    — F1/P/R/TNR at every T ∈ [0,1]
  3. Two-threshold sweep       — deferral rate, T1_FNR, T1_FPR over (T_low, T_high) grid
  4. Operating points          — at ~10%, ~25%, ~50% deferral targets

Output layout:
  {output_dir}/{model_key}/
    raw_scores.json
    threshold_analysis.json
    fig_score_distributions.png
    fig_threshold_sweep.png
    fig_two_threshold_{dataset}.png

Usage:
    # Detoxify-M (rerun or use cached):
    python code/phase4_two_tier/analyze_threshold_tier1.py \\
        --model detoxify_multilingual \\
        --output_dir ~/code/results/tier1_comparison \\
        --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv

    # CitizenLab:
    python code/phase4_two_tier/analyze_threshold_tier1.py \\
        --model citizenlab \\
        --output_dir ~/code/results/tier1_comparison \\
        --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv

    # Custom HuggingFace classifier:
    python code/phase4_two_tier/analyze_threshold_tier1.py \\
        --model hf_classifier \\
        --hf_model_id unitary/multilingual-toxic-xlm-roberta \\
        --hf_toxic_label toxic \\
        --output_dir ~/code/results/tier1_comparison \\
        --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv

    # Reuse cached scores (skip inference):
    python code/phase4_two_tier/analyze_threshold_tier1.py \\
        --model citizenlab \\
        --scores_json ~/code/results/tier1_comparison/citizenlab/raw_scores.json \\
        --output_dir ~/code/results/tier1_comparison

Author: Ural Seyfullah
"""

# ── CVE-2025-32434 bypass ─────────────────────────────────────────────────────
# Cluster runs torch < 2.6; transformers >= 4.49 blocks torch.load() on older
# versions. The check lives in import_utils but is imported by name into
# modeling_utils — so we must patch BOTH the source module and the local
# binding in modeling_utils (patching only the source has no effect on the
# already-bound reference in modeling_utils).
# Safe: unitary/multilingual-toxic-xlm-roberta is a trusted HuggingFace model.
try:
    import transformers.utils.import_utils as _hf_import_utils
    _hf_import_utils.check_torch_load_is_safe = lambda: None
except (ImportError, AttributeError):
    pass
try:
    import transformers.modeling_utils as _hf_modeling_utils
    _hf_modeling_utils.check_torch_load_is_safe = lambda: None
except (ImportError, AttributeError):
    pass
# ─────────────────────────────────────────────────────────────────────────────

import gc
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_hatecheck_fr(cache_dir: str) -> List[Dict]:
    from datasets import load_dataset as _ld
    ds = _ld("Paul/hatecheck-french", cache_dir=cache_dir)
    return [
        {'text': item['test_case'], 'label': 1 if item['label_gold'] == 'hateful' else 0}
        for item in ds['test']
    ]


def load_french_hate_superset(cache_dir: str, max_samples: int = None) -> List[Dict]:
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
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def load_reddit_fr(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]


def load_hatecheck_en(cache_dir: str) -> List[Dict]:
    from datasets import load_dataset as _ld
    ds = _ld("Paul/hatecheck", cache_dir=cache_dir)
    return [
        {'text': item['test_case'], 'label': 1 if item['label_gold'] == 'hateful' else 0}
        for item in ds['test']
    ]


def load_reddit_en(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]


def load_toxigen(path: str, max_samples: int = 5000) -> List[Dict]:
    """ToxiGen: implicit toxicity in English. Label: human_annotation >= 0.5 → hateful."""
    import json as _json
    samples = []
    with open(path) as f:
        for line in f:
            item = _json.loads(line)
            text = item.get('text', '') or ''
            ann = item.get('human_annotation', item.get('toxicity_human', None))
            if not text.strip() or ann is None:
                continue
            label = 1 if float(ann) >= 0.5 else 0
            samples.append({'text': text, 'label': label})
            if len(samples) >= max_samples:
                break
    return samples


def load_openai(path: str) -> List[Dict]:
    """OpenAI moderation dataset: category_scores presence → label 1 if any score > 0."""
    import json as _json
    samples = []
    with open(path) as f:
        for line in f:
            item = _json.loads(line)
            text = item.get('prompt', '') or ''
            if not text.strip():
                continue
            # label: any category flagged
            cats = item.get('categories', {}) or {}
            label = 1 if any(cats.values()) else 0
            samples.append({'text': text, 'label': label})
    return samples


def load_civil_comments(cache_dir: str, max_samples: int = 5000) -> List[Dict]:
    """Civil Comments: toxicity >= 0.5 → label 1."""
    from datasets import load_dataset as _ld
    ds = _ld("google/civil_comments", cache_dir=cache_dir)
    samples = []
    for item in ds['test']:
        text = item.get('text', '') or ''
        if not text.strip():
            continue
        label = 1 if float(item.get('toxicity', 0)) >= 0.5 else 0
        samples.append({'text': text, 'label': label})
        if len(samples) >= max_samples:
            break
    return samples


# ── Model-specific inference ──────────────────────────────────────────────────

def _score_detoxify(samples: List[Dict], device: str) -> List[Dict]:
    """Detoxify-multilingual: returns toxicity probability in [0,1]."""
    from detoxify import Detoxify
    from tqdm import tqdm
    model = Detoxify('multilingual', device=device)
    results = []
    for s in tqdm(samples, desc="detoxify-multilingual", unit="sample", leave=False):
        try:
            score = float(model.predict(s['text'])['toxicity'])
            results.append({'score': score, 'label': s['label']})
        except Exception as e:
            print(f"  Warning: skipped sample — {e}")
    del model
    gc.collect()
    return results


def _score_citizenlab(samples: List[Dict], device: str) -> List[Dict]:
    """CitizenLab XLM-RoBERTa: uses negative-sentiment probability as toxicity proxy.
    Labels are [Negative, Neutral, Positive] — index 0 is the toxicity signal.
    Note: this is a sentiment model repurposed as a toxicity filter. Score calibration
    on hate speech is expected to be imperfect."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm

    model_id = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, use_safetensors=True, torch_dtype=torch.float32,
    )
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    results = []
    for s in tqdm(samples, desc="citizenlab", unit="sample", leave=False):
        try:
            inputs = tokenizer(s['text'], return_tensors="pt",
                               truncation=True, max_length=512)
            if device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                probs = model(**inputs).logits.softmax(dim=-1)
                # [Negative, Neutral, Positive] — negative sentiment used as toxicity proxy
                negative_prob = probs[0][0].item()
            results.append({'score': negative_prob, 'label': s['label']})
        except Exception as e:
            print(f"  Warning: skipped sample — {e}")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _score_hf_classifier(samples: List[Dict], model_id: str,
                          toxic_label: str, device: str) -> List[Dict]:
    """Generic HuggingFace text-classification pipeline.
    Returns the probability of the class matching --hf_toxic_label.
    If the label is not found in the model's output, falls back to 1 - safe_prob.

    Works with:
      - Binary classifiers: 'toxic'/'non-toxic', 'LABEL_1'/'LABEL_0', etc.
      - Probability-output classifiers (softmax over classes)
    """
    import torch
    from transformers import pipeline
    from tqdm import tqdm

    _device = 0 if (device == 'cuda' and torch.cuda.is_available()) else -1
    print(f"  Loading {model_id} via pipeline (device={'cuda' if _device==0 else 'cpu'})...")
    classifier = pipeline(
        "text-classification",
        model=model_id,
        device=_device,
        truncation=True,
        max_length=512,
        top_k=None,          # return all label probs
    )

    results = []
    for s in tqdm(samples, desc=model_id.split('/')[-1], unit="sample", leave=False):
        try:
            preds = classifier(s['text'])  # list of {label, score}
            # preds may be [[{label,score},...]] or [{label,score},...]
            if isinstance(preds[0], list):
                preds = preds[0]
            label_scores = {p['label'].lower(): p['score'] for p in preds}
            toxic_key = toxic_label.lower()
            if toxic_key in label_scores:
                score = label_scores[toxic_key]
            else:
                # Fallback: 1 - max non-toxic label probability
                non_toxic_keys = [k for k in label_scores if k != toxic_key]
                score = 1.0 - max(label_scores.get(k, 0.0) for k in non_toxic_keys) \
                        if non_toxic_keys else 0.5
                print(f"  Warning: label '{toxic_label}' not found in {list(label_scores.keys())}; "
                      f"using fallback score={score:.3f}")
            results.append({'score': score, 'label': s['label']})
        except Exception as e:
            print(f"  Warning: skipped sample — {e}")

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def collect_scores(samples: List[Dict], args) -> List[Dict]:
    """Dispatch to the correct model inference function."""
    device = args.device
    if args.model == 'detoxify_multilingual':
        return _score_detoxify(samples, device)
    elif args.model == 'citizenlab':
        return _score_citizenlab(samples, device)
    elif args.model == 'hf_classifier':
        if not args.hf_model_id:
            raise ValueError("--hf_model_id is required when --model=hf_classifier")
        return _score_hf_classifier(samples, args.hf_model_id, args.hf_toxic_label, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")


# ── Analysis (model-agnostic) ─────────────────────────────────────────────────

def single_threshold_sweep(scored: List[Dict], thresholds: np.ndarray) -> Dict:
    labels = np.array([s['label'] for s in scored])
    scores = np.array([s['score'] for s in scored])
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos

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
                        t_highs: np.ndarray) -> List[Dict]:
    labels = np.array([s['label'] for s in scored])
    scores = np.array([s['score'] for s in scored])
    n = len(labels)
    if n == 0:
        return []

    grid = []
    for t_low in t_lows:
        for t_high in t_highs:
            if t_low >= t_high:
                continue
            safe_mask   = scores < t_low
            unsafe_mask = scores > t_high
            defer_mask  = ~safe_mask & ~unsafe_mask

            n_safe   = int(safe_mask.sum())
            n_unsafe = int(unsafe_mask.sum())
            n_defer  = int(defer_mask.sum())

            hateful_in_safe = int((safe_mask & (labels == 1)).sum())
            safe_in_unsafe  = int((unsafe_mask & (labels == 0)).sum())
            tier1_fnr = hateful_in_safe / n_safe   if n_safe   > 0 else 0.0
            tier1_fpr = safe_in_unsafe  / n_unsafe if n_unsafe > 0 else 0.0

            grid.append({
                't_low':         float(t_low),
                't_high':        float(t_high),
                'coverage':      float((n_safe + n_unsafe) / n),
                'deferral_rate': float(n_defer / n),
                'tier1_fnr':     tier1_fnr,
                'tier1_fpr':     tier1_fpr,
                'n_safe':        n_safe,
                'n_unsafe':      n_unsafe,
                'n_defer':       n_defer,
            })
    return grid


def find_operating_points(grid: List[Dict]) -> List[Dict]:
    targets = [('low_deferral',  0.10),
               ('mid_deferral',  0.25),
               ('high_deferral', 0.50)]
    points = []
    for label, target in targets:
        candidates = sorted(grid, key=lambda r: abs(r['deferral_rate'] - target))[:20]
        best = min(candidates, key=lambda r: r['tier1_fnr'] + r['tier1_fpr'])
        points.append({'label': label, 'target_deferral': target, **best})
    return points


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_score_distributions(scored_by_dataset: Dict, output_dir: Path, model_label: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scored_by_dataset),
                             figsize=(5 * len(scored_by_dataset), 4), sharey=False)
    if len(scored_by_dataset) == 1:
        axes = [axes]

    for ax, (name, scored) in zip(axes, scored_by_dataset.items()):
        safe_scores   = [s['score'] for s in scored if s['label'] == 0]
        unsafe_scores = [s['score'] for s in scored if s['label'] == 1]
        bins = np.linspace(0, 1, 51)
        ax.hist(safe_scores,   bins=bins, alpha=0.6, color='steelblue', density=True,
                label=f'safe (n={len(safe_scores)})')
        ax.hist(unsafe_scores, bins=bins, alpha=0.6, color='tomato',    density=True,
                label=f'hateful (n={len(unsafe_scores)})')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='T=0.5')
        ax.set_title(name.replace('_', '\n'), fontsize=10)
        ax.set_xlabel('Toxicity score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    fig.suptitle(f'{model_label}: score distributions by class', fontsize=12)
    fig.tight_layout()
    path = output_dir / 'fig_score_distributions.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_threshold_sweep(sweep_by_dataset: Dict, output_dir: Path, model_label: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sweep_by_dataset),
                             figsize=(5 * len(sweep_by_dataset), 4), sharey=False)
    if len(sweep_by_dataset) == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, sweep_by_dataset.items()):
        rows = result['sweep']
        ts = [r['t'] for r in rows]
        ax.plot(ts, [r['f1']        for r in rows], label='F1',       color='purple')
        ax.plot(ts, [r['precision'] for r in rows], label='Precision', color='green',     linestyle='--')
        ax.plot(ts, [r['recall']    for r in rows], label='Recall',    color='orange',    linestyle='--')
        ax.plot(ts, [r['tnr']       for r in rows], label='TNR',       color='steelblue', linestyle=':')
        best_t  = result['best']['t']
        best_f1 = result['best']['f1']
        ax.axvline(best_t, color='black', linestyle='--', linewidth=1,
                   label=f'best T={best_t:.2f} (F1={best_f1:.3f})')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, label='T=0.5 (default)')
        ax.set_title(name.replace('_', '\n'), fontsize=10)
        ax.set_xlabel('Threshold T')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    fig.suptitle(f'{model_label}: single-threshold sweep', fontsize=12)
    fig.tight_layout()
    path = output_dir / 'fig_threshold_sweep.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_two_threshold_heatmaps(grid: List[Dict], dataset_name: str,
                                output_dir: Path, model_label: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
        ('deferral_rate', 'Deferral rate\n(fraction sent to Tier 2)',      'YlOrRd'),
        ('tier1_fnr',     'Tier 1 FNR\n(hateful slipping through as safe)', 'Reds'),
        ('tier1_fpr',     'Tier 1 FPR\n(safe wrongly flagged)',             'Blues'),
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

    fig.suptitle(f'{model_label} two-threshold analysis — {dataset_name}', fontsize=12)
    fig.tight_layout()
    path = output_dir / f'fig_two_threshold_{dataset_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(dataset_name: str, single: Dict, operating_points: List[Dict]):
    print(f"\n{'='*65}")
    print(f"  {dataset_name}")
    print(f"{'='*65}")
    print(f"  Samples: {single['n_pos'] + single['n_neg']}  "
          f"(hateful={single['n_pos']}, safe={single['n_neg']})")
    b = single['best']
    default_f1 = next((r['f1'] for r in single['sweep'] if abs(r['t'] - 0.5) < 0.01), None)
    print(f"\n  Best single threshold: T={b['t']:.2f}  "
          f"F1={b['f1']:.3f}  P={b['precision']:.3f}  R={b['recall']:.3f}  "
          f"TPR={b['tpr']:.3f}  TNR={b['tnr']:.3f}")
    if default_f1 is not None:
        print(f"  (vs default T=0.5: F1={default_f1:.3f})")

    print(f"\n  Two-threshold operating points:")
    print(f"  {'Label':<18} {'T_low':>6} {'T_high':>7} {'Deferral':>9} "
          f"{'T1 FNR':>8} {'T1 FPR':>8} {'Coverage':>9}")
    print(f"  {'-'*70}")
    for p in operating_points:
        print(f"  {p['label']:<18} {p['t_low']:>6.2f} {p['t_high']:>7.2f} "
              f"{p['deferral_rate']:>9.1%} {p['tier1_fnr']:>8.1%} "
              f"{p['tier1_fpr']:>8.1%} {p['coverage']:>9.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

MODEL_LABELS = {
    'detoxify_multilingual': 'Detoxify-multilingual',
    'citizenlab':            'CitizenLab-XLM-RoBERTa',
    'hf_classifier':        None,   # filled from --hf_model_id
}


def main():
    parser = argparse.ArgumentParser(
        description='Tier 1 model comparison — two-threshold analysis for two-tier moderation')
    parser.add_argument('--model', required=True,
                        choices=['detoxify_multilingual', 'citizenlab', 'hf_classifier'],
                        help='Which Tier 1 model to evaluate')
    parser.add_argument('--hf_model_id', default=None,
                        help='HuggingFace model ID (required when --model=hf_classifier). '
                             'Recommended: unitary/multilingual-toxic-xlm-roberta')
    parser.add_argument('--hf_toxic_label', default='toxic',
                        help="Label name that means 'toxic/unsafe' in the HF model's output "
                             "(default: 'toxic'). Check the model card if unsure.")
    parser.add_argument('--output_dir',   required=True,
                        help='Root output dir — results written to {output_dir}/{model_key}/')
    parser.add_argument('--cache_dir',    default=str(Path.home() / 'datasets/cache'))
    parser.add_argument('--reddit_fr_path',
                        default=str(Path.home() / 'datasets/reddit/balanced/data-fr/test-fr.csv'))
    # Optional extra datasets — include if path/flag is provided
    parser.add_argument('--reddit_en_path', default=None,
                        help='Path to Reddit-EN CSV (text, label columns). Optional.')
    parser.add_argument('--toxigen_path', default=None,
                        help='Path to ToxiGen JSONL. Optional.')
    parser.add_argument('--openai_path', default=None,
                        help='Path to OpenAI moderation JSONL. Optional.')
    parser.add_argument('--include_hatecheck_en', action='store_true',
                        help='Auto-download and include HateCheck-EN (English).')
    parser.add_argument('--include_civil_comments', action='store_true',
                        help='Auto-download and include Civil Comments (English, max 5000).')
    parser.add_argument('--max_samples_toxigen', type=int, default=5000,
                        help='Max ToxiGen samples (default: 5000).')
    parser.add_argument('--max_samples_fhs', type=int, default=None,
                        help='Max French Hate Superset samples (default: all ~18k).')
    parser.add_argument('--max_samples_civil', type=int, default=5000,
                        help='Max Civil Comments samples (default: 5000).')
    parser.add_argument('--scores_json', default=None,
                        help='Path to a previously saved raw_scores.json — skips model inference')
    parser.add_argument('--device', default='cpu',
                        help='cuda or cpu (default: cpu — all Tier 1 candidates fit on CPU)')
    parser.add_argument('--t_step', type=float, default=0.05,
                        help='Step size for two-threshold grid (default: 0.05)')
    args = parser.parse_args()

    # Model label for titles/filenames
    model_key = args.model if args.model != 'hf_classifier' else \
                (args.hf_model_id or 'hf_classifier').replace('/', '_')
    model_label = MODEL_LABELS.get(args.model) or model_key

    output_dir = Path(args.output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  Tier 1 threshold analysis: {model_label}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}")

    # ── Inference or load cached scores ──────────────────────────────────────
    scores_path = output_dir / 'raw_scores.json'

    if args.scores_json:
        print(f"Loading cached scores from {args.scores_json} ...")
        with open(args.scores_json) as f:
            scored_by_dataset = json.load(f)
    else:
        datasets = {
            'hatecheck_fr':         load_hatecheck_fr(args.cache_dir),
            'french_hate_superset': load_french_hate_superset(args.cache_dir, args.max_samples_fhs),
            'reddit_fr':            load_reddit_fr(args.reddit_fr_path),
        }
        # Optional datasets — include only if paths/flags provided
        if args.include_hatecheck_en:
            datasets['hatecheck_en'] = load_hatecheck_en(args.cache_dir)
        if args.reddit_en_path:
            datasets['reddit_en'] = load_reddit_en(args.reddit_en_path)
        if args.toxigen_path:
            datasets['toxigen'] = load_toxigen(args.toxigen_path, args.max_samples_toxigen)
        if args.openai_path:
            datasets['openai'] = load_openai(args.openai_path)
        if args.include_civil_comments:
            datasets['civil_comments'] = load_civil_comments(args.cache_dir, args.max_samples_civil)
        scored_by_dataset = {}
        for name, samples in datasets.items():
            print(f"\nScoring {name} ({len(samples)} samples) with {model_label}...")
            scored_by_dataset[name] = collect_scores(samples, args)

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
        single = single_threshold_sweep(scored, thresholds)
        grid   = two_threshold_sweep(scored, t_grid, t_grid)
        ops    = find_operating_points(grid)
        single_results[name]   = single
        two_thresh_grids[name] = grid
        all_ops[name]          = ops
        print_summary(name, single, ops)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    try:
        plot_score_distributions(scored_by_dataset, output_dir, model_label)
        plot_threshold_sweep(single_results, output_dir, model_label)
        for name, grid in two_thresh_grids.items():
            plot_two_threshold_heatmaps(grid, name, output_dir, model_label)
    except ImportError as e:
        print(f"  Warning: plotting skipped — {e} (install matplotlib)")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        'model':              model_key,
        'model_label':        model_label,
        'hf_model_id':        args.hf_model_id,
        'single_threshold':   single_results,
        'operating_points':   all_ops,
    }
    results_path = output_dir / 'threshold_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {results_path}")
    print(f"\nDone — results in {output_dir}")


if __name__ == '__main__':
    main()
