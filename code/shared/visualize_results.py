#!/usr/bin/env python3
"""
visualize_results.py — Phase 1 Baseline Visualizations

Generates 7 publication-quality figures from full_baseline_v3/ and
hatecheck_analysis/ result directories.

Figures produced:
  fig1_f1_heatmap.png           — F1 across all 10 models × 8 datasets
  fig2_vram_vs_fr_f1.png        — VRAM vs HC-FR F1 (thesis deployability argument)
  fig3_tpr_vs_tnr.png           — Sensitivity vs specificity, EN and FR
  fig4a_hatecheck_fr_funcs.png  — HateCheck FR per-functionality breakdown
  fig4b_hatecheck_en_funcs.png  — HateCheck EN per-functionality breakdown
  fig5_en_vs_fr_bar.png         — HC-EN vs HC-FR F1 gap per model
  fig6_efficiency_frontier.png  — Inference latency vs HC-FR F1
  fig7_radar_deployable.png     — Radar: deployable tier comparison

Usage:
    python code/visualize_results.py \
        --results_dir ~/code/results \
        --output_dir  code/docs/figures

    # If results synced locally:
    python code/visualize_results.py \
        --results_dir code/results \
        --output_dir  code/docs/figures
"""

import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed — heatmaps use matplotlib fallback. "
          "Install with: pip install seaborn")


# ── Constants ─────────────────────────────────────────────────────────────────

# JSON model name → short display label
MODEL_MAP = {
    'detoxify-multilingual':    'Detox-M',
    'detoxify-unbiased':        'Detox-U',
    'EthicalEye':               'EthEye',
    'CitizenLab-XLM-RoBERTa':   'CitizenLab',
    'KoalaAI-Text-Moderation':  'KoalaAI',
    'Llama-Guard-3-1B':         'LG-1B',
    'Llama-Guard-3-8B':         'LG-8B',
    'ShieldGemma-2b':           'SG-2b',
    'ShieldGemma-9b':           'SG-9b',
    'Mistral-7B-Instruct-v0.3': 'Mistral-7B',
}

# Short label → full name for legends
MODEL_FULL = {
    'Detox-M':    'detoxify-multilingual',
    'Detox-U':    'detoxify-unbiased',
    'EthEye':     'EthicalEye',
    'CitizenLab': 'CitizenLab',
    'KoalaAI':    'KoalaAI',
    'LG-1B':      'Llama-Guard-3-1B',
    'LG-8B':      'Llama-Guard-3-8B',
    'SG-2b':      'ShieldGemma-2b',
    'SG-9b':      'ShieldGemma-9b',
    'Mistral-7B': 'Mistral-7B',
}

# Canonical model order: sorted by HC-FR F1 descending
MODEL_ORDER = ['SG-9b', 'LG-8B', 'SG-2b', 'Mistral-7B', 'Detox-M',
               'LG-1B', 'CitizenLab', 'EthEye', 'Detox-U', 'KoalaAI']

# Dataset directory key → display label
DATASET_MAP = {
    'hatecheck_fr':         'HC-FR',
    'french_hate_superset': 'FR-Hate',
    'reddit_fr':            'Red-FR',
    'hatecheck_en':         'HC-EN',
    'toxigen':              'ToxiGen',
    'openai':               'OpenAI',
    'civil_comments':       'CivComm',
    'reddit_en':            'Red-EN',
}

# Column order in figures: French datasets left, English right
DATASET_ORDER = ['HC-FR', 'FR-Hate', 'Red-FR', 'HC-EN', 'ToxiGen', 'OpenAI', 'CivComm', 'Red-EN']
N_FRENCH = 3   # number of French dataset columns

# VRAM tier label + color per model
TIER_INFO = {
    'SG-9b':      ('≥14 GB — not viable',      '#c0392b'),
    'LG-8B':      ('≥14 GB — not viable',      '#c0392b'),
    'Mistral-7B': ('≥14 GB — not viable',      '#c0392b'),
    'SG-2b':      ('~5.7 GB — sweet spot',     '#e67e22'),
    'LG-1B':      ('~3 GB — mid-tier',         '#f1c40f'),
    'Detox-M':    ('≤1.1 GB — deployable',     '#27ae60'),
    'CitizenLab': ('≤1.1 GB — deployable',     '#27ae60'),
    'EthEye':     ('≤1.1 GB — deployable',     '#27ae60'),
    'Detox-U':    ('≤1.1 GB — deployable',     '#27ae60'),
    'KoalaAI':    ('≤1.1 GB — not viable (FR)','#95a5a6'),
}

# Per-model colors for multi-model line/bar plots
MODEL_COLORS = {m: plt.cm.tab10(i / 10) for i, m in enumerate(MODEL_ORDER)}


# ── Style ─────────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
        'axes.labelsize':    12,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.fontsize':   10,
        'figure.dpi':        150,
        'savefig.dpi':       150,
        'savefig.bbox':      'tight',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
    })


def save_fig(fig, output_dir: Path, name: str):
    path = output_dir / name
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_result(baseline_dir: Path, dataset_key: str, model_raw: str) -> dict | None:
    """Look in flat dir first, fall back to run_1/."""
    fname = model_raw.replace('/', '_') + '.json'
    for subdir in [Path('.'), Path('run_1')]:
        p = baseline_dir / subdir / dataset_key / fname
        if p.exists():
            return _load_json(p)
    return None


def load_baseline(results_dir: Path) -> pd.DataFrame:
    """Load F1 + deployability metrics from full_baseline_v3/."""
    baseline_dir = results_dir / 'full_baseline_v3'
    if not baseline_dir.exists():
        print(f"  Warning: {baseline_dir} not found — baseline figures will be skipped.")
        return pd.DataFrame()

    rows = []
    for model_raw, model_short in MODEL_MAP.items():
        for dataset_key, dataset_disp in DATASET_MAP.items():
            data = _find_result(baseline_dir, dataset_key, model_raw)
            if data is None:
                continue
            rows.append({
                'model':         model_short,
                'dataset':       dataset_disp,
                'f1':            data.get('f1', 0.0),
                'tpr':           data.get('true_positive_rate', 0.0),
                'tnr':           data.get('true_negative_rate', 0.0),
                'fpr':           data.get('false_positive_rate', 0.0),
                'gpu_mb':        data.get('gpu_memory_mb', 0.0),
                'ms_per_sample': data.get('avg_inference_ms', 0.0),
                'energy_kwh':    data.get('energy_kwh', 0.0),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("  Warning: no baseline results loaded.")
    else:
        print(f"  Loaded {len(df)} baseline (model, dataset) pairs from {baseline_dir.name}/")
    return df


def load_hatecheck_metrics(results_dir: Path) -> pd.DataFrame:
    """Load top-level TPR/TNR/F1 from hatecheck_analysis/ for EN and FR."""
    hc_dir = results_dir / 'hatecheck_analysis'
    if not hc_dir.exists():
        print(f"  Warning: {hc_dir} not found — hatecheck figures will be skipped.")
        return pd.DataFrame()

    rows = []
    for model_raw, model_short in MODEL_MAP.items():
        fname = model_raw.replace('/', '_') + '.json'
        row = {'model': model_short}
        for lang in ('en', 'fr'):
            p = hc_dir / f'hatecheck_{lang}' / fname
            if p.exists():
                d = _load_json(p)
                row[f'f1_{lang}']  = d.get('f1', 0.0)
                row[f'tpr_{lang}'] = d.get('true_positive_rate', 0.0)
                row[f'tnr_{lang}'] = d.get('true_negative_rate', 0.0)
            else:
                row[f'f1_{lang}']  = None
                row[f'tpr_{lang}'] = None
                row[f'tnr_{lang}'] = None
        rows.append(row)

    df = pd.DataFrame(rows).set_index('model')
    n_loaded = df['f1_fr'].notna().sum()
    print(f"  Loaded hatecheck metrics for {n_loaded}/{len(df)} models (FR).")
    return df


def load_functionality(results_dir: Path, lang: str) -> pd.DataFrame:
    """Load per-functionality correct-rates; rows=funcs, cols=models."""
    hc_dir = results_dir / 'hatecheck_analysis' / f'hatecheck_{lang}'
    if not hc_dir.exists():
        print(f"  Warning: {hc_dir} not found.")
        return pd.DataFrame()

    func_data: dict[str, dict[str, float]] = {}
    loaded = 0
    for model_raw, model_short in MODEL_MAP.items():
        fname = model_raw.replace('/', '_') + '.json'
        p = hc_dir / fname
        if not p.exists():
            continue
        try:
            d = _load_json(p)
            for func, cat in d.get('by_functionality', {}).items():
                func_data.setdefault(func, {})[model_short] = cat.get('correct_rate', 0.0)
            loaded += 1
        except Exception as e:
            print(f"    Warning: could not load {p.name}: {e}")

    if not func_data:
        print(f"  Warning: no functionality data for hatecheck_{lang}.")
        return pd.DataFrame()

    df = pd.DataFrame(func_data).T
    cols = [m for m in MODEL_ORDER if m in df.columns]
    df = df[cols]
    print(f"  Loaded {lang.upper()} functionality data: {len(df)} funcs × {len(cols)} models.")
    return df


def _func_type(name: str) -> str:
    """Determine if a HateCheck functionality is hateful or non-hateful by suffix."""
    return 'non_hateful' if name.endswith('_nh') else 'hateful'


# ── Figure helpers ────────────────────────────────────────────────────────────

def _tier_legend_handles():
    return [
        mpatches.Patch(color='#27ae60', label='≤1.1 GB — deployable (CPU-feasible)'),
        mpatches.Patch(color='#f1c40f', label='~3 GB — mid-tier'),
        mpatches.Patch(color='#e67e22', label='~5.7 GB — sweet spot'),
        mpatches.Patch(color='#c0392b', label='≥14 GB — not viable for Shareish'),
        mpatches.Patch(color='#95a5a6', label='≤1.1 GB — not viable (French-only issue)'),
    ]


def _annotate_scatter(ax, x, y, label, dx, dy):
    ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.7))


# ── Figure 1: F1 Heatmap ─────────────────────────────────────────────────────

def fig1_f1_heatmap(baseline_df: pd.DataFrame, output_dir: Path):
    """10 models × 8 datasets F1 heatmap. French datasets left, English right."""
    if baseline_df.empty:
        print("  Fig 1 skipped.")
        return

    pivot = baseline_df.pivot(index='model', columns='dataset', values='f1')
    rows  = [m for m in MODEL_ORDER if m in pivot.index]
    cols  = [d for d in DATASET_ORDER if d in pivot.columns]
    pivot = pivot.loc[rows, cols]

    # Build annotation matrix: value or '—' for missing
    annot = pivot.map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')
    data  = pivot.fillna(-0.01)   # -0.01 renders as lowest color (near-white)

    fig, ax = plt.subplots(figsize=(13, 7))
    if HAS_SEABORN:
        sns.heatmap(data, annot=annot, fmt='', cmap='Blues',
                    vmin=0, vmax=1, linewidths=0.5, linecolor='#dddddd',
                    ax=ax, annot_kws={'size': 9})
    else:
        im = ax.imshow(data.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, label='F1')
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, annot.iloc[i, j], ha='center', va='center', fontsize=9)
        ax.set_xticks(range(len(cols)));  ax.set_xticklabels(cols)
        ax.set_yticks(range(len(rows)));  ax.set_yticklabels(rows)

    # Separator between French and English columns
    n_fr = sum(1 for c in cols if c in ('HC-FR', 'FR-Hate', 'Red-FR'))
    ax.axvline(x=n_fr, color='#222222', linewidth=2.5)
    tr = ax.get_xaxis_transform()
    ax.text(n_fr / 2,        -0.55, '← French datasets →', ha='center',
            fontsize=9.5, color='#c0392b', fontweight='bold', transform=tr)
    ax.text(n_fr + (len(cols) - n_fr) / 2, -0.55, '← English datasets →', ha='center',
            fontsize=9.5, color='#2980b9', fontweight='bold', transform=tr)

    ax.set_title('F1 Score — 10 Models × 8 Datasets  (v3 baseline, sorted by HC-FR)', pad=22)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    save_fig(fig, output_dir, 'fig1_f1_heatmap.png')


# ── Figure 2: VRAM vs HC-FR F1 ───────────────────────────────────────────────

def fig2_vram_scatter(baseline_df: pd.DataFrame, output_dir: Path):
    """The thesis deployability argument in one figure."""
    if baseline_df.empty:
        print("  Fig 2 skipped.")
        return

    deploy = baseline_df.groupby('model')[['gpu_mb', 'ms_per_sample']].mean()
    hc_fr  = baseline_df[baseline_df['dataset'] == 'HC-FR'].set_index('model')['f1']
    deploy['hc_fr_f1'] = hc_fr

    fig, ax = plt.subplots(figsize=(11, 7))

    # Per-model annotation offsets (manually tuned to avoid overlap)
    offsets = {
        'SG-9b':      ( 1500,  0.006), 'LG-8B':      ( 1500,  0.006),
        'Mistral-7B': ( 1000, -0.018), 'SG-2b':      (-3000, -0.030),
        'LG-1B':      (  200,  0.010), 'Detox-M':    (  100,  0.010),
        'CitizenLab': (  100, -0.022), 'EthEye':     (  100, -0.036),
        'Detox-U':    (  100,  0.010), 'KoalaAI':    (  100, -0.022),
    }

    for model in MODEL_ORDER:
        if model not in deploy.index or pd.isna(deploy.loc[model, 'hc_fr_f1']):
            continue
        row   = deploy.loc[model]
        color = TIER_INFO.get(model, ('', '#888888'))[1]
        size  = max(80, row['ms_per_sample'] * 3.5)
        ax.scatter(row['gpu_mb'], row['hc_fr_f1'], s=size, color=color,
                   zorder=5, alpha=0.88, edgecolors='white', linewidth=1.5)
        dx, dy = offsets.get(model, (200, 0.006))
        _annotate_scatter(ax, row['gpu_mb'], row['hc_fr_f1'], model, dx, dy)

    # Shareish deployment ceiling
    ax.axvline(x=6000, color='#c0392b', linestyle='--', linewidth=1.8, zorder=3)
    ax.text(6400, 0.06, 'Shareish\nVRAM ceiling\n(~6 GB)',
            color='#c0392b', fontsize=9, va='bottom')
    ax.axvspan(0, 6000, alpha=0.03, color='#27ae60')   # deployable zone shading

    ax.set_xscale('log')
    ax.set_xlabel('Peak GPU Memory (MB, log scale)', fontsize=11)
    ax.set_ylabel('HateCheck-FR F1 Score', fontsize=11)
    ax.set_title('Deployability vs. French Performance\n'
                 '(dot size proportional to inference time ms/sample)', pad=14)
    ax.set_xlim(200, 40000)
    ax.set_ylim(-0.02, 1.02)

    # Size legend (ms/sample reference dots)
    for ms, label in [(6, '6 ms'), (40, '40 ms'), (150, '150 ms')]:
        ax.scatter([], [], s=max(80, ms * 3.5), color='#888888',
                   alpha=0.5, label=f'{label}/sample')
    size_legend = ax.legend(loc='upper left', fontsize=8.5, title='Dot size = ms/sample',
                            title_fontsize=8.5, framealpha=0.9)
    ax.add_artist(size_legend)
    ax.legend(handles=_tier_legend_handles(), loc='lower right',
              fontsize=8.5, framealpha=0.9)

    save_fig(fig, output_dir, 'fig2_vram_vs_fr_f1.png')


# ── Figure 3: TPR vs TNR ─────────────────────────────────────────────────────

def fig3_tpr_tnr(hatecheck_df: pd.DataFrame, output_dir: Path):
    """Sensitivity vs specificity for EN (left) and FR (right)."""
    if hatecheck_df.empty:
        print("  Fig 3 skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, lang, title in [
        (axes[0], 'en', 'HateCheck — English'),
        (axes[1], 'fr', 'HateCheck — French'),
    ]:
        for model in MODEL_ORDER:
            if model not in hatecheck_df.index:
                continue
            tpr = hatecheck_df.loc[model, f'tpr_{lang}']
            tnr = hatecheck_df.loc[model, f'tnr_{lang}']
            if tpr is None or tnr is None or pd.isna(tpr):
                continue
            color = TIER_INFO.get(model, ('', '#888888'))[1]
            ax.scatter(tnr, tpr, s=130, color=color, zorder=5,
                       edgecolors='white', linewidth=1.5)
            # Nudge annotations to avoid overlap
            nudge = {'SG-9b': (-0.13, 0.01), 'SG-2b': (-0.11, -0.03),
                     'LG-8B': (0.01, 0.01), 'Mistral-7B': (0.01, -0.03),
                     'Detox-M': (0.01, 0.01), 'LG-1B': (0.01, -0.03),
                     'CitizenLab': (0.01, 0.01), 'EthEye': (0.01, -0.03),
                     'Detox-U': (0.01, 0.01), 'KoalaAI': (0.01, -0.03)}
            dx, dy = nudge.get(model, (0.01, 0.01))
            ax.text(tnr + dx, tpr + dy, model, fontsize=8.5, fontweight='bold')

        # Quadrant guide lines
        ax.axhline(y=0.5, color='#aaaaaa', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.axvline(x=0.5, color='#aaaaaa', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.text(0.97, 0.97, '★ Ideal', ha='right', va='top', fontsize=9,
                color='#27ae60', fontweight='bold', transform=ax.transAxes)
        ax.text(0.97, 0.03, 'High FP rate', ha='right', va='bottom', fontsize=8,
                color='#888888', style='italic', transform=ax.transAxes)
        ax.text(0.03, 0.97, 'Misses hate', ha='left', va='top', fontsize=8,
                color='#888888', style='italic', transform=ax.transAxes)

        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('TNR — Specificity  (avoids false alarms)', fontsize=10)
        ax.set_ylabel('TPR — Sensitivity  (catches hate speech)', fontsize=10)
        ax.set_title(title, fontsize=12)

    fig.suptitle('Sensitivity vs Specificity — HateCheck EN and FR',
                 fontsize=13, fontweight='bold', y=1.01)

    handles = [mpatches.Patch(color=TIER_INFO[m][1], label=MODEL_FULL[m])
               for m in MODEL_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.09), framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, output_dir, 'fig3_tpr_vs_tnr.png')


# ── Figure 4: Functionality Heatmaps ─────────────────────────────────────────

def fig4_functionality_heatmap(results_dir: Path, output_dir: Path):
    """HateCheck per-functionality correct-rate heatmap for FR and EN."""
    for lang, fname, title in [
        ('fr', 'fig4a_hatecheck_fr_funcs.png',
         'HateCheck FR — Per-Functionality Correct-Rate  (deployment language)'),
        ('en', 'fig4b_hatecheck_en_funcs.png',
         'HateCheck EN — Per-Functionality Correct-Rate'),
    ]:
        df = load_functionality(results_dir, lang)
        if df.empty:
            print(f"  Fig 4 ({lang}) skipped.")
            continue

        # Sort: non-hateful (NH) rows first, then hateful (H)
        nh_funcs = sorted([f for f in df.index if _func_type(f) == 'non_hateful'])
        h_funcs  = sorted([f for f in df.index if _func_type(f) == 'hateful'])
        ordered  = nh_funcs + h_funcs
        df = df.loc[ordered]
        n_nh = len(nh_funcs)

        # Clean up row labels
        df.index = [
            f.replace('_nh', ' ⬥NH').replace('_h', ' ⬥H')
             .replace('counter_quote', 'counter-quote')
             .replace('counter_ref', 'counter-ref')
             .replace('derog_dehum', 'derog-dehumanise')
             .replace('derog_impl', 'derog-implicit')
             .replace('negate_neg', 'negate-neg')
             .replace('negate_pos', 'negate-pos')
             .replace('profanity', 'profanity')
             .replace('slur_reclaimed', 'slur-reclaimed')
             .replace('slur_h', 'slur ⬥H')
             .replace('slur_homonym', 'slur-homonym')
             .replace('spell_char_del', 'obfusc-char-del')
             .replace('spell_leet', 'obfusc-leet')
             .replace('target_group', 'target-group')
             .replace('target_indiv', 'target-indiv')
             .replace('threat_dir', 'threat-direct')
            for f in df.index
        ]

        fig_h = max(8, len(df) * 0.55)
        fig, ax = plt.subplots(figsize=(13, fig_h))

        data  = df.astype(float)
        annot = data.map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')

        if HAS_SEABORN:
            sns.heatmap(data, annot=annot, fmt='', cmap='RdYlGn',
                        vmin=0, vmax=1, linewidths=0.4, linecolor='#eeeeee',
                        ax=ax, annot_kws={'size': 8})
        else:
            im = ax.imshow(data.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, label='Correct-rate')
            ax.set_xticks(range(len(data.columns)))
            ax.set_xticklabels(data.columns, rotation=30, ha='right')
            ax.set_yticks(range(len(data.index)))
            ax.set_yticklabels(data.index)
            for i in range(len(data)):
                for j in range(len(data.columns)):
                    ax.text(j, i, annot.iloc[i, j], ha='center', va='center', fontsize=7.5)

        # Separator between NH and H sections
        ax.axhline(y=n_nh, color='#333333', linewidth=2.5)

        # Section labels in the left margin
        ax.text(-0.02, (n_nh / 2) / len(df), 'Non-Hateful',
                ha='right', va='center', fontsize=8.5, color='#c0392b',
                fontweight='bold', rotation=90, transform=ax.transAxes)
        ax.text(-0.02, (n_nh + len(h_funcs) / 2) / len(df), 'Hateful',
                ha='right', va='center', fontsize=8.5, color='#27ae60',
                fontweight='bold', rotation=90, transform=ax.transAxes)

        ax.set_title(title, pad=14)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)
        ax.set_xlabel('')

        plt.tight_layout()
        save_fig(fig, output_dir, fname)


# ── Figure 5: EN vs FR Bar ───────────────────────────────────────────────────

def fig5_en_vs_fr_bar(hatecheck_df: pd.DataFrame, output_dir: Path):
    """HC-EN vs HC-FR F1 grouped bar chart + French gap subplot."""
    if hatecheck_df.empty:
        print("  Fig 5 skipped.")
        return

    models = [m for m in MODEL_ORDER if m in hatecheck_df.index]
    en_f1  = [hatecheck_df.loc[m, 'f1_en'] or 0.0 for m in models]
    fr_f1  = [hatecheck_df.loc[m, 'f1_fr'] or 0.0 for m in models]
    gap    = [e - f for e, f in zip(en_f1, fr_f1)]

    x, w = np.arange(len(models)), 0.36

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    # Top panel: grouped bars
    bars_en = ax_top.bar(x - w / 2, en_f1, w, label='HC-EN F1',
                         color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_fr = ax_top.bar(x + w / 2, fr_f1, w, label='HC-FR F1',
                         color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.5)

    for bar in list(bars_en) + list(bars_fr):
        h = bar.get_height()
        if h > 0.03:
            ax_top.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(models, fontsize=10)
    ax_top.set_ylabel('F1 Score')
    ax_top.set_ylim(0, 1.08)
    ax_top.set_xlim(-0.6, len(models) - 0.4)
    ax_top.axhline(y=0.5, color='#aaaaaa', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_top.set_title('HateCheck: English vs French F1  (models sorted by HC-FR)',
                     fontweight='bold')
    ax_top.legend(fontsize=10)
    ax_top.set_xticklabels([])   # shared x-axis, labels on bottom panel

    # Bottom panel: gap bars
    gap_colors = ['#c0392b' if g > 0.35 else '#e67e22' if g > 0.15 else '#27ae60'
                  for g in gap]
    ax_bot.bar(x, gap, color=gap_colors, alpha=0.82, edgecolor='white', linewidth=0.5)
    ax_bot.axhline(y=0, color='black', linewidth=0.9)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(models, fontsize=10)
    ax_bot.set_ylabel('EN − FR gap')
    ax_bot.set_xlim(-0.6, len(models) - 0.4)
    ax_bot.set_ylim(-0.05, max(gap) * 1.15 + 0.05)
    ax_bot.tick_params(axis='x', labelsize=10)

    save_fig(fig, output_dir, 'fig5_en_vs_fr_bar.png')


# ── Figure 6: Efficiency Frontier ────────────────────────────────────────────

def fig6_efficiency_frontier(baseline_df: pd.DataFrame, output_dir: Path):
    """Inference time vs HC-FR F1 (dot size encodes VRAM)."""
    if baseline_df.empty:
        print("  Fig 6 skipped.")
        return

    deploy = baseline_df.groupby('model')[['gpu_mb', 'ms_per_sample']].mean()
    hc_fr  = baseline_df[baseline_df['dataset'] == 'HC-FR'].set_index('model')['f1']
    deploy['hc_fr_f1'] = hc_fr

    fig, ax = plt.subplots(figsize=(11, 7))

    # Per-model annotation offsets
    offsets = {
        'SG-9b':      (  4,  0.006), 'LG-8B':      (  8,  0.010),
        'Mistral-7B': (  8, -0.020), 'SG-2b':      ( -5, -0.030),
        'LG-1B':      (  2,  0.010), 'Detox-M':    (  0.3,  0.010),
        'CitizenLab': (  0.3, -0.024), 'EthEye':   (  0.3, -0.038),
        'Detox-U':    (  0.3,  0.010), 'KoalaAI':  (  0.3, -0.024),
    }

    for model in MODEL_ORDER:
        if model not in deploy.index or pd.isna(deploy.loc[model, 'hc_fr_f1']):
            continue
        row   = deploy.loc[model]
        color = TIER_INFO.get(model, ('', '#888888'))[1]
        size  = max(80, row['gpu_mb'] / 60)   # size ∝ VRAM
        ax.scatter(row['ms_per_sample'], row['hc_fr_f1'], s=size, color=color,
                   zorder=5, alpha=0.88, edgecolors='white', linewidth=1.5)
        dx, dy = offsets.get(model, (1, 0.006))
        _annotate_scatter(ax, row['ms_per_sample'], row['hc_fr_f1'], model, dx, dy)

    ax.set_xscale('log')
    ax.set_xlabel('Average Inference Time (ms/sample, log scale)', fontsize=11)
    ax.set_ylabel('HateCheck-FR F1 Score', fontsize=11)
    ax.set_title('Efficiency Frontier — Inference Speed vs French Performance\n'
                 '(dot size ∝ VRAM; upper-left = fast and accurate)', pad=14)
    ax.set_ylim(-0.02, 1.02)

    # VRAM size legend
    for vram_mb, label in [(500, '0.5 GB'), (3000, '3 GB'), (16000, '16 GB')]:
        ax.scatter([], [], s=max(80, vram_mb / 60), color='#888888',
                   alpha=0.5, label=label)
    size_legend = ax.legend(loc='upper left', fontsize=8.5,
                            title='Dot size = VRAM', title_fontsize=8.5, framealpha=0.9)
    ax.add_artist(size_legend)
    ax.legend(handles=_tier_legend_handles(), loc='lower right',
              fontsize=8.5, framealpha=0.9)

    save_fig(fig, output_dir, 'fig6_efficiency_frontier.png')


# ── Figure 7: Radar Chart ────────────────────────────────────────────────────

def fig7_radar(baseline_df: pd.DataFrame, output_dir: Path):
    """Radar: compare the three deployable-tier models across 6 datasets."""
    if baseline_df.empty:
        print("  Fig 7 skipped.")
        return

    radar_models   = ['Detox-M', 'LG-1B', 'SG-2b']
    radar_labels   = ['HC-FR', 'FR-Hate', 'Red-FR', 'HC-EN', 'CivComm', 'Red-EN']
    radar_colors   = ['#27ae60', '#f1c40f', '#e67e22']

    pivot = baseline_df.pivot(index='model', columns='dataset', values='f1')
    missing = [m for m in radar_models if m not in pivot.index]
    if missing:
        print(f"  Fig 7 skipped: missing models {missing}.")
        return

    N      = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, color in zip(radar_models, radar_colors):
        vals = [pivot.loc[model, ds] if ds in pivot.columns else 0.0
                for ds in radar_labels]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2.2, color=color,
                label=MODEL_FULL.get(model, model))
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontsize=8, color='#777777')
    ax.set_title('Deployable Tier Comparison\n'
                 '(models viable for Shareish — sorted by French performance)',
                 pad=22, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12), fontsize=10)
    ax.grid(True, alpha=0.35)

    save_fig(fig, output_dir, 'fig7_radar_deployable.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate Phase 1 baseline visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results_dir', default='~/code/results',
                        help='Root results directory (default: ~/code/results)')
    parser.add_argument('--output_dir',  default='code/docs/figures',
                        help='Output directory for figures (default: code/docs/figures)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir  = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}\n")

    set_style()

    print("Loading data...")
    baseline_df  = load_baseline(results_dir)
    hatecheck_df = load_hatecheck_metrics(results_dir)

    print("\nGenerating figures...")
    fig1_f1_heatmap(baseline_df, output_dir)
    fig2_vram_scatter(baseline_df, output_dir)
    fig3_tpr_tnr(hatecheck_df, output_dir)
    fig4_functionality_heatmap(results_dir, output_dir)
    fig5_en_vs_fr_bar(hatecheck_df, output_dir)
    fig6_efficiency_frontier(baseline_df, output_dir)
    fig7_radar(baseline_df, output_dir)

    print(f"\nDone. {len(list(output_dir.glob('*.png')))} figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
