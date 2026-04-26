"""
Dataset Statistics Explorer — Phase 1, Pass 2
=============================================
Computes comprehensive statistics for each of the 8 evaluation datasets.

For every dataset:
  1. Binary label derivation — exact formula, including how missing values
     and string→int conversions are handled (for non-trivial cases)
  2. Label distribution — counts, percentages, balance ratio
  3. Text length — mean / median / std / min / max in words AND characters
     computed over the FULL dataset (no sampling)
  4. Null / missing value audit per field
  5. Breakdown statistics where meaningful:
       HateCheck  : per functionality type
       FHS        : per sub-dataset (dataset column)
       OpenAI     : per category flag + co-occurrence
       Reddit     : per subreddit + length by label
       Civil Cmts : per split + toxicity score histogram + sub-scores

Run on the cluster after reviewing Pass 1 output.

Usage:
  python explore_datasets_pass2.py \\
      --cache_dir ~/datasets/cache \\
      --openai_path ~/datasets/openai/samples-1680.jsonl \\
      --toxigen_path ~/datasets/toxigen/toxigen_train.jsonl \\
      --reddit_en_path ~/datasets/reddit/balanced/data-en/test-en.csv \\
      --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv \\
      --output ~/code/results/dataset_exploration/statistics.txt
"""

import torch  # noqa: F401 — loads libstdc++ symbols needed by pandas C extensions
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ── formatting ────────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72

def hdr(title):
    print(f"\n{SEP}\n  STATISTICS — {title}\n{SEP}")

def sub(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def kv(key, value, width=28):
    print(f"  {(key + ':'):{width}s}  {value}")


# ── shared helpers ─────────────────────────────────────────────────────────────

def length_stats(texts):
    """Compute word and char length statistics over an iterable of strings."""
    words = np.array([len(str(t).split()) for t in texts])
    chars = np.array([len(str(t))         for t in texts])
    def fmt(arr):
        return {
            "mean":   round(float(arr.mean()), 1),
            "median": round(float(np.median(arr)), 1),
            "std":    round(float(arr.std()), 1),
            "min":    int(arr.min()),
            "max":    int(arr.max()),
        }
    return {"words": fmt(words), "chars": fmt(chars)}

def print_length_stats(st):
    w, c = st["words"], st["chars"]
    print(f"  {'Text length (words)':28s}  "
          f"mean={w['mean']:7.1f}  median={w['median']:7.1f}  "
          f"std={w['std']:7.1f}  min={w['min']:5d}  max={w['max']:6d}")
    print(f"  {'Text length (chars)':28s}  "
          f"mean={c['mean']:7.1f}  median={c['median']:7.1f}  "
          f"std={c['std']:7.1f}  min={c['min']:5d}  max={c['max']:6d}")

def label_block(labels_arr):
    """Print label distribution. labels_arr: numpy int array of 0/1."""
    n     = len(labels_arr)
    n_pos = int(labels_arr.sum())
    n_neg = n - n_pos
    ratio = n_pos / n_neg if n_neg > 0 else float('inf')
    direction = "toward positive" if ratio > 1 else "toward negative"
    print(f"  {'Total rows:':28s}  {n:,}")
    print(f"  {'  Positive (label=1):':28s}  {n_pos:,}  ({n_pos/n*100:.1f}%)")
    print(f"  {'  Negative (label=0):':28s}  {n_neg:,}  ({n_neg/n*100:.1f}%)")
    print(f"  {'  Balance ratio:':28s}  {ratio:.3f}:1  ({direction})")

def null_audit(df, cols):
    sub("Null / missing value audit")
    for col in cols:
        n_null = int(df[col].isna().sum())
        n      = len(df)
        print(f"  {col:30s}  {n_null:6,} nulls  ({n_null/n*100:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset stat functions
# ═══════════════════════════════════════════════════════════════════════════════

def stats_hatecheck(name, hf_id, cache_dir):
    from datasets import load_dataset
    hdr(f"{name}  |  {hf_id}")

    print("""
  Binary label derivation:
    label_gold == 'hateful'     →  1  (positive / unsafe)
    label_gold == 'non-hateful' →  0  (negative / safe)
    Direct expert annotation — no aggregation or threshold needed.
""")

    ds    = load_dataset(hf_id, cache_dir=cache_dir)
    df    = ds['test'].to_pandas()
    df['label'] = (df['label_gold'] == 'hateful').astype(int)

    label_block(df['label'].values)
    print_length_stats(length_stats(df['test_case']))
    null_audit(df, ['test_case', 'label_gold'])

    # Per-functionality breakdown
    sub("Label distribution per functionality")
    grp = df.groupby('functionality')['label'].agg(['sum', 'count'])
    grp['neg']  = grp['count'] - grp['sum']
    grp['pos%'] = grp['sum'] / grp['count'] * 100
    grp = grp.sort_values('pos%', ascending=False)
    print(f"  {'Functionality':40s}  {'N':>5}  {'Pos':>5}  {'Neg':>5}  {'Pos%':>6}")
    for fn, row in grp.iterrows():
        print(f"  {fn:40s}  {int(row['count']):5d}  "
              f"{int(row['sum']):5d}  {int(row['neg']):5d}  {row['pos%']:5.1f}%")


def stats_fhs(cache_dir):
    from datasets import load_dataset
    hdr("French Hate Speech Superset  |  manueltonneau/french-hate-speech-superset")

    print("""
  Binary label derivation:
    labels == 1  →  1  (positive / hateful)
    labels == 0  →  0  (negative / safe)
    Field type: int64 — already binary, no conversion needed.
    Harmonised by Tonneau et al. from 5 source datasets before release.
""")

    ds = load_dataset("manueltonneau/french-hate-speech-superset", cache_dir=cache_dir)
    df = ds['train'].to_pandas()

    label_block(df['labels'].values)
    print_length_stats(length_stats(df['text']))
    null_audit(df, ['text', 'labels', 'target', 'dataset', 'source'])

    # Per sub-dataset
    sub("Label distribution per sub-dataset")
    grp = df.groupby(['dataset', 'source'])['labels'].agg(['sum', 'count']).reset_index()
    grp['neg']  = grp['count'] - grp['sum']
    grp['pos%'] = grp['sum'] / grp['count'] * 100
    grp = grp.sort_values('count', ascending=False)
    print(f"  {'Sub-dataset':20s}  {'Source':20s}  {'N':>6}  {'Pos':>6}  {'Neg':>6}  {'Pos%':>6}")
    for _, row in grp.iterrows():
        print(f"  {row['dataset']:20s}  {row['source']:20s}  "
              f"{int(row['count']):6d}  {int(row['sum']):6d}  "
              f"{int(row['neg']):6d}  {row['pos%']:5.1f}%")

    # Per nb_annotators
    sub("Label distribution per number of annotators")
    grp2 = df.groupby('nb_annotators')['labels'].agg(['sum', 'count'])
    grp2['neg']  = grp2['count'] - grp2['sum']
    grp2['pos%'] = grp2['sum'] / grp2['count'] * 100
    print(f"  {'Annotators':12s}  {'N':>6}  {'Pos':>6}  {'Neg':>6}  {'Pos%':>6}")
    for ann, row in grp2.iterrows():
        print(f"  {ann:<12}  {int(row['count']):6d}  {int(row['sum']):6d}  "
              f"{int(row['neg']):6d}  {row['pos%']:5.1f}%")

    # Per target
    sub("Label distribution per target group (top 15)")
    grp3 = df.groupby('target')['labels'].agg(['sum', 'count'])
    grp3['pos%'] = grp3['sum'] / grp3['count'] * 100
    grp3 = grp3.sort_values('count', ascending=False).head(15)
    print(f"  {'Target':25s}  {'N':>6}  {'Pos':>6}  {'Pos%':>6}")
    for tgt, row in grp3.iterrows():
        print(f"  {str(tgt):25s}  {int(row['count']):6d}  {int(row['sum']):6d}  {row['pos%']:5.1f}%")


def stats_toxigen(path):
    hdr(f"ToxiGen  |  {path}")

    print("""
  Binary label derivation:
    prompt_label == '1'  →  1  (positive / toxic implicit hate)
    prompt_label == '0'  →  0  (negative / benign)
    Field type: string in JSON — requires explicit str→int cast.
    Label source: machine-generated from GPT-3 ALICE prompting.
    Human labels exist for ~8K examples only (annotator_labels field).
""")

    raw = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    df  = pd.DataFrame(raw)
    df['label'] = df['prompt_label'].astype(int)

    label_block(df['label'].values)
    print_length_stats(length_stats(df['generation']))
    null_audit(df, ['generation', 'prompt_label', 'group'])

    # Per target group
    sub("Label distribution per target group")
    grp = df.groupby('group')['label'].agg(['sum', 'count'])
    grp['neg']  = grp['count'] - grp['sum']
    grp['pos%'] = grp['sum'] / grp['count'] * 100
    grp = grp.sort_values('count', ascending=False)
    print(f"  {'Group':35s}  {'N':>7}  {'Pos':>7}  {'Neg':>7}  {'Pos%':>6}")
    for g, row in grp.iterrows():
        print(f"  {g:35s}  {int(row['count']):7d}  {int(row['sum']):7d}  "
              f"{int(row['neg']):7d}  {row['pos%']:5.1f}%")

    # Generation method
    sub("Label distribution per generation method")
    grp2 = df.groupby('generation_method')['label'].agg(['sum', 'count'])
    grp2['pos%'] = grp2['sum'] / grp2['count'] * 100
    print(f"  {'Method':12s}  {'N':>8}  {'Pos':>8}  {'Pos%':>6}")
    for gm, row in grp2.iterrows():
        print(f"  {gm:12s}  {int(row['count']):8d}  {int(row['sum']):8d}  {row['pos%']:5.1f}%")


def stats_openai(path):
    hdr(f"OpenAI Moderation Dataset  |  {path}")

    print("""
  Binary label derivation:
    label = OR(S, H, V, HR, SH, S3, H2, V2)
    Each flag is an integer (0 or 1); missing fields treated as 0.
    A row is positive if ANY of the 8 flags equals 1.

    Category meanings:
      S  = Sexual              H  = Hate
      V  = Violence            HR = Harassment
      SH = Self-harm           S3 = Sexual/minors
      H2 = Hate/threatening    V2 = Violence/graphic
""")

    raw  = [json.loads(l) for l in open(path) if l.strip()]
    df   = pd.DataFrame(raw)
    cats = ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
    for c in cats:
        if c not in df.columns:
            df[c] = 0
    df[cats] = df[cats].fillna(0).astype(int)
    df['label'] = df[cats].any(axis=1).astype(int)

    label_block(df['label'].values)
    print_length_stats(length_stats(df['prompt']))
    null_audit(df, ['prompt'])

    # Per-category
    sub("Per-category flag statistics")
    cat_names = {
        'S':'Sexual', 'H':'Hate', 'V':'Violence', 'HR':'Harassment',
        'SH':'Self-harm', 'S3':'Sexual/minors', 'H2':'Hate/threatening',
        'V2':'Violence/graphic'
    }
    n = len(df)
    print(f"  {'Flag':4s}  {'Name':20s}  {'Rows w/ field':>13}  {'Flagged':>8}  {'Flag%':>7}")
    for c in cats:
        # Count rows where field was originally present (not NaN before fill)
        original_df = pd.DataFrame(raw)
        present = int(original_df[c].notna().sum()) if c in original_df.columns else 0
        flagged = int(df[c].sum())
        print(f"  {c:4s}  {cat_names[c]:20s}  {present:13,}  {flagged:8,}  {flagged/n*100:6.1f}%")

    # Co-occurrence
    sub("Category co-occurrence (top 15 combinations)")
    combos = Counter(
        '+'.join(c for c in cats if row[c] == 1) or '(none)'
        for _, row in df.iterrows()
    )
    for combo, cnt in combos.most_common(15):
        print(f"  {combo:35s}  {cnt:5,}  ({cnt/n*100:.1f}%)")


def stats_civil_comments(cache_dir):
    from datasets import load_dataset
    hdr("Civil Comments  |  google/civil_comments")

    print("""
  Binary label derivation:
    toxicity >= 0.5  →  1  (positive / toxic)
    toxicity <  0.5  →  0  (negative / safe)
    'toxicity' is a float32 in [0,1] = mean rating across ~8 crowdsourced
    annotators. Threshold 0.5 is standard for this dataset (Jigsaw/Google).

  Statistics reported per split (train / validation / test).
""")

    ds = load_dataset("google/civil_comments", cache_dir=cache_dir)
    threshold = 0.5
    score_cols = ['severe_toxicity', 'obscene', 'threat',
                  'insult', 'identity_attack', 'sexual_explicit']

    for split_name in ['train', 'validation', 'test']:
        print(f"\n{'━'*72}")
        print(f"  Split: {split_name}")
        print(f"{'━'*72}")

        df  = ds[split_name].to_pandas()
        n   = len(df)
        tox = df['toxicity'].values

        df['label'] = (tox >= threshold).astype(int)
        label_block(df['label'].values)
        print_length_stats(length_stats(df['text']))

        # Toxicity score histogram
        sub(f"Toxicity score distribution — {split_name}")
        bins = np.arange(0, 1.1, 0.1)
        counts, _ = np.histogram(tox, bins=bins)
        for i, c in enumerate(counts):
            lo, hi = bins[i], bins[i+1]
            bar = '█' * int(c / n * 50)
            print(f"  [{lo:.1f},{min(hi,1.0):.1f})  {c:9,}  {c/n*100:5.1f}%  {bar}")

        # Sub-score positivity
        sub(f"Sub-score positivity rates (threshold=0.5) — {split_name}")
        print(f"  {'Column':25s}  {'Positive':>10}  {'Pos%':>7}")
        for col in score_cols:
            pos = int((df[col] >= threshold).sum())
            print(f"  {col:25s}  {pos:10,}  {pos/n*100:6.1f}%")

        # Null audit
        sub(f"Null / missing value audit — {split_name}")
        for col in ['text', 'toxicity'] + score_cols:
            n_null = int(df[col].isna().sum())
            print(f"  {col:30s}  {n_null:6,} nulls  ({n_null/n*100:.2f}%)")


def stats_reddit(name, path):
    hdr(f"Reddit {name}  |  {path}")

    print("""
  Binary label derivation:
    label column is int64 — already binary (0 or 1). No conversion needed.

  ⚠ Label provenance UNCONFIRMED from file alone.
    Working hypothesis (from Pass 1 cross-tab analysis):
      label = 1  →  comment removed by moderators (rule-based proxy)
      label = 0  →  comment not removed
    This is NOT a toxicity annotation. Confirm with source paper.
""")

    df = pd.read_csv(path)
    label_block(df['label'].values)
    print_length_stats(length_stats(df['text'].fillna('')))
    null_audit(df, ['text', 'label', 'subreddit'])

    # Per subreddit
    sub("Label distribution per subreddit")
    grp = df.groupby('subreddit')['label'].agg(['sum', 'count'])
    grp['neg']  = grp['count'] - grp['sum']
    grp['pos%'] = grp['sum'] / grp['count'] * 100
    grp = grp.sort_values('count', ascending=False)
    print(f"  {'Subreddit':25s}  {'N':>6}  {'Pos':>6}  {'Neg':>6}  {'Pos%':>6}")
    for sr, row in grp.iterrows():
        print(f"  {sr:25s}  {int(row['count']):6d}  {int(row['sum']):6d}  "
              f"{int(row['neg']):6d}  {row['pos%']:5.1f}%")

    # Text length by label
    sub("Text length (words) by label")
    print(f"  {'Label':8s}  {'N':>7}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  {'Min':>5}  {'Max':>6}")
    for lbl in [0, 1]:
        texts = df[df['label'] == lbl]['text'].fillna('').tolist()
        words = np.array([len(str(t).split()) for t in texts])
        print(f"  label={lbl}  {len(texts):7,}  "
              f"{words.mean():7.1f}  {np.median(words):7.1f}  "
              f"{words.std():7.1f}  {words.min():5d}  {words.max():6d}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       default='all')
    parser.add_argument('--cache_dir',      default=str(Path.home() / 'datasets/cache'))
    parser.add_argument('--openai_path',    default=str(Path.home() / 'datasets/openai/samples-1680.jsonl'))
    parser.add_argument('--toxigen_path',   default=str(Path.home() / 'datasets/toxigen/toxigen_train.jsonl'))
    parser.add_argument('--reddit_en_path', default=str(Path.home() / 'datasets/reddit/balanced/data-en/test-en.csv'))
    parser.add_argument('--reddit_fr_path', default=str(Path.home() / 'datasets/reddit/balanced/data-fr/test-fr.csv'))
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    all_keys = ['hatecheck_en', 'hatecheck_fr', 'french_hate_superset',
                'toxigen', 'openai', 'civil_comments', 'reddit_en', 'reddit_fr']
    keys = all_keys if args.datasets == 'all' else [k.strip() for k in args.datasets.split(',')]

    if args.output:
        class Tee:
            def __init__(self, *files): self.files = files
            def write(self, d):
                for f in self.files: f.write(d)
            def flush(self):
                for f in self.files: f.flush()
        fout = open(args.output, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.__stdout__, fout)

    print(f"\n{'#'*72}")
    print(f"#  Dataset Statistics Explorer — Pass 2")
    print(f"#  Datasets: {', '.join(keys)}")
    print(f"{'#'*72}")

    dispatch = {
        'hatecheck_en':         lambda: stats_hatecheck("HateCheck EN", "Paul/hatecheck", args.cache_dir),
        'hatecheck_fr':         lambda: stats_hatecheck("HateCheck FR", "Paul/hatecheck-french", args.cache_dir),
        'french_hate_superset': lambda: stats_fhs(args.cache_dir),
        'toxigen':              lambda: stats_toxigen(args.toxigen_path),
        'openai':               lambda: stats_openai(args.openai_path),
        'civil_comments':       lambda: stats_civil_comments(args.cache_dir),
        'reddit_en':            lambda: stats_reddit("EN", args.reddit_en_path),
        'reddit_fr':            lambda: stats_reddit("FR", args.reddit_fr_path),
    }

    for key in keys:
        fn = dispatch.get(key)
        if fn is None:
            print(f"\n[SKIP] Unknown key: {key}")
            continue
        try:
            fn()
        except Exception as e:
            print(f"\n[ERROR] {key}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'#'*72}")
    print(f"#  Pass 2 complete.")
    print(f"{'#'*72}")


if __name__ == '__main__':
    main()
