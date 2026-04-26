"""
Dataset Structure Explorer — Phase 1, Pass 1
=============================================
For each of the 8 evaluation datasets, this script prints:

  1. RAW STRUCTURE  — splits, columns, dtypes, full label taxonomy, all
                      metadata fields, sample rows. No assumptions about
                      which field is the ground truth label.

  2. TECHNICAL SHEET — a structured thesis-ready summary (populated from
                       the data itself). Fields that cannot be determined
                       without human inspection are marked [TBD — see raw
                       output above].

Run this on the cluster. Bring the output back before writing Pass 2
(statistics), which must use the label fields and binarisation logic
confirmed here.

Usage:
  python explore_datasets.py \\
      --cache_dir ~/datasets/cache \\
      --openai_path ~/datasets/openai/samples-1680.jsonl \\
      --toxigen_path ~/datasets/toxigen/toxigen_train.jsonl \\
      --reddit_en_path ~/datasets/reddit/balanced/data-en/test-en.csv \\
      --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv \\
      --output explore_structure.txt
"""

import torch  # noqa: F401 — loads libstdc++ symbols needed by pandas C extensions
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# ── formatting ────────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72
SEP3 = "·" * 72

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def techsub(title):
    print(f"\n{SEP3}\n  ◆ TECHNICAL SHEET — {title}\n{SEP3}")

def field(name, value, width=26):
    print(f"  {(name + ':'):{width}s}  {value}")


# ── tech sheet helper ─────────────────────────────────────────────────────────

def techsheet(
    name,
    hf_id_or_path,
    task,
    languages,
    domain,
    source_platform,
    annotation_method,
    license_,
    splits,          # dict: split_name → n_rows
    label_field,
    label_taxonomy,  # list of (value, count) tuples
    n_total,
    text_field,
    avg_len_words,   # approximate, computed from sample or full set
    extra_notes="",
    label_provenance_note="",
    citation="",
):
    techsub(name)
    field("Name",               name)
    field("HF ID / Path",       hf_id_or_path)
    field("Task",               task)
    field("Language(s)",        languages)
    field("Domain",             domain)
    field("Source platform",    source_platform)
    field("Annotation method",  annotation_method)
    field("License",            license_)
    field("Citation",           citation if citation else "[TBD — check dataset card]")
    print()
    field("Splits",             str(splits))
    field("Total rows",         f"{n_total:,}")
    field("Text field",         f"'{text_field}'")
    field("Label field",        f"'{label_field}'")
    print()
    print(f"  {'Label taxonomy':26s}")
    for val, cnt in label_taxonomy:
        pct = cnt / n_total * 100 if n_total else 0
        print(f"    {str(val)!r:35s}  {cnt:7,}  ({pct:.1f}%)")
    print()
    field("Approx avg text len", f"{avg_len_words} words")
    if label_provenance_note:
        field("Label provenance",   label_provenance_note)
    if extra_notes:
        field("Notes",              extra_notes)
    print(f"\n{SEP3}")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset explorers
# ═══════════════════════════════════════════════════════════════════════════════

def explore_hatecheck(name, hf_id, cache_dir):
    from datasets import load_dataset
    hdr(f"RAW STRUCTURE — {name}  |  {hf_id}")
    ds = load_dataset(hf_id, cache_dir=cache_dir)

    print(f"\nSplits: {list(ds.keys())}")
    splits_info = {}
    for split_name, split in ds.items():
        splits_info[split_name] = len(split)
        print(f"\n  split='{split_name}'  n={len(split):,}")
        print(f"  Columns : {split.column_names}")
        print(f"  Features: {split.features}")

    split = ds[list(ds.keys())[0]]
    n_total = len(split)

    label_counts = []
    if 'label_gold' in split.column_names:
        counts = Counter(split['label_gold'])
        label_counts = sorted(counts.items(), key=lambda x: -x[1])
        print(f"\nlabel_gold — all distinct values ({len(counts)}):")
        for v, c in label_counts:
            print(f"  {v!r:35s}  {c:6,}  ({c/n_total*100:.1f}%)")

    if 'functionality' in split.column_names:
        counts = Counter(split['functionality'])
        print(f"\nfunctionality — all distinct values ({len(counts)}):")
        for v, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {v!r:44s}  {c:4,}")

    meta = [c for c in split.column_names if c not in ('test_case', 'label_gold', 'functionality')]
    if meta:
        print(f"\nOther columns: {meta}")
        for col in meta:
            top = Counter(str(row.get(col, ''))[:80] for row in split).most_common(5)
            print(f"  '{col}': {top}")

    sub("Sample rows (3)")
    for i, row in enumerate(list(split)[:3]):
        print(f"  [{i}] {json.dumps(dict(row), ensure_ascii=False)[:400]}")

    # Avg word count
    texts = [row['test_case'] for row in split if row.get('test_case')]
    avg_words = round(sum(len(t.split()) for t in texts) / len(texts), 1) if texts else 0

    lang = "French" if "french" in hf_id.lower() or "fr" in hf_id.lower() else "English"
    techsheet(
        name            = name,
        hf_id_or_path   = hf_id,
        task            = "Hate speech detection (functional testing)",
        languages       = lang,
        domain          = "Synthetic / template-based test cases",
        source_platform = "Expert-authored (not from a real platform)",
        annotation_method = "Expert annotation — binary (hateful / non_hateful). "
                            "Templates cover controlled linguistic phenomena.",
        license_        = "CC BY 4.0",
        splits          = splits_info,
        label_field     = "label_gold",
        label_taxonomy  = label_counts,
        n_total         = n_total,
        text_field      = "test_case",
        avg_len_words   = avg_words,
        label_provenance_note = "Direct expert annotation. 'hateful' → label=1, all else → label=0.",
        extra_notes     = (f"functionality column: {len(Counter(split['functionality']))} "
                           "distinct types covering 29-34 linguistic phenomena."),
        citation        = ("Röttger et al. (2021) HateCheck: Functional Tests for Hate Speech "
                           "Detection Models. ACL 2021. / "
                           "Röttger et al. (2022) Multilingual HateCheck. EMNLP 2022."),
    )


def explore_french_hate_superset(cache_dir):
    from datasets import load_dataset
    hdr("RAW STRUCTURE — French Hate Speech Superset  |  manueltonneau/french-hate-speech-superset")
    ds = load_dataset("manueltonneau/french-hate-speech-superset", cache_dir=cache_dir)

    print(f"\nSplits: {list(ds.keys())}")
    splits_info = {}
    for split_name, split in ds.items():
        splits_info[split_name] = len(split)
        print(f"\n  split='{split_name}'  n={len(split):,}")
        print(f"  Columns : {split.column_names}")
        print(f"  Features: {str(split.features)[:800]}")

    split     = ds[list(ds.keys())[0]]
    n_total   = len(split)

    all_label_vals = []
    if 'labels' in split.column_names:
        for row in split:
            lv = row.get('labels')
            if isinstance(lv, list):
                all_label_vals.extend(str(l).strip() for l in lv)
            elif lv is not None:
                all_label_vals.append(str(lv).strip())
        counts = Counter(all_label_vals)
        label_counts = sorted(counts.items(), key=lambda x: -x[1])
        print(f"\n'labels' — all distinct values ({len(counts)}):")
        for v, c in label_counts:
            print(f"  {v!r:40s}  {c:7,}")
    else:
        label_counts = []

    meta = [c for c in split.column_names if c not in ('text', 'labels')]
    if meta:
        print(f"\nOther columns: {meta}")
        for col in meta:
            n_unique = len(set(str(row.get(col,'')) for row in split))
            top = Counter(str(row.get(col,''))[:60] for row in split).most_common(10)
            print(f"  '{col}': {n_unique} unique values")
            for v, c in top:
                print(f"    {v!r:50s}  {c:6,}")

    sub("Sample rows (5)")
    for i, row in enumerate(list(split)[:5]):
        print(f"  [{i}] {json.dumps(dict(row), ensure_ascii=False)[:500]}")

    texts     = [row['text'] for row in split if row.get('text','').strip()]
    avg_words = round(sum(len(t.split()) for t in texts) / len(texts), 1) if texts else 0

    src_col = next((c for c in split.column_names
                    if 'source' in c.lower() or 'dataset' in c.lower()), None)
    src_note = (f"Source column '{src_col}' present — "
                f"{len(set(str(row.get(src_col,'')) for row in split))} distinct source corpora."
                if src_col else
                "No source/dataset column found — cannot determine per-corpus origin from file alone.")

    techsheet(
        name            = "French Hate Speech Superset",
        hf_id_or_path   = "manueltonneau/french-hate-speech-superset",
        task            = "Hate speech detection",
        languages       = "French",
        domain          = "Aggregated multi-source corpora (social media, news, other)",
        source_platform = "Multiple — aggregated from several French hate speech datasets",
        annotation_method = "Varies by source corpus (crowdsourced / expert). "
                            "Labels are multi-valued strings; binarised in our pipeline.",
        license_        = "[TBD — check HuggingFace dataset card]",
        splits          = splits_info,
        label_field     = "labels",
        label_taxonomy  = label_counts[:20],   # top 20 values; full list in raw output
        n_total         = n_total,
        text_field      = "text",
        avg_len_words   = avg_words,
        label_provenance_note = (
            "Multi-label field. Our binarisation: any value NOT in "
            "{none, normal, non-hateful, non_hateful, 0, nohate} → label=1 (hateful). "
            "Verify completeness of safe_strings against the full label list above."),
        extra_notes     = src_note,
        citation        = "Tonneau et al. — see HuggingFace dataset card for full citation list.",
    )


def explore_toxigen(path):
    hdr(f"RAW STRUCTURE — ToxiGen  |  {path}")
    raw = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    n_total = len(raw)
    print(f"\nTotal rows: {n_total:,}")

    key_counts = Counter()
    for item in raw:
        for k in item:
            key_counts[k] += 1
    print(f"\nFields (with row coverage):")
    for k, c in sorted(key_counts.items(), key=lambda x: -x[1]):
        print(f"  {k!r:35s}  present in {c:,} / {n_total:,} rows  ({c/n_total*100:.0f}%)")

    label_counts_all = {}
    for label_field in ('prompt_label', 'label', 'human_label', 'annotator_labels'):
        if label_field in key_counts:
            vals = Counter(str(item.get(label_field)) for item in raw if label_field in item)
            label_counts_all[label_field] = vals
            print(f"\n'{label_field}' — all distinct values ({len(vals)}):")
            for v, c in sorted(vals.items(), key=lambda x: -x[1]):
                print(f"  {v!r:30s}  {c:6,}  ({c/n_total*100:.1f}%)")

    for grp_field in ('target_groups', 'target_group'):
        if grp_field in key_counts:
            groups = Counter()
            for item in raw:
                g = item.get(grp_field, [])
                for gg in (g if isinstance(g, list) else [g]):
                    groups[str(gg)] += 1
            print(f"\n'{grp_field}' — {len(groups)} distinct groups (top 20):")
            for g, c in sorted(groups.items(), key=lambda x: -x[1])[:20]:
                print(f"  {g!r:40s}  {c:6,}")

    for score_field in ('annotator_sentiment', 'roberta_prediction', 'generation_method'):
        if score_field in key_counts:
            vals = Counter(str(item.get(score_field)) for item in raw if score_field in item)
            print(f"\n'{score_field}': {dict(vals)}")

    sub("Sample rows (5)")
    for i, item in enumerate(raw[:5]):
        print(f"  [{i}] {json.dumps(item, ensure_ascii=False)[:500]}")

    text_field_name = 'generation' if 'generation' in key_counts else 'text'
    texts     = [item.get(text_field_name,'') for item in raw if item.get(text_field_name,'').strip()]
    avg_words = round(sum(len(t.split()) for t in texts) / len(texts), 1) if texts else 0

    # Primary label field for our pipeline
    primary_label = 'prompt_label' if 'prompt_label' in key_counts else list(label_counts_all.keys())[0] if label_counts_all else 'unknown'
    primary_counts = list(label_counts_all.get(primary_label, {}).items())

    techsheet(
        name            = "ToxiGen",
        hf_id_or_path   = path,
        task            = "Implicit hate speech detection",
        languages       = "English",
        domain          = "Machine-generated statements (adversarial, GPT-3 with ALICE technique)",
        source_platform = "Synthetic — no real platform; generated to fool existing classifiers",
        annotation_method = ("prompt_label: machine-generated label from generation prompt. "
                             "human_label / annotator_labels (if present): crowdsourced validation "
                             "on a subset (~8K). We use prompt_label for the full set."),
        license_        = "MIT (code); data license in GitHub repository",
        splits          = {"train (local file)": n_total},
        label_field     = primary_label,
        label_taxonomy  = sorted(primary_counts, key=lambda x: -x[1]),
        n_total         = n_total,
        text_field      = text_field_name,
        avg_len_words   = avg_words,
        label_provenance_note = (
            f"'{primary_label}' = 1 → toxic (implicit hate); = 0 → benign. "
            "Generated by GPT-3 ALICE prompting, not human-annotated for all examples. "
            "Subsampled to 5,000 in our evaluation (random seed 42)."),
        extra_notes     = "13 target minority groups. ~95% implicit toxicity (no slurs). "
                          "Designed to be adversarially hard for existing classifiers.",
        citation        = ("Hartvigsen et al. (2022) ToxiGen: A Large-Scale Machine-Generated "
                           "Dataset for Adversarial and Implicit Hate Speech Detection. ACL 2022."),
    )


def explore_openai(path):
    hdr(f"RAW STRUCTURE — OpenAI Moderation  |  {path}")
    raw = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    n_total = len(raw)
    print(f"\nTotal rows: {n_total:,}")

    key_counts = Counter()
    for item in raw:
        for k in item:
            key_counts[k] += 1
    print(f"\nFields (with row coverage):")
    for k, c in sorted(key_counts.items(), key=lambda x: -x[1]):
        print(f"  {k!r:10s}  present in {c:,} / {n_total:,} rows  ({c/n_total*100:.0f}%)")

    category_keys = ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
    category_names = {
        'S': 'Sexual',          'H': 'Hate',
        'V': 'Violence',        'HR': 'Harassment',
        'SH': 'Self-harm',      'S3': 'Sexual/minors',
        'H2': 'Hate/threatening','V2': 'Violence/graphic',
    }
    present_cats = [k for k in category_keys if k in key_counts]
    print(f"\nModeration category fields: {present_cats}")
    cat_counts = []
    for k in present_cats:
        vals     = [item.get(k) for item in raw if k in item]
        types    = Counter(type(v).__name__ for v in vals)
        distinct = Counter(str(v) for v in vals).most_common(5)
        n_true   = sum(1 for v in vals if v)
        cat_counts.append((k, n_true))
        print(f"  {k:5s} ({category_names.get(k,'?'):20s})  "
              f"types={str(types):20s}  distinct={str(distinct)[:40]}  n_flagged={n_true:,}")

    n_any = sum(1 for item in raw if any(item.get(k) for k in present_cats))
    n_safe = n_total - n_any
    print(f"\nOverall: flagged_any={n_any:,}  safe={n_safe:,}")

    combos = Counter(tuple(k for k in present_cats if item.get(k)) for item in raw)
    print(f"\nCategory co-occurrence (top 10):")
    for combo, c in sorted(combos.items(), key=lambda x: -x[1])[:10]:
        print(f"  {('+'.join(combo) or '(none)'):35s}  {c:5,}")

    sub("Sample rows (5)")
    for i, item in enumerate(raw[:5]):
        print(f"  [{i}] {json.dumps(item, ensure_ascii=False)[:500]}")

    texts     = [item.get('prompt','') for item in raw if item.get('prompt','').strip()]
    avg_words = round(sum(len(t.split()) for t in texts) / len(texts), 1) if texts else 0

    taxonomy = [("(any flag — toxic)", n_any), ("(no flag — safe)", n_safe)]
    for k in present_cats:
        cnt = next((c for kk, c in cat_counts if kk == k), 0)
        taxonomy.append((f"{k}: {category_names.get(k,'?')}", cnt))

    techsheet(
        name            = "OpenAI Moderation Dataset",
        hf_id_or_path   = path,
        task            = "Multi-category content moderation",
        languages       = "Multilingual (predominantly English)",
        domain          = "Real user prompts submitted to OpenAI models (adversarial / red-team)",
        source_platform = "OpenAI API — real-world production prompts",
        annotation_method = ("Human annotation by OpenAI. 8 independent binary category flags. "
                             "Our evaluation collapses to binary: any flag = 1 (unsafe)."),
        license_        = "MIT",
        splits          = {"test (local file)": n_total},
        label_field     = f"{present_cats} (8 binary category flags)",
        label_taxonomy  = taxonomy,
        n_total         = n_total,
        text_field      = "prompt",
        avg_len_words   = avg_words,
        label_provenance_note = (
            "8 independent binary flags: S (Sexual), H (Hate), V (Violence), "
            "HR (Harassment), SH (Self-harm), S3 (Sexual/minors), H2 (Hate/threatening), "
            "V2 (Violence/graphic). Our binary label = OR of all 8 flags. "
            "Multi-label cardinality: see co-occurrence table in raw output."),
        extra_notes     = ("Public subset only (1,680 samples). Full production training set "
                           "not released. Mix of real + synthetic data for rare categories."),
        citation        = ("OpenAI (2022) New and Improved Content Moderation Tooling. "
                           "OpenAI Blog. / Markov et al. (2023)."),
    )


def explore_civil_comments(cache_dir):
    from datasets import load_dataset
    hdr("RAW STRUCTURE — Civil Comments  |  google/civil_comments")
    ds = load_dataset("google/civil_comments", cache_dir=cache_dir)

    print(f"\nSplits: {list(ds.keys())}")
    splits_info = {}
    for split_name, split in ds.items():
        splits_info[split_name] = len(split)
        print(f"\n  split='{split_name}'  n={len(split):,}")
        print(f"  Columns: {split.column_names}")
    split = ds[list(ds.keys())[0]]
    n_total = len(split)
    print(f"\nFeatures:\n  {split.features}")

    # Full toxicity distribution (all rows, no sampling)
    tox = split['toxicity']
    brackets = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
                (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
    n_above = sum(1 for v in tox if v >= 0.5)
    print(f"\n'toxicity' score distribution (full split, n={n_total:,}, binarised at 0.5):")
    for lo, hi in brackets:
        c   = sum(1 for v in tox if lo <= v < hi)
        bar = '█' * int(c / n_total * 50)
        print(f"  [{lo:.1f},{hi:.1f})  {c:8,}  {c/n_total*100:5.1f}%  {bar}")
    print(f"\n  toxic (>=0.5): {n_above:,}  ({n_above/n_total*100:.1f}%)")
    print(f"  safe  (<0.5):  {n_total-n_above:,}  ({(n_total-n_above)/n_total*100:.1f}%)")

    # All score columns
    score_cols = [c for c in split.column_names if c != 'text']
    print(f"\nAll score columns: {score_cols}")
    print(f"\n  {'Column':30s}  {'>=0.5 (n)':>10s}  {'>=0.5 (%)':>10s}")
    for col in score_cols:
        vals = split[col]
        if vals and isinstance(vals[0], (int, float)):
            pos = sum(1 for v in vals if v >= 0.5)
            print(f"  {col:30s}  {pos:>10,}  {pos/n_total*100:>9.1f}%")

    sub("Sample rows (5)")
    for i, row in enumerate(list(split)[:5]):
        print(f"  [{i}] {json.dumps(dict(row), ensure_ascii=False)[:500]}")

    texts     = [row['text'] for row in list(split)[:10000] if row.get('text','').strip()]
    avg_words = round(sum(len(t.split()) for t in texts) / len(texts), 1) if texts else 0

    taxonomy  = [
        ("toxic (toxicity >= 0.5)", n_above),
        ("safe  (toxicity < 0.5)",  n_total - n_above),
    ]

    techsheet(
        name            = "Civil Comments",
        hf_id_or_path   = "google/civil_comments",
        task            = "Toxicity detection",
        languages       = "English",
        domain          = "Online news article comment sections (The Guardian, NYT, etc.)",
        source_platform = "Civil Comments platform (comment plugin for news sites)",
        annotation_method = ("Jigsaw/Google crowdsourced annotation. "
                             "Continuous toxicity score [0,1] (mean of annotator ratings). "
                             "Binarised in our pipeline at 0.5 threshold."),
        license_        = "CC BY 4.0",
        splits          = splits_info,
        label_field     = "toxicity (float) → binary at 0.5",
        label_taxonomy  = taxonomy,
        n_total         = n_total,
        text_field      = "text",
        avg_len_words   = avg_words,
        label_provenance_note = (
            "'toxicity' is a continuous float [0,1]. "
            "Additional sub-scores available: severe_toxicity, obscene, "
            "identity_attack, insult, threat, sexual_explicit. "
            "Evaluation subsamples 5,000 rows (random seed 42)."),
        extra_notes     = ("~8 annotators per comment. Also annotated for identity mentions "
                           "(race, religion, gender, etc.) — useful for bias analysis."),
        citation        = ("Borkan et al. (2019) Nuanced Metrics for Measuring Unintended Bias "
                           "with Real Data for Text Classification. WWW 2019."),
    )


def explore_reddit(name, path):
    import pandas as pd
    hdr(f"RAW STRUCTURE — Reddit {name}  |  {path}")
    df = pd.read_csv(path)

    n_total = len(df)
    print(f"\nShape  : {n_total:,} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")

    print(f"\nNull counts per column:")
    for col in df.columns:
        n_null = df[col].isna().sum()
        print(f"  '{col}': {n_null:,} nulls ({n_null/n_total*100:.1f}%)")

    print(f"\nRaw first 10 rows:")
    print(df.head(10).to_string())

    # Every non-text column: full value distribution + cross-tab with label
    for col in df.columns:
        if col == 'text':
            continue
        n_unique = df[col].nunique()
        vc = df[col].value_counts(dropna=False)
        print(f"\n'{col}' — {n_unique} unique values:")
        for val, cnt in vc.head(30).items():
            print(f"  {str(val)!r:45s}  {cnt:7,}  ({cnt/n_total*100:.1f}%)")

        if col != 'label' and 'label' in df.columns and n_unique < 500:
            try:
                xtab = df.groupby([col, 'label'], dropna=False).size().unstack(fill_value=0)
                print(f"  cross-tab with 'label':\n{xtab.to_string()}")
            except Exception:
                pass

    sub("10 label=1 examples")
    if 'label' in df.columns:
        for _, row in df[df['label'] == 1].head(10).iterrows():
            print(f"  {str(row.get('text',''))[:300]!r}")
        sub("10 label=0 examples")
        for _, row in df[df['label'] == 0].head(10).iterrows():
            print(f"  {str(row.get('text',''))[:300]!r}")

    texts     = df['text'].dropna().tolist()
    avg_words = round(sum(len(str(t).split()) for t in texts) / len(texts), 1) if texts else 0

    label_counts: list = []
    if 'label' in df.columns:
        vc = df['label'].value_counts().sort_index()
        label_counts = [(str(v), int(c)) for v, c in vc.items()]

    other_cols = [c for c in df.columns if c not in ('text', 'label')]

    lang  = "French" if name.upper() == "FR" else "English"
    techsheet(
        name            = f"Reddit {name}",
        hf_id_or_path   = path,
        task            = "[TBD — determine from label provenance]",
        languages       = lang,
        domain          = "Reddit comments (popular subreddits — informal register)",
        source_platform = "Reddit",
        annotation_method = ("[TBD — label provenance unknown from file alone. "
                             "Check source paper / dataset README. "
                             f"Extra columns: {other_cols if other_cols else 'none — only text + label'}. "
                             "See cross-tab above for clues.]"),
        license_        = "[TBD — check source paper]",
        splits          = {f"test file ({name})": n_total},
        label_field     = "label",
        label_taxonomy  = label_counts,
        n_total         = n_total,
        text_field      = "text",
        avg_len_words   = avg_words,
        label_provenance_note = (
            "UNKNOWN. Hypotheses: (A) moderator removal action (rule-based proxy), "
            "(B) human annotation for toxicity/hate speech. "
            "Confirm by checking extra columns and source paper before citing results."),
        extra_notes     = (f"File: {path}. "
                           f"Extra columns beyond text+label: "
                           f"{other_cols if other_cols else 'none'}. "
                           "Dataset described as 'balanced' in path — likely downsampled."),
        citation        = "[TBD — identify source paper from dataset README or supervisor]",
    )


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
    parser.add_argument('--output', default=None,
                        help='Write output to file in addition to stdout')
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
    print(f"#  Dataset Explorer — Pass 1: Structure + Technical Sheets")
    print(f"#  Datasets: {', '.join(keys)}")
    print(f"{'#'*72}")
    print(f"\nEach dataset section prints:")
    print(f"  1. RAW STRUCTURE  — all columns, all label values, sample rows")
    print(f"  2. TECHNICAL SHEET — thesis-ready summary; [TBD] marks fields")
    print(f"                       requiring human confirmation after inspection")

    dispatch = {
        'hatecheck_en':         lambda: explore_hatecheck("HateCheck EN", "Paul/hatecheck", args.cache_dir),
        'hatecheck_fr':         lambda: explore_hatecheck("HateCheck FR", "Paul/hatecheck-french", args.cache_dir),
        'french_hate_superset': lambda: explore_french_hate_superset(args.cache_dir),
        'toxigen':              lambda: explore_toxigen(args.toxigen_path),
        'openai':               lambda: explore_openai(args.openai_path),
        'civil_comments':       lambda: explore_civil_comments(args.cache_dir),
        'reddit_en':            lambda: explore_reddit("EN", args.reddit_en_path),
        'reddit_fr':            lambda: explore_reddit("FR", args.reddit_fr_path),
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
    print(f"#  Pass 1 complete.")
    print(f"#  Next steps:")
    print(f"#    1. Review all [TBD] fields — especially Reddit label provenance")
    print(f"#    2. Verify safe_strings completeness for French Hate Superset")
    print(f"#    3. Confirm ToxiGen label field (prompt_label vs human_label)")
    print(f"#    4. Bring this output back to implement Pass 2 (statistics)")
    print(f"{'#'*72}")


if __name__ == '__main__':
    main()
