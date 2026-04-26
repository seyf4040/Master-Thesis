#!/usr/bin/env python3
"""
evaluate_two_tier.py — End-to-end evaluation of the two-tier moderation system

Pairs a Tier 1 lightweight model (fast, CPU) with Tier 2 SG-2b + LoRA (GPU) and evaluates
the combined system on Reddit-FR.

Three-zone logic:
  score < t_low              → Tier 1 passes as SAFE    (high confidence)
  score > t_high             → Tier 1 flags as UNSAFE   (high confidence)
  t_low ≤ score ≤ t_high    → defer to Tier 2

Primary success criterion: combined F1 ≥ Tier 2 alone (SG-2b baseline F1 = 0.662)
while achieving deferral_rate ≤ 60% (≥ 40% traffic handled by Tier 1 alone).

Tier 2 inference:
  ShieldGemma-2b makes predictions by comparing next-token logits for "Yes" vs "No"
  at the end of its prompt. No generation required — a single forward pass per sample.

Output:
  {output_dir}/
    results.json           — full metrics and per-sample decisions
    summary.txt            — human-readable table

Usage:
    # Run with fine-tuned Tier 1 checkpoint (recommended after Track B):
    python code/phase4_two_tier/evaluate_two_tier.py \\
        --tier1_model_id     ~/code/results/tier1_detoxify_finetuned/best \\
        --tier1_t_low        0.15 \\
        --tier1_t_high       0.85 \\
        --tier2_adapter_dir  ~/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best \\
        --reddit_fr_path     ~/datasets/reddit/balanced/data-fr/test-fr.csv \\
        --output_dir         ~/code/results/two_tier_eval

    # Use held-out test set from fine-tuning run (recommended for fair comparison):
    python code/phase4_two_tier/evaluate_two_tier.py \\
        --test_set_json  ~/code/results/tier1_detoxify_finetuned/test_set.json \\
        [... other args ...]

    # Also compute Tier 2-alone baseline (runs SG-2b on all samples — slower):
    python code/phase4_two_tier/evaluate_two_tier.py \\
        --full_comparison \\
        [... other args ...]

Author: Ural Seyfullah
"""

import gc
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch


# ── Prompt template (matches finetune_lora.py and run_full_baseline_v3.py) ────

def _shieldgemma_prompt(text: str) -> str:
    return (
        "You are a policy expert trying to help determine whether a user "
        "message is in violation of safety policies.\n\n"
        "<start_of_turn>user\n"
        f"{text}\n"
        "<end_of_turn>\n\n"
        "Does the above message contain hate speech, harassment, violence, "
        "or other harmful content? Answer only 'Yes' or 'No'.\n"
        "<start_of_turn>model\n"
    )


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_reddit_fr(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]


def load_test_set_json(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data['samples']


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels: List[int], preds: List[int]) -> Dict:
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    n  = len(labels)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fnr  = fn / (tp + fn) if (tp + fn) > 0 else 0.0   # hateful items missed
    fpr  = fp / (tn + fp) if (tn + fp) > 0 else 0.0   # safe items wrongly flagged
    acc  = (tp + tn) / n if n > 0 else 0.0

    return {
        'f1': f1, 'precision': prec, 'recall': rec,
        'accuracy': acc, 'fnr': fnr, 'fpr': fpr,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': n,
    }


# ── Tier 1 scoring ────────────────────────────────────────────────────────────

def score_tier1(samples: List[Dict], model_id: str, toxic_label: str,
                device: str) -> List[float]:
    """
    Score all samples with Tier 1 model using the HuggingFace text-classification pipeline.
    Returns toxicity score ∈ [0,1] for each sample (same order as input).
    """
    from transformers import pipeline
    from tqdm import tqdm

    _device = 0 if (device == 'cuda' and torch.cuda.is_available()) else -1
    print(f"\n  Loading Tier 1 model: {model_id} ...")
    classifier = pipeline(
        "text-classification",
        model=model_id,
        device=_device,
        truncation=True,
        max_length=512,
        top_k=None,
    )

    scores = []
    toxic_key = toxic_label.lower()
    n_fallback = 0
    for s in tqdm(samples, desc="Tier 1 scoring", unit="sample", leave=False):
        preds = classifier(s['text'])
        if isinstance(preds[0], list):
            preds = preds[0]
        label_scores = {p['label'].lower(): p['score'] for p in preds}
        if toxic_key in label_scores:
            score = label_scores[toxic_key]
        else:
            # Fallback: 1 - max(non-toxic probability)
            non_toxic = [v for k, v in label_scores.items() if k != toxic_key]
            score = 1.0 - max(non_toxic) if non_toxic else 0.5
            n_fallback += 1
        scores.append(score)

    if n_fallback > 0:
        print(f"  Warning: {n_fallback} samples used fallback scoring "
              f"(label '{toxic_label}' not found in model output)")

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


# ── Tier 2 scoring ────────────────────────────────────────────────────────────

def load_tier2(base_model_id: str, adapter_dir: str,
               cache_dir: str, device: str):
    """Load SG-2b base model + LoRA adapter. Returns (model, tokenizer, yes_id, no_id)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"\n  Loading Tier 2 base model: {base_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)

    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map='auto',
        cache_dir=cache_dir,
    )

    print(f"  Applying LoRA adapter from: {adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    # Token IDs for the label tokens. ShieldGemma uses "Yes" (unsafe) and "No" (safe).
    yes_ids = tokenizer('Yes', add_special_tokens=False)['input_ids']
    no_ids  = tokenizer('No',  add_special_tokens=False)['input_ids']
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer produced empty IDs for 'Yes'/'No' — wrong tokenizer?")
    yes_id = yes_ids[0]
    no_id  = no_ids[0]
    print(f"  Label token IDs — 'Yes' (unsafe): {yes_id}, 'No' (safe): {no_id}")

    return model, tokenizer, yes_id, no_id


def predict_tier2_batch(texts: List[str], model, tokenizer,
                        yes_id: int, no_id: int, device: str,
                        max_length: int = 512) -> List[Tuple[int, float]]:
    """
    Run Tier 2 (ShieldGemma-2b LoRA) on a list of texts.
    For each text, computes P(Yes) / (P(Yes) + P(No)) from next-token logits.
    Returns list of (prediction, prob_yes): prediction=1 (unsafe) or 0 (safe).
    Processes one sample at a time for reliability with variable-length prompts.
    """
    from tqdm import tqdm
    results = []
    model.eval()
    _device = next(model.parameters()).device

    for text in tqdm(texts, desc="Tier 2 scoring", unit="sample", leave=False):
        prompt = _shieldgemma_prompt(text)
        inputs = tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=max_length
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            next_token_logits = model(**inputs).logits[0, -1, :]
        # Compare only Yes/No logits — normalise to a binary probability
        yes_no = torch.tensor(
            [next_token_logits[no_id].item(), next_token_logits[yes_id].item()]
        ).softmax(dim=-1)
        prob_yes = yes_no[1].item()
        results.append((int(prob_yes >= 0.5), prob_yes))

    return results


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test data ────────────────────────────────────────────────────────
    if args.test_set_json:
        print(f"\nLoading held-out test set from {args.test_set_json} ...")
        samples = load_test_set_json(args.test_set_json)
    else:
        print(f"\nLoading Reddit-FR from {args.reddit_fr_path} ...")
        samples = load_reddit_fr(args.reddit_fr_path)

    print(f"  {len(samples)} samples  "
          f"(toxic={sum(s['label'] for s in samples)}, "
          f"safe={sum(1-s['label'] for s in samples)})")

    labels = [s['label'] for s in samples]
    texts  = [s['text']  for s in samples]
    n = len(samples)

    # ── Tier 1 scoring ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Tier 1: {args.tier1_model_id}")
    print(f"  T_low={args.tier1_t_low}  T_high={args.tier1_t_high}")
    print(f"{'='*60}")

    t1_scores = score_tier1(samples, args.tier1_model_id,
                            args.tier1_toxic_label, args.device)

    # ── Three-zone classification ─────────────────────────────────────────────
    tier1_safe_mask   = [s < args.tier1_t_low  for s in t1_scores]
    tier1_unsafe_mask = [s > args.tier1_t_high for s in t1_scores]
    defer_mask        = [not s and not u for s, u in zip(tier1_safe_mask, tier1_unsafe_mask)]

    n_t1_safe   = sum(tier1_safe_mask)
    n_t1_unsafe = sum(tier1_unsafe_mask)
    n_defer     = sum(defer_mask)

    print(f"\n  Tier 1 routing:")
    print(f"    SAFE   (score < {args.tier1_t_low:.2f}): {n_t1_safe:4d}  ({n_t1_safe/n:.1%})")
    print(f"    DEFER  (uncertain zone):              {n_defer:4d}  ({n_defer/n:.1%})")
    print(f"    UNSAFE (score > {args.tier1_t_high:.2f}): {n_t1_unsafe:4d}  ({n_t1_unsafe/n:.1%})")

    # Tier 1 alone predictions (for FNR/FPR breakdown)
    tier1_preds = [
        0 if safe else (1 if unsafe else -1)   # -1 = deferred (resolved by Tier 2)
        for safe, unsafe in zip(tier1_safe_mask, tier1_unsafe_mask)
    ]

    # ── Tier 1 FNR / FPR audit ────────────────────────────────────────────────
    # How many hateful items does Tier 1 pass as "safe" (without deferral)?
    t1_passed_hateful = sum(
        1 for i, safe in enumerate(tier1_safe_mask) if safe and labels[i] == 1
    )
    t1_flagged_safe = sum(
        1 for i, unsafe in enumerate(tier1_unsafe_mask) if unsafe and labels[i] == 0
    )
    n_hateful = sum(labels)
    n_safe    = n - n_hateful
    tier1_fnr = t1_passed_hateful / n_hateful if n_hateful > 0 else 0.0
    tier1_fpr = t1_flagged_safe   / n_safe    if n_safe    > 0 else 0.0
    print(f"\n  Tier 1 alone (confident zone only):")
    print(f"    FNR = {tier1_fnr:.1%}  ({t1_passed_hateful} hateful passed as SAFE)")
    print(f"    FPR = {tier1_fpr:.1%}  ({t1_flagged_safe} safe wrongly flagged)")

    # ── Tier 2 scoring (deferred samples only) ────────────────────────────────
    deferred_texts  = [texts[i]  for i, d in enumerate(defer_mask) if d]
    deferred_labels = [labels[i] for i, d in enumerate(defer_mask) if d]

    tier2_preds_deferred = []
    tier2_probs_deferred = []

    if n_defer > 0:
        print(f"\n{'='*60}")
        print(f"  Tier 2: {args.tier2_base_model}")
        print(f"  Adapter: {args.tier2_adapter_dir}")
        print(f"  Running on {n_defer} deferred samples ({n_defer/n:.1%} of total)")
        print(f"{'='*60}")

        model, tokenizer, yes_id, no_id = load_tier2(
            args.tier2_base_model, args.tier2_adapter_dir,
            args.cache_dir, args.device,
        )

        t2_results = predict_tier2_batch(
            deferred_texts, model, tokenizer, yes_id, no_id, args.device
        )
        tier2_preds_deferred = [r[0] for r in t2_results]
        tier2_probs_deferred = [r[1] for r in t2_results]

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n  No samples deferred to Tier 2.")

    # ── Tier 2 alone (full set) ───────────────────────────────────────────────
    tier2_all_preds = None
    if args.full_comparison:
        print(f"\n{'='*60}")
        print(f"  Tier 2 alone (full test set — for baseline comparison)")
        print(f"{'='*60}")
        model, tokenizer, yes_id, no_id = load_tier2(
            args.tier2_base_model, args.tier2_adapter_dir,
            args.cache_dir, args.device,
        )
        all_results   = predict_tier2_batch(texts, model, tokenizer, yes_id, no_id, args.device)
        tier2_all_preds = [r[0] for r in all_results]
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Assemble combined predictions ─────────────────────────────────────────
    combined_preds = list(tier1_preds)  # copy; deferred slots are -1
    defer_iter = iter(tier2_preds_deferred)
    for i, d in enumerate(defer_mask):
        if d:
            combined_preds[i] = next(defer_iter)

    # ── Compute metrics ───────────────────────────────────────────────────────
    combined_metrics = compute_metrics(labels, combined_preds)

    tier1_only_preds = [
        p if p != -1 else int(t1_scores[i] >= 0.5)   # fallback: T1 at 0.5 for deferred
        for i, p in enumerate(tier1_preds)
    ]
    tier1_only_metrics = compute_metrics(labels, tier1_only_preds)

    tier2_alone_metrics = compute_metrics(labels, tier2_all_preds) if tier2_all_preds else None

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  n={n}  (toxic={n_hateful}, safe={n_safe})")
    print(f"  Tier 1: {args.tier1_model_id}")
    print(f"  Tier 2: {args.tier2_base_model} + {args.tier2_adapter_dir}")
    print(f"  T_low={args.tier1_t_low}  T_high={args.tier1_t_high}")
    print()
    print(f"  {'Metric':<20} {'Combined':>10} {'Tier1-only':>12} "
          + (f"{'Tier2-alone':>12}" if tier2_alone_metrics else ""))
    print(f"  {'-'*58}")
    for key in ('f1', 'precision', 'recall', 'accuracy', 'fnr', 'fpr'):
        c  = f"{combined_metrics[key]:.3f}"
        t1 = f"{tier1_only_metrics[key]:.3f}"
        line = f"  {key:<20} {c:>10} {t1:>12}"
        if tier2_alone_metrics:
            t2 = f"{tier2_alone_metrics[key]:.3f}"
            line += f" {t2:>12}"
        print(line)
    print(f"  {'deferral_rate':<20} {n_defer/n:>10.1%} {'—':>12}"
          + (f" {'0%':>12}" if tier2_alone_metrics else ""))
    print(f"  {'-'*58}")
    print(f"\n  Key: FNR = hateful items missed  |  FPR = safe items wrongly flagged")

    # ── Save results ──────────────────────────────────────────────────────────
    # Per-sample decision log
    sample_log = []
    t2_iter = iter(zip(tier2_probs_deferred, tier2_preds_deferred)) if n_defer > 0 else iter([])
    t2_i = 0
    deferred_indices = [i for i, d in enumerate(defer_mask) if d]
    deferred_results_map = {}
    for i, (pred_i, prob_i) in enumerate(zip(tier2_preds_deferred, tier2_probs_deferred)):
        deferred_results_map[deferred_indices[i]] = (pred_i, prob_i)

    for i in range(n):
        tier = None
        t2_pred = None
        t2_prob = None
        if tier1_safe_mask[i]:
            tier = 1
            final_pred = 0
        elif tier1_unsafe_mask[i]:
            tier = 1
            final_pred = 1
        else:
            tier = 2
            final_pred, t2_prob = deferred_results_map.get(i, (combined_preds[i], None))
            t2_pred = final_pred
        sample_log.append({
            'idx':        i,
            'label':      labels[i],
            'tier1_score': t1_scores[i],
            'tier':       tier,
            'final_pred': final_pred,
            't2_prob':    t2_prob,
            'correct':    int(final_pred == labels[i]),
        })

    results = {
        'config': {
            'tier1_model_id':   args.tier1_model_id,
            'tier1_toxic_label': args.tier1_toxic_label,
            'tier1_t_low':      args.tier1_t_low,
            'tier1_t_high':     args.tier1_t_high,
            'tier2_base_model': args.tier2_base_model,
            'tier2_adapter_dir': args.tier2_adapter_dir,
            'n_samples':        n,
            'n_hateful':        n_hateful,
            'n_safe':           n_safe,
        },
        'routing': {
            'n_tier1_safe':   n_t1_safe,
            'n_tier1_unsafe': n_t1_unsafe,
            'n_deferred':     n_defer,
            'deferral_rate':  n_defer / n,
            'tier1_fnr':      tier1_fnr,
            'tier1_fpr':      tier1_fpr,
        },
        'combined_metrics':     combined_metrics,
        'tier1_only_metrics':   tier1_only_metrics,
        'tier2_alone_metrics':  tier2_alone_metrics,
        'samples':              sample_log,
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved → {results_path}")

    # Human-readable summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Two-Tier Moderation Evaluation — Reddit-FR\n")
        f.write(f"{'='*60}\n")
        f.write(f"Tier 1 : {args.tier1_model_id}\n")
        f.write(f"Tier 2 : {args.tier2_base_model} + {args.tier2_adapter_dir}\n")
        f.write(f"T_low  : {args.tier1_t_low}  T_high: {args.tier1_t_high}\n")
        f.write(f"n      : {n}  (toxic={n_hateful}, safe={n_safe})\n\n")
        f.write(f"Routing:\n")
        f.write(f"  Tier 1 SAFE   : {n_t1_safe:4d}  ({n_t1_safe/n:.1%})\n")
        f.write(f"  Deferred      : {n_defer:4d}  ({n_defer/n:.1%})\n")
        f.write(f"  Tier 1 UNSAFE : {n_t1_unsafe:4d}  ({n_t1_unsafe/n:.1%})\n")
        f.write(f"  Tier 1 FNR    : {tier1_fnr:.1%}\n")
        f.write(f"  Tier 1 FPR    : {tier1_fpr:.1%}\n\n")
        f.write(f"{'Metric':<20} {'Combined':>10} {'Tier1-only':>12}"
                + (f" {'Tier2-alone':>12}" if tier2_alone_metrics else "") + "\n")
        f.write(f"{'-'*58}\n")
        for key in ('f1', 'precision', 'recall', 'accuracy', 'fnr', 'fpr'):
            line = f"{key:<20} {combined_metrics[key]:>10.3f} {tier1_only_metrics[key]:>12.3f}"
            if tier2_alone_metrics:
                line += f" {tier2_alone_metrics[key]:>12.3f}"
            f.write(line + "\n")
    print(f"  Summary saved    → {summary_path}")
    print(f"\nDone.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end evaluation of the two-tier moderation system on Reddit-FR',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Data source
    data_grp = parser.add_mutually_exclusive_group(required=True)
    data_grp.add_argument('--reddit_fr_path',
                          help='Path to Reddit-FR CSV (text, label columns)')
    data_grp.add_argument('--test_set_json',
                          help='Path to test_set.json from finetune_detoxify_tier1.py '
                               '(held-out samples — recommended for scientific rigour)')

    # Tier 1
    parser.add_argument('--tier1_model_id', required=True,
                        help='HF model ID or local path for Tier 1 model. '
                             'Recommended: ~/code/results/tier1_detoxify_finetuned/best')
    parser.add_argument('--tier1_toxic_label', default='toxic',
                        help="Label name meaning 'toxic' in Tier 1 output (default: 'toxic')")
    parser.add_argument('--tier1_t_low',  type=float, required=True,
                        help='Lower threshold: score < t_low → SAFE')
    parser.add_argument('--tier1_t_high', type=float, required=True,
                        help='Upper threshold: score > t_high → UNSAFE')

    # Tier 2
    parser.add_argument('--tier2_base_model', default='google/shieldgemma-2b',
                        help='Tier 2 base model HF ID (default: google/shieldgemma-2b)')
    parser.add_argument('--tier2_adapter_dir', required=True,
                        help='Path to SG-2b LoRA adapter. '
                             'Recommended: ~/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best')

    # Misc
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results.json and summary.txt')
    parser.add_argument('--cache_dir',
                        default=str(Path.home() / 'datasets/cache'))
    parser.add_argument('--device', default='cuda',
                        help='Device for Tier 2 GPU inference (default: cuda). '
                             'Tier 1 always runs on CPU via HF pipeline.')
    parser.add_argument('--full_comparison', action='store_true', default=False,
                        help='Also run Tier 2 on the full test set to compute Tier-2-alone baseline '
                             '(slower — doubles Tier 2 inference time)')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
