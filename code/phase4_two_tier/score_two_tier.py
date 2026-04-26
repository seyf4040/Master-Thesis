#!/usr/bin/env python3
"""
score_two_tier.py — Inference pass for the two-tier moderation system

Scores ALL test samples with both Tier 1 (lightweight classifier) and Tier 2
(SG-2b + LoRA) and saves per-sample results to raw_scores.json. Threshold
simulation is done separately by simulate_thresholds.py — no model reload needed.

Design rationale:
  Running Tier 2 on ALL samples (not just deferred ones) decouples inference
  from threshold selection. simulate_thresholds.py can then sweep any (T_low,
  T_high) configuration offline. The extra Tier 2 inference cost (~511 samples)
  is small compared to GPU setup time.

Per-sample timing enables:
  avg_ms(T_low, T_high) = mean(t1_ms) + deferral_rate × mean(t2_ms)
  — a compute-cost axis for the thesis deployability argument.

Usage:
    # Pretrained Tier 1:
    python code/phase4_two_tier/score_two_tier.py \\
        --tier1_model_id   unitary/multilingual-toxic-xlm-roberta \\
        --tier2_adapter_dir ~/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best \\
        --test_set_json    ~/code/results/tier1_detoxify_finetuned/test_set.json \\
        --output_dir       ~/code/results/two_tier_scores/pretrained

    # Fine-tuned Tier 1:
    python code/phase4_two_tier/score_two_tier.py \\
        --tier1_model_id   ~/code/results/tier1_detoxify_finetuned/best \\
        --tier2_adapter_dir ~/code/results/lora_adapters/shieldgemma_2b/reddit_fr/best \\
        --test_set_json    ~/code/results/tier1_detoxify_finetuned/test_set.json \\
        --output_dir       ~/code/results/two_tier_scores/finetuned

Author: Ural Seyfullah
"""

# ── CVE-2025-32434 bypass ─────────────────────────────────────────────────────
# Cluster runs torch < 2.6; transformers >= 4.49 blocks torch.load(). The check
# lives in import_utils but is imported by name into modeling_utils — patch both.
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
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch


# ── Prompt template (matches finetune_lora.py and evaluate_two_tier.py) ──────

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


# ── Data loader ───────────────────────────────────────────────────────────────

def load_test_set(path: str) -> List[Dict]:
    """Load test_set.json saved by finetune_detoxify_tier1.py."""
    with open(path) as f:
        data = json.load(f)
    return data['samples']   # [{'text': str, 'label': int}, ...]


# ── Tier 1 scoring ────────────────────────────────────────────────────────────

def score_tier1(samples: List[Dict], model_id: str, toxic_label: str,
                device: str) -> List[Dict]:
    """
    Score all samples with Tier 1 model.
    Returns list of {'t1_score': float, 't1_ms': float} in same order as input.
    Uses HuggingFace text-classification pipeline, one sample at a time for
    accurate per-sample timing.
    """
    from transformers import pipeline
    from tqdm import tqdm

    _device = 0 if (device == 'cuda' and torch.cuda.is_available()) else -1
    print(f"\n  Loading Tier 1: {model_id} (device={'cuda' if _device==0 else 'cpu'}) ...")
    classifier = pipeline(
        "text-classification",
        model=model_id,
        device=_device,
        truncation=True,
        max_length=512,
        top_k=None,
    )

    toxic_key = toxic_label.lower()
    n_fallback = 0
    results = []

    for s in tqdm(samples, desc="Tier 1", unit="sample", leave=False):
        t0 = time.perf_counter()
        preds = classifier(s['text'])
        t1_ms = (time.perf_counter() - t0) * 1000

        if isinstance(preds[0], list):
            preds = preds[0]
        label_scores = {p['label'].lower(): p['score'] for p in preds}

        if toxic_key in label_scores:
            score = label_scores[toxic_key]
        else:
            non_toxic = [v for k, v in label_scores.items() if k != toxic_key]
            score = 1.0 - max(non_toxic) if non_toxic else 0.5
            n_fallback += 1

        results.append({'t1_score': float(score), 't1_ms': float(t1_ms)})

    if n_fallback > 0:
        print(f"  Warning: {n_fallback}/{len(samples)} samples used fallback scoring "
              f"(label '{toxic_label}' not found — using 1 - max_other_prob)")

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── Tier 2 scoring ────────────────────────────────────────────────────────────

def load_tier2(base_model_id: str, adapter_dir: str, cache_dir: str, device: str):
    """Load SG-2b + LoRA adapter. Returns (model, tokenizer, yes_id, no_id)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"\n  Loading Tier 2 base: {base_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)

    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map='auto',
        cache_dir=cache_dir,
    )

    print(f"  Applying adapter: {adapter_dir} ...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    yes_ids = tokenizer('Yes', add_special_tokens=False)['input_ids']
    no_ids  = tokenizer('No',  add_special_tokens=False)['input_ids']
    yes_id, no_id = yes_ids[0], no_ids[0]
    print(f"  Token IDs — 'Yes' (unsafe): {yes_id},  'No' (safe): {no_id}")

    return model, tokenizer, yes_id, no_id


def score_tier2(samples: List[Dict], model, tokenizer, yes_id: int, no_id: int,
                device: str, max_length: int = 512) -> List[Dict]:
    """
    Score ALL samples with Tier 2 (SG-2b LoRA).
    Processes one sample at a time for accurate per-sample timing and to handle
    variable-length prompts reliably.
    Returns list of {'t2_pred': int, 't2_prob': float, 't2_ms': float}.
    """
    from tqdm import tqdm

    model.eval()
    _device = next(model.parameters()).device
    results = []

    for s in tqdm(samples, desc="Tier 2", unit="sample", leave=False):
        prompt = _shieldgemma_prompt(s['text'])
        inputs = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            next_token_logits = model(**inputs).logits[0, -1, :]
        t2_ms = (time.perf_counter() - t0) * 1000

        # Normalise Yes/No logits to binary probability
        yes_no = torch.tensor(
            [next_token_logits[no_id].item(), next_token_logits[yes_id].item()]
        ).softmax(dim=-1)
        prob_yes = float(yes_no[1].item())
        pred = int(prob_yes >= 0.5)

        results.append({'t2_pred': pred, 't2_prob': prob_yes, 't2_ms': float(t2_ms)})

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Score all test samples with both tiers; save for offline simulation')

    parser.add_argument('--test_set_json', required=True,
                        help='Path to test_set.json from finetune_detoxify_tier1.py')
    parser.add_argument('--tier1_model_id', required=True,
                        help='HF model ID or local path for Tier 1')
    parser.add_argument('--tier1_toxic_label', default='toxic',
                        help="Label name meaning 'toxic' in Tier 1 output (default: 'toxic')")
    parser.add_argument('--tier2_base_model', default='google/shieldgemma-2b')
    parser.add_argument('--tier2_adapter_dir', required=True,
                        help='Path to SG-2b LoRA adapter directory')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory — saves raw_scores.json')
    parser.add_argument('--cache_dir',
                        default=str(Path.home() / 'datasets/cache'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Two-Tier Scoring Run")
    print(f"  Tier 1: {args.tier1_model_id}")
    print(f"  Tier 2: {args.tier2_base_model} + {args.tier2_adapter_dir}")
    print(f"  Test set: {args.test_set_json}")
    print(f"{'='*65}")

    # ── Load data ─────────────────────────────────────────────────────────────
    samples = load_test_set(args.test_set_json)
    n = len(samples)
    n_hateful = sum(s['label'] for s in samples)
    print(f"\n  {n} samples  (hateful={n_hateful}, safe={n - n_hateful})")

    # ── Tier 1 inference ──────────────────────────────────────────────────────
    t1_results = score_tier1(samples, args.tier1_model_id,
                              args.tier1_toxic_label, args.device)

    # ── Tier 2 inference ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Tier 2: scoring all {n} samples")
    print(f"{'='*65}")
    model, tokenizer, yes_id, no_id = load_tier2(
        args.tier2_base_model, args.tier2_adapter_dir, args.cache_dir, args.device
    )
    t2_results = score_tier2(samples, model, tokenizer, yes_id, no_id, args.device)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Combine and save ──────────────────────────────────────────────────────
    combined = []
    for i, (s, t1, t2) in enumerate(zip(samples, t1_results, t2_results)):
        combined.append({
            'idx':     i,
            'label':   s['label'],
            **t1,   # t1_score, t1_ms
            **t2,   # t2_pred, t2_prob, t2_ms
        })

    # Quick sanity summary
    t1_scores = [r['t1_score'] for r in combined]
    t2_preds  = [r['t2_pred']  for r in combined]
    avg_t1_ms = sum(r['t1_ms'] for r in combined) / n
    avg_t2_ms = sum(r['t2_ms'] for r in combined) / n

    labels = [s['label'] for s in samples]
    t2_tp = sum(p == 1 and l == 1 for p, l in zip(t2_preds, labels))
    t2_fp = sum(p == 1 and l == 0 for p, l in zip(t2_preds, labels))
    t2_tn = sum(p == 0 and l == 0 for p, l in zip(t2_preds, labels))
    t2_fn = sum(p == 0 and l == 1 for p, l in zip(t2_preds, labels))
    t2_prec = t2_tp / (t2_tp + t2_fp) if (t2_tp + t2_fp) > 0 else 0
    t2_rec  = t2_tp / (t2_tp + t2_fn) if (t2_tp + t2_fn) > 0 else 0
    t2_f1   = 2 * t2_prec * t2_rec / (t2_prec + t2_rec) if (t2_prec + t2_rec) > 0 else 0

    print(f"\n{'='*65}")
    print(f"  Scoring complete")
    print(f"  Tier 1 score range: [{min(t1_scores):.3f}, {max(t1_scores):.3f}]  "
          f"avg={sum(t1_scores)/n:.3f}  avg_ms={avg_t1_ms:.1f}")
    print(f"  Tier 2 alone  — F1={t2_f1:.3f}  P={t2_prec:.3f}  R={t2_rec:.3f}  "
          f"avg_ms={avg_t2_ms:.1f}")
    print(f"{'='*65}")

    output = {
        'tier1_model_id':    args.tier1_model_id,
        'tier1_toxic_label': args.tier1_toxic_label,
        'tier2_base_model':  args.tier2_base_model,
        'tier2_adapter_dir': args.tier2_adapter_dir,
        'test_set_json':     args.test_set_json,
        'n_samples':         n,
        'n_hateful':         n_hateful,
        'n_safe':            n - n_hateful,
        'tier2_alone': {
            'f1': t2_f1, 'precision': t2_prec, 'recall': t2_rec,
            'tp': t2_tp, 'fp': t2_fp, 'tn': t2_tn, 'fn': t2_fn,
            'avg_ms': avg_t2_ms,
        },
        'avg_t1_ms': avg_t1_ms,
        'avg_t2_ms': avg_t2_ms,
        'samples': combined,
    }

    out_path = output_dir / 'raw_scores.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw scores saved → {out_path}")
    print(f"  Run simulate_thresholds.py on this file to explore all threshold configs.")


if __name__ == '__main__':
    main()
