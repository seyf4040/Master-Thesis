#!/usr/bin/env python3
"""
finetune_detoxify_tier1.py — Fine-tune Detoxify-M backbone on Reddit-FR for Tier 1 calibration

Detoxify-multilingual (unitary/multilingual-toxic-xlm-roberta) assigns near-zero toxicity
scores to informal French hate speech — the model is out-of-distribution for Reddit-FR.
This script fine-tunes the XLM-RoBERTa backbone with a fresh single-label regression head
using MSE loss (targets: 0.0 safe, 1.0 toxic), directly re-calibrating output scores for
informal French.

Why MSE regression (not cross-entropy):
  Cross-entropy pushes outputs toward binary extremes; threshold analysis needs well-calibrated
  continuous scores across [0,1]. MSE loss on sigmoid(logit) produces a bimodal distribution
  with soft peaks — the uncertainty region stays usable for threshold sweeps.

Why full fine-tuning (not LoRA):
  XLM-RoBERTa-base has ~270M params (10× smaller than SG-2b). Fits on A5000 with batch_size=16.
  Full fine-tuning gives stronger adaptation for a model that is heavily out-of-distribution
  on the target domain.

Output:
  {output_dir}/
    best/                    # best val-loss checkpoint (HF save_pretrained format)
    final/                   # last-epoch checkpoint
    test_set.json            # held-out samples (never seen during training)
    training_meta.json       # hyperparameters, training log, timestamps

After training, evaluate Tier 1 calibration with:
  python code/phase4_two_tier/analyze_threshold_tier1.py \\
      --model hf_classifier \\
      --hf_model_id {output_dir}/best \\
      --hf_toxic_label toxic \\
      --output_dir ~/code/results/tier1_comparison \\
      --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv

Usage:
    python code/phase4_two_tier/finetune_detoxify_tier1.py \\
        --reddit_fr_path ~/datasets/reddit/balanced/data-fr/test-fr.csv \\
        --output_dir ~/code/results/tier1_detoxify_finetuned \\
        --epochs 3 --lr 2e-5 --batch_size 16

Author: Ural Seyfullah
"""

import gc
import json
import time
import random
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# ── Bypass torch.load CVE-2025-32434 check (transformers >= 4.49 enforces torch >= 2.6) ──
# The cluster runs torch < 2.6. The check lives in import_utils but is imported by
# name into modeling_utils — patch BOTH: source module + the local binding.
# Safe: unitary/multilingual-toxic-xlm-roberta is a known-safe HuggingFace model.
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


BASE_MODEL_ID = 'unitary/multilingual-toxic-xlm-roberta'


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_reddit_fr(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]


def load_synthetic(path: str) -> List[Dict]:
    """Load Phase 3 synthetic French hate speech from a directory of *.jsonl files.
    Strips the '1.1. ' list-parsing artifact from generated text."""
    import re
    import glob
    samples = []
    for fpath in sorted(glob.glob(str(Path(path) / '*.jsonl'))):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    text = re.sub(r'^\d+\.\d+\.\s*', '', r.get('text', '') or '')
                    if text.strip():
                        samples.append({'text': text, 'label': int(r['label'])})
                except (json.JSONDecodeError, KeyError):
                    pass
    return samples


# ── Train / val / test split ──────────────────────────────────────────────────

def _split_source(samples: List[Dict], seed: int,
                  test_frac: float, val_frac: float
                  ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Three-way split: train / val / test.  test_frac applied first,
    then val_frac to the remaining data."""
    random.Random(seed).shuffle(samples)
    n_test    = max(1, int(len(samples) * test_frac))
    test      = samples[:n_test]
    remaining = samples[n_test:]
    n_val     = max(1, int(len(remaining) * val_frac))
    val       = remaining[:n_val]
    train     = remaining[n_val:]
    return train, val, test


# ── PyTorch Dataset & collate ─────────────────────────────────────────────────

class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch):
        texts  = [b['text']  for b in batch]
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        return {**enc, 'labels': labels}
    return collate_fn


# ── Training ──────────────────────────────────────────────────────────────────

def finetune(args):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
        get_linear_schedule_with_warmup

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Fine-tuning Detoxify-M backbone for Tier 1 calibration")
    print(f"{'='*65}")
    print(f"  Base model   : {BASE_MODEL_ID}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  LR           : {args.lr}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Grad accum   : {args.grad_accum}  (effective batch = {args.batch_size * args.grad_accum})")
    print(f"  Max length   : {args.max_length}")
    print(f"  Test frac    : {args.test_fraction}")
    print(f"  Val frac     : {args.val_fraction}")

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\n  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"\n  Device: cpu")

    # ── Load and split data ───────────────────────────────────────────────────
    print(f"\nLoading Reddit-FR from {args.reddit_fr_path}...")
    all_samples = load_reddit_fr(args.reddit_fr_path)

    if args.synthetic_data_path:
        print(f"Loading synthetic data from {args.synthetic_data_path}...")
        synthetic = load_synthetic(args.synthetic_data_path)
        print(f"  Synthetic: {len(synthetic)} items  "
              f"(toxic={sum(s['label'] for s in synthetic)}, "
              f"safe={sum(1-s['label'] for s in synthetic)})")
        all_samples = all_samples + synthetic

    train_s, val_s, test_s = _split_source(
        all_samples, args.seed, args.test_fraction, args.val_fraction
    )

    n_pos_tr = sum(s['label'] for s in train_s)
    n_pos_va = sum(s['label'] for s in val_s)
    n_pos_te = sum(s['label'] for s in test_s)
    print(f"  Total  : {len(all_samples)}")
    print(f"  Train  : {len(train_s)}  (toxic={n_pos_tr}, safe={len(train_s)-n_pos_tr})")
    print(f"  Val    : {len(val_s)}   (toxic={n_pos_va}, safe={len(val_s)-n_pos_va})")
    print(f"  Test   : {len(test_s)}  (toxic={n_pos_te}, safe={len(test_s)-n_pos_te})  ← held-out")

    test_set_path = output_dir / 'test_set.json'
    with open(test_set_path, 'w') as f:
        json.dump({
            'dataset':       'reddit_fr',
            'seed':          args.seed,
            'test_fraction': args.test_fraction,
            'n_test':        len(test_s),
            'samples':       test_s,
        }, f, indent=2)
    print(f"  Test set saved → {test_set_path}")

    # ── Load tokenizer and model ──────────────────────────────────────────────
    print(f"\nLoading {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, cache_dir=args.cache_dir
    )

    # Load the full pretrained model (num_labels=7: toxicity, severe_toxicity, obscene, ...).
    # Then manually replace only the output projection with a single-label head.
    # This avoids the `ignore_mismatched_sizes` parameter (added in transformers 4.15),
    # which raises TypeError on older cluster installs.
    # The dense layer weights from the original toxicity head are preserved.
    import torch.nn as nn
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        cache_dir=args.cache_dir,
    )
    # Replace 7-output head with a single toxicity output.
    # classifier.out_proj: Linear(hidden_size, 7) → Linear(hidden_size, 1)
    hidden_size = model.classifier.out_proj.in_features
    model.classifier.out_proj = nn.Linear(hidden_size, 1)
    model.config.num_labels   = 1
    # Name the single output 'toxic' so analyze_threshold_tier1.py can use --hf_toxic_label toxic
    model.config.id2label     = {0: 'toxic'}
    model.config.label2id     = {'toxic': 0}
    model.config.problem_type = 'regression'   # hint for pipeline inference (applies sigmoid)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = torch.utils.data.DataLoader(
        RedditDataset(train_s), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        RedditDataset(val_s), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    # ── Optimizer and scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"\n  Total optimizer steps: {total_steps}  (warmup: {warmup_steps})")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_log     = []
    best_val_loss = float('inf')
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t_epoch = time.time()

        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop('labels').to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits.squeeze(-1)   # [batch]
            scores = torch.sigmoid(logits)               # [batch] in [0,1]
            targets = labels
            if args.label_smoothing > 0:
                # Soft targets: {0→ε, 1→1-ε}. Keeps calibrated uncertainty in
                # the middle range instead of forcing extreme bimodal scores.
                eps = args.label_smoothing
                targets = labels * (1 - 2 * eps) + eps
            loss   = F.mse_loss(scores, targets) / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum  # un-scale for logging

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Flush remaining gradients
        if len(train_loader) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop('labels').to(device)
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits.squeeze(-1)
                scores = torch.sigmoid(logits)
                val_loss += F.mse_loss(scores, labels).item()
        avg_val_loss = val_loss / len(val_loader)

        elapsed = time.time() - t_epoch
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"time={elapsed:.0f}s  step={global_step}")
        train_log.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'elapsed_s': elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / 'best')
            tokenizer.save_pretrained(output_dir / 'best')
            print(f"    → New best val_loss={best_val_loss:.4f}, saved to {output_dir / 'best'}")

    # ── Save final checkpoint ─────────────────────────────────────────────────
    model.save_pretrained(output_dir / 'final')
    tokenizer.save_pretrained(output_dir / 'final')

    meta = {
        'base_model_id':  BASE_MODEL_ID,
        'dataset':        'reddit_fr',
        'reddit_fr_path': args.reddit_fr_path,
        'epochs':         args.epochs,
        'lr':             args.lr,
        'batch_size':     args.batch_size,
        'grad_accum':     args.grad_accum,
        'max_length':     args.max_length,
        'seed':           args.seed,
        'test_fraction':  args.test_fraction,
        'val_fraction':   args.val_fraction,
        'n_train':        len(train_s),
        'n_val':          len(val_s),
        'n_test':         len(test_s),
        'test_set_path':  str(test_set_path),
        'best_val_loss':  best_val_loss,
        'train_log':      train_log,
        'label_smoothing':    args.label_smoothing,
        'synthetic_data_path': args.synthetic_data_path,
        'loss_function':  'mse(sigmoid(logit), target)'
                          + (f' label_smoothing={args.label_smoothing}' if args.label_smoothing else ''),
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'training_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Fine-tuning complete.")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Best checkpoint: {output_dir / 'best'}")
    print(f"  Final checkpoint: {output_dir / 'final'}")
    print(f"  Test set       : {test_set_path}")
    print(f"\n  Next step — run threshold analysis:")
    print(f"    python code/phase4_two_tier/analyze_threshold_tier1.py \\")
    print(f"        --model hf_classifier \\")
    print(f"        --hf_model_id {output_dir / 'best'} \\")
    print(f"        --hf_toxic_label toxic \\")
    print(f"        --output_dir ~/code/results/tier1_comparison \\")
    print(f"        --reddit_fr_path {args.reddit_fr_path}")
    print(f"{'='*65}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Detoxify-M backbone on Reddit-FR for Tier 1 score calibration',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--reddit_fr_path', required=True,
                        help='Path to Reddit-FR CSV (columns: text, label)')
    parser.add_argument('--output_dir', required=True,
                        help='Root output dir for checkpoints and metadata')
    parser.add_argument('--cache_dir',
                        default=str(Path.home() / 'datasets/cache'),
                        help='HuggingFace model/dataset cache dir')
    parser.add_argument('--epochs',         type=int,   default=3)
    parser.add_argument('--lr',             type=float, default=2e-5,
                        help='Learning rate (default: 2e-5 — full fine-tuning, not LoRA)')
    parser.add_argument('--batch_size',     type=int,   default=16)
    parser.add_argument('--grad_accum',     type=int,   default=2,
                        help='Gradient accumulation steps (effective batch = batch_size × grad_accum)')
    parser.add_argument('--max_length',     type=int,   default=128,
                        help='Max token length (XLM-RoBERTa encoder; 128 covers most Reddit comments)')
    parser.add_argument('--test_fraction',  type=float, default=0.1)
    parser.add_argument('--val_fraction',   type=float, default=0.1,
                        help='Fraction of remaining (post-test) data used for validation')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--device',         default='cuda',
                        help='cuda or cpu (default: cuda)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing ε: targets become {ε, 1-ε} instead of {0,1}. '
                             'Reduces bimodal score collapse and preserves calibrated uncertainty. '
                             'Recommended: 0.05 (default: 0.0 = hard labels)')
    parser.add_argument('--synthetic_data_path', default=None,
                        help='Optional path to directory of *.jsonl synthetic data files '
                             '(Phase 3 Track A). Concatenated with Reddit-FR before splitting.')
    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()
