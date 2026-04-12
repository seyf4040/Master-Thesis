#!/usr/bin/env python3
"""
finetune_lora.py — LoRA fine-tuning for content moderation LLMs (Phase 2)

Supported models:
  - llama_guard_1b  → meta-llama/Llama-Guard-3-1B
  - shieldgemma_2b  → google/shieldgemma-2b

Training format (causal LM, label-tokens-only loss):
  - Llama Guard:  <llama_guard_prompt(text)> + "safe" or "unsafe"
  - ShieldGemma:  <shieldgemma_prompt(text)> + "No" or "Yes"

Usage:
    python code/finetune_lora.py \\
        --model llama_guard_1b \\
        --dataset french_hate_superset \\
        --output_dir ~/code/results/lora_adapters \\
        --epochs 3 --lr 2e-4 --lora_r 16 --lora_alpha 32 \\
        --batch_size 4 --max_length 512 \\
        --cache_dir ~/datasets/cache

    python code/finetune_lora.py \\
        --model shieldgemma_2b \\
        --dataset reddit_fr \\
        --output_dir ~/code/results/lora_adapters \\
        --epochs 3 --lr 2e-4 --lora_r 8 --lora_alpha 16 \\
        --batch_size 4 --max_length 512 \\
        --cache_dir ~/datasets/cache

Adapter weights are saved to:
    {output_dir}/{model}/{dataset}/
"""

import gc
import json
import time
import random
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


# ── Prompt templates (identical to run_full_baseline_v3.py) ──────────────────

def _llama_guard_prompt(text: str) -> str:
    """Official Llama Guard 3 prompt. Model should output 'safe' or 'unsafe'."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Task: Check if there is unsafe content in 'User' messages according to our safety policy.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        "S1: Violent Crimes. S2: Non-Violent Crimes. S3: Sex Crimes.\n"
        "S4: Child Exploitation. S5: Defamation. S6: Specialized Advice.\n"
        "S7: Privacy. S8: Intellectual Property. S9: Indiscriminate Weapons.\n"
        "S10: Hate. S11: Self-Harm. S12: Sexual Content.\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        f"<BEGIN CONVERSATION>\nUser: {text}\n<END CONVERSATION>\n\n"
        "Provide your safety assessment:\n"
        "- First line must read 'safe' or 'unsafe'.\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def _shieldgemma_prompt(text: str) -> str:
    """ShieldGemma harm-assessment prompt. Model should output 'Yes' or 'No'."""
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


# ── Dataset loaders (identical to run_full_baseline_v3.py) ───────────────────

Sample = Dict  # {'text': str, 'label': int}


def load_hatecheck_fr(cache_dir: str, max_samples: Optional[int] = None) -> List[Sample]:
    from datasets import load_dataset as _ld
    ds = _ld("Paul/hatecheck-french", cache_dir=cache_dir)
    samples = [
        {'text': item['test_case'], 'label': 1 if item['label_gold'] == 'hateful' else 0}
        for item in ds['test']
    ]
    return samples[:max_samples] if max_samples else samples


def load_hatecheck_en(cache_dir: str, max_samples: Optional[int] = None) -> List[Sample]:
    from datasets import load_dataset as _ld
    ds = _ld("Paul/hatecheck", cache_dir=cache_dir)
    samples = [
        {'text': item['test_case'], 'label': 1 if item['label_gold'] == 'hateful' else 0}
        for item in ds['test']
    ]
    return samples[:max_samples] if max_samples else samples


def load_french_hate_superset(cache_dir: str, max_samples: Optional[int] = None) -> List[Sample]:
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
    if max_samples and len(samples) > max_samples:
        random.Random(42).shuffle(samples)
        samples = samples[:max_samples]
    return samples


def load_toxigen(path: str, max_samples: int = 5000) -> List[Sample]:
    raw = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            raw.append(json.loads(line.strip()))
    if max_samples and len(raw) > max_samples:
        raw = random.Random(42).sample(raw, max_samples)
    samples = []
    for item in raw:
        text  = item.get('generation') or item.get('text', '')
        label = item.get('prompt_label', item.get('label'))
        if text and label is not None:
            samples.append({'text': text, 'label': int(label)})
    return samples


def load_openai(path: str, max_samples: Optional[int] = None) -> List[Sample]:
    label_keys = ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
    samples = []
    with open(path, 'r') as f:
        for line in f:
            item  = json.loads(line)
            label = 1 if any(item.get(k, 0) for k in label_keys) else 0
            samples.append({'text': item.get('prompt', ''), 'label': label})
    return samples[:max_samples] if max_samples else samples


def load_civil_comments(cache_dir: str, max_samples: int = 5000) -> List[Sample]:
    from datasets import load_dataset as _ld
    ds    = _ld("google/civil_comments", cache_dir=cache_dir)
    split = ds['train']
    indices = random.Random(42).sample(range(len(split)), min(max_samples, len(split)))
    samples = []
    for idx in indices:
        item = split[idx]
        text = item.get('text', '') or ''
        if text.strip():
            samples.append({'text': text, 'label': 1 if item.get('toxicity', 0) >= 0.5 else 0})
    return samples


def load_reddit(path: str, max_samples: Optional[int] = None) -> List[Sample]:
    import pandas as pd
    df = pd.read_csv(path)
    samples = [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]
    return samples[:max_samples] if max_samples else samples


ALL_DATASET_KEYS = [
    'hatecheck_fr', 'hatecheck_en', 'french_hate_superset',
    'toxigen', 'openai', 'civil_comments', 'reddit_en', 'reddit_fr',
]


def load_dataset_samples(dataset_key: str, args) -> List[Sample]:
    if dataset_key == 'hatecheck_fr':
        return load_hatecheck_fr(args.cache_dir, args.max_samples)
    elif dataset_key == 'hatecheck_en':
        return load_hatecheck_en(args.cache_dir, args.max_samples)
    elif dataset_key == 'french_hate_superset':
        return load_french_hate_superset(args.cache_dir, args.max_samples)
    elif dataset_key == 'toxigen':
        return load_toxigen(args.toxigen_path, args.max_samples or 5000)
    elif dataset_key == 'openai':
        return load_openai(args.openai_path, args.max_samples)
    elif dataset_key == 'civil_comments':
        return load_civil_comments(args.cache_dir, args.max_samples or 5000)
    elif dataset_key == 'reddit_en':
        return load_reddit(args.reddit_en_path, args.max_samples)
    elif dataset_key == 'reddit_fr':
        return load_reddit(args.reddit_fr_path, args.max_samples)
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")


# ── Model configs ─────────────────────────────────────────────────────────────

MODEL_IDS = {
    'llama_guard_1b': 'meta-llama/Llama-Guard-3-1B',
    'shieldgemma_2b': 'google/shieldgemma-2b',
}

# LoRA target modules per architecture
# Llama Guard 3 1B is based on Llama 3.2 (standard LlamaForCausalLM)
# ShieldGemma 2B is based on Gemma 2
LORA_TARGET_MODULES = {
    'llama_guard_1b': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
    'shieldgemma_2b': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
}

# Label tokens used during inference (must match run_full_baseline_v3.py)
LABEL_TOKENS = {
    'llama_guard_1b': {1: 'unsafe', 0: 'safe'},
    'shieldgemma_2b': {1: 'Yes',    0: 'No'},
}

PROMPT_FN = {
    'llama_guard_1b': _llama_guard_prompt,
    'shieldgemma_2b': _shieldgemma_prompt,
}


# ── Training data preparation ─────────────────────────────────────────────────

def make_token_ids(sample: Sample, model_key: str, tokenizer, max_length: int):
    """
    Build input_ids and labels for one training example.
    Labels are -100 for all prompt tokens (masked from loss),
    and the actual token IDs for the label token(s) + EOS.
    """
    prompt_fn   = PROMPT_FN[model_key]
    label_token = LABEL_TOKENS[model_key][sample['label']]

    prompt_ids = tokenizer(
        prompt_fn(sample['text']), add_special_tokens=False
    )['input_ids']

    target_ids = tokenizer(
        label_token, add_special_tokens=False
    )['input_ids']

    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = (prompt_ids + target_ids + eos)[:max_length]
    labels    = ([-100] * len(prompt_ids) + target_ids + eos)[:max_length]

    return {'input_ids': input_ids, 'labels': labels}


class ModerationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Sample], model_key: str, tokenizer, max_length: int):
        self.examples = []
        skipped = 0
        for s in samples:
            ex = make_token_ids(s, model_key, tokenizer, max_length)
            # Skip examples where the entire sequence is prompt (label was truncated)
            if all(l == -100 for l in ex['labels']):
                skipped += 1
                continue
            self.examples.append(ex)
        if skipped:
            print(f"  Warning: {skipped} examples skipped (label truncated by max_length)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id: int):
    """Left-pad to the longest sequence in the batch; labels padded with -100."""
    max_len = max(len(ex['input_ids']) for ex in batch)
    input_ids_padded = []
    labels_padded    = []
    attention_masks  = []
    for ex in batch:
        pad_len = max_len - len(ex['input_ids'])
        input_ids_padded.append([pad_token_id] * pad_len + ex['input_ids'])
        labels_padded.append(   [-100]         * pad_len + ex['labels'])
        attention_masks.append( [0]            * pad_len + [1] * len(ex['input_ids']))
    return {
        'input_ids':      torch.tensor(input_ids_padded, dtype=torch.long),
        'labels':         torch.tensor(labels_padded,    dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks,  dtype=torch.long),
    }


# ── LoRA fine-tuning ──────────────────────────────────────────────────────────

def finetune(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, TaskType

    model_id   = MODEL_IDS[args.model]
    output_dir = Path(args.output_dir) / args.model / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"LoRA Fine-tuning: {args.model} on {args.dataset}")
    print(f"{'='*70}")
    print(f"Base model    : {model_id}")
    print(f"Output dir    : {output_dir}")
    print(f"Epochs        : {args.epochs}")
    print(f"LR            : {args.lr}")
    print(f"LoRA r        : {args.lora_r}")
    print(f"LoRA alpha    : {args.lora_alpha}")
    print(f"LoRA dropout  : {args.lora_dropout}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Grad accum    : {args.grad_accum}")
    print(f"Max length    : {args.max_length}")
    print(f"Test fraction : {args.test_fraction}")
    print(f"Val fraction  : {args.val_fraction} (of remaining after test split)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:   {torch.cuda.get_device_name(0)}")
        print(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}...")
    all_samples = load_dataset_samples(args.dataset, args)
    random.Random(args.seed).shuffle(all_samples)

    # Three-way split: held-out test / val (early stopping) / train
    n_test    = max(1, int(len(all_samples) * args.test_fraction))
    test_s    = all_samples[:n_test]
    remaining = all_samples[n_test:]
    n_val     = max(1, int(len(remaining) * args.val_fraction))
    val_s     = remaining[:n_val]
    train_s   = remaining[n_val:]

    n_pos_te = sum(s['label'] for s in test_s)
    n_pos_tr = sum(s['label'] for s in train_s)
    n_pos_va = sum(s['label'] for s in val_s)
    print(f"  Total : {len(all_samples)} samples")
    print(f"  Test  : {len(test_s)}  (unsafe={n_pos_te}, safe={len(test_s)-n_pos_te})  ← held-out, never seen during training")
    print(f"  Train : {len(train_s)} (unsafe={n_pos_tr}, safe={len(train_s)-n_pos_tr})")
    print(f"  Val   : {len(val_s)}   (unsafe={n_pos_va}, safe={len(val_s)-n_pos_va})")

    # Save test set so the eval script can load the exact same held-out samples
    test_set_path = output_dir / 'test_set.json'
    with open(test_set_path, 'w') as f:
        json.dump({
            'dataset':       args.dataset,
            'seed':          args.seed,
            'test_fraction': args.test_fraction,
            'n_test':        len(test_s),
            'samples':       test_s,
        }, f, indent=2)
    print(f"  Test set saved → {test_set_path}")

    # ── Load tokenizer & model ────────────────────────────────────────────────
    print(f"\nLoading tokenizer and model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map='auto',
        cache_dir=args.cache_dir,
    )
    base_model.config.use_cache = False  # required for gradient checkpointing

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES[args.model],
        bias='none',
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ── Build datasets & loaders ──────────────────────────────────────────────
    train_ds = ModerationDataset(train_s, args.model, tokenizer, args.max_length)
    val_ds   = ModerationDataset(val_s,   args.model, tokenizer, args.max_length)
    print(f"\n  Train examples after filtering: {len(train_ds)}")
    print(f"  Val   examples after filtering: {len(val_ds)}")

    pad_id = tokenizer.pad_token_id
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps   = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps  = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n  Total optimizer steps : {total_steps}  (warmup: {warmup_steps})")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_log = []
    best_val_loss = float('inf')
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t_epoch = time.time()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Flush remaining gradients at end of epoch
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
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
        avg_val_loss = val_loss / len(val_loader) if val_loader else float('nan')

        elapsed = time.time() - t_epoch
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"time={elapsed:.0f}s  step={global_step}")

        entry = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'elapsed_s': elapsed,
        }
        train_log.append(entry)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / 'best')
            tokenizer.save_pretrained(output_dir / 'best')
            print(f"    → New best val_loss={best_val_loss:.4f}, saved to {output_dir / 'best'}")

    # Save final checkpoint and training metadata
    model.save_pretrained(output_dir / 'final')
    tokenizer.save_pretrained(output_dir / 'final')

    meta = {
        'model':          args.model,
        'base_model_id':  model_id,
        'dataset':        args.dataset,
        'epochs':         args.epochs,
        'lr':             args.lr,
        'lora_r':         args.lora_r,
        'lora_alpha':     args.lora_alpha,
        'lora_dropout':   args.lora_dropout,
        'batch_size':     args.batch_size,
        'grad_accum':     args.grad_accum,
        'max_length':     args.max_length,
        'seed':           args.seed,
        'test_fraction':  args.test_fraction,
        'val_fraction':   args.val_fraction,
        'n_test':         len(test_s),
        'n_train':        len(train_ds),
        'n_val':          len(val_ds),
        'test_set_path':  str(test_set_path),
        'best_val_loss':  best_val_loss,
        'target_modules': LORA_TARGET_MODULES[args.model],
        'train_log':      train_log,
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'training_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Fine-tuning complete.")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best adapter  : {output_dir / 'best'}")
    print(f"  Final adapter : {output_dir / 'final'}")
    print(f"  Test set      : {test_set_path}  ({len(test_s)} samples)")
    print(f"  Metadata      : {output_dir / 'training_meta.json'}")
    print(f"\nNext step — evaluate baseline then LoRA on the same test set:")
    print(f"  python run_full_baseline_lora.py --test_set_path {test_set_path} \\")
    print(f"      --models llama_guard_1b,shieldgemma_2b --output_dir <results/baseline_on_test>")
    print(f"  python run_full_baseline_lora.py --test_set_path {test_set_path} \\")
    print(f"      --models llama_guard_1b_lora,shieldgemma_2b_lora --lora_adapter_llama ... --output_dir <results/lora_on_test>")
    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LoRA fine-tuning for content moderation LLMs (Phase 2)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model', required=True,
                        choices=list(MODEL_IDS.keys()),
                        help='Model to fine-tune')
    parser.add_argument('--dataset', required=True,
                        choices=ALL_DATASET_KEYS,
                        help='Training dataset key')
    parser.add_argument('--output_dir', required=True,
                        help='Root dir for adapter weights; saved to {output_dir}/{model}/{dataset}/')

    # Dataset paths (same defaults as run_full_baseline_v3.py)
    parser.add_argument('--cache_dir',
                        default=str(Path.home() / 'datasets/cache'))
    parser.add_argument('--openai_path',
                        default=str(Path.home() / 'datasets/openai/samples-1680.jsonl'))
    parser.add_argument('--toxigen_path',
                        default=str(Path.home() / 'datasets/toxigen/toxigen_train.jsonl'))
    parser.add_argument('--reddit_en_path',
                        default=str(Path.home() / 'datasets/reddit/balanced/data-en/test-en.csv'))
    parser.add_argument('--reddit_fr_path',
                        default=str(Path.home() / 'datasets/reddit/balanced/data-fr/test-fr.csv'))
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap total samples loaded (before train/val split)')

    # LoRA hyperparameters
    parser.add_argument('--lora_r',       type=int,   default=16)
    parser.add_argument('--lora_alpha',   type=int,   default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Training hyperparameters
    parser.add_argument('--epochs',       type=int,   default=3)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--grad_accum',   type=int,   default=4,
                        help='Gradient accumulation steps (effective batch = batch_size × grad_accum)')
    parser.add_argument('--max_length',   type=int,   default=512)
    parser.add_argument('--test_fraction', type=float, default=0.2,
                        help='Fraction held out as clean test set (never seen during training). '
                             'Saved to test_set.json for use in run_full_baseline_lora.py.')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of the remaining (post-test) data used for validation / early stopping')
    parser.add_argument('--seed',         type=int,   default=42)

    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()
