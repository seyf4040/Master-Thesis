# Phase 2 — LoRA Fine-tuning Improvements

**Status:** in progress | **Priority:** high

---

## Hyperparameter sweep

The defaults (`r=16`, `alpha=32`, `lr=2e-4`, `epochs=3`) are reasonable but not tuned.

| Param | Current | Try |
|---|---|---|
| `lora_r` | 16 | 4, 8, 32 — smaller r may generalize better on small datasets like Reddit-FR |
| `lr` | 2e-4 | 1e-4, 5e-4 — lower lr may be more stable on the large French Hate Superset |
| `epochs` | 3 | 5 on Reddit-FR — small dataset, model may need more passes |

## Multi-dataset joint adapter

Current setup trains one adapter per dataset. A joint adapter trained on
`french_hate_superset + reddit_fr + hatecheck_fr` might generalize better across all
French datasets instead of specializing on one distribution.

Requires a small change to `finetune_lora.py`: accept a comma-separated `--dataset` list
and concatenate the loaded samples before the train/val split.

## Civil Comments — domain-specific adapter

Civil Comments is the biggest outlier: detoxify dominates (F1 ≈ 0.72) while all large
models fail (<0.32). The mismatch is likely the label definition (toxicity ≥ 0.5 on a
continuous score rather than a binary judgment). An adapter fine-tuned on the Civil
Comments train split would test whether the models can learn this label scheme.

## QLoRA (4-bit quantization + LoRA)

QLoRA (Dettmers et al. 2023) loads the base model in 4-bit NF4, reducing training VRAM
by ~2×. Benefits:
- Fine-tune Llama Guard 3 1B on a **1080Ti (11 GB)** instead of requiring an a5000
- Potentially fine-tune Llama Guard 3 8B within 24 GB

Implementation: add `BitsAndBytesConfig(load_in_4bit=True)` to `finetune_lora.py`.
Requires `bitsandbytes` package — already a common dependency in the HuggingFace ecosystem.

## References
- `code/finetune_lora.py` — current implementation
- `code/slurm_jobs/finetune_lora.sbatch` — SLURM job
- Dettmers et al. (2023) QLoRA: arXiv:2305.14314
