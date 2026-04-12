# Minor Technical Improvements

**Status:** not started | **Priority:** low — quality of life, not blocking

---

## Save per-sample probabilities

Currently the evaluation loop saves only binary predictions in result JSON files.
Raw probability scores are discarded after inference.

**Impact:** threshold sensitivity analysis (note 02) and two-tier simulation (note 05)
both require per-sample scores. Without them, inference must be re-run from scratch.

**Fix:** add a `scores: List[float]` field to the `Result` dataclass in
`run_full_baseline_v3.py` and `run_full_baseline_lora.py`, and append the raw score
(e.g. `detoxify['toxicity']`, `unsafe_prob` for ShieldGemma, etc.) for each sample.
Backward-compatible: old result files simply lack the field.

## Batch inference for ShieldGemma

ShieldGemma currently processes one sample at a time (single forward pass per sample).
Batching would reduce wall time significantly on large datasets:

```python
# Current: loop over single samples
for s in samples:
    inputs = tokenizer(prompt, return_tensors="pt", ...)
    logits = model(**inputs).logits[0, -1, :]

# Improved: batch of N samples padded to same length
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, ...)
logits = model(**inputs).logits[:, -1, :]  # last position per sequence
```

Note: padding changes the last-position index — need to use the actual last non-padded
token per sequence (via `attention_mask`). Worth doing before any large re-runs.

## Complete the 3rd statistical run

`run_2` in the multi-run baseline is missing Reddit-FR results for:
- Llama-Guard-3-8B
- ShieldGemma-9b
- Mistral-7B

These three jobs likely timed out or were interrupted. Running them completes the 3-run
grid needed to compute mean ± std for all model × dataset pairs in the thesis tables.

```bash
python ~/code/run_full_baseline_v3.py \
    --output_dir ~/code/results/full_baseline_v3 \
    --datasets reddit_fr \
    --models llama_guard_8b,shieldgemma_9b,mistral_7b \
    --run_id 2
```
