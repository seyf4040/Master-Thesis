# Two-Tier Architecture End-to-End Evaluation

**Status:** not started | **Priority:** high — core thesis deliverable

---

## Architecture

```
Incoming listing
      │
      ▼
┌─────────────────────────┐
│  Tier 1: detoxify-multi │  ~1 GB VRAM, ~6.6 ms/sample
│  threshold = T₁         │
└───────────┬─────────────┘
            │
     ┌──────┴──────┐
  safe (< T₁)   escalate (≥ T₁)
     │               │
     ▼               ▼
  pass         ┌─────────────────────────┐
               │  Tier 2: LG-3-1B-LoRA  │  ~3 GB VRAM, ~40 ms/sample
               │  threshold = T₂         │
               └───────────┬─────────────┘
                           │
                    ┌──────┴──────┐
                 safe           flag
```

## Key Questions

1. **What T₁ maximizes recall at Tier 1** (so Tier 2 sees all borderline cases)?
   - Want TPR ≈ 1.0 at Tier 1 — no unsafe content should be silently passed
   - FPR at Tier 1 does not matter much (false escalations are just reviewed by Tier 2)

2. **What fraction of samples reach Tier 2** at different T₁ values?
   - If T₁ is too low, almost everything escalates and Tier 2 becomes a bottleneck
   - Target: Tier 2 sees < 20% of traffic in production

3. **What is the combined F1, latency, and energy** of the two-tier system?
   - Compare against running Llama Guard 3 1B alone on every sample

4. **Does the LoRA fine-tuned Llama Guard improve the combined system** vs. the base model?

## Experiment Design

Does not require new inference code. Use Phase 1 result files (per-sample predictions)
to simulate the cascade:

```python
# Pseudocode
for each sample:
    detoxify_score = detoxify_result[sample_id]
    if detoxify_score < T1:
        final_pred = 0  # safe, Tier 1 filtered
    else:
        final_pred = llama_guard_result[sample_id]  # escalated to Tier 2
```

Sweep T₁ in [0.1, 0.5] and compute combined metrics. **Requires per-sample probabilities
to be saved** (see note 02 — Threshold Sensitivity Analysis).

## Why This Matters for the Thesis

The two-tier system is the proposed production architecture for Shareish. Evaluating it
end-to-end — with real numbers on latency, energy, and F1 — is the concrete deliverable
that connects the baseline evaluation (Phase 1) to a real deployment recommendation.

It also directly supports the deployability argument: the combined system processes
most traffic with a 1 GB model and only routes hard cases to the 3 GB model.
