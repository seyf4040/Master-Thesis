# Active Learning with Real Shareish Data (Phase 3)

**Status:** not started | **Priority:** high — planned thesis phase

---

## Motivation

All Phase 1 and Phase 2 experiments use public benchmark datasets. These are useful for
comparison but do not reflect the real Shareish distribution:
- Real listings are short, informal, French-dominant, often mixed French/English
- The content taxonomy differs (sharing economy context vs. general hate speech)
- The label noise profile is different (moderator judgments vs. crowdsourced labels)

Active learning minimizes the number of labeled Shareish examples needed to reach
good performance on real platform content.

## Process

```
1. Seed set: ~100 labeled real Shareish listings (manually annotated)
      │
      ▼
2. Fine-tune model on seed set (LoRA adapter)
      │
      ▼
3. Run inference on unlabeled Shareish listings
      │
      ▼
4. Select most uncertain examples (lowest confidence margin)
   = examples where P(unsafe) is closest to threshold T
      │
      ▼
5. Human labeling of selected batch (~50–100 examples)
      │
      ▼
6. Add to training set → go to step 2
      │
   Repeat until performance plateaus
```

## Key Research Question

**How many labeled Shareish examples are needed** before the active-learning-fine-tuned
model outperforms all Phase 1 baselines on Shareish content?

This is the central empirical question of Phase 3. Hypothesis: fewer than 500, because
the model already has strong linguistic priors from pre-training and Phase 2 fine-tuning.

## Uncertainty Sampling

For Llama Guard (generative), uncertainty = entropy over the `safe`/`unsafe` first-token
probabilities. For ShieldGemma (token-probability), uncertainty = `|P(Yes) - 0.5|`.
Select examples with lowest uncertainty score (closest to the decision boundary).

## Practical Requirements

- Access to the Shareish database (even a static export)
- A labeling interface (could be as simple as a spreadsheet or a basic Streamlit app)
- Coordination with Shareish team for annotation

## Connection to Phase 2

The LoRA fine-tuning pipeline from Phase 2 is reused directly. The only new component is
the uncertainty sampling step between training iterations.
