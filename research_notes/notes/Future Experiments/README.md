# Future Experiments & Areas for Improvement

**Last updated:** 2026-03-30

Ideas and open tasks that go beyond the current thesis scope, grouped by priority.
Each file has its own status, context, and implementation notes.

---

## High Priority

| # | Topic | Status | Why It Matters |
|---|-------|--------|----------------|
| [01](./01%20Phase%202%20LoRA%20Improvements.md) | Phase 2 LoRA improvements | in progress | Hyperparameter sweep, joint adapter, QLoRA for 1080Ti |
| [02](./02%20Threshold%20Sensitivity%20Analysis.md) | Threshold sensitivity analysis | not started | Fixed 0.5 threshold is a simplification — needed before thesis conclusions |
| [05](./05%20Two-Tier%20Architecture%20Evaluation.md) | Two-tier architecture end-to-end eval | not started | Core thesis deliverable — connects baseline to a real deployment recommendation |
| [06](./06%20Active%20Learning%20Phase%203.md) | Active learning with Shareish data | not started | Planned Phase 3 — requires DB access |

## Medium Priority

| # | Topic | Status | Why It Matters |
|---|-------|--------|----------------|
| [03](./03%20Commercial%20Listing%20Detection.md) | Commercial listing detection | not started | Shareish-specific moderation axis — high real-world impact |
| [04](./04%20HateCheck%20Analysis%20Gaps.md) | HateCheck analysis gaps | not started | ShieldGemma and CitizenLab missing from v3 per-functionality breakdown |

## Low Priority

| # | Topic | Status | Why It Matters |
|---|-------|--------|----------------|
| [07](./07%20Minor%20Technical%20Improvements.md) | Minor technical improvements | not started | Per-sample scores, ShieldGemma batching, complete 3rd statistical run |

---

## Quick Notes

- **Immediate blockers**: notes 02 and 05 both require per-sample probabilities to be
  saved during inference — currently they are not. Fix this before the next evaluation run.
- **Easiest win**: note 07 (complete 3rd statistical run for Reddit-FR) is a single
  `sbatch` command.
- **Highest thesis impact**: notes 05 (two-tier eval) and 02 (threshold analysis) are the
  most directly tied to the deployability argument.
- **Requires external access**: notes 03 and 06 depend on the Shareish database — plan
  this with the Shareish team.
