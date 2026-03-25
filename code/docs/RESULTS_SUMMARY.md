# Results Summary

Last updated: **2026-03-25**

---

## Coverage

| Results dir | Models | Datasets | Notes |
|-------------|:------:|:--------:|-------|
| `full_baseline/` (flat) | 6 | 7 | Small models only; missing Llama-8B, ShieldGemma-9b, Mistral; missing reddit_fr |
| `full_baseline/run_1/` | 9 | 7 | All large models included; missing reddit_fr; Mistral/ShieldGemma missing reddit_en |
| `hatecheck_analysis/` | 7 | 2 | Missing ShieldGemma-9b, Mistral-7B, CitizenLab |

**CitizenLab**: no results in any run — crashes due to torch CVE-2025-32434 (needs `torch >= 2.6`).
**reddit_fr**: no results yet — not present in any completed run.

Expected when fully complete: **80 files** (10 models × 8 datasets) in `full_baseline/`.

---

## F1 Scores — `full_baseline/run_1/` (most complete dataset)

| Model | HC-EN | HC-FR | FR-Hate | ToxiGen | OpenAI | CivComm | Reddit-EN |
|-------|:-----:|:-----:|:-------:|:-------:|:------:|:-------:|:---------:|
| **Llama-Guard-3-8B** | **0.939** | **0.879** | 0.354 | 0.546 | **0.785** | 0.110 | 0.160 |
| Mistral-7B-Instruct-v0.3 | 0.922 | 0.781 | **0.390** | **0.667** | 0.762 | 0.293 | — |
| Llama-Guard-3-1B | 0.817 | 0.674 | 0.374 | 0.556 | 0.652 | 0.182 | **0.411** |
| **detoxify-multilingual** | 0.803 | 0.787 | 0.292 | 0.486 | 0.688 | **0.723** | 0.332 |
| EthicalEye | 0.725 | 0.593 | 0.291 | 0.562 | 0.665 | 0.488 | 0.407 |
| detoxify-unbiased | 0.760 | 0.281 | 0.072 | 0.463 | 0.672 | **0.763** | 0.318 |
| KoalaAI-Text-Moderation | 0.548 | 0.002 | 0.006 | 0.396 | **0.838** | 0.207 | 0.179 |
| ShieldGemma-2b | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | — |
| ShieldGemma-9b | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | — |
| CitizenLab | — | — | — | — | — | — | — |

---

## Detailed Metrics — `full_baseline/run_1/`

### HateCheck EN (n=3728)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| Llama-Guard-3-8B | 0.919 | 0.963 | 0.917 | 0.939 | 0.917 | 0.078 |
| Mistral-7B-Instruct-v0.3 | 0.889 | 0.885 | 0.963 | 0.922 | 0.963 | 0.275 |
| Llama-Guard-3-1B | 0.760 | 0.859 | 0.778 | 0.817 | 0.778 | 0.281 |
| detoxify-multilingual | 0.714 | 0.764 | 0.847 | 0.803 | 0.847 | 0.577 |
| detoxify-unbiased | 0.668 | 0.754 | 0.766 | 0.760 | 0.766 | 0.549 |
| EthicalEye | 0.612 | 0.707 | 0.744 | 0.725 | 0.744 | 0.680 |
| KoalaAI-Text-Moderation | 0.542 | 0.849 | 0.405 | 0.548 | 0.405 | 0.158 |
| ShieldGemma-2b | 0.312 | — | — | ~0.000 | ~0.000 | ~0.000 |
| ShieldGemma-9b | 0.313 | — | — | ~0.000 | ~0.000 | ~0.000 |

### HateCheck FR (n=3718)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| Llama-Guard-3-8B | 0.840 | 0.937 | 0.827 | 0.879 | 0.827 | 0.129 |
| detoxify-multilingual | 0.702 | 0.787 | 0.787 | 0.787 | 0.787 | 0.495 |
| Mistral-7B-Instruct-v0.3 | 0.724 | 0.876 | 0.705 | 0.781 | 0.705 | 0.233 |
| Llama-Guard-3-1B | 0.615 | 0.827 | 0.568 | 0.674 | 0.568 | 0.276 |
| EthicalEye | 0.533 | 0.760 | 0.486 | 0.593 | 0.486 | 0.357 |
| detoxify-unbiased | 0.389 | 0.794 | 0.171 | 0.281 | 0.171 | 0.103 |
| KoalaAI-Text-Moderation | 0.302 | — | 0.001 | 0.002 | 0.001 | 0.000 |
| ShieldGemma-2b | 0.301 | — | — | ~0.000 | ~0.000 | ~0.000 |
| ShieldGemma-9b | 0.301 | — | — | ~0.000 | ~0.000 | ~0.000 |

### French Hate Superset (n=18071)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| Mistral-7B-Instruct-v0.3 | 0.737 | 0.441 | 0.350 | 0.390 | 0.350 | 0.140 |
| Llama-Guard-3-1B | 0.569 | 0.287 | 0.536 | 0.374 | 0.536 | 0.420 |
| Llama-Guard-3-8B | 0.708 | 0.378 | 0.333 | 0.354 | 0.333 | 0.174 |
| detoxify-multilingual | 0.711 | 0.355 | 0.248 | 0.292 | 0.248 | 0.143 |
| EthicalEye | 0.729 | 0.391 | 0.231 | 0.291 | 0.231 | 0.114 |
| detoxify-unbiased | 0.757 | 0.430 | 0.039 | 0.072 | 0.039 | 0.016 |
| KoalaAI-Text-Moderation | 0.760 | 0.619 | 0.003 | 0.006 | 0.003 | 0.001 |
| ShieldGemma-2b | 0.760 | — | 0.001 | ~0.000 | ~0.000 | ~0.000 |
| ShieldGemma-9b | 0.760 | — | — | ~0.000 | ~0.000 | ~0.000 |

### ToxiGen (n=5000)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| Mistral-7B-Instruct-v0.3 | 0.737 | 0.872 | 0.540 | 0.667 | 0.540 | 0.075 |
| EthicalEye | 0.666 | 0.780 | 0.439 | 0.562 | 0.439 | 0.118 |
| Llama-Guard-3-1B | 0.646 | 0.717 | 0.454 | 0.556 | 0.454 | 0.171 |
| Llama-Guard-3-8B | 0.683 | 0.906 | 0.390 | 0.546 | 0.390 | 0.039 |
| detoxify-multilingual | 0.653 | 0.874 | 0.337 | 0.486 | 0.337 | 0.046 |
| detoxify-unbiased | 0.642 | 0.865 | 0.316 | 0.463 | 0.316 | 0.047 |
| KoalaAI-Text-Moderation | 0.619 | 0.876 | 0.256 | 0.396 | 0.256 | 0.034 |
| ShieldGemma-2b | 0.512 | — | 0.001 | ~0.000 | ~0.000 | ~0.000 |
| ShieldGemma-9b | 0.511 | — | — | ~0.000 | ~0.000 | ~0.000 |

### OpenAI Moderation (n=1680)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| KoalaAI-Text-Moderation | 0.910 | 0.945 | 0.753 | 0.838 | 0.753 | 0.020 |
| Llama-Guard-3-8B | 0.867 | 0.789 | 0.782 | 0.785 | 0.782 | 0.094 |
| Mistral-7B-Instruct-v0.3 | 0.842 | 0.715 | 0.814 | 0.762 | 0.814 | 0.146 |
| detoxify-multilingual | 0.820 | 0.744 | 0.640 | 0.688 | 0.640 | 0.099 |
| detoxify-unbiased | 0.816 | 0.753 | 0.607 | 0.672 | 0.607 | 0.090 |
| EthicalEye | 0.772 | 0.612 | 0.728 | 0.665 | 0.728 | 0.208 |
| Llama-Guard-3-1B | 0.733 | 0.547 | 0.807 | 0.652 | 0.807 | 0.301 |
| ShieldGemma-2b | 0.689 | — | — | ~0.000 | ~0.000 | ~0.000 |
| ShieldGemma-9b | 0.689 | — | — | ~0.000 | ~0.000 | ~0.000 |

### Civil Comments (n=5000)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| detoxify-unbiased | 0.961 | 0.730 | 0.798 | 0.763 | 0.798 | 0.025 |
| detoxify-multilingual | 0.949 | 0.636 | 0.836 | 0.723 | 0.836 | 0.041 |
| EthicalEye | 0.882 | 0.372 | 0.708 | 0.488 | 0.708 | 0.103 |
| Mistral-7B-Instruct-v0.3 | 0.852 | 0.236 | 0.385 | 0.293 | 0.385 | 0.107 |
| KoalaAI-Text-Moderation | 0.914 | 0.386 | 0.141 | 0.207 | 0.141 | 0.019 |
| Llama-Guard-3-1B | 0.677 | 0.114 | 0.451 | 0.182 | 0.451 | 0.303 |
| Llama-Guard-3-8B | 0.907 | 0.227 | 0.073 | 0.110 | 0.073 | 0.022 |
| ShieldGemma-2b | 0.920 | — | — | ~0.000 | ~0.000 | ~0.000 |
| ShieldGemma-9b | 0.920 | — | — | ~0.000 | ~0.000 | ~0.000 |

### Reddit EN (n=56462)

| Model | Acc | Prec | Rec | F1 | TPR | FPR |
|-------|:---:|:----:|:---:|:--:|:---:|:---:|
| Llama-Guard-3-1B | 0.551 | 0.557 | 0.326 | 0.411 | 0.326 | 0.240 |
| EthicalEye | 0.577 | 0.625 | 0.302 | 0.407 | 0.302 | 0.168 |
| detoxify-multilingual | 0.568 | 0.649 | 0.223 | 0.332 | 0.223 | 0.112 |
| detoxify-unbiased | 0.564 | 0.643 | 0.211 | 0.318 | 0.211 | 0.109 |
| KoalaAI-Text-Moderation | 0.546 | 0.686 | 0.103 | 0.179 | 0.103 | 0.043 |
| Llama-Guard-3-8B | 0.543 | 0.684 | 0.091 | 0.160 | 0.091 | 0.039 |
| Mistral-7B | — | — | — | — | — | — |
| ShieldGemma-2b | — | — | — | — | — | — |
| ShieldGemma-9b | — | — | — | — | — | — |

---

## HateCheck Analysis — `hatecheck_analysis/`

| Model | EN F1 | EN TPR | EN TNR | FR F1 | FR TPR | FR TNR |
|-------|:-----:|:------:|:------:|:-----:|:------:|:------:|
| Llama-Guard-3-8B | 0.939 | 0.917 | 0.922 | 0.879 | 0.827 | 0.871 |
| Llama-Guard-3-1B | 0.819 | 0.785 | 0.712 | 0.692 | 0.588 | 0.742 |
| detoxify-multilingual | 0.803 | 0.847 | 0.423 | 0.787 | 0.787 | 0.505 |
| detoxify-unbiased | 0.760 | 0.766 | 0.451 | 0.281 | 0.171 | 0.897 |
| EthicalEye | 0.725 | 0.744 | 0.320 | 0.593 | 0.486 | 0.643 |
| KoalaAI-Text-Moderation | 0.548 | 0.405 | 0.842 | 0.002 | 0.001 | 1.000 |
| ShieldGemma-2b | ~0.000 | ~0.000 | ~1.000 | ~0.000 | ~0.000 | ~1.000 |

---

## Deployability — `full_baseline/run_1/` (avg across available datasets)

| Model | GPU MB | ms/sample | kWh/dataset | Viable for Shareish? |
|-------|:------:|:---------:|:-----------:|:--------------------:|
| detoxify-unbiased | 496 | 10 | 0.006 | ✅ Yes |
| KoalaAI-Text-Moderation | 598 | 22 | 0.012 | ✅ Yes (EN only) |
| EthicalEye | 1078 | 9 | 0.005 | ✅ Yes |
| detoxify-multilingual | 1078 | 10 | 0.005 | ✅ Yes |
| Llama-Guard-3-1B | 2955 | 51 | 0.028 | ⚠️ Marginal (needs 4GB) |
| ShieldGemma-2b | 5047 | 1944 | 0.805 | ❌ No (too slow) |
| Mistral-7B-Instruct-v0.3 | 13909 | 205 | 0.123 | ❌ No (13GB VRAM) |
| Llama-Guard-3-8B | 15547 | 128 | 0.142 | ❌ No (15GB VRAM) |
| ShieldGemma-9b | 17766 | 3169 | 1.873 | ❌ No (18GB + very slow) |

---

## Key Observations

1. **Best accuracy overall**: Llama-Guard-3-8B (F1=0.939 HC-EN, 0.879 HC-FR) but requires ~15GB VRAM — not viable for Shareish. Fails badly on Civil Comments (0.110) and Reddit-EN (0.160).

2. **Best deployable model**: `detoxify-multilingual` — consistent F1 ~0.72–0.80 on both EN/FR hate speech, only 1GB VRAM, 10ms/sample. Clear candidate for pre-filter tier.

3. **Mistral-7B is the best large model for French**: F1=0.390 on FR-Hate superset (best of all models) and 0.781 HC-FR. Better than Llama-Guard-3-8B on noisy real-world French data, but 13GB VRAM rules it out for direct deployment.

4. **ShieldGemma is definitively broken**: Both 2b and 9b produce F1≈0 across **all 7 datasets**. Always predicts "safe". This is confirmed across civil_comments, FR-hate, toxigen, openai, HC-EN, HC-FR. Not a threshold issue — the prompt elicits no meaningful discrimination.

5. **KoalaAI is English-only**: F1=0.838 on OpenAI (best of all models), but 0.002 HC-FR and 0.006 FR-Hate. Completely unusable for French content moderation.

6. **detoxify-unbiased degrades severely in French**: F1=0.760 EN vs 0.281 FR on HateCheck. detoxify-multilingual is the correct choice for bilingual deployment.

7. **Reddit-EN is the hardest dataset**: All models struggle — best is Llama-Guard-3-1B at F1=0.411. Large models (Llama-8B=0.160, Mistral=missing) underperform here, suggesting domain mismatch with the Shareish safety prompt.

8. **EthicalEye is the most consistent**: Mid-range F1 (0.407–0.725) across all datasets, never collapsing. Best single-model generalist within deployable VRAM constraints.

9. **Civil Comments favours Detoxify**: detoxify-unbiased (0.763) and detoxify-multilingual (0.723) dominate. Instruction-tuned LLMs (Llama-Guard-3-8B=0.110, Mistral=0.293) perform poorly — likely because Civil Comments contains subtle/mild toxicity that doesn't match the Shareish policy prompt categories.

10. **CitizenLab**: Still no results — torch CVE-2025-32434 blocker. Needs `torch >= 2.6`.

---

## Missing Results

| What | Why | Action |
|------|-----|--------|
| CitizenLab (all datasets) | torch < 2.6 CVE crash | Upgrade torch on cluster |
| reddit_fr (all models) | Not in any completed run | Resubmit with `--datasets reddit_fr` |
| Mistral-7B + ShieldGemma on reddit_en | Possibly hit time limit or VRAM guard | Check logs, resubmit |
