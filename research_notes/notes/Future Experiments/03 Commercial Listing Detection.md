# Commercial Listing Detection (Shareish-Specific Task)

**Status:** not started | **Priority:** medium — high real-world impact for Shareish

---

## Task Definition

Flag listings on Shareish that propose to sell items or include monetary compensation.
This is against the platform's solidarity principles.

```
Label 1 (flag):  "Vends vélo 50€, bon état"
                 "Proposition: cours de guitare, 20€/h"
                 "Looking for a babysitter, will pay"

Label 0 (ok):    "Je donne des tomates de mon jardin"
                 "Cherche quelqu'un pour m'aider à déménager"
                 "Proposons des cours de français bénévolement"
```

## Data Collection Options

No public dataset exists for this task. Three realistic sources:

### Option 1 — Distant supervision from Shareish DB (recommended first step)
Use the `price` field in the Shareish database:
- `price > 0` → label = 1 (commercial)
- `price IS NULL` or `price = 0` → label = 0 (solidarity)

Zero manual annotation needed if the field is reliably filled. Could yield thousands of
labeled examples for free. **Start here.**

### Option 2 — Manual annotation of a DB export
Export listings and manually label a sample. Highest quality, most effort.
Even 500 examples is likely sufficient given the surface-level nature of the signal.

### Option 3 — Synthetic data generation
Prompt Claude/GPT-4 to generate French listing texts (commercial and solidarity).
Fast but risks distribution shift — synthetic listings may not match real platform noise.

## Recommended Approach

**Before any fine-tuning, test a regex baseline:**

```python
import re
COMMERCIAL_RE = re.compile(
    r'(\d+\s*€|\d+\s*euro|\bvend[s]?\b|à vendre|\bprix\b|\btarif\b'
    r'|\brémunér|\bpay[eé]\b|\bcompensation\b)',
    re.IGNORECASE
)
```

Evaluate on a labeled sample. If F1 > 0.90, a regex is the right tool — no model needed.
If it misses paraphrases ("je cède contre une petite participation"), proceed to fine-tuning.

## Model Recommendation

| Option | Pros | Cons |
|---|---|---|
| Regex | Zero cost, instant | Brittle, misses paraphrases |
| CamemBERT classifier | Native French, 111M params, 450MB, < 1ms/sample | Adds a different model family to the pipeline |
| Llama Guard 3 1B + LoRA | Consistent with Phase 2 pipeline | Overkill for a surface-level pattern |

**CamemBERT (`camembert-base`) is the best fit** if fine-tuning is needed:
- Designed as a classifier (not a generative model)
- Pre-trained specifically on French text
- Much lighter and faster than Llama Guard at inference
- Easily fine-tuned with a standard sequence classification head

## If Using the Existing LoRA Pipeline

Add to `finetune_lora.py`:

```python
def load_shareish_listings(path: str, max_samples=None):
    import pandas as pd
    df = pd.read_csv(path)  # expects columns: 'text', 'label'
    samples = [
        {'text': str(row['text']), 'label': int(row['label'])}
        for _, row in df.iterrows()
        if str(row.get('text', '')).strip()
    ]
    return samples[:max_samples] if max_samples else samples
```

And a new prompt template that redefines "unsafe" in the Shareish context — the label
tokens (`safe`/`unsafe`) and the inference loop stay unchanged.
