# LoRA Fine-tuning — Detailed Explanation

> Context: used in Phase 2 of the thesis to fine-tune Llama Guard 3 1B and ShieldGemma 2B
> on French hate speech data (French Hate Superset, Reddit-FR).

---

## 1. The Problem LoRA Solves

Full fine-tuning means updating every parameter of a pre-trained model. For a 1B-parameter
model in bfloat16, that is ~2 GB just to store the weights — but during training you also
need gradients (another ~2 GB) and optimizer states (Adam keeps a running mean and variance
per parameter, so another ~8 GB). Total: ~12 GB for a 1B model. For 7B+ models this
becomes prohibitive on a single GPU.

LoRA (Low-Rank Adaptation, Hu et al. 2021) trains a tiny number of additional parameters
instead of touching the original weights at all.

---

## 2. The Core Idea: Low-Rank Decomposition

Every weight matrix W in a transformer layer can theoretically change during training by
some delta ΔW:

```
W_new = W_original + ΔW
```

LoRA's insight: **the weight updates that actually matter for a downstream task live in a
low-dimensional subspace**. Rather than storing a full ΔW (which has the same shape as W),
you approximate it as a product of two small matrices:

```
ΔW = B × A
```

Where:
- W has shape (d_out, d_in)  — e.g. (4096, 4096) for a large attention projection
- A has shape (r, d_in)      — a small "down-projection", initialized randomly
- B has shape (d_out, r)     — a small "up-projection", initialized to zero
- r is the **rank** — a hyperparameter, typically 4–64

At the start of training B=0, so ΔW=0 and the model behaves exactly like the original.
As training progresses, A and B learn the task-specific update.

The parameter count comparison:
- Full ΔW: d_out × d_in = 4096 × 4096 = **16.7M** parameters
- LoRA (r=16): r × d_in + d_out × r = 16×4096 + 4096×16 = **131K** parameters — 128× fewer

---

## 3. The Forward Pass with LoRA

During inference (and training), the adapted layer computes:

```
output = x @ W^T + x @ A^T @ B^T × (alpha / r)
```

Or equivalently (merged form):

```
output = x @ (W + B×A)^T × (alpha / r)
```

The `alpha / r` scaling factor is important:
- `alpha` (lora_alpha) is a fixed hyperparameter, typically 2× the rank
- Scaling by `alpha/r` means that increasing r does not automatically increase the
  magnitude of the LoRA update — the two hyperparameters decouple rank (expressivity)
  from step size (learning rate sensitivity)
- In practice, keeping `alpha = 2 × r` is a safe default

**The original W is frozen throughout training.** Only A and B receive gradients.

---

## 4. Which Matrices to Target

LoRA can be applied to any linear layer. In transformer models the candidates are:

| Module | Description | Size (7B model) |
|---|---|---|
| `q_proj` | Query projection in attention | (4096, 4096) |
| `k_proj` | Key projection in attention | (4096, 1024) |
| `v_proj` | Value projection in attention | (4096, 1024) |
| `o_proj` | Output projection in attention | (4096, 4096) |
| `gate_proj` | Gate in SwiGLU MLP | (14336, 4096) |
| `up_proj` | Up-projection in SwiGLU MLP | (14336, 4096) |
| `down_proj` | Down-projection in SwiGLU MLP | (4096, 14336) |

The original LoRA paper targeted only q and v. Later work (QLoRA, full LoRA) found that
including all 7 modules improves performance significantly at still a small total param
count. This thesis uses all 7 target modules.

For a 1B Llama Guard model with r=16, the total trainable parameters are roughly:

```
16 layers × 7 modules × 2 × r × avg_dim ≈ 16 × 7 × 2 × 16 × ~2000 ≈ 7M parameters
```

vs. ~1000M total — about 0.7% of the model trained.

---

## 5. Training Format for Causal LMs

Llama Guard and ShieldGemma are causal language models. They do not have a classification
head — they generate text autoregressively. To fine-tune them for binary classification,
we frame the task as next-token prediction:

```
Input:   [prompt describing the text + task]
Target:  "safe" or "unsafe"   (Llama Guard)
         "No"   or "Yes"      (ShieldGemma)
```

The prompt is identical to the one used during Phase 1 inference (so the fine-tuned model
is evaluated the same way). The model learns to assign high probability to the correct
label token given the prompt.

### 5.1 Label-Only Loss Masking

For causal LM training, the standard cross-entropy loss is computed over every output
token. But we do not want the model to "learn" to reproduce the prompt — we only want it
to learn the label token. So we mask the prompt tokens from the loss:

```python
input_ids = [prompt_token_1, prompt_token_2, ..., label_token, EOS]
labels    = [-100,           -100,           ..., label_token, EOS]
```

`-100` is PyTorch's ignore index: positions where labels=-100 contribute zero to the loss.
The gradient flows only through the label token(s).

This means the model is trained on a very short effective sequence — usually 1–2 tokens
of supervision per example — but this is sufficient because the prompt provides all the
context.

### 5.2 Why Not Classification Fine-tuning?

An alternative would be to add a linear classification head on top of the LM and fine-tune
that. This is common for BERT-style models. For causal LMs it is less natural because:

1. The model was pre-trained to generate text, not to produce a hidden-state classification
2. The existing inference code (Phase 1) already uses text generation / token probabilities
3. Keeping the same inference format means Phase 1 and Phase 2 results are directly comparable

---

## 6. Hyperparameters and Their Effect

| Hyperparameter | Default (this thesis) | Effect |
|---|---|---|
| `lora_r` | 16 | Rank — higher = more expressive but more parameters and risk of overfitting |
| `lora_alpha` | 32 | Scaling (= 2×r). Controls effective learning rate of LoRA weights |
| `lora_dropout` | 0.05 | Dropout applied to LoRA layers. Mild regularization |
| `lr` | 2e-4 | Higher than typical full fine-tuning because LoRA weights start at ~0 |
| `epochs` | 3 | Small datasets (FHS, Reddit-FR) risk overfitting beyond 3–5 epochs |
| `batch_size` | 4 | Per-GPU batch |
| `grad_accum` | 4 | Effective batch = 4×4 = 16. Stabilizes training without memory cost |
| `max_length` | 512 | Prompt + label truncated to 512 tokens |
| `val_fraction` | 0.1 | 10% held out. Monitors overfitting, selects best checkpoint |

**On rank choice**: for a small fine-tuning dataset like French Hate Superset (~tens of
thousands of examples), r=16 is a reasonable middle ground. r=4 might underfit; r=64
would overfit unless you have 100k+ examples. The rule of thumb is roughly r ≈ log2(n_samples).

**On learning rate**: LoRA typically uses lr=1e-4 to 3e-4, much higher than full
fine-tuning (1e-5 to 5e-5). The LoRA weights start near zero and need a larger step to
move into a useful region quickly.

---

## 7. Gradient Accumulation

The GPU computes gradients for one mini-batch, but updates the optimizer only every
`grad_accum` steps. This simulates a larger effective batch without the memory cost:

```
effective_batch = batch_size × grad_accum = 4 × 4 = 16
```

With batch=16 and sequences of length 512 in bfloat16, the activation memory would be
large. Gradient accumulation splits this into 4 micro-batches of 4, each processed
sequentially, then their gradients are summed before the weight update.

The training loss and the gradient norm behave as if batch=16 was used, but only 4
examples are in GPU memory at once.

---

## 8. Checkpointing Strategy

Two checkpoints are saved:

- **`best/`**: saved whenever val_loss improves. Used for evaluation. Guards against
  overfitting — if the model starts memorizing training examples, val_loss rises and the
  best checkpoint remains from the epoch before that happened.
- **`final/`**: saved at the end of training regardless. Useful for comparing against
  best/ to diagnose overfitting.

The `training_meta.json` file records the full hyperparameter set, per-epoch loss, and
best val loss. This is embedded in the `summary.json` of the evaluation run for
reproducibility.

---

## 9. Adapter Storage

After training, only A and B matrices are saved (not the full model):

```
lora_adapters/
  llama_guard_1b/
    french_hate_superset/
      best/
        adapter_config.json    ← LoRA config (r, alpha, target_modules, …)
        adapter_model.safetensors  ← A and B matrices only (~30MB for r=16)
      final/
        adapter_config.json
        adapter_model.safetensors
      training_meta.json       ← hyperparams + loss curves
    reddit_fr/
      best/ ...
  shieldgemma_2b/
    french_hate_superset/ ...
    reddit_fr/ ...
```

At inference time (run_full_baseline_lora.py), the HuggingFace PEFT library:
1. Loads the full base model (frozen)
2. Loads the adapter weights from the directory
3. Injects the ΔW = B×A into the model transparently
4. The resulting model behaves like the fine-tuned version

The base model weights are never modified and never saved again — the adapter is the only
artifact.

---

## 10. Relation to the Thesis Argument

LoRA is directly aligned with the deployability thesis argument:

- **VRAM during inference**: same as the base model. The adapter adds ~0 overhead at
  inference time (the ΔW is merged into W once at load time).
- **VRAM during training**: the base model is frozen, so no gradients for 99%+ of
  parameters. Training a 1B model with LoRA requires ~4–5 GB vs ~12 GB for full fine-tuning.
  This means Llama Guard 3 1B **can be fine-tuned on a 1080Ti (11 GB)** with LoRA.
- **Adapter size on disk**: ~30 MB per adapter (r=16) vs ~2 GB for a full model copy.
  Multiple fine-tuned variants can be stored cheaply.
- **Swap cost**: a Shareish operator could maintain one base model and multiple adapters
  (e.g., one per language or domain), swapping adapters at runtime with near-zero overhead.

---

## 11. References

- Hu, E. et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**.
  arXiv:2106.09685
- Dettmers, T. et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**.
  arXiv:2305.14314 (combines LoRA with 4-bit quantization — relevant for 1080Ti use)
- HuggingFace PEFT documentation: https://huggingface.co/docs/peft
- `code/finetune_lora.py` — implementation used in this thesis
