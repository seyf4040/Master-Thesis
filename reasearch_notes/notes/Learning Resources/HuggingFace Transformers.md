# HuggingFace Transformers

**Date:** October 29, 2025  
**Context:** Shareish Content Moderation Thesis

---

```table-of-contents
title: 
style: nestedList # TOC style (nestedList|nestedOrderedList|inlineFirstLevel)
minLevel: 2 # Include headings from the specified level
maxLevel: 2 # Include headings up to the specified level
include: 
exclude: 
includeLinks: true # Make headings clickable
hideWhenEmpty: false # Hide TOC if no headings are found
debugInConsole: false # Print debug info in Obsidian console
```
## Licensing & Limits Quick Reference

### ✅ Key Facts

### Free
### **License**
- **Transformers Library:** Apache 2.0 (open-source, commercial use allowed)
- **Most Models:** Apache 2.0 or MIT (BERT, RoBERTa, DistilBERT, etc.)

---

### 🔒 Privacy & Deployment

#### **Local Deployment (`from_pretrained()`)**

```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# ✅ Runs 100% locally on your machine
# ✅ No data sent to external servers
# ✅ GDPR compliant
# ✅ Zero usage limits
# ✅ Zero cost per request
```

#### **What Happens:**

1. **First time:** Downloads model weights from Hugging Face Hub (one-time, free)
2. **Cached locally:** Stored in `~/.cache/huggingface/`
3. **All inference:** Runs on your hardware (no internet needed after download)

---

### 💰 Cost Breakdown

|Item|Cost|Notes|
|---|---|---|
|Library|**FREE**|Apache 2.0 license|
|Model weights|**FREE**|Download once, use forever|
|Inference|**FREE**|No per-request fees|
|Usage limits|**NONE**|Process unlimited data|
|Only cost|Hardware|Your GPU/CPU electricity (~$10-30/month)|

---

### ⚠️ Important Distinctions

#### **✅ Local Deployment (What You're Using)**

- Unlimited, free, private
- Model runs on your machine
- No API calls during inference

#### **❌ Hugging Face Inference API (Different Service)**

- Uses their servers (costs apply)
- Has rate limits
- **You're NOT using this**

---

### 📋 Recommended Models (All Free & Open)

|Model|License|Size|Use Case|
|---|---|---|---|
|**bert-base-uncased**|Apache 2.0|440MB|General text classification|
|**distilbert-base-uncased**|Apache 2.0|250MB|Faster, lighter BERT|
|**roberta-base**|MIT|500MB|Better performance than BERT|
|**xlm-roberta-base**|MIT|1GB|Multilingual support|

All suitable for content moderation, all completely free.

---

### 📝 For Your Thesis

#### **Can You Use It?**

✅ Academic research  
✅ Commercial deployment (Shareish platform)  
✅ Modify and distribute  
✅ No attribution required in outputs (but cite in thesis)

#### **Must Do:**

- Include license notice in code
- Cite properly in thesis (Wolf et al., 2020)
- Check individual model licenses

#### **Example Citation:**

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Wolf, Thomas and others",
    booktitle = "EMNLP 2020: System Demonstrations",
    year = "2020",
    pages = "38--45"
}
```

---

### 🔍 How to Verify License

```python
# Check model page on Hugging Face Hub
# https://huggingface.co/bert-base-uncased
# Look for "License:" field

# Or in code:
from transformers import AutoConfig
config = AutoConfig.from_pretrained("bert-base-uncased")
print(config._name_or_path)
```

---

### 🎯 Bottom Line
- ✅ **100% free** to use
- ✅ **No limits** on number of inferences
- ✅ **Fully private** (data stays local)
- ✅ **No hidden costs** or quotas
- ✅ **Academic integrity compliant**
- ✅ **Production ready**

**Only cost:** Your own hardware (GPU/CPU electricity)

---

## Model Storage & Fine-Tuning Quick Guide

### ✅ Can You Download & Store Locally?

**YES** - Multiple ways:

#### **Method 1: Automatic Caching (Easiest)**

```python
# First time: Downloads and caches automatically
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# Stored at: ~/.cache/huggingface/

# Next time: Loads from cache (no download)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

#### **Method 2: Explicit Local Storage (Recommended)**

```python
# Download once
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Save to your project
model.save_pretrained("./my_models/bert-base")
tokenizer.save_pretrained("./my_models/bert-base")

# Load later (offline-capable)
model = AutoModelForSequenceClassification.from_pretrained("./my_models/bert-base")
```

#### **Method 3: Offline Mode**

```python
# Force offline loading (no internet access)
model = AutoModelForSequenceClassification.from_pretrained(
    "./my_models/bert-base",
    local_files_only=True  # Ensures no internet calls
)
```

---

### Fine-Tuning Methods

#### **Method 1: Trainer API (Easiest - Recommended)**

```python
from transformers import Trainer, TrainingArguments

# 1. Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3  # safe, borderline, toxic
)

# 2. Prepare Shareish data
shareish_data = {
    "text": ["message1", "message2", ...],
    "label": [0, 2, ...]  # your labels
}

# 3. Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch"
)

# 4. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 5. Fine-tune!
trainer.train()

# 6. Save fine-tuned model
model.save_pretrained("./models/shareish-moderation-v1")
```

#### **Method 2: Manual Training Loop (Advanced)**

```python
# Full control over training process
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

#### **Method 3: Few-Shot Learning (Small Data)**

```python
# Use zero-shot for bootstrapping labels
classifier = pipeline("zero-shot-classification")
result = classifier(text, ["safe", "borderline", "toxic"])
```

---

### Complete Workflow for Shareish

#### **Phase 1: Download Base Model (One-Time)**

```python
# Download and save locally
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.save_pretrained("./models/bert-base")

# ✅ Now you can work offline forever
```

#### **Phase 2: Collect Shareish Data**

```python
# Format your data
data = {
    "text": ["message1", "message2", ...],
    "label": [0, 1, 2, ...]  # your moderation labels
}
```

#### **Phase 3: Fine-Tune on Shareish Data**

```python
# Load local model
model = AutoModelForSequenceClassification.from_pretrained("./models/bert-base")

# Fine-tune with your data
trainer = Trainer(model=model, ...)
trainer.train()

# Save fine-tuned version
model.save_pretrained("./models/shareish-finetuned-v1")
```

#### **Phase 4: Deploy for Inference**

```python
# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    "./models/shareish-finetuned-v1"
)

# Moderate new messages
inputs = tokenizer("New message", return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax()
```

---

### Key Benefits for Thesis

#### **Storage**

✅ Download once, use forever  
✅ No repeated downloads  
✅ Work completely offline  
✅ Version control friendly  
✅ Easy backup and sharing

#### **Fine-Tuning**

✅ Adapt to Shareish-specific content  
✅ Improve accuracy on your domain  
✅ Handle platform-specific language  
✅ Continuous improvement with feedback loop  
✅ Full control over training process

---

### Evaluation Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# After training, evaluate thoroughly
predictions = model.predict(test_dataset)
labels = test_dataset["label"]

# Classification report
print(classification_report(labels, predictions, 
                          target_names=["safe", "borderline", "toxic"]))

# Confusion matrix
print(confusion_matrix(labels, predictions))
```

---

### Important Notes

⚠️ **Always version your models**

```python
model.save_pretrained("./models/shareish-v1.0-2025-10-29")
```

⚠️ **Document training parameters**

```python
# Save training configuration
training_args.save_to_json("./models/shareish-v1/training_config.json")
```

⚠️ **Test before deployment**

```python
# Evaluate on held-out test set
# Check for bias and fairness
# Validate on edge cases
```

---

## Loading Models

### Overview

Hugging Face Transformers provides pretrained models that can be loaded and used with minimal code. The library handles model downloading, weight loading, and configuration automatically through the `from_pretrained()` method.

---

### 1. Core Concepts

#### 1.1 Architecture vs. Checkpoint

|Term|Definition|Example|
|---|---|---|
|**Architecture**|The model's skeleton/structure|BERT, Llama, Mistral|
|**Checkpoint**|Trained weights for a specific architecture|`google-bert/bert-base-uncased`|

> **Note:** The term "model" is often used interchangeably for both architecture and checkpoint.

#### 1.2 Model Files Structure

Every model contains:

- **`configuration.py`**: Model attributes (hidden layers, vocab size, activation functions, etc.)
- **`modeling.py`**: Layer definitions and mathematical operations
- **Weight files**: Preferably in `safetensors` format (more secure and faster than pickle)

---

### 2. Model Classes

#### 2.1 AutoClass (Recommended)

Automatically selects the correct model class based on configuration.

```python
from transformers import AutoModelForSequenceClassification

# Load model for text classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    torch_dtype="auto",
    device_map="auto"
)
```

**Advantages:**

- No need to know exact class names
- Easy switching between models and tasks
- Consistent API across different architectures

**Common AutoClass variants:**

- `AutoModel` - Base model outputting hidden states
- `AutoModelForSequenceClassification` - Text classification
- `AutoModelForCausalLM` - Language generation
- `AutoModelForQuestionAnswering` - Question answering
- `AutoModelForTokenClassification` - Named Entity Recognition (NER)

#### 2.2 Model-Specific Classes

Direct loading using architecture-specific classes.

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
```

**Use when:** You need architecture-specific functionality or optimizations.

---

### 3. Loading Models

#### 3.1 Basic Loading

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("model-name")
```

#### 3.2 From Local Directory

```python
model = AutoModelForSequenceClassification.from_pretrained("/path/to/local/model")
```

#### 3.3 Multiple Frameworks Support

Available for PyTorch, TensorFlow, and Flax:

```python
# PyTorch (default)
from transformers import AutoModelForSequenceClassification

# TensorFlow
from transformers import TFAutoModelForSequenceClassification

# Flax
from transformers import FlaxAutoModelForSequenceClassification
```

---

### 4. Memory Optimization for Large Models

#### 4.1 The Memory Challenge

Loading large models involves:

1. Creating model with random weights
2. Loading pretrained weights
3. Placing pretrained weights on model

This requires **2× model size** in memory temporarily.

#### 4.2 Sharded Checkpoints

**Automatic sharding:** Models >10GB are split into smaller files (default: 5GB per shard)

```python
from transformers import AutoModel

# Save with sharding
model = AutoModel.from_pretrained("biomistral/biomistral-7b")
model.save_pretrained("./my_model", max_shard_size="5GB")

# Load sharded model
new_model = AutoModel.from_pretrained("./my_model")
```

**Index file structure:**

- `metadata`: Total model size
- `weight_map`: Maps parameters to shard files

#### 4.3 Big Model Inference

**Requirements:** Accelerate v0.9.0+ and PyTorch v1.9.0+

**Features:**

1. Uses PyTorch `meta` device (only metadata, no actual data)
2. Weights loaded directly without duplication
3. Automatic device distribution (GPU → CPU → disk)

```python
from transformers import AutoModelForCausalLM

# Enable Big Model Inference
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b",
    device_map="auto"  # Automatic device placement
)
```

**Manual device mapping:**

```python
device_map = {
    "model.layers.1": 0,        # GPU 0
    "model.layers.14": 1,       # GPU 1
    "model.layers.31": "cpu",   # CPU
    "lm_head": "disk"           # Offload to disk
}

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    device_map=device_map
)
```

#### 4.4 Data Type Optimization

**Default:** Models load in `torch.float32` (32-bit precision)

**Optimization:** Use lower precision to reduce memory

```python
import torch
from transformers import AutoModelForCausalLM

# Explicit float16
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b",
    torch_dtype=torch.float16  # Half precision
)

# Automatic dtype
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b",
    torch_dtype="auto"  # Use stored dtype
)
```

**Memory savings:** ~50% reduction with float16 vs. float32

---

### 5. Custom Models

#### 5.1 Loading Custom Models

Models with custom code not in Transformers library.

```python
from transformers import AutoModelForImageClassification

# Requires explicit trust
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d",
    trust_remote_code=True
)
```

#### 5.2 Security Considerations

**⚠️ Warning:** Custom models can execute arbitrary code.

**Best practice:** Load from specific commit hash

```python
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"

model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d",
    trust_remote_code=True,
    revision=commit_hash  # Pin to specific version
)
```

**Hub safety:**

- All repositories undergo malware scanning
- Still exercise caution with trust_remote_code

---

### 6. Key Code Patterns

#### 6.1 Complete Loading Example

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,  # toxic, borderline, safe
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare input
text = "Example message to moderate"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

#### 6.2 Saving Fine-tuned Models

```python
# After fine-tuning
model.save_pretrained("./shareish-moderation-model")
tokenizer.save_pretrained("./shareish-moderation-model")

# Reload later
model = AutoModelForSequenceClassification.from_pretrained(
    "./shareish-moderation-model"
)
```

---

### 7. Important Considerations

#### 7.1 Security

- ✅ Use `safetensors` format (default)
- ✅ Pin versions with commit hashes
- ⚠️ Be cautious with `trust_remote_code=True`

#### 7.2 Performance

- Use `device_map="auto"` for large models
- Consider mixed precision (`torch.float16`)
- Monitor memory usage with `model.hf_device_map`

#### 7.3 Reproducibility

- Always specify model version/commit
- Document `torch_dtype` and device configuration
- Save configuration files alongside models

---

### 8. Next Steps for Thesis

1. **Explore task-specific models:**
    - Text classification for content categories
    - Token classification for sensitive entity detection
    - Zero-shot classification for flexible categories
2. **Study tokenization:**
    - Understand tokenizer preprocessing
    - Handle multilingual text properly
    - Manage special tokens and padding
3. **Learn fine-tuning:**
    - Trainer API for supervised learning
    - Few-shot learning approaches
    - Active learning integration
4. **Set up evaluation:**
    - Metrics for moderation (precision, recall, F1)
    - Confusion matrices for category analysis
    - Fairness and bias evaluation

---

### 10. Resources

- **Hugging Face Hub:** [https://huggingface.co/models](https://huggingface.co/models)
- **Transformers Documentation:** [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **Accelerate Library:** [https://huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate)
- **Hugging Face Fine-tuning Guide:** https://huggingface.co/docs/transformers/training
- **Trainer API Docs:** https://huggingface.co/docs/transformers/main_classes/trainer

---

**Last Updated:** October 29, 2025  
**License:** All Hugging Face Transformers code is Apache 2.0 licensed (open-source compliant)