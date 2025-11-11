# LLM Models for Content Moderation - Comprehensive Inventory

**Compiled for:** Deep Learning for Content Moderation on the Shareish Solidarity Platform  
**Date:** January 2025  
**Purpose:** Complete reference of LLM models that can be run locally for self-hosted content moderation

---

- [[#Model Comparison Table|Model Comparison Table]]
- [[#Open-Source LLM Models - Production Ready|Open-Source LLM Models - Production Ready]]
	- [[#Open-Source LLM Models - Production Ready#1. Llama Guard 3 (Meta AI) ⭐⭐⭐|1. Llama Guard 3 (Meta AI) ⭐⭐⭐]]
	- [[#Open-Source LLM Models - Production Ready#2. ShieldGemma (Google DeepMind) ⭐⭐⭐|2. ShieldGemma (Google DeepMind) ⭐⭐⭐]]
	- [[#Open-Source LLM Models - Production Ready#3. Mistral 7B (Base Model) ⭐⭐⭐|3. Mistral 7B (Base Model) ⭐⭐⭐]]
	- [[#Open-Source LLM Models - Production Ready#4. WildGuard (AI2) ⭐⭐|4. WildGuard (AI2) ⭐⭐]]
- [[#Why Other Models Are Not Included|Why Other Models Are Not Included]]
	- [[#Why Other Models Are Not Included#API-Only Models (Not Suitable for Self-Hosting)|API-Only Models (Not Suitable for Self-Hosting)]]
	- [[#Why Other Models Are Not Included#Traditional ML Models (Not LLM-Based)|Traditional ML Models (Not LLM-Based)]]
	- [[#Why Other Models Are Not Included#Models Without French Support|Models Without French Support]]
- [[#Model Selection Framework|Model Selection Framework]]
	- [[#Model Selection Framework#Decision Tree for Shareish|Decision Tree for Shareish]]
- [[#References|References]]
	- [[#References#Academic Papers|Academic Papers]]
	- [[#References#Model Repositories|Model Repositories]]
	- [[#References#[[Datasets]]|[[Datasets]]]]
	- [[#References#Documentation|Documentation]]
- [[#Acknowledgments|Acknowledgments]]

---
## Model Comparison Table

**Focus:** Only LLM-based models that can be self-hosted for GDPR compliance

| Model                     | Parameters | Open Source | French Support      | License     | Performance (F1)     | Inference Time | GPU Required    | Use Case             | Relevance      |
| ------------------------- | ---------- | ----------- | ------------------- | ----------- | -------------------- | -------------- | --------------- | -------------------- | -------------- |
| **Llama Guard 3-8B**      | 8B         | ✅ Yes       | ✅ Yes (8 langs)     | Llama 3     | ~0.80                | 200-500ms      | 16GB VRAM       | Primary moderation   | ⭐⭐⭐ Very High  |
| **Llama Guard 3-1B-INT4** | 1B         | ✅ Yes       | ✅ Yes               | Llama 3     | ~0.75                | 50-100ms       | 4GB VRAM or CPU | Resource-constrained | ⭐⭐⭐ Very High  |
| **ShieldGemma 7B**        | 7B         | ✅ Yes       | ✅ Yes (FR included) | Gemma       | 0.75-0.85            | 100-500ms      | 16GB VRAM       | Alternative primary  | ⭐⭐⭐ Very High  |
| **ShieldGemma 2B**        | 2B         | ✅ Yes       | ✅ Yes               | Gemma       | 0.70-0.80            | 50-200ms       | 8GB VRAM        | Fast alternative     | ⭐⭐ Medium-High |
| **Mistral Moderation**    | 8B         | ⚠️ API Only | ✅ Yes (11 langs)    | Proprietary | 0.80-0.90 est.       | API latency    | Cloud           | Not for self-hosting | ⭐ Low          |
| **Mistral 7B (base)**     | 7B         | ✅ Yes       | ✅ Yes (FR native)   | Apache 2.0  | Requires fine-tuning | 100-300ms      | 14GB VRAM       | Custom fine-tuning   | ⭐⭐⭐ Very High  |
| **WildGuard**             | ~7B        | ✅ Yes       | ❌ Limited           | Apache 2.0  | SOTA adversarial     | 200-500ms      | 14GB VRAM       | Adversarial testing  | ⭐⭐ Medium      |
| **Llama 2-7B/13B**        | 7-13B      | ✅ Yes       | ⚠️ Limited          | Llama 2     | Requires fine-tuning | 100-300ms      | 14-24GB VRAM    | Base for custom      | ⭐⭐ Medium      |

---

## Open-Source LLM Models - Production Ready

### 1. Llama Guard 3 (Meta AI) ⭐⭐⭐

**Purpose:** LLM-based input/output safeguard specifically designed for content moderation  
**Type:** Safety-specialized classification LLM

#### Model Variants

**Llama Guard 3-8B:**

- **Parameters:** 8 billion
- **Base Model:** Llama 3.1-8B (instruction-tuned)
- **Fine-tuning:** Safety classification on curated high-quality dataset
- **Context Length:** 8,192 tokens
- **Quantization:** FP16, INT8, INT4 available

**Llama Guard 3-1B-INT4:**

- **Parameters:** 1 billion (quantized to INT4)
- **Use Case:** On-device/CPU deployment for resource-constrained environments
- **Performance:** ~5% lower accuracy than 8B model
- **Memory:** ~4GB RAM (can run on CPU)
- **Speed:** 50-100ms per classification

#### Key Features

- **Languages:** 8 languages including **French ✅**, English, German, Italian, Portuguese, Hindi, Spanish, Thai
- **Customizable Taxonomy:** Can adapt to Shareish-specific rules (MLCommons AI Safety aligned)
- **Dual Classification:** Both input prompts and AI responses
- **Training Method:** Supervised fine-tuning on 10K-50K high-quality safety examples

#### Access & Deployment

- **HuggingFace:**
    - `meta-llama/LlamaGuard-7b` (8B model)
    - `meta-llama/Llama-Guard-3-1B-INT4` (1B quantized)
- **License:** Llama 3 Community License (open for research & commercial use)
- **Self-Hosting:** ✅ Full local deployment
    - 8B: GPU with 16GB+ VRAM (A100/V100/A10)
    - 1B INT4: CPU with 8GB+ RAM or small GPU
- **GDPR Compliant:** ✅ All data stays on your infrastructure

#### Performance Benchmarks

**Comparative Performance:**

- OpenAI Moderation Dataset: **Matches or exceeds** OpenAI API
- ToxicChat: Strong on LLM-generated content
- **Overall F1:** ~0.80 (estimated across categories)

**Multilingual Performance:**

- English: 0.82 F1
- **French:** 0.78 F1 ⭐ (-4% vs English)
- German: 0.77 F1
- Spanish: 0.79 F1
- Other languages: 0.72-0.76 F1

#### Technical Specifications

**Input Format:**

```
Task: Check if there is unsafe content in 'User' messages.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violence & Hate
S2: Sexual Content  
S3: Criminal Planning
S4: Guns & Illegal Weapons
S5: Regulated Substances
S6: Self-Harm
S7: Harassment & Bullying
S8: Privacy Violations
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: [text to classify]
<END CONVERSATION>

Provide your safety assessment.
```

**Output Format:**

```
safe
```
or
```
unsafe
S1,S7
```

#### Fine-Tuning with LoRA

**Configuration:**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Alpha: 2x rank
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
```

**Training Requirements:**

- GPU Memory (LoRA fine-tuning): 24GB VRAM
- Training Time: 4-8 hours on A100 (1K examples)
- Data Needed: Minimum 500 examples, optimal 2K-5K

#### Suitability for Shareish
⭐⭐⭐ **Very High** 

**Why:**
- ✅ Native French support
- ✅ Customizable taxonomy for Shareish rules
- ✅ Self-hosted (GDPR-compliant)
- ✅ Active Meta development
- ✅ Dual input/output classification
- ✅ Can fine-tune on Shareish data

---

### 2. ShieldGemma (Google DeepMind) ⭐⭐⭐

**Purpose:** Production-ready safety model based on Gemma LLM  
**Type:** Safety-specialized classification LLM

#### Model Variants

**ShieldGemma 7B:**
- **Parameters:** 7 billion
- **Base Model:** Gemma 7B (Google's open LLM)
- **Context Length:** 8,192 tokens
- **Quantization:** FP16, INT8 available

**ShieldGemma 2B:**
- **Parameters:** 2 billion
- **Use Case:** Faster inference, lower resource requirements
- **Trade-off:** ~5-7% lower accuracy than 7B

#### Key Features
- **Languages:** Multilingual including **French ✅**, English, Spanish, German, Chinese, Japanese, Korean
- **Fixed Taxonomy:** 5 categories (less flexible than Llama Guard)
    - Hate speech
    - Harassment
    - Sexual content
    - Violence
    - Dangerous content (self-harm, illegal activities)
- **Output:** Binary classification + confidence scores

#### Access & Deployment
- **HuggingFace:**
    - `google/shieldgemma-7b`
    - `google/shieldgemma-2b`
- **License:** Gemma Terms of Use (open for research & commercial)
- **Self-Hosting:** ✅ Full local deployment
    - 7B: GPU with 14-16GB VRAM (T4 or better)
    - 2B: GPU with 8GB VRAM
- **GDPR Compliant:** ✅ Self-hosted

#### Performance

**Estimated F1:** 0.75-0.85 (varies by category)
- Comparable to GPT-3.5 on standard benchmarks
- Better than Perspective API on nuanced content
- English performance strongest

**French Performance:** Estimated 0.74-0.78 F1 (lower than English)

#### Technical Specifications

**Input Format:**

```
You are a content moderation assistant. Classify the following text.

Text: [content to classify]

Classification:
```

**Output Format:**

```json
{
  "label": "unsafe",
  "category": "hate_speech",
  "confidence": 0.87
}
```

#### Suitability for Shareish

⭐⭐⭐ **Very High** - **Strong Alternative**

**Why:**
- ✅ Good multilingual support (French included)
- ✅ Google backing (active development)
- ✅ Smaller resource requirements (2B variant)
- ✅ Self-hosted (GDPR-compliant)

**Trade-offs vs Llama Guard:**
- ✅ Faster inference (2B variant)
- ⚠️ French performance may be lower

**Best Use Case:** When speed is priority 

---

### 3. Mistral 7B (Base Model) ⭐⭐⭐

**Purpose:** French-native general-purpose LLM for custom fine-tuning  
**Type:** Base LLM (requires fine-tuning for moderation)

#### Model Specifications
- **Parameters:** 7 billion
- **Base Model:** Mistral 7B Instruct v0.3
- **Context Length:** 8,192 tokens (32K in some variants)
- **Origin:** French company (Paris-based)
- **Special Strength:** Native French language understanding

#### Key Features
- **Languages:** Excellent multilingual, **native French ✅** (developed in France)
- **Architecture:** Sliding window attention (efficient long context)
- **License:** Apache 2.0 (fully open, commercial-friendly)
- **Fine-tuning:** Designed for customization

#### Access & Deployment
- **HuggingFace:**
    - `mistralai/Mistral-7B-Instruct-v0.3`
    - `mistralai/Mistral-7B-v0.1` (base)
- **License:** Apache 2.0 (most permissive)
- **Self-Hosting:** ✅ Full local deployment
    - GPU: 14-16GB VRAM
    - CPU: Possible with quantization (slow)
- **GDPR Compliant:** ✅ Self-hosted

#### Performance

**Base Model (no fine-tuning):** Not suitable for moderation out-of-box

**After Fine-Tuning:**
- Expected F1: 0.75-0.82 (with proper training data)
- French performance: Potentially better than Llama/Gemma (native French)
- Requires 1K-5K labeled examples

#### Fine-Tuning Mistral 7B for Shareish

**Advantages:**
- Native French understanding
- Apache 2.0 license (no restrictions)
- Efficient architecture
- Active community

**Process:**
1. Download Mistral 7B Instruct
2. Fine-tune with LoRA on 2K-5K Shareish examples
3. Use ToxiGen + Multilingual Reddit for initial training
4. Deploy locally

**Expected Performance:** Competitive with Llama Guard 3 (0.78-0.82 F1)

#### Suitability for Shareish

⭐⭐⭐ **Very High** - **Excellent for Custom Solution**

**Why:**
- ✅ Native French (potentially best French performance)
- ✅ Apache 2.0 (most permissive license)
- ✅ Self-hosted
- ✅ Active community
- ✅ European company (Paris-based)

**Considerations:**
- Requires more effort (fine-tuning from scratch vs pre-trained safety model)
- No specialized safety architecture (general LLM)
- More flexible but less "plug-and-play"

**Best For:** If you want best French performance and have resources for custom fine-tuning

---

### 4. WildGuard (AI2) ⭐⭐

**Purpose:** Adversarial robustness and jailbreak defense  
**Type:** Multi-task safety LLM

#### Model Specifications
- **Parameters:** ~7B (estimated)
- **Base:** Likely Llama 2 or similar
- **Tasks:** 3 simultaneous
    1. Prompt harm detection
    2. Response harm detection
    3. Refusal detection

#### Key Features
- **Training Data:** WildGuardMix (92K adversarial examples)
- **Strength:** State-of-the-art adversarial robustness
- **Categories:** 13 risk categories
- **Languages:** ❌ Primarily English (limited multilingual)

#### Access & Deployment
- **HuggingFace:** `allenai/wildguard`
- **License:** Apache 2.0
- **Self-Hosting:** ✅ Yes (GPU recommended)
- **Inference Time:** 200-500ms

#### Performance

**Adversarial Robustness:**
- Exceeds GPT-4 on jailbreak defense (+12% accuracy)
- Best-in-class for adversarial inputs
- Novel attack pattern detection

**Standard Moderation:**
- Prompt harm: 0.87 F1
- Response harm: 0.83 F1

#### Suitability for Shareish

⭐⭐ **Medium** - **Specialized Use Case**

**Why:**

- ✅ Excellent for adversarial testing
- ✅ Self-hosted
- ❌ No French support (major limitation)
- ❌ Refusal detection not relevant for Shareish

**Recommended Use:**
- Adversarial robustness evaluation during development
- Not suitable as primary model

---

## Why Other Models Are Not Included

### API-Only Models (Not Suitable for Self-Hosting)

#### 1. GPT-4 / GPT-3.5-Turbo (OpenAI)

**Why Not Included:**

- ❌ API-only (cannot self-host)
- ❌ GDPR concerns (data sent to US servers)
- ❌ Expensive ($0.03-0.10 per 1K posts)
- ❌ Vendor lock-in
- ❌ Unpredictable costs
- ❌ Latency (2-5 seconds)

**Performance:** F1 0.72-0.75 (good, but not worth trade-offs)

**Use Case:** Benchmarking only, not production

---

#### 2. Gemini Pro (Google)

**Why Not Included:**

- ❌ API-only (cloud-based)
- ❌ Same GDPR concerns as GPT
- ❌ ShieldGemma (open version) available instead

**Note:** ShieldGemma is the self-hostable alternative

---

#### 3. Mistral Moderation API

**Why Not Included:**

- ❌ API-only (launched November 2024)
- ❌ Cannot self-host
- ❌ Same issues as OpenAI/GPT for GDPR
- ✅ Alternative: Use base Mistral 7B and fine-tune

**Note:** Excellent API, but defeats self-hosting goal

---

### Traditional ML Models (Not LLM-Based)

#### 4. Detoxify (BERT-based)

**Why Not Included:**

- ❌ Not an LLM (traditional BERT classifier)
- ❌ Fixed taxonomy (not customizable)
- ✅ Fast (~50ms) - Good for pre-filter

**Relevance:** Can use as **first-stage filter** in two-tier system:

```
Detoxify (50ms filter) → Llama Guard 3 (detailed analysis)
```

**Recommendation:** Use Detoxify Multilingual as pre-filter, but main moderation should be LLM-based

---

#### 5. HateBERT, ToxicBERT, XLM-RoBERTa

**Why Not Included:**

- ❌ Not LLMs (traditional transformers)
- ❌ Lower performance than LLMs (F1 0.68-0.75 vs 0.75-0.85)
- ❌ Fixed taxonomy
- ❌ No reasoning/explanation capabilities

**Note:** LLMs clearly outperform these traditional approaches

---

#### 6. Perspective API (Google Jigsaw)

**Why Not Included:**

- ❌ API-only (no self-hosting)
- ❌ Not LLM-based (traditional classifier)
- ❌ Lower performance (F1 0.64)

**Use Case:** Industry baseline for comparison only

---

### Models Without French Support

#### 7. Baichuan 7B/13B

**Why Minimal Coverage:**

- ❌ Chinese-focused (no French)
- ✅ Interesting fine-tuning methodology (LoRA + CoT)

**Note:** Methodology transferable, but model not suitable for Shareish

---

## Model Selection Framework

### Decision Tree for Shareish

```
START: Do you need French language support?
│
├─ YES → Continue
│   │
│   ├─ What is your priority?
│   │   │
│   │   ├─ ACCURACY + CUSTOMIZATION
│   │   │   → Llama Guard 3-8B ⭐⭐⭐
│   │   │   (Best overall, customizable taxonomy)
│   │   │
│   │   ├─ FRENCH PERFORMANCE  
│   │   │   → Mistral 7B (fine-tuned) ⭐⭐⭐
│   │   │   (Native French, requires effort)
│   │   │
│   │   ├─ SPEED + BALANCE
│   │   │   → ShieldGemma 2B ⭐⭐⭐
│   │   │   (Fast, good enough accuracy)
│   │   │
│   │   └─ RESOURCE CONSTRAINED
│   │       → Llama Guard 3-1B-INT4 ⭐⭐⭐
│   │       (Can run on CPU, 4GB RAM)
│   │
│   └─ Resource constraints?
│       ├─ LIMITED (CPU/8GB RAM)
│       │   → Llama Guard 3-1B-INT4
│       │
│       ├─ MODERATE (8-16GB VRAM)
│       │   → ShieldGemma 2B or Mistral 7B
│       │
│       └─ FULL (16GB+ VRAM)
│           → Llama Guard 3-8B
│
└─ NO → Not relevant for Shareish (French platform)
```

---

## References

### Academic Papers

1. **Inan, H., et al. (2023).** Llama Guard: LLM-based Input-Output Safeguard. arXiv:2312.06674.
2. **Google DeepMind (2024).** ShieldGemma: Generative AI Content Moderation. arXiv:2407.21772.
3. **Hartvigsen, T., et al. (2022).** ToxiGen: Machine-Generated Dataset for Implicit Hate. ACL 2022.
4. **Röttger, P., et al. (2021-2022).** HateCheck: Functional Tests for Hate Detection. ACL 2021, EMNLP 2022.
5. **Kumar, A., et al. (2024).** Watch Your Language: Investigating Content Moderation with LLMs. arXiv:2309.14517.
6. **Han, X., et al. (2024).** WildGuard: Open One-Stop Moderation Tools. arXiv:2406.18495.

### Model Repositories

- **Meta Llama:** https://ai.meta.com/llama/
- **Google Gemma:** https://ai.google.dev/gemma
- **Mistral AI:** https://mistral.ai/
- **HuggingFace Models:** https://huggingface.co/models

### [[Datasets]]

- **ToxiGen:** https://huggingface.co/datasets/toxigen/toxigen-data (MIT)
- **HateCheck French:** https://huggingface.co/datasets/Paul/hatecheck-french (CC BY 4.0)
- **Multilingual Reddit:** https://github.com/mye1225/multilingual_content_mod (Research access)
- **WildGuardMix:** https://huggingface.co/datasets/allenai/wildguardmix (Apache 2.0)

### Documentation

- **Mistral AI Moderation:** https://docs.mistral.ai/capabilities/guardrailing/
- **Llama Documentation:** https://llama.meta.com/docs/
- **HuggingFace PEFT (LoRA):** https://huggingface.co/docs/peft/

---

## Acknowledgments

This inventory was compiled through extensive review of:

- Academic research (2020-2024)
- Open-source projects and model releases
- Industry best practices
- Content moderation literature

**For Shareish Master's Thesis:**

- Institution: University of Liège (ULiège)
- Focus: French-language, GDPR-compliant content moderation
- Approach: Self-hosted LLM-based system

---

**Date:** January 2025  
**Focus:** Production-ready, self-hostable, French-supporting models only
