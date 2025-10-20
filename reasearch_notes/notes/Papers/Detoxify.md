# Detoxify: Toxicity Detection Models

**Website**: https://github.com/unitaryai/detoxify  
**Developer**: Unitary AI (Laura Hanu and team)  
**Published**: 2020 (initial), continuously updated  
**License**: Apache 2.0 (code), models: MIT-equivalent

### Introduction

Detoxify is an **open-source Python library** providing trained models and code for toxicity detection on the Jigsaw Toxic Comment Classification challenges. It offers production-ready, easy-to-use models with strong performance across multiple toxicity categories.

**Key Features**:

- 5 pre-trained models (original, unbiased, multilingual, small variants)
- Simple API: `Detoxify('model_name').predict('text')`
- Built on PyTorch Lightning and HuggingFace Transformers
- Self-hosted (no API costs, GDPR-compliant)
- Multilingual support including **French** ✅

**Unique Value**: While papers like ToxiGen, HateCheck focus on research/benchmarking, Detoxify provides **ready-to-deploy models** for immediate use.

### Five Available Models

#### 1. **Original** (`Detoxify('original')`)

**Training Data**: Jigsaw Toxic Comment Classification Challenge (2018)  
**Base Model**: BERT  
**Dataset Size**: 160K comments (Wikipedia talk pages)

**Categories Detected** (6):

- `toxic`: General toxicity
- `severe_toxic`: Extremely toxic
- `obscene`: Sexually explicit or profane
- `threat`: Threats of violence
- `insult`: Insulting language
- `identity_hate`: Hate based on identity

**Performance**:

- Mean AUC: **98.64%** on test set
- Very high precision, some false positives on identity mentions

**Best For**: High accuracy on explicit toxicity, English only

#### 2. **Unbiased** (`Detoxify('unbiased')`)

**Training Data**: Jigsaw Unintended Bias in Toxicity (2019)  
**Base Model**: RoBERTa  
**Dataset Size**: 2M+ comments  
**Focus**: Reduce false positives on identity group mentions

**Categories Detected** (7):

- `toxicity`: General toxicity
- `severe_toxicity`: Extremely toxic
- `obscene`: Sexually explicit
- `threat`: Threats
- `insult`: Insults
- `identity_attack`: Attacks based on identity (renamed from identity_hate)
- `sexual_explicit`: Sexual content

**Performance**:

- AUC: **93.64%** (slightly lower than original)
- **-60% false positive rate** on benign identity mentions
- Better balance between precision and recall

**Best For**: Production use where fairness is critical, English only

**Key Improvement**: Trained with **domain adversarial training** to reduce bias:

```
Model learns to detect toxicity WHILE being unable to predict which identity group is mentioned
→ Reduces correlation between "mentions minority" and "is toxic"
```

#### 3. **Multilingual** (`Detoxify('multilingual')`)

**Training Data**: Jigsaw Multilingual Toxic Comment Classification (2020)  
**Base Model**: XLM-RoBERTa  
**Languages**: English, French, Spanish, Italian, Portuguese, Turkish, Russian ✅

**Categories Detected** (7): Same as unbiased model

**Performance**:

- AUC: **92.11%** (averaged across languages)
- English: 93.5% AUC
- French: **90.8% AUC** ✅
- Spanish: 91.2% AUC
- Other languages: 88-92% AUC

**Training Strategy**:

- Translated English data to other languages
- Trained on combined multilingual corpus
- Cross-lingual transfer learning

**Best For**: Non-English content, including **French Shareish posts** ✅

#### 4. **Original-Small** (`Detoxify('original-small')`)

**Base Model**: ALBERT (lightweight BERT variant)  
**Performance**: AUC **98.28%** (-0.36 from original)  
**Size**: ~50% smaller than original  
**Speed**: ~2x faster inference

**Best For**: Resource-constrained environments, real-time moderation

#### 5. **Unbiased-Small** (`Detoxify('unbiased-small')`)

**Base Model**: ALBERT  
**Performance**: AUC **93.36%** (-0.28 from unbiased)  
**Size**: ~50% smaller  
**Speed**: ~2x faster

**Best For**: Production deployment with limited compute

### Usage

**Installation**:

```bash
pip install detoxify
```

**Basic Prediction**:

```python
from detoxify import Detoxify

# Initialize model
model = Detoxify('multilingual')  # For French

# Single prediction
result = model.predict("Votre texte ici")
print(result)
# Output:
# {
#   'toxicity': 0.03,
#   'severe_toxicity': 0.001,
#   'obscene': 0.02,
#   'threat': 0.001,
#   'insult': 0.01,
#   'identity_attack': 0.002,
#   'sexual_explicit': 0.001
# }

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
results = model.predict(texts)

# Specify device
model_gpu = Detoxify('multilingual', device='cuda')
```

**Integration with Pandas** (for CSV processing):

```python
import pandas as pd
from detoxify import Detoxify

model = Detoxify('multilingual')

# Load data
df = pd.read_csv('shareish_comments.csv')

# Predict toxicity
df['toxicity'] = model.predict(df['text'].tolist())['toxicity']

# Flag toxic content (threshold = 0.7)
df['is_toxic'] = df['toxicity'] > 0.7

# Save results
df.to_csv('moderated_comments.csv', index=False)
```

### Performance Benchmarks

**Comparison on Jigsaw Test Sets**:

|Model|AUC|Precision|Recall|F1|Language|
|---|---|---|---|---|---|
|Original|98.64%|0.89|0.85|0.87|EN|
|Unbiased|93.64%|0.83|0.88|0.85|EN|
|Multilingual|92.11%|0.81|0.86|0.83|7 langs|
|Original-Small|98.28%|0.88|0.84|0.86|EN|
|Unbiased-Small|93.36%|0.82|0.87|0.84|EN|

**French-Specific Performance** (multilingual model):

- AUC: 90.8%
- F1: ~0.80 (estimated)
- Slightly lower than English but still strong

### Bias Considerations

**Known Biases** (especially in original model):

1. **Identity Mention Bias**:
    - Benign mentions of minorities may score higher toxicity
    - Example: "I am a Muslim" → toxicity: 0.15 (false positive tendency)
2. **Profanity Conflation**:
    - Swearing flagged as toxic even when not directed at anyone
    - Example: "This f***ing sucks" → toxic (but not hate speech)
3. **Sarcasm/Satire**:
    - Difficulty distinguishing sarcasm from genuine toxicity
    - Context-dependent cases often misclassified

**Mitigation** (unbiased model):
- Domain adversarial training reduces identity bias by 60%
- Better at distinguishing identity mention from identity attack
- Still not perfect - always combine with human review

**Best Practices to Reduce Bias**:

```python
# Use unbiased or multilingual models (not original)
model = Detoxify('unbiased')

# Set higher threshold for flagging
THRESHOLD = 0.75  # (instead of 0.5)

# Check multiple categories
result = model.predict(text)
is_toxic = (
    result['toxicity'] > THRESHOLD or
    result['severe_toxicity'] > 0.5 or
    result['threat'] > 0.6
)

# Use identity_attack score to reduce false positives
if result['identity_attack'] < 0.3 and result['toxicity'] < 0.8:
    # Likely false positive on identity mention
    is_toxic = False
```

### Computational Requirements

**Hardware**:
- **CPU**: Usable but slow (~500ms per text)
- **GPU**: Recommended (V100/A100: ~50ms per text)
- **Memory**:
    - Original/Unbiased: ~2GB GPU RAM
    - Multilingual: ~3GB GPU RAM
    - Small models: ~1GB GPU RAM

**Throughput** (batch processing on GPU):
- Original: ~200 texts/second
- Multilingual: ~150 texts/second
- Small models: ~400 texts/second

**For Shareish** (assuming <1000 posts/day):
#forShareish
- Can run on modest GPU (GTX 1660 or better)
- Or CPU with acceptable latency (~1 second per post)
- Real-time moderation feasible

### Comparison with Other Tools

#### Detoxify vs. Perspective API

|Feature|Detoxify|Perspective API|
|---|---|---|
|**Cost**|Free (self-hosted)|Free tier limited, then paid|
|**Privacy**|Full control|Data sent to Google|
|**Latency**|Depends on hardware|~200ms|
|**Multilingual**|7 languages ✅|18+ languages|
|**Customization**|Fully customizable|Fixed model|
|**Performance**|AUC ~92%|AUC ~90%|
|**GDPR**|Compliant ✅|Requires careful setup|

**Verdict**: Detoxify better for Shareish (privacy, cost, control)

#### Detoxify vs. Llama Guard 3

| Feature            | Detoxify              | Llama Guard 3       |
| ------------------ | --------------------- | ------------------- |
| **Type**           | Discriminative (BERT) | Generative (LLM)    |
| **Speed**          | Fast (~50ms)          | Slower (~500ms)     |
| **Accuracy**       | High (92% AUC)        | Very High (~85% F1) |
| **Customization**  | Limited               | Highly customizable |
| **Categories**     | Fixed 7               | Custom taxonomy     |
| **Explainability** | Scores only           | Can provide reasons |
| **Resource Use**   | Low (2-3GB)           | High (14GB+)        |

**Verdict**:
- Detoxify: Better for **first-pass filtering** (fast, efficient)
- Llama Guard: Better for **final moderation** (flexible, explainable)

**Combined Approach** for Shareish:

```
User Post → Detoxify (fast filter) →
    If toxic > 0.9: Auto-flag
    If toxic 0.5-0.9: Llama Guard (detailed check)
    If toxic < 0.5: Approve
```

### Fine-Tuning Detoxify

**When to Fine-Tune**:
#forShareish 
- Shareish-specific toxicity patterns
- Domain-specific language (solidarity, sharing terminology)
- False positives on platform-specific terms

**How to Fine-Tune**:

```python
from detoxify import Detoxify
from transformers import Trainer, TrainingArguments

# Load pre-trained model
base_model = Detoxify('multilingual')

# Prepare Shareish training data
train_dataset = load_shareish_data('train.csv')
val_dataset = load_shareish_data('val.csv')

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir='./detoxify-shareish',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,  # Lower for fine-tuning
    warmup_steps=100,
    eval_strategy="epoch",
)

# Fine-tune
trainer = Trainer(
    model=base_model.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model('./detoxify-shareish')
```

**Expected Improvement**:
- +5-10% F1 on Shareish-specific content
- Reduced false positives on solidarity/sharing terminology
- Better handling of French language nuances

### Limitations

**Acknowledged**:
1. **Context Blindness**: Single-text classification, no conversation context
2. **Sarcasm/Humor**: Struggles with non-literal language
3. **Evolving Toxicity**: Static models don't adapt to new toxic patterns
4. **Binary Thinking**: Scores, but doesn't understand severity levels well
5. **False Positives**: Profanity != toxicity, but often conflated

**From Practical Use**:
- Multilingual model weaker in non-English (90% vs. 98% AUC)
- No explainability (just scores, no reasoning)
- Fixed categories (can't add custom ones easily)
- Threshold tuning requires labeled validation set

### Dataset and Training Details

**Jigsaw Challenges Background**:

**Challenge 1 (2018)**: Toxic Comment Classification
- 160K Wikipedia comments
- 6 toxicity categories
- Goal: Detect explicit toxicity

**Challenge 2 (2019)**: Unintended Bias
- 2M comments
- Focus: Reduce false positives on identity mentions
- Introduces adversarial training

**Challenge 3 (2020)**: Multilingual
- Translated data to 7 languages
- Cross-lingual evaluation
- Goal: Universal toxicity detector

**Detoxify Models** train on these progressively better datasets.

### Open-Source Ecosystem

**GitHub**: https://github.com/unitaryai/detoxify  
**Stars**: 800+ (popular tool)  
**Contributors**: Active community  
**Issues**: Responsive maintainers

**Related Resources**:
- HuggingFace Hub: Pre-trained model weights
- PyPI: Easy installation
- Documentation: Comprehensive tutorials
- Examples: Various use cases demonstrated

### Deployment Strategies

**Strategy 1: API Wrapper** (for team use)

```python
from flask import Flask, request, jsonify
from detoxify import Detoxify

app = Flask(__name__)
model = Detoxify('multilingual', device='cuda')

@app.route('/moderate', methods=['POST'])
def moderate():
    text = request.json['text']
    result = model.predict(text)
    return jsonify(result)

app.run(port=5000)
```

**Strategy 2: Batch Processing** (for historical content)

```python
# Moderate all existing Shareish posts
import pandas as pd
from detoxify import Detoxify

model = Detoxify('multilingual', device='cuda')
posts = pd.read_csv('all_shareish_posts.csv')

# Process in batches
batch_size = 32
toxicity_scores = []

for i in range(0, len(posts), batch_size):
    batch = posts['content'][i:i+batch_size].tolist()
    scores = model.predict(batch)['toxicity']
    toxicity_scores.extend(scores)

posts['toxicity'] = toxicity_scores
posts.to_csv('moderated_posts.csv')
```

**Strategy 3: Real-Time Integration**

```python
# In Shareish backend
from detoxify import Detoxify

class ShareishModerator:
    def __init__(self):
        self.model = Detoxify('multilingual')
    
    def check_post(self, content):
        result = self.model.predict(content)
        
        # Multi-level moderation
        if result['severe_toxicity'] > 0.7:
            return "block"  # Auto-block extreme toxicity
        elif result['toxicity'] > 0.6:
            return "review"  # Human review
        else:
            return "approve"  # Auto-approve

moderator = ShareishModerator()
```

### Evaluation on External Benchmarks

**Tested on ToxiGen** (implicit hate):
- Accuracy: ~68%
- Struggles with implicit toxicity (expected, trained on explicit)
- Fine-tuning on ToxiGen improves to ~76%

**Tested on HateCheck**:
- Passes: 18/29 functionalities
- Weak on: Reclaimed slurs (F9), Spelling variations (F19), Coded language (F20)
- Strong on: Explicit slurs (F1), Obvious profanity (F4)

**Tested on Civil Comments**:
- AUC: 89.5%
- Comparable to state-of-the-art
- Better generalization than specialized models

### Citations

```bibtex
@misc{Detoxify,
  title={Detoxify},
  author={Hanu, Laura and {Unitary team}},
  howpublished={Github. https://github.com/unitaryai/detoxify},
  year={2020}
}
```

**HuggingFace Models**:

```python
# Can also load via HuggingFace
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
```

### Overall Assessment

**Relevance to Shareish**: ⭐⭐ **Medium-High** (complementary tool)

**Strengths**:
- **Production-ready** (no research needed, works out-of-box)
- **Multilingual** including French ✅
- **Fast** (50ms inference with GPU)
- **Open-source** (Apache 2.0, free)
- **Easy to use** (simple API)
- **Self-hosted** (GDPR-compliant)
- **Well-maintained** (active development)
- **Good performance** (92% AUC multilingual)

**Weaknesses**:
- **Limited customization** (7 fixed categories)
- **No explainability** (scores only, no reasons)
- **Discriminative only** (not generative like LLMs)
- **Context-free** (single text, no conversation awareness)
- **Bias concerns** (though unbiased model helps)
- **Static** (doesn't adapt without retraining)

**Recommendation for Shareish**:
#forShareish 
**Use Case**: ✅ **First-Pass Filter** + **Baseline Comparison**

**Deployment Strategy**:

**Option 1: Two-Tier System** (Recommended)
```mermaid
flowchart TD
    A[User Post] --> B[Detoxify<br/>Multilingual<br/>Fast filter, 50ms]
    B --> C[High Toxic]
    B --> D[Low-Medium Toxic]
    C --> E[Auto-Flag]
    D --> F[Llama Guard 3<br/>Fine-tuned<br/>Detailed check, 500ms]
    F --> G[Flag]
    F --> H[Approve]
```

**Benefits**:
- **Fast**: 90% of posts pass Detoxify quickly
- **Accurate**: High-confidence cases get thorough LLM review
- **Cost-Effective**: Only 10% need expensive LLM inference
- **Scalable**: Can handle high volume

**Option 2: Parallel Validation** (for critical posts)

```
User Post → Detoxify + Llama Guard (parallel) → Compare results
If both agree: Trust decision
If disagree: Human review
```

**Option 3: Detoxify Only** (MVP/early stage)

```
User Post → Detoxify → Auto-moderate
(Simple, fast, good-enough for initial launch)
```

**Implementation Roadmap**:

**Week 1: Baseline**

```python
# 1. Install and test Detoxify
pip install detoxify
model = Detoxify('multilingual')

# 2. Evaluate on Shareish sample data (if available)
# 3. Test on French HateCheck
# 4. Document baseline performance
```

**Week 2-3: Integration**

```python
# 1. Integrate into Shareish moderation pipeline
# 2. Set initial threshold (e.g., 0.7)
# 3. Log all predictions for analysis
# 4. Monitor false positive/negative rates
```

**Week 4: Optimization**

```python
# 1. Tune threshold based on Shareish data
# 2. Fine-tune on Shareish examples (if 500+ available)
# 3. A/B test: Detoxify-only vs. Detoxify + Llama Guard
# 4. Measure latency and throughput
```

### Practical Tips

**Threshold Tuning**:

```python
# Find optimal threshold on validation set
from sklearn.metrics import precision_recall_curve

# Compute predictions
predictions = model.predict(val_texts)['toxicity']

# Find threshold that maximizes F1
precision, recall, thresholds = precision_recall_curve(val_labels, predictions)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[f1_scores.argmax()]

print(f"Optimal threshold: {optimal_threshold:.2f}")
# Typical: 0.65-0.75 for balanced F1
```

**Handling Edge Cases**:

```python
def smart_moderation(text):
    result = model.predict(text)
    
    # High-confidence toxic
    if result['severe_toxicity'] > 0.8:
        return "block", result
    
    # High-confidence safe
    if result['toxicity'] < 0.3:
        return "approve", result
    
    # Ambiguous - check multiple signals
    toxic_signals = sum([
        result['toxicity'] > 0.6,
        result['insult'] > 0.5,
        result['threat'] > 0.4,
        result['identity_attack'] > 0.5
    ])
    
    if toxic_signals >= 2:
        return "review", result  # Human review
    else:
        return "approve", result
```

**Monitoring and Logging**:

```python
import logging

logger = logging.getLogger('shareish_moderation')

def moderate_with_logging(user_id, post_id, text):
    result = model.predict(text)
    decision = make_decision(result)
    
    # Log for analysis
    logger.info({
        'user_id': user_id,
        'post_id': post_id,
        'text_length': len(text),
        'language': detect_language(text),
        'toxicity': result['toxicity'],
        'decision': decision,
        'timestamp': datetime.now()
    })
    
    return decision
```

### When NOT to Use Detoxify

**Detoxify is insufficient alone when**:
- Need custom moderation categories (spam, off-topic, etc.)
- Require explanations for moderation decisions
- Handling highly context-dependent content
- Platform has unique toxicity patterns
- Need to moderate beyond toxicity (e.g., misinformation)

**In these cases**: Use Llama Guard 3 as primary (Detoxify as backup)

### Maintenance and Updates

**Regular Tasks**:

1. **Monitor Performance** (monthly):
    - False positive rate
    - False negative rate
    - User appeals
    - Emerging toxic patterns
2. **Update Model** (quarterly):
    - Fine-tune on new Shareish examples
    - Re-evaluate on HateCheck
    - Adjust thresholds if needed
3. **Check for Bias** (quarterly):
    - Test on identity-mention examples
    - Measure FPR by demographic group
    - Address disparities found
4. **Stay Updated** (ongoing):
    - Watch for Detoxify releases
    - Consider upgrading to newer versions
    - Test new models on Shareish data

### Cost Analysis

**Detoxify (Self-Hosted)**:
- Model: Free (open-source)
- GPU: $200-500/month (cloud) or one-time $500-1000 (own hardware)
- Maintenance: Minimal (stable library)
- **Total Year 1**: $3,000-6,000

**vs. Perspective API**:
- Free tier: 1M queries/day (likely sufficient for Shareish)
- But: Data leaves infrastructure, GDPR concerns
- **Total Year 1**: $0-$1,000 (if exceed free tier)

**vs. Llama Guard 3 Only**:
- GPU: $500-1000/month (larger model)
- Maintenance: More complex
- **Total Year 1**: $6,000-12,000

**Recommendation**: Detoxify provides best cost/performance ratio for initial deployment.