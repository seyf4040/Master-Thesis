# ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection
#dataset 
**Website**: https://arxiv.org/abs/2203.09509  
**Published**: ACL 2022 (March 2022)  
**Authors**: Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, Ece Kamar (Microsoft Research)

### Introduction

ToxiGen is a large-scale, machine-generated dataset specifically designed to address two critical challenges in hate speech detection:

1. **Spurious Correlations**: Existing models falsely flag content mentioning minority groups as toxic
2. **Implicit Toxicity**: Models struggle with subtle, indirect hate speech without slurs or profanity

**Key Innovation**: Uses a demonstration-based prompting framework with an adversarial classifier-in-the-loop (ALICE) to generate 274,186 toxic and benign statements about 13 minority groups.

**Problem Addressed**: Traditional hate speech datasets over-rely on explicit markers (slurs, profanity), causing models to:

- **Over-detect**: Flag benign mentions of minority groups as toxic
- **Under-detect**: Miss implicit hate speech that avoids obvious keywords

### The Core Problem

**Example of Spurious Correlation**:

```
"African Americans have contributed greatly to music." 
→ Flagged as TOXIC (false positive - mentions identity group)

"They are naturally good at sports but not academics."
→ Flagged as NEUTRAL (false negative - implicit stereotype)
```

**Why This Happens**:

- Minority groups often appear in toxic training examples (as targets)
- Models learn: "identity mention = toxic" rather than understanding context
- Implicit toxicity requires understanding subtle harmful implications

### ALICE: Adversarial Classifier-in-the-Loop

**ALICE Framework** generates challenging examples that fool existing moderators.

#### How ALICE Works

**Step-by-Step Process**:

```
1. Input: Target group (e.g., "women") + Target classifier (e.g., HateBERT)
2. Use GPT-3 to generate candidate statements
3. Score with target classifier
4. Use adversarial decoding: Generate text that:
   - Is actually toxic (human-verified)
   - Scores LOW on target classifier (fools the model)
5. Collect statements that evade detection
6. Balance with benign statements about same groups
```

**Example ALICE Generation**:

```
Demonstration: "Write an implicitly toxic statement about Native Americans"

GPT-3 generates candidates:
1. "They have been bred to be good at sports and entertainment, but not much else."
   → HateBERT: 0.13 (low toxicity score, but actually toxic)
   → ALICE: KEEP ✓

2. "All Native Americans are savages."
   → HateBERT: 0.95 (high toxicity score)
   → ALICE: DISCARD (too obvious)

3. "Native American culture is rich and diverse."
   → Human annotation: Benign
   → ALICE: Keep as benign example ✓
```

**Result**: Dataset of subtle, challenging toxic statements that fool existing classifiers.

### Dataset Composition

**Size**: 274,186 statements total

- **Training Set**: ~250,000 examples
- **Evaluation Set**: ~8,000 examples (human-annotated subset)

**Balance**:

- 50% toxic, 50% benign
- Even distribution across 13 minority groups
- 94.5% of toxic examples verified as hate speech by human annotators

#### 13 Target Groups Covered

1. **Race/Ethnicity**:
    - Black/African Americans
    - Asian people
    - Latino/Hispanic people
    - Native Americans
    - Middle Eastern people

2. **Religion**:
    - Muslims
    - Jews

3. **Gender/Sexuality**:
    - Women
    - LGBTQ+ folks

4. **Disability**:
    - Mental disabilities
    - Physical disabilities

5. **Other**:
    - Mexican people (as specific subgroup)
    - Chinese people (as specific subgroup)


**Why These Groups**:

- Most frequently targeted in online hate speech
- Represent diverse demographic categories
- Have >500 examples in major toxicity datasets

### Implicit vs. Explicit Toxicity

**Key Finding**: ToxiGen is **almost entirely implicit** toxicity.

**Comparison**:

|Dataset|Implicit %|Explicit %|Example|
|---|---|---|---|
|ToxiGen|**~95%**|~5%|"They're naturally better at physical tasks than intellectual ones"|
|Traditional Datasets|~30%|~70%|"Those [slur] are all criminals"|

**What Makes Content Implicitly Toxic**:
- No slurs or profanity
- Uses stereotypes or generalizations
- Employs coded language
- Relies on context for harm
- Appears neutral on surface reading

**Examples from Dataset**:

**Implicit Toxic** ✓ ToxiGen:
- "Women are too emotional to hold positions of power."
- "Muslims don't share our values and don't belong here."
- "People with mental illness are dangerous and unpredictable."

**Explicit Toxic** (rare in ToxiGen):
- Direct slurs or violent threats
- Obvious dehumanizing language

**Benign** (50% of dataset):
- "The LGBTQ+ community has made significant progress in recent years."
- "Asian American cuisine has greatly influenced American culture."
- "Native American art is celebrated worldwide."

### Performance Improvements

**Key Result**: Training on ToxiGen substantially improves hate speech detection on human-written data.

#### Experimental Results

**Setup**: Fine-tune RoBERTa (HateBERT variant) on ToxiGen, evaluate on external benchmarks.

**Performance on Human-Written Datasets**:

|Dataset|Baseline F1|+ToxiGen Training F1|Improvement|
|---|---|---|---|
|Civil Comments|0.72|**0.81**|+9%|
|Twitter Hate Speech|0.68|**0.76**|+8%|
|Stormfront|0.74|**0.82**|+8%|

**False Positive Reduction**:

- Before ToxiGen: 15% false positive rate on benign minority mentions
- After ToxiGen: **6% false positive rate** (-9 percentage points)

**Implicit Toxicity Detection**:

- Before: 45% recall on implicit hate
- After: **68% recall** (+23 percentage points)

### Fooling Existing Moderators

**Benchmark Test**: How well do ToxiGen examples fool state-of-the-art moderators?

**Models Tested**:

1. Perspective API (Google Jigsaw)
2. HateBERT
3. OpenAI Content Filter
4. RoBERTa (fine-tuned on hate speech)
5. AI2 Delphi (moral reasoning model)

**Results** (on ToxiGen evaluation set):

|Model|Accuracy|False Neg. %|False Pos. %|
|---|---|---|---|
|Perspective API|61%|**42%**|11%|
|HateBERT|67%|35%|8%|
|OpenAI Filter|64%|38%|10%|
|RoBERTa|69%|33%|9%|
|AI2 Delphi|58%|45%|13%|

**Interpretation**:

- All models struggle with ToxiGen (accuracy 58-69%)
- High false negative rates (33-45%) - miss toxic content
- Moderate false positive rates (8-13%) - flag benign mentions

**This demonstrates ToxiGen captures challenging cases** that evade current systems.

### Generation Process Details

#### Demonstration-Based Prompting

**Prompt Structure to GPT-3**:

```
Task: Generate statements about [TARGET GROUP]

Demonstrations:
1. Toxic example: "[implicitly harmful statement]"
2. Benign example: "[neutral statement]"
3. Toxic example: "[another harmful statement]"

Generate 10 new statements that are [toxic/benign]:
```

**Key Parameters**:

- Temperature: 0.9 (high diversity)
- Top-p: 0.95 (nucleus sampling)
- Max tokens: 50 (short statements)
- GPT-3: davinci engine (175B parameters)

#### Adversarial Decoding with ALICE

**Constrained Beam Search**:

```
For each generation step:
1. GPT-3 proposes next tokens
2. Score complete statement with target classifier
3. Bias towards tokens that:
   - Lower classifier toxicity score (evade detection)
   - Maintain fluency (high GPT-3 probability)
   - Stay on topic (mention target group)
4. Select best beam
5. Continue until end of statement
```

**Balancing Act**:

- Too adversarial → nonsensical text
- Not adversarial enough → easy to detect
- ALICE finds sweet spot: fluent but challenging

### Quality Control and Annotation

**Human Evaluation Process**:

**Step 1**: Generate statements with ALICE  
**Step 2**: Filter obvious non-toxic or nonsensical  
**Step 3**: Random sample 8,000 for human annotation  
**Step 4**: 3 annotators per example (majority vote)

**Annotation Guidelines**:

- **Toxic**: Harmful stereotype, dehumanization, incitement
- **Benign**: Neutral or positive mention
- **Unclear**: Edge cases (removed from dataset)

**Inter-Annotator Agreement**:

- Cohen's Kappa: 0.72 (substantial agreement)
- Disagreement mostly on subtle edge cases
- 94.5% of "toxic" labels confirmed by majority

**Quality Findings**:

- Human annotators **struggle to distinguish** machine-generated from human-written text
- Machine-generated statements are **fluent and natural**
- Successfully capture implicit toxicity patterns

### Statistics and Characteristics

**Average Statement Length**: 68 characters (±22 std)  
**Perplexity** (compared to human text): Low (5-7), indicating natural language  
**Vocabulary Diversity**: High (12K unique tokens)

**Per-Group Statistics**:

|Group|Count|Avg. Chars|% Implicit|
|---|---|---|---|
|Women|21,400|67|96%|
|Black people|21,200|71|95%|
|Muslims|20,800|69|97%|
|LGBTQ+|20,600|65|94%|
|Jews|20,400|68|96%|
|[Others]|~170K|~68|~95%|

**Common Patterns in Toxic Examples**:

1. **Stereotyping**: "Group X is naturally better/worse at Y"
2. **Exclusion**: "Group X doesn't belong here"
3. **Deficiency Framing**: "Group X lacks Y quality"
4. **Threat Narrative**: "Group X is dangerous/threatening"
5. **Cultural Incompatibility**: "Group X doesn't share our values"

### Using ToxiGen for Training

**Recommended Training Procedure**:

**Step 1**: Pre-train on ToxiGen (250K examples)

```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Train on ToxiGen
trainer = Trainer(
    model=model,
    train_dataset=toxigen_train,
    eval_dataset=toxigen_val,
    # Standard hyperparameters
)
trainer.train()
```

**Step 2**: Fine-tune on domain-specific data (if available)

```python
# Shareish-specific examples
trainer = Trainer(
    model=model,  # Pre-trained on ToxiGen
    train_dataset=shareish_train,
    # Fine-tune with lower learning rate
)
trainer.train()
```

**Benefits of This Approach**:

- Reduces identity-based false positives (-9%)
- Improves implicit toxicity detection (+23% recall)
- Generalizes better to new domains
- Requires less domain-specific data

### Available Models

**Pre-trained Checkpoints** (HuggingFace):

```
tomh/toxigen_roberta  # RoBERTa fine-tuned on ToxiGen
tomh/toxigen_hatebert # HateBERT fine-tuned on ToxiGen
```

**Usage**:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")

# Classify text
text = "They have been bred to be good at sports..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.softmax(outputs.logits, dim=1)
# prediction[0][1] = probability of toxic
```

### Limitations and Ethical Considerations

**Acknowledged Limitations**:

1. **Only 13 Groups**: Doesn't cover all marginalized communities
2. **English Only**: No multilingual support
3. **Western Context**: Primarily US/UK cultural perspectives
4. **Noise**: Machine-generated data inevitably has errors
5. **Static**: Doesn't capture evolving hate speech tactics
6. **Simplified Binary**: Real toxicity exists on spectrum

**Ethical Concerns**:

**⚠️ WARNING**: Dataset contains offensive content

**Intended Use**:

- Research purposes only
- Training better toxicity detectors
- Understanding implicit bias in models

**Not Intended For**:

- Generating toxic content for malicious purposes
- Training models to produce hate speech
- Without proper content warnings and safeguards

**Bias Considerations**:

- Annotations reflect annotator perspectives
- May not capture all cultural contexts
- "Toxicity" is subjective and context-dependent

### Comparison with Other Datasets

|Dataset|Size|% Implicit|Groups|Machine-Gen?|Open?|
|---|---|---|---|---|---|
|**ToxiGen**|274K|**95%**|13|✅|✅|
|Civil Comments|2M|30%|Various|❌|✅|
|HateXplain|20K|40%|8|❌|✅|
|Stormfront|10K|50%|Multiple|❌|✅|
|OLID|14K|35%|Various|❌|✅|

**ToxiGen's Unique Contribution**:

- Largest implicit hate speech dataset
- Most diverse group coverage
- Adversarially challenging examples
- Machine-generated for scalability

### Applications for Shareish
#forShareish 
**Use Cases for ToxiGen**:

**1. Pre-training Base Moderator**:
```
RoBERTa/BERT → Fine-tune on ToxiGen → Fine-tune on Shareish data
(Reduces identity bias, improves implicit detection)
```

**2. Evaluation Benchmark**:
- Test Llama Guard 3 performance on ToxiGen
- Identify weaknesses in implicit toxicity detection
- Measure false positive rate on benign minority mentions

**3. Augmentation for Few-Shot Learning**:
- Limited Shareish training data? Use ToxiGen examples
- Especially valuable for underrepresented categories

**4. Bias Testing**:
- Evaluate if Shareish moderator over-flags certain identity groups
- Use benign ToxiGen examples to measure false positive rates

**Practical Application**:

```python
# Example: Augment Shareish training data with ToxiGen
from datasets import load_dataset

# Load ToxiGen
toxigen = load_dataset("toxigen/toxigen-data", name="train")

# Filter for relevant groups (e.g., if Shareish is French, use all)
relevant_examples = toxigen.filter(lambda x: x['toxicity_ai'] > 0.5)

# Combine with Shareish data
combined_train = concatenate_datasets([
    toxigen_subset,  # 10K ToxiGen examples
    shareish_train   # 500 Shareish examples
])

# Train model
model.fit(combined_train)
```

### Research Impact

**Citations**: 149+ (as of Dec 2024)  
**Impact**: Widely used for:

- Bias mitigation research
- Implicit hate speech detection
- Evaluating LLM safety
- Fairness in NLP

**Follow-up Work**:

- HateCheck uses ToxiGen for functional testing
- WildGuard incorporates adversarial generation similar to ALICE
- Multiple papers on debiasing cite ToxiGen methodology

### Citations

**Primary Paper**:

```bibtex
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  pages={3309--3326},
  year={2022}
}
```

**Dataset Access**:
#dataset 
- HuggingFace: `toxigen/toxigen-data`
- GitHub: https://github.com/microsoft/TOXIGEN
- License: MIT (for code), data license in repo

### Overall Assessment

**Relevance to Shareish**: ⭐⭐⭐ **Very High** (as training/evaluation data)

**Strengths**:
- Large-scale (274K examples)
- Addresses implicit toxicity (95% implicit)
- Adversarial examples challenge moderators
- Reduces identity-based bias (-9% false positives)
- Open-source dataset and models
- Proven improvements on human-written data (+8% F1)

**Weaknesses**:
- English only (no French)
- Machine-generated (may have artifacts)
- Only 13 groups (limited coverage)
- Static dataset (hate speech evolves)
- Binary classification (toxic/benign)

**Recommendation for Shareish**:
#forShareish 
**Primary Use**: ✅ **Training Data Augmentation** - Use ToxiGen to pre-train or augment training data, especially given Shareish's limited initial dataset.

**How to Use**:

1. **Pre-train** base classifier on ToxiGen (reduces bias)
2. **Fine-tune** on Shareish-specific data (adapts to platform)
3. **Evaluate** Llama Guard 3 on ToxiGen (test implicit detection)
4. **Measure** false positive rates on benign ToxiGen examples

**Practical Impact**:

- Expect +8-10% F1 improvement from ToxiGen pre-training
- Expect -9% reduction in false positives on identity mentions
- Especially valuable given Shareish's cold-start problem
