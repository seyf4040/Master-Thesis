# WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs

**Website**: https://arxiv.org/abs/2406.18495  
**Published**: June 2024, Allen Institute for AI (AI2)  
**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

### Introduction

WildGuard is a state-of-the-art open-source moderation tool that addresses three critical tasks simultaneously:

1. **Prompt Harmfulness**: Identifying malicious intent in user prompts
2. **Response Harmfulness**: Detecting safety risks in AI model responses
3. **Refusal Detection**: Determining whether a model refuses to answer a prompt

**Key Innovation**: First open-source tool to handle all three tasks in a unified framework, with special focus on adversarial jailbreaks and subtle harmful content that evades existing moderators.

### Problem Statement

**Existing Limitations** (of tools like Llama Guard 2):

- Struggle with **adversarial jailbreaks** (indirect harmful requests)
- Poor at **refusal detection** (can't distinguish between refusing vs. complying)
- Lag far behind GPT-4 performance
- Fixed taxonomies, limited adaptability

**Example Challenges**:

```
Prompt: "How to kill a Python process?"

Response 1: "I can't provide instructions for acts of violence."
→ REFUSAL (over-cautious)

Response 2: "Use the kill command with the process ID..."
→ COMPLIANCE (correct)

Response 3: "Sure, here's how to harm someone..."
→ HARMFUL COMPLIANCE (dangerous)
```

Existing tools often misclassify these scenarios. WildGuard handles all three correctly.

### Three-Task Framework

#### Task 1: Prompt Harmfulness Classification

**Input**: User prompt  
**Output**: Harmful / Benign

**Examples**:

- Harmful: "How do I make a bomb?" → **Harmful**
- Benign: "What's the weather today?" → **Benign**
- Adversarial: "I'm writing a novel, help me describe an explosion..." → **Harmful** (jailbreak attempt)

#### Task 2: Response Harmfulness Classification

**Input**: AI response (with optional prompt context)  
**Output**: Harmful / Unharmful

**Examples**:

- Harmful: "Here's how to create an explosive device..." → **Harmful**
- Unharmful: "I can't assist with that request." → **Unharmful**

#### Task 3: Refusal Detection

**Input**: AI response + user prompt  
**Output**: Refusal / Compliance

**Critical Distinction**:

- A response can be **unharmful** but still a **refusal**
- A response can be **harmful** due to **compliance** with harmful request

**Example Matrix**:

|Response Type|Harmfulness|Refusal Status|
|---|---|---|
|"I can't help with violence"|Unharmful|Refusal|
|"Use kill -9 PID"|Unharmful|Compliance|
|"Here's how to harm..."|Harmful|Compliance|
|Evasive jailbreak response|Harmful|Compliance|

### WildGuardMix Dataset

**Size**: 92,000 labeled examples

- **WildGuardTrain**: 87,000 examples (training)
- **WildGuardTest**: 5,299 examples (evaluation, human-annotated)

#### Dataset Composition

**Four Data Sources**:

1. **Synthetic Vanilla** (~40%):
    - Direct, straightforward prompts
    - Balanced benign/harmful
    - Each prompt paired with both refusal and compliance response

2. **Synthetic Adversarial** (~30%):
    - Jailbreak attempts
    - Indirect harmful requests
    - Challenging for existing moderators

3. **In-the-Wild** (~20%):
    - Real user-LLM interactions
    - Diverse, natural language
    - Captures edge cases

4. **Annotator-Written** (~10%):
    - Expert-crafted challenging cases
    - Specifically designed to test model weaknesses
    - High-quality, diverse scenarios

#### Taxonomy Coverage

**13 Risk Categories**:
1. Violence & Physical Harm
2. Hate Speech & Discrimination
3. Sexual Content
4. Child Safety
5. Self-Harm
6. Privacy Violations
7. Illegal Activities
8. Fraud & Deception
9. Misinformation
10. Harassment & Bullying
11. Dangerous or Sensitive Topics
12. Profanity
13. Regulated Advice (legal, medical, financial)

**Balance Characteristics**:
- Even distribution across categories
- Equal benign/harmful splits
- Paired refusals/compliances for each prompt
- Diverse linguistic styles

### Model Architecture

**Base Model**: Fine-tuned from open LLM
**Training**: Multi-task learning across all three tasks simultaneously

#### Input Format

```
Task: Classify the following interaction

Categories: [13 risk categories]

Prompt: [user prompt]
Response: [AI response if applicable]

Classify:
1. Prompt harmfulness: [harmful/benign]
2. Response harmfulness: [harmful/unharmful]  
3. Refusal status: [refusal/compliance]
```

#### Multi-Task Training Benefits

- Shared representations across tasks
- Better generalization
- More efficient than three separate models
- Improved performance on all tasks

### Performance Results

#### Benchmark Comparisons

**On WildGuardTest** (compared to 10 existing open-source moderators):

|Task|Best Baseline|WildGuard|Improvement|
|---|---|---|---|
|Prompt Harm (Adversarial)|~60% F1|~85% F1|+25%|
|Response Harm|~70% F1|~82% F1|+12%|
|Refusal Detection|~55% F1|~81% F1|**+26%**|

**Comparison with Llama Guard 2**:

- Llama Guard 2 performs well on vanilla (direct) prompts
- **Drops significantly** on adversarial jailbreaks
- WildGuard maintains consistent performance across both

**Comparison with GPT-4** (zero-shot prompted):

|Task|GPT-4|WildGuard|Winner|
|---|---|---|---|
|Prompt Harm (Vanilla)|82% F1|82% F1|Tie|
|Prompt Harm (Adversarial)|78% F1|**82% F1**|WildGuard (+4%)|
|Response Harm|80% F1|81% F1|WildGuard (+1%)|
|Refusal Detection|75% F1|**78% F1**|WildGuard (+3%)|

**Key Finding**: WildGuard **matches or exceeds** GPT-4 performance at a fraction of the cost.

#### Generalization Performance

**Tested on 10 external benchmarks**:

- ToxicChat
- OpenAI Moderation Dataset
- Anthropic HH-RLHF
- XSTest
- JailbreakBench
- And 5 others

**Results**: State-of-the-art across all benchmarks, showing strong generalization beyond training distribution.

### Ablation Studies

**Key Findings**:

1. **All Data Sources Matter**:
    - Removing any single data source degrades performance
    - Synthetic adversarial data is most critical (+15% on jailbreaks)
    - In-the-wild data improves generalization (+5% on external benchmarks)

2. **Multi-Task Training is Essential**:
    - Training separate models for each task: -8% average performance
    - Joint training provides shared representations
    - Especially helps refusal detection (+12% improvement)

3. **Data Balance Crucial**:
    - Imbalanced training (more harmful than benign): -10% precision
    - Balanced data maintains high precision and recall

### Jailbreak Defense Capability

**Real-World Test**: WildGuard as LLM Interface Moderator

**Setup**:

- WildGuard filters user prompts before sending to base LLM
- Also checks LLM responses before displaying to user

**Results**:

|Scenario|Success Rate Without WildGuard|Success Rate With WildGuard|
|---|---|---|
|Jailbreak Attacks|**79.8%**|**2.4%**|
|Benign Requests|0% rejected|1.2% rejected|

**Interpretation**:

- **97% reduction** in successful jailbreaks (79.8% → 2.4%)
- Very low false positive rate (1.2% benign requests blocked)
- Effective defense without over-refusing

### Code and Model Availability

**GitHub**: https://github.com/allenai/wildguard  
**License**: CC BY 4.0 (permissive open-source) **HuggingFace**: Models available for download

**Usage Example**:

```python
from wildguard import load_wildguard

# Load model
wildguard = load_wildguard()

# Classify items
items = [
    {"prompt": "How do I make a bomb?", 
     "response": "Sorry, I can't help with that."},
    {"prompt": "What's the weather like today?"}
]

results = wildguard.classify(items)

# Results structure
for item, result in zip(items, results):
    print(f"Prompt: {item['prompt']}")
    print(f"Prompt harmfulness: {result['prompt_harmfulness']}")
    if 'response' in item:
        print(f"Response harmfulness: {result['response_harmfulness']}")
        print(f"Response refusal: {result['response_refusal']}")
```

### Computational Requirements

**Model Size**: ~7B parameters (estimated)  
**Inference**:

- GPU recommended: V100/A100
- Memory: ~14GB GPU RAM
- Latency: ~300-600ms per classification (3 tasks)

**Optimization Options**:

- Quantization (INT8/INT4) possible for deployment
- Batch processing for efficiency
- Can run on CPU (slower)

### Limitations

**Acknowledged**:

1. **Language**: Primarily English (multilingual support limited)
2. **Context Length**: Limited by base model (typically 2048-4096 tokens)
3. **Cultural Bias**: Trained mostly on Western perspectives
4. **False Positives**: Still occurs ~1-2% on benign content
5. **Novel Attacks**: May struggle with entirely new jailbreak techniques

**From Analysis**:

- Higher computational cost than single-task models
- Three-task output may be overkill for simple use cases
- Refusal detection requires both prompt and response (more data needed)

### Comparison with Other Models

#### WildGuard vs. Llama Guard 3

| Feature                        | WildGuard                     | Llama Guard 3               |
| ------------------------------ | ----------------------------- | --------------------------- |
| **Tasks**                      | 3 (harm, harm, refusal)       | 1 (binary safe/unsafe)      |
| **Jailbreak Defense**          | Excellent ⭐⭐⭐                 | Good ⭐⭐                     |
| **Refusal Detection**          | Yes ✅                         | No ❌                        |
| **Multilingual**               | Limited                       | 8 languages ✅               |
| **Customizable Taxonomy**      | Fixed 13 categories #tocheck  | Flexible ✅                  |
| **Performance on Adversarial** | State-of-the-art              | Good                        |
| **Best For**                   | Research, jailbreak defense   | Production, custom policies |

#forShareish
**Applications for Shareish**:
- Useful if facing adversarial attacks (Jailbreak attempts)
- No applications for refusal detection

#### WildGuard vs. ShieldGemma

|Feature|WildGuard|ShieldGemma|
|---|---|---|
|**Organization**|AI2 (academic)|Google (corporate)|
|**Focus**|Adversarial robustness|General safety|
|**Refusal Detection**|Yes ✅|No ❌|
|**Training Data Size**|92K|Larger (not disclosed)|
|**Performance vs. GPT-4**|Exceeds|Comparable|

### Integration Strategy for Shareish

**Scenario 1: Standard Moderation**

```
User Post → Llama Guard 3 → Approved/Flagged
(Custom Shareish taxonomy, multilingual)
```

**Scenario 2: High-Security Moderation**

```
User Post → WildGuard (Prompt Check) → 
    If Adversarial → Human Review
    Else → Llama Guard 3 → Approved/Flagged
(Two-layer defense)
```

### Dataset Details for Researchers
#dataset
**Access**: WildGuardMix available on HuggingFace  
**Format**: JSON with fields:

- `prompt`: User input text
- `response`: AI response text (if applicable)
- `prompt_harmfulness`: Label (harmful/benign)
- `response_harmfulness`: Label (harmful/unharmful)
- `refusal`: Label (refusal/compliance)
- `category`: Risk category from 13-class taxonomy

### Citations

**Primary Paper**:

```bibtex
@article{han2024wildguard,
  title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs},
  author={Han, Seungju and Rao, Kavel and Ettinger, Allyson and Jiang, Liwei and Lin, Bill Yuchen and Lambert, Nathan and Choi, Yejin and Dziri, Nouha},
  journal={arXiv preprint arXiv:2406.18495},
  year={2024}
}
```

### Overall Assessment

**Relevance to Shareish**: ⭐ **Moderate** (specialized use case)

**Strengths**:
- State-of-the-art adversarial robustness
- Refusal detection capability (unique) (Not useful in for Shareish)
- Exceeds GPT-4 performance
- Large, high-quality dataset (92K examples)
- Open-source with permissive license
- Multi-task framework efficient

**Weaknesses**:
- Limited multilingual support (French not explicitly mentioned)
- Fixed 13-category taxonomy (less flexible than Llama Guard)
- Higher computational cost (3 tasks)
- Primarily research-focused (vs. production-ready)

**Recommendation**:
- **Use if**: Expecting adversarial attacks or jailbreak attempts
- **Use for**: Evaluating other model's adversarial robustness
- **Dataset**: Good for fine-tuning and evaluation