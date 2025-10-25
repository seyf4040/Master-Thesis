# Few-Shot Learning: Concepts and Methods

**Table of Contents**

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Few-Shot Learning Methods](#few-shot-learning-methods)
    - [Metric-Based Methods](#metric-based-methods)
    - [Model-Based Methods](#model-based-methods)
    - [Optimization-Based Methods](#optimization-based-methods)
    - [Prompting-Based Methods (for LLMs)](#prompting-based-methods-for-llms)
- [Application to Content Moderation](#application-to-content-moderation)
- [Practical Implementation Considerations](#practical-implementation-considerations)
- [References](#references)

---

## Introduction

Few-shot learning (FSL) is a machine learning paradigm that enables models to learn new tasks or classes from a very limited number of training examples. Unlike traditional supervised learning which requires thousands of labeled samples, few-shot learning can generalize from just a handful of examples (typically 1-10 samples per class).

**Why Few-Shot Learning?**

- Addresses the cold-start problem when new platforms have limited labeled data
- Reduces annotation costs and effort
- Enables rapid adaptation to new content types or policy changes
- Mimics human learning capability (humans can learn concepts from few examples)

**Common Terminology:**

- **N-way K-shot**: A task with N classes and K examples per class
- **Support set**: The small set of labeled examples provided for learning
- **Query set**: The unlabeled examples the model must classify
- **Meta-learning**: Learning to learn; training on multiple tasks to enable quick adaptation

---

## Core Concepts

### The Few-Shot Learning Problem

Traditional machine learning assumes abundant labeled data. Few-shot learning operates under extreme data scarcity:

```
Traditional ML:  Thousands of examples → Train model → Predict
Few-Shot ML:     5-10 examples → Adapt/Learn → Predict
```

### Episode-Based Training

Few-shot learning models are typically trained using episodic training:

1. Sample a small support set from available classes
2. Sample a query set from the same classes
3. Model learns to classify query examples using only the support set
4. Repeat with different random class combinations

This forces the model to learn how to learn from few examples rather than memorizing specific classes.

**Source:** Vinyals, O., Blundell, C., Lillicrap, T., & Wierstra, D. (2016). Matching networks for one shot learning. _Advances in Neural Information Processing Systems_, 29. https://arxiv.org/abs/1606.04080

---

## Few-Shot Learning Methods

### Metric-Based Methods

These methods learn an embedding space where similar examples cluster together. Classification is performed by measuring similarity (distance) to support set examples.

#### 1. **Siamese Networks**

Siamese networks learn to map inputs into an embedding space where similarity can be measured directly.

**Architecture:**

- Two identical neural networks (shared weights)
- Learn an embedding function that brings similar examples closer
- Use contrastive loss or triplet loss

**How it works:**

```
Input pairs → [Shared Encoder] → Embeddings → Distance Metric → Similarity Score
```

**Implementation approach:**

```python
# Pseudo-code structure
def siamese_network(input_a, input_b):
    embedding_a = shared_encoder(input_a)
    embedding_b = shared_encoder(input_b)
    distance = euclidean_distance(embedding_a, embedding_b)
    return similarity_score(distance)
```

**Source:** Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. _ICML Deep Learning Workshop_. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

#### 2. **Prototypical Networks**

Prototypical networks create a prototype representation for each class by averaging embeddings of support examples, then classify query examples based on distance to prototypes.

**Key idea:**

- Each class is represented by a single prototype (mean of support embeddings)
- Classification is based on nearest prototype

**Process:**

```
1. Embed all support examples
2. Compute class prototypes (mean of embeddings per class)
3. Embed query example
4. Assign to nearest prototype class
```

**Source:** Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. _Advances in Neural Information Processing Systems_, 30. https://arxiv.org/abs/1703.05175

#### 3. **Matching Networks**

Matching networks use attention mechanisms to compare query examples against the entire support set, enabling classification through weighted nearest neighbors.

**Distinctive features:**

- Uses attention over support set
- Employs fully contextual embeddings (each example embedding depends on others)
- Differentiable nearest neighbor classifier

**Source:** Vinyals, O., Blundell, C., Lillicrap, T., & Wierstra, D. (2016). Matching networks for one shot learning. _Advances in Neural Information Processing Systems_, 29. https://arxiv.org/abs/1606.04080

#### 4. **Relation Networks**

Instead of using fixed distance metrics (Euclidean, cosine), relation networks learn a deep neural network to compute similarity between samples.

**Innovation:**

- Learnable similarity metric (not hand-crafted)
- More flexible than fixed distance functions

**Source:** Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H., & Hospedales, T. M. (2018). Learning to compare: Relation network for few-shot learning. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 1199-1208. https://arxiv.org/abs/1711.06025

---

### Model-Based Methods

These approaches use models with internal memory or rapid parameter adaptation to quickly learn from few examples.

#### 1. **Memory-Augmented Neural Networks (MANN)**

MANNs extend neural networks with external memory mechanisms, enabling rapid encoding and retrieval of new information.

**Key components:**

- External memory matrix
- Attention-based read/write operations
- Designed to quickly store and retrieve support set information

**Notable implementation:** Neural Turing Machines, Differentiable Neural Computers

**Source:** Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016). Meta-learning with memory-augmented neural networks. _International Conference on Machine Learning_, 1842-1850. https://arxiv.org/abs/1605.06065

#### 2. **Meta Networks**

Meta networks consist of a base learner and a meta learner that provides fast weights for rapid adaptation.

**Architecture:**

- Base learner: Task-specific network
- Meta learner: Generates parameters for base learner from few examples

**Source:** Munkhdalai, T., & Yu, H. (2017). Meta networks. _International Conference on Machine Learning_, 2554-2563. https://arxiv.org/abs/1703.00837

---

### Optimization-Based Methods

These methods explicitly optimize for rapid adaptation through careful algorithm design.

#### 1. **Model-Agnostic Meta-Learning (MAML)**

MAML is one of the most influential few-shot learning algorithms. It learns initial model parameters that can be quickly fine-tuned with few gradient steps.

**Core principle:**

- Find initialization that is sensitive to small changes in task
- Few gradient steps from this initialization lead to good task-specific models

**Process:**

```
1. Initialize model parameters θ
2. For each task:
   a. Sample support set
   b. Compute adapted parameters with few gradient steps
   c. Evaluate on query set
3. Update θ based on query set performance across tasks
```

**Advantages:**

- Model-agnostic (works with any gradient-based model)
- Simple and elegant conceptually
- Strong empirical performance

**Source:** Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. _International Conference on Machine Learning_, 1126-1135. https://arxiv.org/abs/1703.03400

#### 2. **Reptile**

Reptile is a simpler alternative to MAML that performs well with less computational cost.

**Difference from MAML:**

- Doesn't require computing second-order derivatives
- Moves parameters toward task-specific parameters after training

**Source:** Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. _arXiv preprint_. https://arxiv.org/abs/1803.02999

---

### Prompting-Based Methods (for LLMs)

Modern large language models enable few-shot learning through in-context learning with prompts.

#### 1. **In-Context Learning (ICL)**

LLMs can perform tasks by providing examples directly in the prompt without any parameter updates.

**How it works:**

```
Prompt structure:
[Task description]

Example 1: [input] → [output]
Example 2: [input] → [output]
...
Example K: [input] → [output]

Now classify: [new input] → 
```

**Advantages:**

- No training required
- Instant adaptation
- Works across diverse tasks

**Source:** Brown, T. B., et al. (2020). Language models are few-shot learners. _Advances in Neural Information Processing Systems_, 33, 1877-1901. https://arxiv.org/abs/2005.14165

#### 2. **Chain-of-Thought (CoT) Few-Shot**

Enhances few-shot prompting by including reasoning steps in examples.

**Structure:**

```
Example 1:
Input: [text]
Reasoning: [step-by-step analysis]
Output: [classification]
```

**Source:** Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_, 35, 24824-24837. https://arxiv.org/abs/2201.11903

#### 3. **SetFit: Efficient Few-Shot Learning**

SetFit combines sentence transformers with contrastive learning for highly efficient few-shot text classification.

**Process:**

1. Fine-tune sentence transformer on few labeled examples
2. Generate sentence embeddings
3. Train simple classifier (e.g., logistic regression) on embeddings

**Advantages:**

- Extremely sample-efficient (8-64 examples)
- No prompts required
- Fast training and inference

**Source:** Tunstall, L., et al. (2022). Efficient few-shot learning without prompts. _arXiv preprint_. https://arxiv.org/abs/2209.11055

**Implementation:** Open-source at https://github.com/huggingface/setfit

---

## Application to Content Moderation

Few-shot learning is particularly relevant for content moderation on Shareish due to the cold-start problem and limited initial labeled data.

### Why Few-Shot Learning for Content Moderation?

**Challenges addressed:**

1. **Limited initial data**: New platforms don't have thousands of moderated examples
2. **Evolving policies**: Rules change, requiring quick adaptation without extensive retraining
3. **New violation types**: Emerging harmful content patterns need rapid detection
4. **Multilingual support**: Limited labeled data in some languages (e.g., French)
5. **Cost efficiency**: Reduces annotation burden on moderators

### Recommended Approaches for Shareish

#### 1. **Hybrid: SetFit + Rule-Based System**

**Rationale:**

- SetFit excels at text classification with 8-64 examples
- Open-source and actively maintained
- Can complement rule-based policies

**Implementation strategy:**

```
Initial deployment:
1. Define clear moderation categories
2. Collect 20-30 examples per category from:
   - Similar platforms (with appropriate licenses)
   - Synthetic examples
   - Initial manual moderation
3. Fine-tune SetFit model
4. Deploy alongside rule-based checks
5. Continuously collect real moderation decisions
6. Retrain periodically with growing dataset
```

**Source:** Tunstall, L., et al. (2022). Efficient few-shot learning without prompts. https://arxiv.org/abs/2209.11055

#### 2. **Prototypical Networks for Image Moderation**

For visual content moderation (profile pictures, shared images):

**Approach:**

- Pre-trained image encoder (e.g., CLIP, ResNet)
- Prototypical network for few-shot classification
- Categories: safe, NSFW, violent, spam/scam

**Source:** Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. https://arxiv.org/abs/1703.05175

#### 3. **LLM In-Context Learning (Zero/Few-Shot)**

For complex policy interpretation and edge cases:

**Strategy:**

- Use open LLMs (e.g., Llama, Mistral) for privacy
- Design prompts with few examples of policy violations
- Handle nuanced cases that rules might miss

**Example prompt structure:**

```
You are a content moderator for Shareish, a solidarity platform.

Policy: [Brief policy description]

Violations include:
- Example 1: "[text]" → Violates because [reason]
- Example 2: "[text]" → Safe because [reason]
- Example 3: "[text]" → Violates because [reason]

Classify: "[new text]"
Analysis: [reasoning]
Decision: [safe/review/remove]
```

**Source:** Brown, T. B., et al. (2020). Language models are few-shot learners. https://arxiv.org/abs/2005.14165

### Feedback Loop Architecture

Critical for continuous improvement with limited initial data:

```
User posts content
    ↓
AI system evaluates (few-shot model + rules)
    ↓
Decision: Auto-approve / Flag for review / Auto-remove
    ↓
If flagged → Human moderator reviews
    ↓
Moderator decision becomes training data
    ↓
Periodically retrain model with accumulated data
    ↓
Model improves over time
```

---

## Practical Implementation Considerations

### Data Requirements

**Initial dataset (per category):**

- Minimum: 5-10 examples (true few-shot)
- Recommended: 20-50 examples (better reliability)
- Optimal: 100+ examples (approaching standard supervised learning)

**Data quality over quantity:**

- Diverse examples covering edge cases
- Clear, unambiguous labels
- Representative of real platform content

### Evaluation Metrics

For content moderation, consider:

- **Precision**: Avoid false positives (don't remove legitimate content)
- **Recall**: Catch actual violations
- **F1-Score**: Balance between precision and recall
- **Human agreement**: Alignment with human moderator decisions

Reference the project's existing document on [[Metrics Choice]] for detailed discussion.

### Ethical Considerations

1. **Transparency**: Users should understand moderation decisions
2. **Bias mitigation**: Few examples can amplify biases; ensure diverse examples
3. **Human oversight**: Always maintain human review for uncertain cases
4. **Privacy**: Use on-device or self-hosted models when possible (GDPR compliance)

**Source:** Gorwa, R., Binns, R., & Katzenbach, C. (2020). Algorithmic content moderation: Technical and political challenges in the automation of platform governance. _Big Data & Society_, 7(1). https://doi.org/10.1177/2053951719897945

### Computational Resources

**Efficient approaches for limited resources:**

- SetFit: Runs on CPU, minimal requirements
- Prototypical Networks: Lightweight after pre-training
- Smaller LLMs (7B parameters): Can run locally with quantization

**Recommended setup:**

- GPU: Optional but beneficial (NVIDIA GTX 1080 or better)
- RAM: 16GB minimum, 32GB recommended for LLMs
- Storage: SSD for faster data loading

---

## References

### Core Few-Shot Learning Papers

1. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. _ICML Deep Learning Workshop_. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    
2. Vinyals, O., Blundell, C., Lillicrap, T., & Wierstra, D. (2016). Matching networks for one shot learning. _Advances in Neural Information Processing Systems_, 29. https://arxiv.org/abs/1606.04080
    
3. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. _Advances in Neural Information Processing Systems_, 30. https://arxiv.org/abs/1703.05175
    
4. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. _International Conference on Machine Learning_, 1126-1135. https://arxiv.org/abs/1703.03400
    
5. Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H., & Hospedales, T. M. (2018). Learning to compare: Relation network for few-shot learning. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 1199-1208. https://arxiv.org/abs/1711.06025
    

### LLM and Modern Approaches

6. Brown, T. B., et al. (2020). Language models are few-shot learners. _Advances in Neural Information Processing Systems_, 33, 1877-1901. https://arxiv.org/abs/2005.14165
    
7. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_, 35, 24824-24837. https://arxiv.org/abs/2201.11903
    
8. Tunstall, L., et al. (2022). Efficient few-shot learning without prompts. _arXiv preprint_. https://arxiv.org/abs/2209.11055
    
    - GitHub: https://github.com/huggingface/setfit

### Memory and Meta-Learning

9. Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016). Meta-learning with memory-augmented neural networks. _International Conference on Machine Learning_, 1842-1850. https://arxiv.org/abs/1605.06065
    
10. Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. _arXiv preprint_. https://arxiv.org/abs/1803.02999
    

### Content Moderation Context

11. Gorwa, R., Binns, R., & Katzenbach, C. (2020). Algorithmic content moderation: Technical and political challenges in the automation of platform governance. _Big Data & Society_, 7(1). https://doi.org/10.1177/2053951719897945

### Survey Papers

12. Wang, Y., Yao, Q., Kwok, J. T., & Ni, L. M. (2020). Generalizing from a few examples: A survey on few-shot learning. _ACM Computing Surveys_, 53(3), 1-34. https://arxiv.org/abs/1904.05046
    
13. Hospedales, T., Antoniou, A., Micaelli, P., & Storkey, A. (2021). Meta-learning in neural networks: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 44(9), 5149-5169. https://arxiv.org/abs/2004.05439
    

---

_This document provides a foundation for understanding few-shot learning techniques applicable to the Shareish content moderation system. For implementation details, refer to the linked papers and open-source repositories._