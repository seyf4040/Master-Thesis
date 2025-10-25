# HuggingFace Model Performance

---

**HuggingFace**: A platform where one can find AI models for various applications, including natural language processing and content moderation.

This report documents the testing and evaluation of different content moderation models available on the HuggingFace platform. The focus is to compare their performance across multiple datasets, analyzing how well they handle toxic or unsafe content without any initial fine-tuning.

## Testing Procedure

- All models will be systematically evaluated across all selected datasets.
- **No fine-tuning** will be applied in this phase to ensure a consistent baseline comparison.
- Model outputs will be standardized to a **binary decision**: 'fine' or 'removed' ('toxic', 'non-toxic', ...). This allows for uniform performance metrics across different datasets and models.
- A sample will be classified as 'removed' if the probability of the unsafe/toxic category exceeds a **threshold of 0.5**.
- **Key Metrics**:
    - **AUC (Area Under Curve)**: Assesses the model’s ability to distinguish between safe and unsafe content.
    - **Accuracy**: Measures the proportion of correct predictions.

## Datasets

### 1. Kaggle Toxic Comment Challenge

- **Overview**: A well-known dataset for testing AI moderation systems, containing user comments labeled across multiple categories of toxicity.
- **Categories**: Toxic, severe toxic, obscene, threat, insult, identity hate.
- **Processing**: Annotations were filtered and harmonized into a binary label system (fine/removed).
- **Objective**: Evaluate the effectiveness of models in accurately identifying various forms of toxicity.

### 2. Wikipedia Attack

- **Overview**: A dataset derived from Wikipedia discussions, containing samples labeled as personal attacks or non-attacks.
- **Categories**: Attack, non-attack.
- **Processing**: Multi-annotation entries were consolidated using a majority voting system, with a customizable threshold for 'attack' classification.
- **Objective**: Assess model performance in detecting personal attacks within a high-context environment.

### 3. KoalaAI - Text Classification (empty ?)

- **Overview**: A content moderation dataset from KoalaAI, aimed at classifying text into safe and unsafe categories.
- **Categories**: Multiple, including harassment (H), hate speech (SH), violence (V), and other harmful types.
- **Processing**: Mapped various unsafe categories to a unified binary classification of fine/removed.
- **Objective**: Test how well models can generalize across a diverse range of harmful content types.

---

## Models Under Evaluation

### 1. **EthicalEye** (autopilot-ai/EthicalEye)

- **Overview**: A model specifically designed for detecting unsafe content in online comments and discussions.
- **Categories**: Safe, Un-safe.
- **Testing Details**: Predictions are based on the model's probability output for 'Un-safe', with binary classification using the 0.5 threshold.

### 2. **KoalaAI Model**

- **Overview**: A model trained for fine-grained content moderation, targeting multiple harmful content categories.
- **Categories**: H, SH, V, S, HR, V2, S3, H2 (various forms of unsafe content).
- **Testing Details**: Outputs are filtered to focus on the highest probability of any non-safe label, with the 0.5 threshold standard.

### 3. **CitizenLab Model**

- **Overview**: A binary toxicity classifier built to distinguish between toxic and non-toxic content.
- **Categories**: Toxic, Non-toxic.
- **Testing Details**: A separate prediction logic due to its binary output structure, still adhering to the 0.5 decision threshold.

---

## Preliminary Results

Model Performance Metrics:
### Wikipedia attack

|     | Model      |      AUC | Accuracy |
| --: | :--------- | -------: | -------: |
|   0 | EthicalEye | 0.880904 |    0.968 |
|   1 | KoalaAI    | 0.821862 |     0.99 |
|   2 | citizenlab | 0.803306 |    0.988 |

### Kaggle Toxic Comment Classification Challenge

|     | Model      |      AUC | Accuracy |
| --: | :--------- | -------: | -------: |
|   0 | EthicalEye | 0.977854 |    0.874 |
|   1 | KoalaAI    | 0.928208 |    0.922 |
|   2 | citizenlab | 0.758994 |    0.916 |

---

## Observations

### Wikipedia Attack Dataset

- **EthicalEye** had the highest AUC (**0.88**) with good accuracy (**96.8%**), indicating effective distinction between categories.
- **KoalaAI** had a slightly lower AUC (**0.82**) but excelled in accuracy (**99%**), indicating reliable classification despite less nuance.
- **CitizenLab** showed the lowest AUC (**0.80**) but still had a strong accuracy (**98.8%**).

### Kaggle Toxic Comment Classification Challenge

- **EthicalEye** led with an AUC of **0.98**, though accuracy was lower at **87.4%**.
- **KoalaAI** performed consistently well, with an AUC of **0.93** and accuracy of **92.2%**.
- **CitizenLab** had a lower AUC (**0.76**) but maintained solid accuracy at **91.6%**.

## Summery

1. **EthicalEye** is best for nuanced detection, leading in AUC on both datasets.
2. **KoalaAI** offers the highest accuracy, particularly for simpler binary classifications.
3. **CitizenLab** is reliable for general accuracy but needs refinement for subtle distinctions.

---
## Next Steps

- Consider fine-tuning some models on specific datasets to boost performance in targeted scenarios.
- Expand the dataset pool to include more **diverse content types** (e.g., social media data, news comments).
- Further optimize prediction functions to reduce computation time and enhance real-time evaluation capabilities.
- Explore additional metrics like **Precision, Recall, and F1 Score** for a deeper analysis of model strengths and weaknesses.

---

## Conclusion

This phase has established a clear baseline for comparing content moderation models available on HuggingFace across a standardized set of datasets. Results demonstrate variability in model performance depending on the dataset context, reinforcing the need for further refinement and dataset-specific optimizations.

