# Predicting the Type and Target of Offensive Posts in Social Media
**Website**: https://paperswithcode.com/paper/predicting-the-type-and-target-of-offensive  
**Related to**: OLID Dataset (Offensive Language Identification Dataset)

### Introduction
Extends offensive language detection to include:
1. **Type of offense**: Targeted vs. untargeted
2. **Target category**: Individual, group, other

### OLID Dataset Taxonomy
**Level A**: Is post offensive?
- Offensive (OFF)
- Not offensive (NOT)

**Level B** (if offensive): Type of offense
- Targeted insult (TIN)
- Untargeted profanity (UNT)

**Level C** (if targeted): Target of insult
- Individual (IND)
- Group (GRP)  
- Other (OTH)

### Approach
**Hierarchical Classification:**
- First classify: offensive vs. not
- Then classify: type of offense
- Finally classify: target category

**Models:**
- BERT-based classifiers
- Bi-LSTM with attention
- Ensemble methods

### Dataset Details
**OLID (Offensive Language Identification Dataset):**
- 14,100 tweets
- Hierarchical annotations
- Available on GitHub (free with citation)
- Widely used benchmark

### Evaluation
Separate metrics for each classification level:
- Level A (offensive detection): F1 ~ 0.80
- Level B (type): F1 ~ 0.68
- Level C (target): F1 ~ 0.47 (most challenging)

### Overall
**Multi-label approach** more sophisticated than binary classification. Understanding **type and target** enables:
- Priority ranking (targeted harassment worse than general profanity)
- Context-specific rules (different thresholds for different categories)
- Better user reporting (what type of content was violated)

**Limitation**: Twitter-specific, English-only

**Relevance**: High - hierarchical classification could improve Shareish moderation granularity.