# Evaluation Framework 

**Master's Thesis**: Deep Learning for Content Moderation on Shareish  
**Student**: Seyfullah Ural | **Date**: October 2025

---
```table-of-contents
title: ## 📋 Table of Contents
minLevel:2
maxLevel:2
```
---

## 1. Framework Philosophy

**Key Principle**: _"Accuracy is necessary but insufficient for content moderation legitimacy."_

Moving beyond traditional accuracy-centric evaluation to include:

- **Technical Performance** (Does it work?)
- **Legitimacy** (Is it fair and transparent?)
- **Operational Efficiency** (Is it fast and affordable?)
- **Robustness** (Does it handle edge cases?)

---

## 2. Proposed Evaluation Framework

### 2.1 Five Evaluation Dimensions

```
┌────────────────────────────────────────────────┐
│         SHAREISH EVALUATION FRAMEWORK          │
├────────────────────────────────────────────────┤
│  1. Technical Performance (Classification)     │
│  2. Legitimacy (Fairness & Transparency)       │
│  3. Operational Efficiency (Speed & Cost)      │
│  4. Functional Robustness (Edge Cases)         │
│  5. Real-World Impact (User Experience)        │
└────────────────────────────────────────────────┘
```

---

### 2.2 Dimension 1: Technical Performance

**Primary Metrics:**

|Metric|Target|Why|
|---|---|---|
|**F1 Score**|≥ 0.82|Balance precision/recall (most papers use this)|
|**Precision**|≥ 0.85|Minimize false positives (user trust)|
|**Recall**|≥ 0.80|Catch violations (platform safety)|
|**MCC**|≥ 0.65|Better for imbalanced data|

**Per-Category F1**: evaluate each violation type separately

---

### 2.3 Dimension 2: Legitimacy Metrics

Based on "Content Moderation by LLM: From Accuracy to Legitimacy"

#### Consistency
**Test**: Same input → Same output  
**Method**: 100 samples × 5 runs  
**Target**: ≥ 95% identical outputs

#### Fairness
**Test**: Equal performance across demographic groups  
**Metrics**:
- False Positive Rate disparity: < 0.10
- False Negative Rate disparity: < 0.10
- Special attention to reclaimed slurs (< 20% FPR)

**Test Groups**: Religious groups, ethnicities, LGBTQ+, gender, French dialects

#### Explainability
**Requirement**: 100% of flagged content has explanation  
**Format**:

```
Decision: Flagged
Reason: Hate speech targeting [group]
Specific phrase: [highlighted text]
Confidence: 87%
```

#### Transparency

**Metric**: Expected Calibration Error (ECE) < 0.10  
**Meaning**: If model says 80% confident, it should be right ~80% of the time

---

### 2.4 Dimension 3: Operational Efficiency

|Metric|Target|Why|
|---|---|---|
|**Latency (p95)**|< 500ms|User experience|
|**Cost per 1K posts**|< €0.05|Small platform constraint|
|**Human workload reduction**|≥ 60%|Scalability|

**Two-Tier Architecture Cost Analysis:**

```
90% posts → Detoxify (€0.001/1K) = €0.0009
10% posts → Llama Guard (€0.05/1K) = €0.005
Total: €0.006 per 1,000 posts (vs €0.05 Llama-only)
Savings: 88%
```

---

### 2.5 Dimension 4: Functional Robustness

#### HateCheck Functional Testing

**Framework**: 29 specific functionalities  
**Target**: Pass ≥ 25/29 (pass = 70% accuracy per functionality)

**Key Weaknesses to Address:**

- F9: Reclaimed slurs (50% → target 75%)
- F13-F16: Negation handling (55% → target 80%)
- F17-F20: Spelling variations (45% → target 75%)

#### Implicit Toxicity (ToxiGen Dataset)

|Metric|Baseline|Target|
|---|---|---|
|F1 on implicit hate|~0.65|≥ 0.75|
|FPR on benign identity mentions|~12%|< 7%|

#### Adversarial Robustness (WildGuard)

- Jailbreak success rate: < 10%
- False positives on benign edge cases: < 3%

---

### 2.6 Dimension 5: Real-World Impact

**User Experience:**
- User trust score (survey): ≥ 3.8/5
- Appeal success rate: 15-25% (healthy system)
- Time to appeal resolution: < 24 hours

**Moderator Impact:**
- AI-human agreement: ≥ 85%
- Human review load: 8-15% of posts (down from 100%)
- High-quality escalations: ≥ 80%

**Platform Health:**
- User reports of toxic content: -40%
- Community satisfaction: +20%

---

## 3. Summary of Metrics from Papers

### 3.1 Most Common Metrics

|Metric Category|Usage|Shareish Adoption|
|---|---|---|
|**Precision, Recall, F1**|88% of papers|✅ Primary metrics|
|**ROC-AUC, PR-AUC**|53% of papers|✅ Secondary|
|**Fairness metrics**|18% of papers|✅ Legitimacy requirement|
|**Latency/Cost**|24% / 9%|✅ Operational constraint|
|**Functional testing**|6% (HateCheck)|✅ Critical for robustness|

### 3.2 Key Papers and Their Metrics

|Paper|Year|Primary Metrics|Performance|
|---|---|---|---|
|**Traditional ML baseline**|2018|F1|~0.70|
|**GPT-3.5** (Watch Your Language)|2024|F1, Precision, Recall|0.72-0.75|
|**GPT-4**|2024|F1|0.75|
|**Llama Guard 3**|2024|F1 (estimated)|~0.80|
|**ShieldGemma**|2024|F1|0.75-0.85|
|**HateCheck**|2021|Pass rate (29 functionalities)|Typical: 15-20/29|
|**ToxiGen**|2022|F1 improvement on implicit|+8% baseline|
|**WildGuard**|2024|Jailbreak success rate|2.4% (vs 79.8% baseline)|

**Key Finding**: LLMs outperform traditional ML by +10-20% F1, but accuracy alone is insufficient.

---

## 4. Implementation Plan

### Phased Evaluation (16 Weeks)

**Phase 1: Baseline (Weeks 1-2)**
- Datasets: HateCheck French, ToxiGen sample, Shareish pilot (200-500)
- Metrics: F1, Precision, Recall, HateCheck pass rate, Latency, Cost
- Baselines: Detoxify, Perspective API, Llama Guard 3 (zero-shot)

**Phase 2: Optimization (Weeks 3-10)**
- Fine-tune on ToxiGen (274K) + HateCheck weaknesses
- Implement two-tier (Detoxify → Llama Guard)
- Target: F1 +10-15%, HateCheck 25+/29, Latency <100ms avg

**Phase 3: Legitimacy (Weeks 11-12)**
- Consistency testing (100 samples × 5 runs)
- Fairness analysis (demographic groups)
- Explainability implementation
- Calibration analysis

**Phase 4: Adversarial Testing (Week 13)**
- HateCheck edge cases
- WildGuard jailbreak attempts
- Shareish-specific adversarial examples

**Phase 5: Real-World Pilot (Weeks 14-16)**
- Shadow mode deployment
- Measure AI-human agreement, user feedback
- Success: ≥85% agreement, ≥60% workload reduction

---

## 5. Quick Reference Tables

### 5.1 Metric Definitions

|Metric|Formula|Use Case|
|---|---|---|
|Precision|TP/(TP+FP)|Minimize false positives|
|Recall|TP/(TP+FN)|Catch all violations|
|F1|2×(P×R)/(P+R)|Balanced performance|
|MCC|(TP×TN-FP×FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]|Imbalanced data|
|ECE|Expected Calibration Error|Confidence accuracy|

### 5.2 Performance Targets Summary

|Dimension|Key Metric|Target|Priority|
|---|---|---|---|
|**Technical**|F1 Score|≥ 0.82|P0|
|**Technical**|Per-category F1|≥ 0.80-0.90|P0|
|**Legitimacy**|Consistency|≥ 95%|P1|
|**Legitimacy**|FPR disparity|< 0.10|P1|
|**Legitimacy**|Explainability|100% coverage|P1|
|**Operational**|Latency (p95)|< 500ms|P1|
|**Operational**|Cost per 1K|< €0.05|P1|
|**Robustness**|HateCheck pass|≥ 25/29|P0|
|**Robustness**|Implicit F1|≥ 0.75|P1|
|**Impact**|Human agreement|≥ 85%|P2|
|**Impact**|Workload reduction|≥ 60%|P2|

---

## 6. Key Takeaways

1. **Beyond Accuracy**: Use multi-dimensional evaluation (technical + legitimacy + operational + robustness + impact)
    
2. **Primary Metrics**: F1 score, HateCheck pass rate, FPR disparity, Latency, Cost
    
3. **Literature Insight**: Modern LLMs achieve F1 ≈ 0.75-0.85, but 88% of papers use only 1 metric (insufficient)
    
4. **Critical Gap**: Most papers ignore fairness, consistency, and explainability - these are required for legitimacy

---

## References

**Key Papers Informing Framework:**

- Content Moderation by LLM: From Accuracy to Legitimacy (2024)
- HateCheck: Functional Tests for Hate Speech Detection (2021)
- Watch Your Language: Investigating CM with LLMs (2024)
- The Oversight of Content Moderation by AI (2024)
- ToxiGen: Adversarial and Implicit Hate Speech Detection (2022)
- WildGuard: Open One-Stop Moderation Tools (2024)
- Llama Guard 3: Multilingual Safety Classifier (2024)
- Critical Analysis of Metrics in AI (2020)