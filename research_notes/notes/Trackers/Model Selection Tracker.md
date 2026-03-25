# Model Selection Tracker

**Master's Thesis**: Deep Learning for Content Moderation on Shareish  
**Student**: Seyfullah Ural  
**Phase**: Model Selection & Baseline Evaluation  
**Last Updated**: [DATE]

---

## 📋 Quick Status Dashboard

| Model                | Status       | Technical | Legitimacy | Efficiency | Robustness | Final Score |
| -------------------- | ------------ | --------- | ---------- | ---------- | ---------- | ----------- |
| **Detoxify Multi**   | ⬜ Not tested | -         | -          | -          | -          | -           |
| **Llama Guard 3-8B** | ⬜ Not tested | -         | -          | -          | -          | -           |
| **ShieldGemma 7B**   | ⬜ Not tested | -         | -          | -          | -          | -           |
| Mistral              | ⬜ Not tested | -         | -          | -          | -          | -           |
| **Two-Tier Hybrid**  | ⬜ Not tested | -         | -          | -          | -          | -           |

**Legend**: ✅ Passed | ⚠️ Needs improvement | ❌ Failed | ⬜ Not tested

---

## 1. Technical Performance Results

### 1.1 Standard Classification Metrics

| Model                | Precision | Recall | F1 Score | MCC | Target Met? |
| -------------------- | --------- | ------ | -------- | --- | ----------- |
| **Detoxify Multi**   |           |        |          |     |             |
| **Llama Guard 3-8B** |           |        |          |     |             |
| **ShieldGemma 7B**   |           |        |          |     |             |
| Mistral              |           |        |          |     |             |
| **Two-Tier Hybrid**  |           |        |          |     |             |

**Targets**: F1 ≥ 0.82 | Precision ≥ 0.85 | Recall ≥ 0.80 | MCC ≥ 0.65

---

### 1.2 Per-Category Performance (F1 Scores)

| Model                | Hate Speech | Harassment | Violence | Sexual | Self-Harm | Avg |
| -------------------- | ----------- | ---------- | -------- | ------ | --------- | --- |
| **Detoxify Multi**   |             |            |          |        |           |     |
| **Llama Guard 3-8B** |             |            |          |        |           |     |
| **ShieldGemma 7B**   |             |            |          |        |           |     |
| Mistral              |             |            |          |        |           |     |
| **Two-Tier Hybrid**  |             |            |          |        |           |     |

**Target**: All categories ≥ 0.80

---

### 1.3 Confusion Matrix Summary

#### Detoxify Multilingual

```
Predicted:     Safe    Toxic
Actual:
Safe           [  ]    [  ]    (FPR: %)
Toxic          [  ]    [  ]    (FNR: %)
```

#### Llama Guard 3-8B

```
Predicted:     Safe    Toxic
Actual:
Safe           [  ]    [  ]    (FPR: %)
Toxic          [  ]    [  ]    (FNR: %)
```

#### ShieldGemma 7B

```
Predicted:     Safe    Toxic
Actual:
Safe           [  ]    [  ]    (FPR: %)
Toxic          [  ]    [  ]    (FNR: %)
```

#### Two-Tier Hybrid

```
Predicted:     Safe    Toxic
Actual:
Safe           [  ]    [  ]    (FPR: %)
Toxic          [  ]    [  ]    (FNR: %)
```

---

### 1.4 ROC-AUC / PR-AUC

|Model|ROC-AUC|PR-AUC|Notes|
|---|---|---|---|
|**Detoxify Multi**||||
|**Llama Guard 3-8B**||||
|**ShieldGemma 7B**||||
|**Two-Tier Hybrid**||||

---

## 2. Legitimacy Metrics

### 2.1 Consistency Testing (100 samples × 5 runs)

|Model|Identical Outputs|Consistency %|Target Met?|
|---|---|---|---|
|**Detoxify Multi**|/500|%||
|**Llama Guard 3-8B**|/500|%||
|**ShieldGemma 7B**|/500|%||
|**Two-Tier Hybrid**|/500|%||

**Target**: ≥ 95% consistency

---

### 2.2 Fairness Analysis (Demographic Groups)

#### False Positive Rate Disparity

|Model|Religious|LGBTQ+|Ethnic|Gender|French Dialects|Max Disparity|Target Met?|
|---|---|---|---|---|---|---|---|
|**Detoxify Multi**|%|%|%|%|%|||
|**Llama Guard 3-8B**|%|%|%|%|%|||
|**ShieldGemma 7B**|%|%|%|%|%|||
|**Two-Tier Hybrid**|%|%|%|%|%|||

**Target**: Disparity < 0.10 (10%)

---

#### False Negative Rate Disparity

|Model|Religious|LGBTQ+|Ethnic|Gender|French Dialects|Max Disparity|Target Met?|
|---|---|---|---|---|---|---|---|
|**Detoxify Multi**|%|%|%|%|%|||
|**Llama Guard 3-8B**|%|%|%|%|%|||
|**ShieldGemma 7B**|%|%|%|%|%|||
|**Two-Tier Hybrid**|%|%|%|%|%|||

**Target**: Disparity < 0.10 (10%)

---

#### Reclaimed Slurs Performance

|Model|FPR on Reclaimed Slurs|Sample Size|Target Met?|
|---|---|---|---|
|**Detoxify Multi**|%|||
|**Llama Guard 3-8B**|%|||
|**ShieldGemma 7B**|%|||
|**Two-Tier Hybrid**|%|||

**Target**: FPR < 20%

---

### 2.3 Explainability

|Model|Explanations Provided|Quality Score (1-5)|Notes|
|---|---|---|---|
|**Detoxify Multi**|%|/5||
|**Llama Guard 3-8B**|%|/5||
|**ShieldGemma 7B**|%|/5||
|**Two-Tier Hybrid**|%|/5||

**Target**: 100% coverage, Quality ≥ 3/5

---

### 2.4 Calibration (Expected Calibration Error)

|Model|ECE|Calibration Plot|Target Met?|
|---|---|---|---|
|**Detoxify Multi**||[Link/Image]||
|**Llama Guard 3-8B**||[Link/Image]||
|**ShieldGemma 7B**||[Link/Image]||
|**Two-Tier Hybrid**||[Link/Image]||

**Target**: ECE < 0.10

---

## 3. Operational Efficiency

### 3.1 Latency Measurements

|Model|Mean (ms)|Median (ms)|P95 (ms)|P99 (ms)|Target Met?|
|---|---|---|---|---|---|
|**Detoxify Multi**||||||
|**Llama Guard 3-8B**||||||
|**ShieldGemma 7B**||||||
|**Two-Tier Hybrid**||||||

**Target**: P95 < 500ms

---

### 3.2 Throughput

|Model|Predictions/sec|Batch Size|GPU Used|Target Met?|
|---|---|---|---|---|
|**Detoxify Multi**|||||
|**Llama Guard 3-8B**|||||
|**ShieldGemma 7B**|||||
|**Two-Tier Hybrid**|||||

---

### 3.3 Memory Usage

|Model|RAM (MB)|GPU Memory (MB)|Model Size (MB)|Notes|
|---|---|---|---|---|
|**Detoxify Multi**|||||
|**Llama Guard 3-8B**|||||
|**ShieldGemma 7B**|||||
|**Two-Tier Hybrid**|||||

---

### 3.4 Cost Estimation (10K predictions/day)

|Model|Daily Cost|Monthly Cost|Cost per 1K|GPU Type|Target Met?|
|---|---|---|---|---|---|
|**Detoxify Multi**|$|$|$|||
|**Llama Guard 3-8B**|$|$|$|||
|**ShieldGemma 7B**|$|$|$|||
|**Two-Tier Hybrid**|$|$|$|||

**Target**: < €0.05 per 1K predictions

**Two-Tier Breakdown** (if applicable):

- Tier 1 (Detoxify): % of traffic → $ daily
- Tier 2 (LLM): % of traffic → $ daily
- Escalation rate: %
- Savings vs LLM-only: %

---

### 3.5 Model Complexity

|Model|Parameters (M)|FLOPs (G)|Inference Time/Param|Notes|
|---|---|---|---|---|
|**Detoxify Multi**|||||
|**Llama Guard 3-8B**|||||
|**ShieldGemma 7B**|||||

---

## 4. Functional Robustness

### 4.1 HateCheck French (29 Functionalities)

|Functionality|Detoxify|Llama Guard|ShieldGemma|Two-Tier|Target|
|---|---|---|---|---|---|
|F1: Hate identity-directed|%|%|%|%|70%|
|F2: Hate derogatory|%|%|%|%|70%|
|F3: Hate threatening|%|%|%|%|70%|
|F4: Hate dehumanizing|%|%|%|%|70%|
|F5: Hate explicit|%|%|%|%|70%|
|F6: Hate implicit|%|%|%|%|70%|
|F7: Non-hate profanity|%|%|%|%|70%|
|F8: Non-hate slurs (ref group)|%|%|%|%|70%|
|F9: Non-hate reclaimed slurs|%|%|%|%|70%|
|F10: Non-hate criticizing hate|%|%|%|%|70%|
|F11: Non-hate counter speech|%|%|%|%|70%|
|F12: Non-hate positive|%|%|%|%|70%|
|F13: Negation (hate → non-hate)|%|%|%|%|70%|
|F14: Negation (non-hate → hate)|%|%|%|%|70%|
|F15: Negation (double negative)|%|%|%|%|70%|
|F16: Negation (tricky)|%|%|%|%|70%|
|F17: Spelling (leetspeak)|%|%|%|%|70%|
|F18: Spelling (space added)|%|%|%|%|70%|
|F19: Spelling (char deleted)|%|%|%|%|70%|
|F20: Spelling (char substituted)|%|%|%|%|70%|
|F21: Context (targets)|%|%|%|%|70%|
|F22: Context (reference group)|%|%|%|%|70%|
|F23: Context (neutral phrases)|%|%|%|%|70%|
|F24: Context (homonyms)|%|%|%|%|70%|
|F25: Ident (ment groups)|%|%|%|%|70%|
|F26: Ident (non-prot groups)|%|%|%|%|70%|
|F27: Ident (male vs female)|%|%|%|%|70%|
|F28: Ident (pers vs others)|%|%|%|%|70%|
|F29: Ident (phrase templates)|%|%|%|%|70%|

**Overall Pass Rate**: __/29 | __/29 | __/29 | __/29

**Target**: ≥ 25/29 functionalities passing (70% accuracy each)

---

### 4.2 Weak Functionalities Deep Dive

#### Most Critical Failures (Accuracy < 50%)

|Model|Functionality|Accuracy|Root Cause|Action Plan|
|---|---|---|---|---|
|||%|||
|||%|||
|||%|||

---

### 4.3 Implicit Hate (ToxiGen)

|Model|F1 on Implicit|FPR on Benign Mentions|Sample Size|Target Met?|
|---|---|---|---|---|
|**Detoxify Multi**||%|||
|**Llama Guard 3-8B**||%|||
|**ShieldGemma 7B**||%|||
|**Two-Tier Hybrid**||%|||

**Targets**: F1 ≥ 0.75, FPR < 7%

---

### 4.4 Adversarial Robustness

#### Jailbreak Attempts (if applicable for LLMs)

|Model|Total Attempts|Successful Jailbreaks|Success Rate|Target Met?|
|---|---|---|---|---|
|**Llama Guard 3-8B**|||%||
|**ShieldGemma 7B**|||%||
|**Two-Tier Hybrid**|||%||

**Target**: < 10% success rate

---

#### Edge Case Performance

|Model|Leetspeak|Obfuscation|Context-Dependent|Avg Score|
|---|---|---|---|---|
|**Detoxify Multi**|%|%|%|%|
|**Llama Guard 3-8B**|%|%|%|%|
|**ShieldGemma 7B**|%|%|%|%|
|**Two-Tier Hybrid**|%|%|%|%|

---

## 5. Real-World Impact (Pilot Phase)

### 5.1 AI-Human Agreement

|Model|Total Reviewed|Agreements|Disagreements|Agreement Rate|Target Met?|
|---|---|---|---|---|---|
|**Selected Model**||||%||

**Target**: ≥ 85%

**Disagreement Breakdown**:

- AI flagged, Human approved: (%)
- AI approved, Human flagged: (%)

---

### 5.2 User Experience Metrics

|Metric|Value|Target|Met?|
|---|---|---|---|
|User trust score (survey)|/5|≥ 3.8/5||
|Appeal rate|%|15-25%||
|Appeal success rate|%|15-25%||
|Avg resolution time|hours|< 24h||

---

### 5.3 Moderator Impact

|Metric|Before AI|After AI|Change|Target|
|---|---|---|---|---|
|Daily review load|posts|posts|%|-60%|
|High-quality escalations|%|%||≥ 80%|
|Time per review|min|min|%||

---

### 5.4 Platform Health

|Metric|Baseline|After Deployment|Change|Target|
|---|---|---|---|---|
|User reports of toxic content|/day|/day|%|-40%|
|Community satisfaction|/5|/5|%|+20%|
|False positive complaints|/day|/day|||

---

## 6. Dataset Performance Summary

### 6.1 Performance by Dataset

|Dataset|Detoxify|Llama Guard|ShieldGemma|Two-Tier|
|---|---|---|---|---|
|**HateCheck French** (F1)|||||
|**ToxiGen** (F1)|||||
|**Shareish Pilot** (F1)|||||
|**French Hate Superset** (F1)|||||

---

### 6.2 Dataset-Specific Insights

#### HateCheck French

- Strengths: [Which categories/functionalities performed best?]
- Weaknesses: [Which failed?]
- Action items: [How to improve?]

#### ToxiGen

- Strengths:
- Weaknesses:
- Action items:

#### Shareish Pilot

- Strengths:
- Weaknesses:
- Action items:

---

## 7. Model Comparison Matrix

### 7.1 Weighted Scoring System

**Weights** (Total = 1.0):

- Technical Performance: 0.30
- Legitimacy: 0.25
- Operational Efficiency: 0.20
- Functional Robustness: 0.20
- Real-World Impact: 0.05 (only if pilot completed)

|Model|Technical (×0.30)|Legitimacy (×0.25)|Efficiency (×0.20)|Robustness (×0.20)|Impact (×0.05)|**Total**|
|---|---|---|---|---|---|---|
|**Detoxify Multi**|/30|/25|/20|/20|/5|**/100**|
|**Llama Guard 3-8B**|/30|/25|/20|/20|/5|**/100**|
|**ShieldGemma 7B**|/30|/25|/20|/20|/5|**/100**|
|**Two-Tier Hybrid**|/30|/25|/20|/20|/5|**/100**|

**Scoring Guide**:

- 90-100: Excellent, exceeds targets
- 80-89: Good, meets all targets
- 70-79: Acceptable, meets most targets
- 60-69: Needs improvement
- <60: Insufficient

---

### 7.2 Strengths & Weaknesses Summary

#### Detoxify Multilingual

## **Strengths**:

## **Weaknesses**:

**Best Use Case**:

---

#### Llama Guard 3-8B

## **Strengths**:

## **Weaknesses**:

**Best Use Case**:

---

#### ShieldGemma 7B

## **Strengths**:

## **Weaknesses**:

**Best Use Case**:

---

#### Two-Tier Hybrid

## **Strengths**:

## **Weaknesses**:

**Best Use Case**:

---

## 8. Decision Log

### Week 1-2: Baseline Testing

**Date**: [DATE] **Decision**: **Rationale**: **Next Steps**:

---

### Week 3-4: Performance Analysis

**Date**: [DATE] **Decision**: **Rationale**: **Next Steps**:

---

### Week 5-6: Legitimacy & Fairness Testing

**Date**: [DATE] **Decision**: **Rationale**: **Next Steps**:

---

### Week 7-8: Robustness Testing

**Date**: [DATE] **Decision**: **Rationale**: **Next Steps**:

---

### Final Selection Decision

**Date**: [DATE] **Selected Model(s)**: **Rationale**:

**Key Factors**: 1. 2. 3.

## **Trade-offs Accepted**:

## **Rejected Alternatives & Why**:

---

## 9. Issues & Resolutions

|Date|Issue|Model(s) Affected|Resolution|Status|
|---|---|---|---|---|
||||||
||||||
||||||

---

## 10. Timeline Tracker

|Week|Planned Activity|Actual Activity|Status|Blockers|
|---|---|---|---|---|
|1|Dataset download & setup||⬜||
|2|Baseline testing (Technical)||⬜||
|3|Efficiency measurements||⬜||
|4|Cost analysis||⬜||
|5|Legitimacy testing||⬜||
|6|Fairness analysis||⬜||
|7|HateCheck robustness||⬜||
|8|ToxiGen & adversarial||⬜||
|9|Pilot preparation||⬜||
|10|Pilot deployment||⬜||
|11|Pilot analysis||⬜||
|12|Final decision||⬜||

**Legend**: ✅ Complete | 🔄 In progress | ⬜ Not started | ⚠️ Delayed | ❌ Blocked

---

## 11. Key Insights & Learnings

### Technical Insights

### Unexpected Findings

### Lessons for Deployment

---

## 12. Next Steps

### Immediate (This Week)

- [ ]
- [ ]
- [ ]

### Short-term (Next 2 Weeks)

- [ ]
- [ ]
- [ ]

### Long-term (After Selection)

- [ ]
- [ ]
- [ ]

---

## 13. Appendix: Testing Environment

### Hardware Configuration

- **CPU**:
- **RAM**:
- **GPU**:
- **Storage**:

### Software Versions

- **Python**:
- **PyTorch**:
- **Transformers**:
- **Detoxify**:
- **CUDA**:

### Dataset Versions

- **HateCheck French**: [Version/Date]
- **ToxiGen**: [Version/Date]
- **French Hate Superset**: [Version/Date]
- **Shareish Pilot**: [Size/Date range]

---

## 14. References & Links

### Code Repositories

- Profiling scripts: [Link]
- Evaluation scripts: [Link]
- Data preprocessing: [Link]

### Documentation

- Model cards: [Link]
- Dataset documentation: [Link]
- Evaluation framework: [Link]

### Key Papers

- [Citation 1]
- [Citation 2]
- [Citation 3]

---

**Document Status**: 🔄 Active tracking  
**Update Frequency**: After each testing phase  
**Next Review**: [DATE]  
**Contact**: [Your email]