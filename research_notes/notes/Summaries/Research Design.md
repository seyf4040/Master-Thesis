# Research Design: Architectures, Datasets, and Testing Protocols

**Master's Thesis**: Deep Learning for Content Moderation on Shareish  
**Date**: October 2025

---

## 1. Architectures to Compare

### 1.1 Selected Models (Rationale-Based)

| Architecture              | Type                     | Parameters | French Support | License    | Rationale                                                 |
| ------------------------- | ------------------------ | ---------- | -------------- | ---------- | --------------------------------------------------------- |
| **Detoxify Multilingual** | XLM-RoBERTa (fine-tuned) | 270M       | ✅              | Apache 2.0 | Baseline ML model; fast, proven multilingual performance  |
| **Llama Guard 3-8B**      | LLM (instruction-tuned)  | 8B         | ✅              | Llama 3    | State-of-the-art safety classifier; customizable taxonomy |
| **ShieldGemma 7B**        | LLM (instruction-tuned)  | 7B         | ✅              | Gemma      | Google's alternative; comparable to Llama Guard           |
| Mistral                   |                          |            |                |            |                                                           |
| **Two-Tier Hybrid**       | Cascade system           | 270M + 8B  | ✅              | Mixed      | Cost-optimized: Detoxify (90%) → Llama Guard (10%)        |
|                           |                          |            |                |            |                                                           |

### 1.2 Architecture Comparison Focus

**Technical Dimensions:**
- Traditional ML (Detoxify) vs. LLM-based (Llama Guard, ShieldGemma, Mistral)
- Single-tier vs. two-tier cascaded approach
- Zero-shot vs. fine-tunable capabilities

**Evaluation Questions:**
1. Do LLMs justify their higher cost vs. Detoxify?
2. Does the two-tier system maintain accuracy while reducing cost?
3. Which architecture best balances performance, fairness, and efficiency for a small solidarity platform?

---

## 2. Datasets to Use

### 2.1 Primary Datasets (French-Focused)

| Dataset                         | Size       | Language   | Purpose                                    | License   |
| ------------------------------- | ---------- | ---------- | ------------------------------------------ | --------- |
| **HateCheck French**            | 3.7K cases | FR         | Functional testing (29 robustness tests)   | CC BY 4.0 |
| **French Hate Speech Superset** | Moderate   | FR         | Training/fine-tuning                       | Open      |
| **ToxiGen**                     | 274K       | EN (multi) | Cold-start training; implicit hate testing | MIT       |


### 2.2 Secondary Datasets (Supplementary)

| Dataset               | Size | Purpose                            |
| --------------------- | ---- | ---------------------------------- |
| **HateCheck English** | 3.7K | Cross-lingual comparison           |
| **OpenAI Moderation** | 1.6K | Taxonomy reference                 |
| **OLID**              | 14K  | Additional offensive language data |

### 2.3 Dataset Usage Strategy

```
┌─────────────────────────────────────────────────────┐
│              DATASET ALLOCATION PLAN                │
├─────────────────────────────────────────────────────┤
│ Training (Cold-Start):                              │
│   → ToxiGen (274K) + French Hate Superset           │
│                                                     │
│ Evaluation (Primary):                               │
│   → HateCheck French (29 functional tests)          │
│   → Shareish Pilot Data (200-500 real posts         │
|                                       or gererated) |
│                                                     │
│ Evaluation (Secondary):                             │
│   → ToxiGen Test Split (implicit hate)              │
│   → HateCheck English (cross-lingual validation)    │
│                                                     │
│ Continuous Improvement:                             │
│   → Shareish Feedback Loop (human corrections)      │
└─────────────────────────────────────────────────────┘
```

**Key Constraint**: No large Shareish dataset available → rely on public datasets + gradual real-data collection via feedback loop.

---

## 3. Testing Protocols

### 3.1 Five-Dimensional Evaluation Framework

Following the comprehensive evaluation approach outlined in the Evaluation Framework document:

| Dimension                     | Key Metrics                                                   | Target                            |
| ----------------------------- | ------------------------------------------------------------- | --------------------------------- |
| **1. Technical Performance**  | F1 Score, Precision, Recall, MCC                              | F1 ≥ 0.82                         |
| **2. Legitimacy**             | Consistency, Fairness (FPR disparity), Explainability         | 95% consistency, <0.10 disparity  |
| **3. Operational Efficiency** | Latency (p95), Cost per 1K posts, Human workload reduction    | <500ms, <€0.05/1K, ≥60% reduction |
| **4. Functional Robustness**  | HateCheck pass rate, Implicit hate F1, Adversarial resistance | ≥25/29 pass, F1≥0.75              |
| **5. Real-World Impact**      | AI-human agreement, User trust, Appeal rates                  | ≥85% agreement                    |

---

### 3.2 Testing Protocol Timeline

#### **Phase 1: Baseline Evaluation**

**Datasets**: HateCheck French, ToxiGen sample, generated dataset pilot (200-500 posts)

**Tests**:
1. **Technical Performance**:
    - Standard metrics: Precision, Recall, F1, MCC
    - Per-category F1 (hate speech, harassment, violence, etc.)
    - Confusion matrix analysis
2. **Operational Efficiency**:
    - Latency benchmarking (p50, p95, p99)
    - Cost estimation per 1,000 posts
    - Throughput testing

**Models**: Detoxify Multilingual, Llama Guard 3-8B (zero-shot), ShieldGemma 7B (zero-shot)

**Deliverable**: Baseline performance report comparing all four architectures

---

#### **Phase 2: Optimization**

**Datasets**: ToxiGen (274K for fine-tuning), French Hate Superset, HateCheck French, generated dataset


**Tests**:
1. **Fine-Tuning Experiments**:
    - Fine-tune Llama Guard on ToxiGen + French data + generated dataset
    - Optimize two-tier thresholds (Detoxify confidence → Llama Guard escalation)
    - A/B test: zero-shot vs. fine-tuned
2. **Robustness Testing** (HateCheck):
    - Run all 29 functional tests
    - Identify weak functionalities (F9: reclaimed slurs, F13-16: negation, F17-20: spelling)
    - Iteratively improve on failure cases

**Target**: F1 improvement +10-15%, HateCheck pass rate ≥25/29, latency <100ms (avg)

**Deliverable**: Optimized model versions with documented improvements

---

#### **Phase 3: Legitimacy Testing 

**Datasets**: HateCheck French + custom adversarial samples

**Tests**:
1. **Consistency**:
    - Run 100 samples × 5 times each
    - Measure output variability
    - Target: ≥95% identical predictions
2. **Fairness**:
    - Demographic group analysis (religious groups, LGBTQ+, ethnicities, French dialects)
    - False Positive Rate disparity across groups
    - False Negative Rate disparity across groups
3. **Explainability**:
    - Verify 100% of flagged content has explanation
    - Quality audit of explanations (human evaluation)


**Deliverable**: Legitimacy audit report with fairness metrics

---

#### **Phase 4: Adversarial Testing**

**Datasets**: HateCheck edge cases, WildGuard adversarial prompts, custom Shareish-specific tests

**Tests**:
1. **Edge Case Resistance**:
    - HateCheck failure cases from Phase 2
    - Implicit toxicity (ToxiGen implicit hate subset)
2. **Adversarial Robustness**:
    - Jailbreak attempts (WildGuard methodology)
    - Spelling variations, leetspeak, obfuscation
    - Target: jailbreak success rate <10%
3. **False Positive Analysis**:
    - Benign identity mentions (e.g., "I'm proud to be [group]")
    - Context-dependent language (solidarity platform-specific)
    - Target: FPR on benign mentions <7%

**Deliverable**: Adversarial robustness report with red-teaming results

---

#### **Phase 5: Real-World Pilot**

**Dataset**: Shareish production data (shadow mode deployment)

**Tests**:
1. **AI-Human Agreement**:
    - Human moderators review AI decisions
    - Measure agreement rate (target ≥85%)
    - Analyze disagreement patterns
2. **User Experience**:
    - User trust survey (target ≥3.8/5)
    - Appeal rate monitoring (healthy: 15-25%)
    - Appeal resolution time (<24 hours)
3. **Operational Impact**:
    - Human review load reduction (target ≥60%)
    - High-quality escalation rate (≥80%)
    - Platform health metrics (user reports -40%, satisfaction +20%)

**Deliverable**: Pilot study report with real-world performance data and deployment recommendation

---

## 4. Summary Table

| Requirement          | Answer                                                                                                                                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Architectures**    | 5 models: Detoxify Multilingual (baseline), Llama Guard 3-8B (SOTA LLM), ShieldGemma 7B (alternative LLM), Mistral, Two-Tier Hybrid (cost-optimized)                                                                                                                     |
| **Datasets**         | Primary: HateCheck French (3.7K), French Hate Superset, ToxiGen (274K), Shareish pilot (200-500). Secondary: HateCheck English, OLID, OpenAI Moderation                                                                                                                  |
| **Testing Protocol** | 5-phase: (1) Baseline evaluation, (2) Optimization with fine-tuning, (3) Legitimacy testing (fairness, consistency, explainability), (4) Adversarial robustness, (5) Real-world pilot. Five evaluation dimensions: Technical, Legitimacy, Efficiency, Robustness, Impact |

---

## 5. Key Decisions and Justifications

### Why These Architectures?
- **Detoxify**: Proven baseline, fast, multilingual
- **Llama Guard 3**: Meta's SOTA safety classifier
- **ShieldGemma**: Google alternative for comparison
- **Mistral**: French tech, local alternative
- **Two-Tier**: Addresses Shareish's cost constraint (small platform)

### Why These Datasets?
- **HateCheck French**: Only robust French functional test suite (29 tests)
- **ToxiGen**: Largest open implicit hate dataset (addresses cold-start problem)
- **French Hate Superset**: Additional French training data
- **Shareish Pilot**: Real-world validation (ethical, consent-based)

### Why This Testing Protocol?
- **Multi-dimensional**: Goes beyond accuracy (legitimacy, fairness, efficiency)
- **Phased**: Allows iterative improvement
- **Evidence-based**: Metrics drawn from literature review (88% of papers use F1, but only 18% test fairness)
- **Ethical**: Includes bias auditing, transparency, and privacy protections

---

**Next Steps**: Supervisor approval → Begin Phase 1 (baseline evaluation) → Document results weekly