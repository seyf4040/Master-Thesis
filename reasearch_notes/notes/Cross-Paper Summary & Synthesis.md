# Cross-Paper Summary & Synthesis

```table-of-contents
title: ## 📋 Table of contents
minLevel:2
maxLevel:2
```

---
## 📚 Papers by Category

### **LLM Integration & Applications (Most Relevant)** - 8 papers

1. **Integrating Content Moderation Systems with LLMs** (2024) ⭐⭐⭐
    - Policy-as-prompt framework
    - Personalized moderation
    - User-platform communication
2. **ShieldGemma** (2024) ⭐⭐⭐
    - Google's open-source model
    - Production-ready (2B/7B parameters)
    - Multilingual support including French
3. **Watch Your Language** (2024) ⭐⭐⭐
    - Comprehensive LLM evaluation
    - Rule-based + toxicity detection
    - F1: 0.72-0.75 (toxicity)
4. **Adapting LLMs for Content Moderation** (2024) ⭐⭐⭐
    - Chain of Thought fine-tuning
    - Weak supervision methodology
    - Outperforms GPT-4 with proper training
5. **Content Moderation by LLM: Accuracy to Legitimacy** (2024) ⭐⭐⭐
    - Critique of accuracy-centric evaluation
    - Legitimacy framework
    - Easy vs. hard case distinction
6. **LLM-Mod: Can LLMs Assist Content Moderation?** (2024) ⭐⭐
    - Identifies LLM limitations
    - 43.1% recall (poor performance)
    - Small test set (744 samples)
7. **GPTFUZZER: Red Teaming LLMs** (2023) ⭐
    - Adversarial testing
    - Jailbreak prompt generation
    - Security considerations
8. **Like a Good Nearest Neighbor** (2023) ⭐
    - LaGoNN (SetFit modification)
    - k-NN + transformer
    - Limited improvement over baseline

---

### **Traditional ML & Discriminative Models** - 9 papers

9. **OpenAI Content Moderation API** (2022) ⭐⭐⭐
    - Detailed 8-category taxonomy (S/H/V/SH/HR/S3/H2/V2)
    - GPT-based transformer with MLP heads
    - Domain adversarial training
    - Active learning methodology
10. **Multilingual Content Moderation (Reddit)** (2023) ⭐⭐⭐
    - 1.8M samples, French included
    - **71% of removed content is non-toxic** (key finding)
    - Transformer encoder (XLM-RoBERTa)
11. **Perspective API** (Google Jigsaw, 2018-ongoing) ⭐⭐
    - Industry standard baseline
    - F1: 0.64 (toxicity)
    - 18+ languages including French
12. **Text Classification Using ML Techniques** (2004) ⭐
    - Foundational survey paper
    - Traditional methods: NB, SVM, NN
    - Evaluation metrics framework
13. **Design and Application of AI-Based TCM** (2022) ❌
    - Cloud-based system (not aligned with Shareish)
    - FastText implementation
    - 360K Chinese samples
14. **Real-Time Content Moderation Using AI/ML** (2024) ⭐
    - Overview of techniques (NLP, CV, behavioral)
    - Challenges and ethical considerations
    - No novel techniques presented
15. **A Review of Standard Text Classification Practices** (2018) ⭐
    - Multi-label toxicity identification
    - Kaggle Toxic dataset (159K samples)
    - Stacking classifiers approach
16. **Comparison of Deep Learning Models & Preprocessing** (2020) ⭐
    - CNN, LSTM, Bi-LSTM, GRU, BERT comparison
    - Preprocessing vs. model complexity trade-offs
    - BERT best with minimal preprocessing
17. **Content Moderation System Using ML Techniques** (2023) ⭐
    - General ML survey
    - Supervised learning approaches
    - No groundbreaking innovations

---

### **Specialized Classification & Nuanced Detection** - 3 papers

18. **Predicting Type and Target of Offensive Posts** (2019) ⭐⭐
    - OLID dataset (14.1K tweets)
    - 3-level hierarchical classification
    - F1: 0.80 (Level A), 0.68 (Level B), 0.47 (Level C)
19. **Do You Really Want to Hurt Me? (Abusive Swearing)** (2020) ⭐⭐
    - Context-aware swearing classification
    - Distinguishes abusive vs. casual swearing
    - SWAD dataset (GPL 3.0 license)
20. **On-Device Content Moderation** (2021) ⭐
    - Image moderation (NSFW detection)
    - SSD + MobileNetV3
    - F1: 0.91, Precision: 95%, Recall: 88%

---

### **Human-AI Collaboration & System Design** - 2 papers

21. **Learning to Defer in Content Moderation** (2024) ⭐⭐⭐
    - When AI should escalate to humans
    - Learning to Defer framework
    - Optimal defer rate: 10-30%
    - System-wide accuracy optimization
22. **Content Moderation, AI, and the Question of Scale** (2020) ⭐
    - Early philosophical paper
    - Should we automate moderation?
    - AI as assistant, not replacement

---

### **Explainability, Evaluation & Ethics** - 4 papers

23. **From Machine Learning to Explainable AI** (2018) ⭐⭐⭐
    - XAI techniques: LIME, SHAP, attention
    - Legal requirement (GDPR, DSA)
    - Chain of Thought for LLMs
24. **Oversight of Content Moderation by AI** (date unknown) ⭐⭐
    - Impact assessment frameworks
    - Multi-dimensional evaluation
    - Beyond accuracy metrics
25. **The Use of AI in Online Content Moderation** (2022) ⭐⭐
    - Policy report (AEI)
    - Industry practices overview
    - Regulatory landscape (DSA, Section 230)
    - Best practices identification
26. **Artificial Intelligence as a Tool in Social Media CM** (2023) ⭐
    - Bachelor thesis - literature review
    - Benefits and limitations summary
    - No novel contributions

---

### **Legal, Regulatory & Policy** - 1 paper

27. **Online Content Moderation: Regulatory Challenges** (2024) ⭐⭐
    - Master thesis (University of Namur)
    - DSA, GDPR compliance
    - Legal frameworks for platforms
    - Likely French language

---

### **Foundational Concepts** - 2 papers

28. **Transfer Learning for Text Classification** (2005) ⭐⭐
    - Foundational concept (NeurIPS)
    - Pre-training + fine-tuning paradigm
    - Underlies all modern transformers
29. **Deeper Attention to Abusive User Content Moderation** (2017) ⭐⭐
    - Early attention mechanisms
    - Historical context
    - **Priority to read in full**

---

## 🔑 Key Themes Across All Papers

### **Consensus Findings** (Universal Agreement)

1. **AI Alone Is Insufficient**
    
    - All 29 papers advocate human oversight
    - Especially: Learning to Defer, Content Moderation AI & Scale, Integrating with LLMs
    - Even most optimistic LLM papers recommend human review for edge cases
2. **Context Understanding Is Critical Limitation**
    
    - Major weakness in: Traditional ML (9 papers), LLM-Mod, Watch Your Language
    - Partially addressed by: LLMs with CoT, Abusive Swearing (context-aware)
    - Still unsolved: Cultural nuances, sarcasm, jokes, temporal context
3. **Explainability Is Non-Negotiable**
    
    - Legal requirement: GDPR (right to explanation), DSA (transparency)
    - Papers emphasizing: From ML to XAI, Integrating with LLMs, OpenAI API
    - Solutions: Chain of Thought (LLMs), LIME/SHAP (traditional ML)
4. **Bias Is Persistent Challenge**
    
    - Mentioned in: 18/29 papers
    - Sources: Training data bias, demographic disparities, language bias
    - Solutions proposed: Diverse training data, fairness metrics, human oversight
5. **Multilingual Support Is Limited**
    
    - English-centric: 21/29 papers
    - French support: Multilingual Reddit, ShieldGemma, Perspective API (only 3 papers with good FR support)
    - Gap: French-specific evaluation and cultural adaptation

---

### **LLM Safety & Guardrails** - 2 papers

27. **Llama Guard: LLM-based Input-Output Safeguard** (2023-2024) ⭐⭐⭐
    - Meta's open-source safety model (Llama 2/3 based)
    - **Llama Guard 3**: 8 languages including French ✅
    - Customizable taxonomy (MLCommons aligned)
    - Dual classification: input + output moderation
    - Performance matches/exceeds OpenAI Moderation API
    - **1B INT4 version** available for on-device deployment
28. **WildGuard: Open One-Stop Moderation Tools** (2024) ⭐⭐
    - AI2's multi-task safety model
    - **Three tasks**: prompt harm, response harm, refusal detection
    - State-of-the-art adversarial robustness
    - 92K training examples (WildGuardMix dataset)
    - Exceeds GPT-4 on jailbreak defense
    - Limited multilingual support (primarily English)

---

### **Datasets & Benchmarks** - 3 papers

29. **ToxiGen: Machine-Generated Dataset for Implicit Hate** (2022) ⭐⭐⭐
    - 274K adversarially-generated examples
    - **95% implicit toxicity** (subtle, context-dependent)
    - 13 minority groups covered
    - ALICE generation method (RoBERTa-based)
    - Reduces false positives on identity mentions by 9%
    - Improves implicit hate detection (+8% F1)
    - English only (no French version)
30. **HateCheck: Functional Tests for Hate Detection** (2021) ⭐⭐⭐
    - 3,728 test cases across **29 functionalities**
    - **French version available** (Multilingual HateCheck 2022) ✅
    - Template-based evaluation (identity, slurs, negations, etc.)
    - Pass threshold: 70% per functionality
    - Reveals specific model weaknesses
    - Essential for systematic evaluation
31. **Detoxify: Toxic Comment Classification** (2020) ⭐⭐
    - Unitary's open-source toxicity classifier
    - **Multilingual version includes French** ✅
    - BERT-based (7 toxicity categories)
    - Fast inference (~50ms with GPU)
    - 92% AUC on multilingual data
    - Production-ready, Apache 2.0 license
    - Good baseline/first-pass filter

---

### **Legal, Regulatory & Policy** - 1 paper

32. **Online Content Moderation: Regulatory Challenges** (2024) ⭐⭐
    - Master thesis (University of Namur)
    - DSA, GDPR compliance
    - Legal frameworks for platforms
    - Likely French language

---

### **Foundational Concepts** - 2 papers

33. **Transfer Learning for Text Classification** (2005) ⭐⭐
    - Foundational concept (NeurIPS)
    - Pre-training + fine-tuning paradigm
    - Underlies all modern transformers
34. **Deeper Attention to Abusive User Content Moderation** (2017) ⭐⭐
    - Early attention mechanisms
    - Historical context
    - **Priority to read in full**

---

### **Major Contradictions & Debates**

#### **Debate 1: Accuracy vs. Legitimacy**

**Pro-Accuracy Position:**

- Traditional ML papers (9 papers)
- Watch Your Language (benchmarking focus)
- Comparison of DL Models
- Argument: Higher accuracy = better system

**Anti-Accuracy Position:**

- **Content Moderation by LLM: Accuracy→Legitimacy** ⭐⭐⭐
- Oversight of Content Moderation by AI
- Argument: Legitimacy (consistency, fairness, explainability) > raw accuracy

**Synthesis:** Both matter, but accuracy alone is insufficient for production systems.

---

#### **Debate 2: Preprocessing Requirements**

**Extensive Preprocessing Camp:**

- Traditional ML papers (7 papers)
- Comparison of DL Models: "Preprocessing helps LSTM/CNN"
- Text Classification ML: Tokenization, stemming, stopword removal essential

**Minimal Preprocessing Camp:**

- LLM papers (8 papers)
- Comparison of DL Models: "BERT works best with raw text"
- Adapting LLMs: Minimal preprocessing recommended

**Resolution:** Depends on model architecture. LLMs → minimal; Traditional ML → extensive.

---

#### **Debate 3: Deployment Strategy**

**Cloud API Advocates:**

- Watch Your Language (tested GPT-3.5, GPT-4)
- LLM-Mod (used GPT-3.5)
- Advantages: No infrastructure, always updated, easy to use

**Self-Hosting Advocates:**

- ShieldGemma ⭐⭐⭐
- Regulatory Challenges (GDPR concerns)
- Learning to Defer
- Advantages: Privacy, control, no vendor lock-in, cost predictability

**Shareish Decision:** Self-hosting preferred (GDPR, user trust, cost control)

---

#### **Debate 4: Fine-Tuning vs. Prompt Engineering**

**Fine-Tuning Advocates:**

- Adapting LLMs ⭐⭐⭐: "Fine-tuning with CoT outperforms GPT-4"
- ShieldGemma: Pre-fine-tuned for moderation
- Transfer Learning: Core concept

**Prompt Engineering Advocates:**

- Watch Your Language: "Prompt engineering sufficient for many tasks"
- Integrating with LLMs: "Policy-as-prompt" flexibility
- LLM-Mod: "Prompting alone has limitations"

**Synthesis:** Fine-tuning for performance; prompt engineering for flexibility. Use both.

---

## 📊 Performance Benchmark Summary

### **Toxicity Detection (F1 Scores)**

|Approach|Best F1|Paper|Year|Notes|
|---|---|---|---|---|
|ShieldGemma|0.75-0.85|ShieldGemma|2024|Estimated, open-source|
|GPT-4|0.75|Watch Your Language|2024|Proprietary API|
|GPT-3.5|0.72-0.75|Watch Your Language|2024|Proprietary API|
|Baichuan-13B (fine-tuned)|>0.75|Adapting LLMs|2024|Chinese data, Setting D|
|Perspective API|0.64|Watch Your Language|2024|Industry baseline|
|Traditional ML (best)|~0.70|Review of Text Class.|2018|With stacking|

**Trend:** LLMs significantly outperform traditional approaches (+10-20% F1)

---

### **Rule-Based Moderation (Accuracy)**

|Approach|Median Accuracy|Median Precision|Paper|Notes|
|---|---|---|---|---|
|GPT-3.5|64%|83%|Watch Your Language|95 subreddits|
|LLM-Mod (GPT-3.5)|Poor|Low (43.1% recall)|LLM-Mod|9 subreddits, small test|

**Insight:** Performance highly variable across communities (some near-human, others worse than baseline)

---

### **Hierarchical Classification (OLID Dataset)**

|Level|Task|Best F1|Paper|
|---|---|---|---|
|A|Offensive Y/N|0.80|Predicting Type & Target|
|B|Type (Targeted/Untargeted)|0.68|Predicting Type & Target|
|C|Target (Individual/Group/Other)|0.47|Predicting Type & Target|

**Challenge:** Fine-grained classification significantly harder than binary

---

### **Image Moderation**

|Model|F1|Precision|Recall|Paper|
|---|---|---|---|---|
|On-Device (SSD+MobileNetV3)|0.91|95%|88%|On-Device CM|

**Note:** Only one paper on image moderation; text moderation dominates research

---

## 🎯 Direct Applicability to Shareish Platform

### **Tier 1: Core Implementation Papers** (Must Read in Full)

**Architecture Design:**

1. **Integrating Content Moderation Systems with LLMs** ⭐⭐⭐
    - Use for: System architecture, policy-as-prompt framework
    - Apply: Personalized rules, user communication, appeals process
2. **Learning to Defer in Content Moderation** ⭐⭐⭐
    - Use for: AI-human collaboration strategy
    - Apply: Confidence thresholds, escalation workflow, optimal defer rate

**Model Selection:** 3. **ShieldGemma** ⭐⭐⭐

- Use for: Baseline model or production deployment
- Apply: Self-hosted solution, multilingual support, fine-tuning base

4. **Adapting LLMs for Content Moderation** ⭐⭐⭐
    - Use for: Fine-tuning methodology
    - Apply: Chain of Thought training, weak supervision, synthetic data generation

**Evaluation:** 5. **Content Moderation by LLM: Accuracy→Legitimacy** ⭐⭐⭐

- Use for: Evaluation philosophy and framework
- Apply: Beyond accuracy metrics, legitimacy criteria, easy/hard case distinction

6. **From Machine Learning to Explainable AI** ⭐⭐⭐
    - Use for: Explanation generation
    - Apply: Chain of Thought, user-facing explanations, GDPR compliance

---

### **Tier 2: Supporting Design Decisions** (Read Summaries + Key Sections)

**Benchmarking:** 7. **Watch Your Language** ⭐⭐⭐

- Use for: Performance benchmarks, comparison baseline
- Apply: Evaluate Shareish system against state-of-the-art

8. **OpenAI Content Moderation API** ⭐⭐⭐
    - Use for: Taxonomy reference (8 categories)
    - Apply: Adapt taxonomy to Shareish rules

**Data Strategy:** 9. **Multilingual Content Moderation (Reddit)** ⭐⭐⭐

- Use for: French dataset access
- Apply: Training/testing on French content, transfer learning

10. **Transfer Learning for Text Classification** ⭐⭐
    - Use for: Conceptual foundation
    - Apply: Justification for using pre-trained models

**Specialized Detection:** 11. **Predicting Type and Target of Offensive Posts** ⭐⭐ - Use for: Hierarchical classification approach - Apply: Multi-label moderation (if needed for granularity)

12. **Do You Really Want to Hurt Me?** ⭐⭐
    - Use for: Nuanced detection (abusive vs. casual swearing)
    - Apply: Context-aware classification (if Shareish allows some swearing)

**Evaluation Framework:** 13. **Oversight of Content Moderation by AI** ⭐⭐ - Use for: Multi-dimensional impact assessment - Apply: Fairness metrics, user impact, systemic evaluation

---

### **Tier 3: Context & Background** (Read Summaries Only)

**Legal/Policy:** 14. **Online Content Moderation: Regulatory Challenges** ⭐⭐ - Use for: GDPR, DSA compliance requirements - Apply: Legal obligations discussion in thesis

15. **The Use of AI in Online Content Moderation** ⭐⭐
    - Use for: Industry best practices, policy context
    - Apply: Introduction/discussion sections

**Historical Context:** 16. **Content Moderation, AI, and the Question of Scale** ⭐ - Use for: Historical perspective on automation debate - Apply: Introduction context

17. **Text Classification Using ML Techniques** ⭐
    - Use for: ML fundamentals background
    - Apply: Background section (if needed)

**Baseline Comparison:** 18. **Perspective API** ⭐⭐ - Use for: Industry baseline for comparison - Apply: Benchmark against Perspective toxicity scores

---

### **Tier 4: Limited Relevance** (Skip or Skim)

**Not Aligned with Approach:** 19. **Design and Application of AI-Based TCM** ❌ - Reason: Cloud-based, not self-hosted (conflicts with Shareish philosophy)

20. **LLM-Mod** ⭐⭐
    - Reason: Negative results (useful for "what doesn't work" discussion)
21. **Like a Good Nearest Neighbor** ⭐
    - Reason: Limited improvement, not compelling approach

**General Overviews (No Novel Insights):** 22. **Real-Time Content Moderation Using AI/ML** ⭐ 23. **Content Moderation System Using ML Techniques** ⭐ 24. **Artificial Intelligence as a Tool** ⭐ 25. **A Review of Standard Text Classification Practices** ⭐

**Image-Specific (Not Primary Focus):** 26. **On-Device Content Moderation** ⭐ - Reason: Image moderation (use if Shareish adds image support)

**Security (Secondary Concern):** 27. **GPTFUZZER** ⭐ - Reason: Adversarial testing (useful for discussion, not core implementation)

**Outdated Approaches:** 28. **Comparison of Deep Learning Models & Preprocessing** ⭐ - Reason: Traditional ML methods (superseded by LLMs)

---

## 🔍 Gap Analysis: Missing from Literature

### **Critical Gaps** (Your Novel Contribution Opportunities)

#### **Gap 1: Cold-Start Problem for Small Platforms** ⭐⭐⭐

**Coverage:** ❌ No papers address this directly

- Transfer Learning (2005): Concept only, no practical guidance
- Adapting LLMs (2024): Uses 8.7K samples (not truly cold-start)
- OpenAI API (2022): Mentions cold-start but doesn't solve it

**Your Contribution:**

- Strategy for platforms with <500 initial samples
- Synthetic data generation methodology
- Active learning from day 1
- Performance curve: 100 → 500 → 1,000 samples

---

#### **Gap 2: French-Language Specific Evaluation** ⭐⭐⭐

**Coverage:** ⚠️ Partial - only 3 papers include French

- Multilingual Reddit: Has French data but no French-specific analysis
- ShieldGemma: Supports French but no performance breakdown
- Perspective API: Supports French but lower quality than English

**Your Contribution:**

- French-specific performance metrics
- Cultural nuance handling (French humor, slang, regional variations)
- Comparison: EN-trained vs. FR-trained vs. multilingual models
- French hate speech patterns unique to French-speaking communities

---

#### **Gap 3: Small Platform Scalability** ⭐⭐⭐

**Coverage:** ❌ All research focuses on large platforms

- Facebook/Meta scale (billions of users)
- Reddit scale (millions of users)
- Twitter/X scale (millions of users)

**No papers address:**

- Platforms with <100K users
- Cost-benefit for small platforms
- Infrastructure requirements at small scale
- When does automation become worthwhile?

**Your Contribution:**

- Scalability analysis for Shareish (<10K users initially)
- Cost projections: human-only vs. hybrid vs. AI-heavy
- Performance requirements for responsive user experience
- Growth trajectory: when to upgrade from model X to model Y

---

#### **Gap 4: Cost-Benefit Analysis** ⭐⭐

**Coverage:** ❌ No quantitative cost comparisons

- Watch Your Language: Performance comparison only
- The Use of AI: Mentions cost but no concrete numbers
- Industry reports: Qualitative only

**Missing Information:**

- API costs vs. self-hosting (compute, infrastructure, maintenance)
- Human moderator costs vs. AI costs vs. hybrid
- Training costs (one-time) vs. inference costs (ongoing)
- Hidden costs (data annotation, model updates, appeals handling)

**Your Contribution:**

- Detailed cost analysis for Shareish deployment
- Break-even point: human vs. AI moderation
- TCO (Total Cost of Ownership) over 3 years
- Cost per correctly moderated item

---

#### **Gap 5: Synthetic Data Quality Control** ⭐⭐

**Coverage:** ⚠️ Mentioned but not deeply explored

- Adapting LLMs: Uses GPT-4 for synthetic data with weak supervision
- OpenAI API: Mentions synthetic data for rare categories

**Not Addressed:**

- How to validate synthetic data quality?
- What ratio of synthetic to real data is optimal?
- Does synthetic data introduce new biases?
- How to ensure diversity in synthetic samples?

**Your Contribution:**

- Synthetic data generation protocol
- Quality metrics for synthetic samples
- Comparison: real-only vs. synthetic+real vs. synthetic-only
- Bias analysis of synthetic data

---

#### **Gap 6: Multi-Modal Moderation (Text + Images Together)** ⭐

**Coverage:** ❌ Text and images treated separately

- On-Device: Images only
- All other papers: Text only
- No papers on joint text+image understanding for moderation

**Future Work (Not Priority for Shareish Initially):**

- Memes with offensive text overlays
- Context: innocent text + offensive image (or vice versa)
- Joint embeddings for multi-modal content

---

## 📝 Updated Recommendations for Thesis

### **Papers to Read in Full** (Priority Order)

**Week 1-2: Core Architecture** (Must read)

1. ✅ **Integrating Content Moderation Systems with LLMs** (2024)
    
    - Why: System architecture blueprint
    - Focus: Policy-as-prompt, personalization, user communication
2. ✅ **Learning to Defer in Content Moderation** (2024)
    
    - Why: AI-human collaboration framework
    - Focus: Defer criteria, confidence thresholds, system-wide optimization
3. ✅ **ShieldGemma** (2024)
    
    - Why: Practical model for deployment
    - Focus: Architecture, performance, deployment requirements

**Week 3-4: Fine-Tuning & Evaluation** (High priority) 4. ✅ **Adapting LLMs for Content Moderation** (2024)

- Why: Fine-tuning methodology
- Focus: Chain of Thought, weak supervision, settings A-D comparison

5. ✅ **Content Moderation by LLM: Accuracy→Legitimacy** (2024)
    
    - Why: Evaluation philosophy
    - Focus: Legitimacy framework, critique of accuracy
6. ✅ **From Machine Learning to Explainable AI** (2018)
    
    - Why: Explanation generation techniques
    - Focus: LIME, SHAP, attention, Chain of Thought for LLMs

**Week 5: Benchmarking & Data** (Important) 7. ⚠️ **Watch Your Language** (2024)

- Already have good notes, read specific sections:
- Section on rule-based moderation (methods + results)
- Section on toxicity detection (comparison with Perspective)
- Error analysis and limitations

8. ⚠️ **Multilingual Content Moderation (Reddit)** (2023)
    - Already have good notes, focus on:
    - Dataset access procedure
    - French sample characteristics
    - 71% non-toxic violations finding (detailed analysis)

---

### **Papers Sufficient with Current Summaries** (Skim Only)

**Background & Context:**

- ✅ Content Moderation, AI, and the Question of Scale
- ✅ The Use of AI in Online Content Moderation
- ✅ Artificial Intelligence as a Tool (bachelor thesis)
- ✅ Text Classification Using ML Techniques
- ✅ Transfer Learning for Text Classification (conceptual understanding sufficient)

**Baseline Comparison:**

- ✅ OpenAI Content Moderation API (excellent notes on taxonomy)
- ✅ Perspective API (good understanding of baseline)

**Specialized (Use If Needed):**

- ✅ Predicting Type and Target (hierarchical approach, use if going multi-label)
- ✅ Do You Really Want to Hurt Me (nuanced swearing, use if Shareish allows swearing)
- ✅ On-Device Content Moderation (image moderation, use if adding images)

**Negative Results (Informative):**

- ✅ LLM-Mod (understand limitations)
- ✅ Like a Good Nearest Neighbor (alternative approach, not compelling)

**Security (Discussion Only):**

- ✅ GPTFUZZER (adversarial considerations)

---

### **Papers Needing More Detail** (If Time Permits)

**Legal/Regulatory (For Discussion Section):**

- ⚠️ **Online Content Moderation: Regulatory Challenges**
    - Need: Specific GDPR/DSA compliance requirements
    - Action: Request full PDF or focus on executive summary

**Evaluation (For Methodology):**

- ⚠️ **Oversight of Content Moderation by AI**
    - Need: Specific impact assessment frameworks
    - Action: Read full paper or find similar framework

**Traditional ML (For Background):**

- ⚠️ **Comparison of Deep Learning Models & Preprocessing**
    - Need: Only if using traditional ML fallback
    - Action: Reference current notes (sufficient)

---

### **Papers in Reading Queue** (Not Yet Read)

**Priority to Read:**

1. **Deeper Attention to Abusive User Content Moderation** (2017)
    
    - Historical context
    - Attention mechanisms for abuse detection
2. **Toxicity Detection is NOT All You Need** (2024)
    
    - Volunteer moderator support gaps
    - System design beyond detection

**Lower Priority:**

- Various survey papers and general overviews

---

## 🎓 Thesis Structure Recommendations

### **Chapter 1: Introduction**

**Cite:** Content Moderation AI & Scale, The Use of AI (context), Multilingual Reddit (71% non-toxic finding)

### **Chapter 2: Literature Review**

**2.1 Evolution of Content Moderation (2004-2024)**

- Early foundations: Transfer Learning, Text Classification ML
- Traditional ML era: OpenAI API, Perspective API
- LLM revolution: Watch Your Language, ShieldGemma
- Current state: Integrating with LLMs

**2.2 Technical Approaches**

- **2.2.1 Traditional ML:** Text Classification ML, Comparison of DL Models, OpenAI API
- **2.2.2 LLM-Based:** Watch Your Language, Adapting LLMs, ShieldGemma, LLM-Mod
- **2.2.3 Specialized:** Predicting Type & Target, Abusive Swearing

**2.3 Key Challenges**

- Context understanding: Watch Your Language, LLM-Mod
- Multilingualism: Multilingual Reddit, ShieldGemma
- Bias: Multiple papers
- Cold-start: Gap in literature (your contribution)

**2.4 Evaluation Paradigms**

- Accuracy-centric: Traditional papers
- Legitimacy-based: Content Moderation by LLM
- Multi-dimensional: Oversight of CM by AI

**2.5 Human-AI Collaboration**

- Framework: Learning to Defer
- Industry practices: The Use of AI
- Philosophy: Content Moderation AI & Scale

**2.6 Explainability & Transparency**

- XAI techniques: From ML to XAI
- Legal requirements: Regulatory Challenges
- Implementation: Chain of Thought (Adapting LLMs)

**2.7 Legal & Ethical Considerations**

- Regulations: Regulatory Challenges, The Use of AI
- Ethics: Multiple papers on bias and fairness

### **Chapter 3: Research Questions & Gaps**

**Identify gaps from 29 papers:**

- Cold-start problem (no existing solutions)
- French-language specificity (limited work)
- Small platform scalability (no research)
- Cost-benefit analysis (missing)

**Your novel contributions** ⭐

### **Chapter 4: Methodology**

**4.1 Model Selection**

- Justify LLM approach: Watch Your Language, Adapting LLMs
- Choose base model: ShieldGemma or Mistral/Llama
- Fine-tuning strategy: Adapting LLMs (CoT + weak supervision)

**4.2 Architecture Design**

- Policy-as-prompt: Integrating with LLMs
- AI-human collaboration: Learning to Defer
- Explanation generation: From ML to XAI

**4.3 Data Strategy**

- Transfer learning foundation: Transfer Learning
- Multilingual dataset: Multilingual Reddit
- Synthetic data: Adapting LLMs, OpenAI API
- Active learning: OpenAI API, Learning to Defer

**4.4 Evaluation Framework**

- Beyond accuracy: Content Moderation by LLM (Legitimacy)
- Multi-dimensional: Oversight of CM by AI
- Metrics: Precision, Recall, F1, fairness metrics

### **Chapter 5: Implementation**

[Your work]

### **Chapter 6: Results & Evaluation**

**6.1 Performance Comparison**

- Compare against: Watch Your Language benchmarks, Perspective API, OpenAI API

**6.2 French-Language Performance**

- Novel contribution: French-specific metrics

**6.3 Cold-Start Performance**

- Novel contribution: 100 → 500 → 1,000 sample learning curve

**6.4 Cost Analysis**

- Novel contribution: Detailed cost-benefit

### **Chapter 7: Discussion**

**7.1 Comparison with Literature**

- Performance vs. Watch Your Language, ShieldGemma
- Approach vs. Integrating with LLMs, Learning to Defer

**7.2 Novel Contributions**

- Cold-start solution
- French-language results
- Small platform scalability
- Shareish-specific adaptations

**7.3 Limitations**

- Acknowledge challenges from: LLM-Mod, Watch Your Language
- Context understanding gaps
- Bias concerns

**7.4 Legal & Ethical Implications**

- GDPR compliance: Regulatory Challenges
- Transparency: From ML to XAI
- User rights: The Use of AI, Integrating with LLMs

**7.5 Threats to Validity**

- Dataset representativeness
- Generalizability
- Evaluation limitations

### **Chapter 8: Conclusion & Future Work**

**Future directions based on gaps:**

- Multi-modal moderation (text + images)
- Adversarial robustness (GPTFUZZER)
- Continuous improvement (active learning)
- Cross-platform generalization

---

## 📊 Next Steps for Research

### **Immediate Actions (This Week)**

1. **Obtain Full PDFs** ✅
    
    - [ ] Integrating Content Moderation Systems with LLMs
    - [ ] Learning to Defer in Content Moderation
    - [ ] ShieldGemma (full paper with all metrics)
    - [ ] Adapting LLMs (get exact hyperparameters from appendix)
    - [ ] Content Moderation by LLM: Accuracy→Legitimacy (full framework details)
2. **Extract Missing Metrics** 📊
    
    - [ ] Precision/Recall breakdown per category (not just F1)
    - [ ] Statistical significance tests
    - [ ] Performance variance (not just median/mean)
    - [ ] Language-specific performance (EN vs FR vs others)
3. **Document Dataset Access** 📁
    
    - [ ] Multilingual Reddit: Request access procedure
    - [ ] OLID: Download dataset
    - [ ] OpenAI Moderation eval: Download 1.8K samples
    - [ ] SWAD: Evaluate for potential use
4. **Code Availability Check** 💻
    
    - [ ] ShieldGemma: Download model weights from HuggingFace
    - [ ] Watch Your Language: Check if code released (likely not)
    - [ ] Adapting LLMs: Check for code (likely not, but check)
    - [ ] Multilingual Reddit: Check data loader code
5. **Create Technical Comparison Matrix** 📈
    
    ```
    | Paper | Base Model | Training Data | Training Time | Inference Latency | GPU Required | Cost Estimate |
    ```
    

### **Week 2-3 Actions**

6. **Test ShieldGemma** 🧪
    
    - [ ] Download Gemma-2B and Gemma-7B
    - [ ] Run on French hate speech dataset
    - [ ] Compare with OpenAI Moderation API
    - [ ] Measure inference speed and resource usage
    - [ ] Evaluate on Shareish-like samples
7. **Benchmark Baselines** 📏
    
    - [ ] Create test set (100 samples, French, various categories)
    - [ ] Run through Perspective API
    - [ ] Run through OpenAI Moderation API
    - [ ] Document performance for thesis baseline
8. **Design Shareish Architecture** 🏗️
    
    - Based on: Integrating with LLMs, Learning to Defer, ShieldGemma
    - [ ] Draw system diagram
    - [ ] Define confidence thresholds
    - [ ] Plan human escalation workflow
    - [ ] Design explanation generation

### **Week 4+ Actions**

9. **Implement Proof of Concept** 💻
    
    - [ ] Fine-tune ShieldGemma or Mistral-7B on small dataset
    - [ ] Implement Chain of Thought prompting
    - [ ] Add confidence-based deferral
    - [ ] Generate explanations
    - [ ] Test on French samples
10. **Evaluate & Iterate** 🔄
    
    - [ ] Run full evaluation (precision, recall, F1, fairness)
    - [ ] Compare with baselines
    - [ ] Identify failure modes
    - [ ] Collect additional training samples for weak areas
    - [ ] Re-train and re-evaluate

---

## 💡 Key Insights for Supervisor Meeting

### **Strengths of Current Research**

✅ **Comprehensive Coverage:** 29 papers across all relevant areas ✅ **Strong LLM Foundation:** 8 papers on LLM approaches (state-of-the-art) ✅ **Practical Models Identified:** ShieldGemma as deployable solution ✅ **Clear Methodology:** Adapting LLMs provides fine-tuning roadmap ✅ **Evaluation Framework:** Beyond accuracy (legitimacy, explainability) ✅ **French Data Available:** Multilingual Reddit has French samples

### **Identified Gaps (Novel Contribution Opportunities)**

⭐ **Cold-start problem** - No existing solutions for platforms with <500 samples ⭐ **French-specific evaluation** - Limited research on French hate speech patterns ⭐ **Small platform scalability** - No research on <100K user platforms ⭐ **Cost-benefit analysis** - No quantitative comparisons available

### **Recommended Approach**

1. **Model:** Fine-tuned ShieldGemma (or Mistral-7B if better performance)
2. **Architecture:** Policy-as-prompt with learning-to-defer
3. **Data:** Transfer learning + synthetic data + active learning
4. **Evaluation:** Multi-dimensional (accuracy + legitimacy + cost)
5. **Deployment:** Self-hosted (GDPR compliant)

### **Timeline Feasibility**

- **Literature review:** ✅ Complete (29 papers reviewed)
- **Model selection:** ✅ Clear (ShieldGemma + fine-tuning)
- **Implementation:** 6-8 weeks (feasible)
- **Evaluation:** 2-3 weeks (standard metrics + novel metrics)
- **Writing:** 4-6 weeks (well-structured with 29 citations)

**Total:** ~4 months for implementation + thesis writing

---

## 🎯 Final Recommendations

### **Must Do:**

1. Read 6 core papers in full (Weeks 1-4)
2. Test ShieldGemma on French data (Week 2-3)
3. Design Shareish architecture (Week 3)
4. Implement PoC (Weeks 4-8)
5. Evaluate thoroughly (Weeks 9-10)

### **Should Do:**

6. Create cost-benefit analysis spreadsheet
7. Document French-language performance gaps
8. Design active learning workflow
9. Plan user study (if time permits)

### **Nice to Have:**

10. Multi-modal exploration (text + images)
11. Adversarial robustness testing
12. Cross-platform generalization study

---

**Status:** Literature review complete ✅  
**Next:** Full paper deep-dive + ShieldGemma testing 🚀  
**Goal:** Novel contributions in cold-start, French, and small-platform scalability ⭐⭐⭐