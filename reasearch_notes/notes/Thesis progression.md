# Thesis Progress Summary

## Deep Learning for Content Moderation on Shareish

**Student:** Seyfullah Ural | **Date:** October 2025

---

## 📊 Current State

### Literature Review: ✅ COMPLETE

- **34 papers** reviewed and synthesized
- Comprehensive comparison table and relationship flowchart
- Gap analysis completed, research focus areas identified

**Key findings:**

- **LLM-based approaches** are state-of-the-art (Llama Guard 3, ShieldGemma)
- **Multi-dimensional evaluation** needed (beyond accuracy: legitimacy, fairness, explainability)
- **French language support** critical but underexplored

### Technical Approach: 🔄 UNDER CONSIDERATION

**Primary Model:** LLM-based

- Multilingual support required
- Customizable safety taxonomy
- Open-source options (Llama Guard 3, Mistral, ShieldGemma)

**Possible Architectures:**

1. **Two-Tier System:**
    - Stage 1: Detoxify (pre-filter) → Fast, low-cost screening
    - Stage 2: LLM-based → Detailed analysis with explanations
2. **Single-Step System:**
    - Direct LLM-based moderation

**Processing Strategy:** To be determined

- Real-time processing vs. batch processing
- Trade-offs: latency vs. throughput vs. cost

**Datasets Identified:**

- **ToxiGen:** 274K adversarial examples (solves cold-start)
- **HateCheck French:** 2K test cases (systematic evaluation)
- **Multilingual Reddit:** 20K French samples

---

## 🎯 What's Next

### Short-Term (Weeks 1-4)

**Goal:** Model selection and baseline testing

- **Compare candidate models** (Llama Guard 3, Mistral, ShieldGemma)
    - Performance on French content
    - Inference latency and cost
    - Customization capabilities
- **Test architectures:**
    - Single-step vs. two-tier system
    - Real-time vs. batch processing trade-offs
- Access and prepare datasets (ToxiGen, HateCheck, Reddit)

**Deliverable:** Architecture decision with comparative analysis

### Mid-Term (Weeks 5-12)

**Goal:** Implementation and optimization

- Implement chosen architecture
- Fine-tune selected model(s)
- Optimize for Shareish requirements (latency, cost, accuracy)
- Comprehensive evaluation (functional + adversarial + legitimacy)

**Deliverable:** Production-ready moderation system

### Long-Term (Weeks 13-20)

**Goal:** Complete thesis

- Draft all chapters (leveraging 34 papers)
- Present architecture justification and results
- Document evaluation and comparative analysis
- Finalize manuscript

**Deliverable:** Complete thesis document

---

## 🎯 Research Focus Areas

Based on literature gaps identified, the thesis will address:
1. **Cold-start problem:** Platforms with limited labeled data (<500 samples)
2. **French language moderation:** Systematic evaluation on French content
3. **Architecture for small platforms:** Comparative analysis of approaches
4. **Comprehensive evaluation:** Beyond accuracy metrics (legitimacy, fairness, cost-effectiveness)

---

## 📈 Timeline (20 Weeks / ~5 Months)

|Phase|Duration|Key Output|
|---|---|---|
|**Model Selection & Testing**|Weeks 1-4|Architecture decision + baseline|
|**Implementation & Fine-tuning**|Weeks 5-10|Optimized system|
|**Evaluation**|Weeks 11-12|Complete results|
|**Thesis Writing**|Weeks 13-20|Final manuscript|

**Target Completion:** March 2026

*This timeline provides a 3-month buffer before the June 2026 deadline, allowing time for supervisor feedback, revisions, and unexpected delays.*

---

## ❓ Questions for Supervisor

1. **Shareish Requirements:**
    - How many messages per day need moderation?
    - What's the acceptable latency? (real-time <500ms vs. batch <5min)
    - Historical moderation data available? (Need 200-500 minimum)

2. **Resources:**
    - GPU access for fine-tuning and inference? 
	    - Alan GPU cluster access available (connection issues to resolve)
    - Budget constraints for cloud services?
	    - Even for testing? if ever needed.

3. **Evaluation Priorities:**
    - Can we pilot on Shareish platform?
    - What matters most: accuracy, speed, or cost?
    - Ethical approval needed for user testing?

4. **Scope:**
    - Text only, or include images later?
    - Preference for specific model family (Meta vs. Mistral vs. Google)?

---

## ✅ Status Summary

**Completed:**
- ✅ Comprehensive literature review (34 papers)
- ✅ Candidate models identified (Llama Guard 3, Mistral, ShieldGemma)
- ✅ Datasets identified and accessible
- ✅ Evaluation framework designed
- ✅ Research focus defined

**Next Critical Decisions (Weeks 1-4):**
- 🔄 Final model selection (requires comparative testing)
- 🔄 Architecture choice (single-step vs. two-tier)
- 🔄 Processing strategy (real-time vs. batch)

**Blockers:**
- ⚠️ Need Shareish requirements clarity (latency, volume)
- ⚠️ Need clear/final definition of "toxic content" or "content to moderate", and Taxonomy
- ⚠️ Find solution for Alan GPU cluster access
- ⚠️ Data access details

---

**Ready to begin comparative testing phase immediately to make informed architecture decisions.**
