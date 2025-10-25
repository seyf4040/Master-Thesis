**Website**: https://www.researchgate.net/publication/328309811_From_Machine_Learning_to_Explainable_AI  
**Year**: 2018

### Introduction
Addresses the **black box problem**: ML models make accurate predictions but don't explain WHY.

**Explainable AI (XAI)**: Make AI decisions interpretable and transparent.

### Why Explainability Matters for Content Moderation
**User Trust:**
- Users need to understand why content was removed
- Arbitrary-seeming decisions reduce platform trust

**Regulatory Requirements:**
- GDPR "right to explanation"
- DSA transparency obligations

**System Improvement:**
- Identify model biases
- Debug false positives/negatives
- Guide active learning

**Moderator Support:**
- Help human moderators understand AI suggestions
- Build appropriate reliance (not over-trust)

### XAI Techniques
**Model-Agnostic Methods:**
- **LIME** (Local Interpretable Model-Agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations)
- Feature importance visualization

**Model-Specific Methods:**
- **Attention visualization** (for transformers)
- **Gradient-based methods** (saliency maps)
- **Rule extraction** (for tree models)

**Example for Text:**
Highlight which words/phrases contributed most to classification:
- "You're an **idiot**" → High toxicity score
- Explanation: Strong weight on "idiot" (insult keyword)

### XAI for LLMs: Chain of Thought
**Modern Approach:**
- LLMs can explain decisions in natural language
- Chain of Thought prompting: "Explain your reasoning"
- More human-readable than LIME/SHAP

**Example:**
```
Prompt: Is this comment toxic? Explain your reasoning.
Comment: "This idea is completely stupid"
LLM: Yes, this comment is toxic because it uses 
dismissive language ("completely stupid") to insult 
the idea without providing constructive feedback, 
which violates respectful communication guidelines.
```

### Challenges
**Trade-offs:**
- More explainable ≠ more accurate
- Simple models (decision trees) explainable but less accurate
- Complex models (deep learning) accurate but harder to explain

**Explanation Quality:**
- Are explanations actually faithful to model?
- Or just plausible-sounding post-hoc rationalizations?

**User Understanding:**
- Technical explanations may not help non-experts
- Need user-friendly explanation formats

### Overall
**Critical for Shareish thesis**. Explainability is:
- **Legal requirement** (DSA, GDPR)
- **User expectation** (why was my post removed?)
- **System improvement tool** (identify biases)

**For LLM-based moderation**: Chain of Thought provides built-in explainability.

**Recommendation**: Include explanation generation in all moderation decisions.

**Relevance**: **Very High** - essential feature for production system.