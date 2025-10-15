**Website**: https://arxiv.org/abs/2402.12237  
**Published**: arXiv preprint, February 2024

### Introduction
Addresses the **critical question**: When should AI make moderation decisions autonomously vs. defer to human moderators?

### Key Concept: Learning to Defer (L2D)
**Principle**: Train AI to recognize its own limitations and **defer difficult cases** to humans.

**Differs from traditional AI:**
- Most models try to predict all cases
- L2D models learn **when they're likely to be wrong**
- Explicitly trained to defer uncertain cases

### Framework
**Two Models:**
1. **Rejection model**: Decides whether to handle or defer
2. **Classification model**: Makes decision if not deferred

**Training objective**: Minimize overall error considering:
- AI error rate on retained cases
- Human decision quality on deferred cases
- Cost of human review

### Metrics
- **Coverage**: % of cases AI handles autonomously
- **Accuracy**: Performance on cases AI handles
- **Defer rate**: % sent to humans
- **System-wide accuracy**: Combined AI + human performance

### Key Findings
- AI should defer cases that are:
  - **Ambiguous** (low confidence)
  - **Context-dependent** (need world knowledge)
  - **Culturally sensitive** (high stakes)
- Optimal defer rate: 10-30% depending on cost/accuracy trade-off

### Human-AI Collaboration
**Advantages:**
- Better overall accuracy than AI-only or human-only
- Reduced human workload (70-90% handled by AI)
- AI learns from human decisions (active learning)

**Challenges:**
- Humans may over-rely on AI suggestions (automation bias)
- Need for diverse human reviewers to avoid bias
- Communication of uncertainty to human reviewers

### Overall
**Highly relevant to Shareish**. Proposes **practical hybrid approach** rather than full automation. Acknowledges AI limitations and explicitly plans for human oversight. The "learning to defer" concept could be integrated into fine-tuned LLM (use confidence scores to determine defer threshold).

**Key Takeaway**: Don't try to automate everything - design for AI-human collaboration from the start.

**Relevance**: **Very High** - directly applicable to system architecture design.