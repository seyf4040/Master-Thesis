# Integrating Content Moderation Systems with Large Language Models

**Website**: https://dl.acm.org/doi/abs/10.1145/3700789  
**Published**: ACM Transactions on the Web, October 2024  
**Authors**: Mirko Franco, Ombretta Gaggi, Claudio E. Palazzi

### Abstract
Online Social Networks rely on content moderation systems to ensure platform and user safety. However, there is growing consensus that such systems are unfair to historically marginalized individuals, fragile users, and minorities. OSN policies are often hardcoded in AI-based classifiers, making personalized content moderation challenging.

### Proposed Approach
**Integration of LLMs into content moderation pipeline to:**
1. Support **personal content moderation** (customizable rules per user)
2. Improve **user-platform communication**
3. Provide **in-depth explanations** for moderation decisions
4. Enable **chat-based appeals** process

### Key Innovation
**Policy-as-Prompt Framework:**
- Instead of hardcoded rules, use LLM prompts
- Can adapt to different personal preferences
- Provides explanations in natural language
- Enables dialogue with users about decisions

### Experiments
**Models Evaluated:**
- GPT-3.5
- LLaMA 2

**Comparison**: Performance compared to commercial products (likely Perspective API, OpenAI Moderation)

### Advantages Over Traditional Systems
- **Flexibility**: Change policies without retraining models
- **Personalization**: Different rules for different user groups
- **Explainability**: Natural language reasoning for decisions
- **User engagement**: Two-way communication about moderation

### Limitations Discussed
- LLM reasoning limitations
- Potential for bias in LLM responses
- Computational costs
- Need for human oversight for complex cases

### Overall
Very relevant paper proposing **practical integration approach**. Addresses critical issues of fairness, personalization, and communication. The "policy-as-prompt" concept aligns well with Shareish needs. However, paper doesn't provide full implementation details or code. Focus is on framework design rather than technical evaluation.

**Key Takeaway**: LLMs can make moderation more flexible and user-centric by replacing hardcoded rules with adaptable prompts.

**Relevance**: **Very High** - directly applicable to Shareish architecture design.
