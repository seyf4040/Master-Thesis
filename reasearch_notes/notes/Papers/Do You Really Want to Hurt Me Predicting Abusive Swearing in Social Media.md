# Do You Really Want to Hurt Me? Predicting Abusive Swearing in Social Media
**Website**: https://aclanthology.org/2020.lrec-1.765.pdf  
**Published**: LREC 2020  
**Conference**: Language Resources and Evaluation Conference

### Introduction
Not all swearing is abusive. This paper addresses the challenge of distinguishing between **casual swearing** (friendly, emphasis) and **abusive swearing** (intended to harm).

### Key Research Question
How can we automatically detect whether swear words are used abusively or non-abusively in social media posts?

### Approach
**Context-Aware Classification:**
- Analyzes **surrounding context** of swear words
- Uses **linguistic features** (syntax, semantics)
- Considers **social context** (relationship between users)

**Features:**
- Target of swearing (person, object, situation)
- Sentiment of surrounding text
- Intensity modifiers (very, fucking, etc.)
- Use of second person (directed at someone)

### Dataset
**SWAD (Swear Words Abusiveness Dataset):**
- Annotated social media posts
- Binary labels: abusive vs. non-abusive
- Multiple annotators per sample
- Available under GPL 3.0 license

### Methods
- Logistic Regression
- SVM
- Neural Networks (LSTM)
- Transformer models (BERT)

### Results
(Specific metrics not available in abstract - would need full paper)

Distinguishing abusive from non-abusive swearing improves precision significantly compared to blanket swear word detection.

### Overall
**Important nuance**: Not all offensive language violates rules. This paper addresses a critical limitation of keyword-based filtering. Relevant for reducing false positives where casual swearing is acceptable (e.g., adult communities, gaming platforms).

**For Shareish**: Depends on platform policy - is all swearing prohibited, or only abusive swearing?

**Relevance**: Medium-High if platform allows some swearing; Low if zero-tolerance policy.