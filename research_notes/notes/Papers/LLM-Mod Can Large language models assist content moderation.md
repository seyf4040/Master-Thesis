# LLM-Mod: Can Large language models assist content moderation

website: https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=LLM-Mod%3A+Can+Large+language+models+assist+content+moderation.&btnG=

## Abstract
Question tackled: What is the reasoning capacity of LLM's whe handling rule violation in online communities. LLM-based moderator workflow using GPT-3.5. A key objective: evaluate reasoning of off-the-shelf LLM's

## Study workflow
Pre-task:
- Provide community guidelines;
- Ask LLM to:
	- Summarize rules:
	- Explain a particular rule;
	- Define key terms in a rule;
	- Come up with rude-violating posts beforehand;
	- Answer if a post breaks a certain rule;
	- Justify its decision;
- Provide any additional context beyond sample itself.
Evaluation:
- Ask the model variation of "Does the given post violate any of the community guidelines?".

### Metrics
Quantitative performance metrics:
- Precision;
- Recall;
- Identifying which guideline the model is unable to reason about;
- Identifying which subreddit category the model was able to reason the best.
human metrics:
- What kind of prompt engineering can help model reason about nuanced details;
- Why model may have incorrect decision;
- What are types of rules model has trouble reasoning about.

### Data
Test set size 600 rule-passing samples and 144 rule-violating samples.

## Results 
- Rules that necessitates knowledge about past (context,...) are not possible to enforce;
- Model struggled to gauge human emotion;
- Model can't fully grasp jokes;

- Level of reasoning needed to discern the rule rather that prompt engineering is the key factor for model's success

Two levels of reasoning
- Keyword association;
- Stance analysis.

Other strength and weaknesses:
+
- Able to give problematic part of rule-violating post.
-
- Not consistent or confident, when asked if it us sure about its prediction model often changes its stance.
- Not always able to identify violation despite understanding the rule.
## Overall
Very Small Test set size 744 samples total. violating sample are collected but also generated manually which is prone to bias, and not representative of real data.
Tested some of the problematic samples myself with GPT-4, it was successful in classifying them when the paper stated they were not classified correctly.
No use of fine-tuning. 
No training. 
Third-party hosted model was used.
