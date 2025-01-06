# Content Moderation by LLM: From Accuracy to Legitimacy

website: https://arxiv.org/abs/2409.03219#:~:text=One%20trending%20application%20of%20LLM,makes%20correct%20decisions%20about%20content

## Introduction
Paper argues that accuracy is not a good metric to reflect performance of LLM in content moderation tasks. And that improving accuracy is not the true advancement LLMs offer, but rather their ability to justify and establish legitimacy.
**Four fields LLM can offer improvements:**
- conduct screening of hard cases from easy cases;
- provide quality explanation for moderation decision;
- assist human reviewers in getting more contextual information;
- facilitate user participation in a more interactive way.
## The Accuracy Discourse
Here Precision and recall are considered similar to accuracy in the sens that they measure capability of making correct decisions, accuracy measures correct decisions and precision and recall measure capacity to avoid erroneous decisions.

- **Impossible to reach 100% accuracy in real life, and dangerous to try to reach it (Overfitting):** Ground truth is not easy to determine;
- **Focus only on individual aspect and not systemic aspect:** System accuracy is not aggregation of accuracy on individual cases but it refers to general performance of the whole moderation system, including metrics like:
	- consistency;
	- predictability;
	- fairness of error distribution across different groups of user, different categories and different periods of time;
- **Failed to recognize easy and hard cases (because they should be dealt with differently):** 
	- Distinction already exists (eg. Meta);
	- Also distinguished in legal systems;
	- Different social impacts;µ
	- Hard cases should not be dealt with by the moderation agent but by human agents;
- **Overlook other important aspects:**
	- Moderation should be seen as a part of governance systems on platforms;
	- No legal compulsion to moderate but voluntary decision to:
		- Prevent regulations;
		- Promote public image;
		- Make product more profitable;
	- Right protection: delineation of free speech when it conflicts with other rights

As accuracy is not perfect, The important questions become:
- who suffers from false positive and false negatives?
- What errors are acceptable?

Alternative: ???


## Limitation of traditional ML
- Heavily relies on manual annotation of training dataset;
	- Costly;
	- Introduce bias;
- Lack flexibility and adaptability;
	- One models can't perform well on different environments;
	- Can not adapt model with time;
- Lack explainability and transparency;
	- Can not provide explanation to user.

Main difference:
ML is target-trained >< LLM is pre-trained on huge corpus of data.

LLM models are trained on a much larger dataset than traditional ML models this enable LLMs to better appreciate context and nuances, to generalize across domains,...

LLM using transformers and especially self-attention layers can better understand context, linking words that are not necessarily neighbors.

LLM decision process is decoupled from training process, allows flexibility, change prompt and you have a model adapted fer a task that could be very different than he previous one.

#todo 
- [ ] Research Transformers
- [ ] Research attention layers, particularly self-attention

## Continuity
Paper then exposes a Legitimacy-based framework for content moderation. 

## Overall
Paper proposes a new way of using LLM for content moderation, instead of using LLM for their accuracy (which is not their best advantage for this paper), use the LLM to scan for easy cases and moderate them, then flag hard cases for human moderation.
Accuracy as a metric to characterise performance is critiqued but no concrete alternative is given. Although the scope changes from individual to systemic (where consistency and predictability are proposed), no other metric for individual cases proposed. 
Very interesting paper, with solid arguments, well thought limitations and a suitable framework.