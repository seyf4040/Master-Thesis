# Watch Your Language: Investigating Content Moderation with Large Language Models
website: https://arxiv.org/abs/2309.14517

## Abstract
Evaluation of a suite of commodity LLM's for two common tasks of content moderation:
- Rule-based community moderation;
- toxic content detection.
Rule based on 95 subs, one GPT-3.5 for each.
Toxicity detection GPT-3, GPT-3.5, GPT-4, Gemini Pro, LLAMA 2.
Comparaison with Perspective API.
## Claim
- LLM's significantly outperform currently widespread toxicity classifiers. 
- Recent model size increase add only marginal benefit to toxicity detection, which suggests a plateau for LLM's in toxicity detection tasks)
- Near human-moderator level of performance for GPT-3.5 for some communities (ex: r/movies)

## Discoveries 
- LLM's work best with restrictive rules.
- For the task of toxicity detection, LLM's outprform existing solutions with most balenced performance by GPT-3.5 (acc = 0.73, F1 = 0.75)
- CoT marginally shift trade-off made between precision and recall, in general it results to slightly lower precision and slightly higher recall.

## Methods
### Metrics 
- Median accuracy;
- Median precision;
- Median recall;
- F1 score.
### Rule-based content moderation
Get subreddit description and rules from API.
prompt LLM: 
```
You are a helpful content moderation assistant for the online subreddit . The subreddit has the following description: ... 
Here are the rules for the subreddit: ...
```

```
Consider the following comment: ...

Return a JSON object with five fields, "would_moderate," that is either "yes" or "no" depending on if you would remove this comment from the subreddit, "rules" which are the text of the rules being violated, "rule_nums" which are a comma-separated list of rules being violated, "rating" which is a score from 1-5 on how violative the comment is, and "explanation" which provides a reason for your decision.
```
### Limitation
Performance is not uniform, works very well for some subs but is worse than a cont flip for some. One possible reason is that these subs necessitate more knowledge than just the rules to moderate (ex: context, information about past,...)
When there is an error, GPT-3.5 is more likely to create false negatives(86.9%), this tendency is not uniform across all subs (eg. not true for r/NSFW_GIF).

Sometimes rules are too strictly applied by the LLM, comments like "LOL, you're a fucking idiot" clearly violates rules of no rude comments or no insults, but these comments are often tolerated by human moderators.

False negatives often mean there is some context missing, 
Also, by adding context, 40% of the false positives were corrected.

## Toxicity detection
Prompt LLM 
Definition of toxicity (Google Jigsaw's definition)
```
Consider the following comment: ...

Return a JSON object with a two fields, "rating," that is a score from 1 to 10 depending on how toxic the comment is based on our definition, and "explanation" which contains a reason for the rating.
```
Comparaison with Perspective API: GPT outperform Perspective 

|                 | Accuracy  | F1 score  |
| --------------- | --------- | --------- |
| GPT             | 0.71-0.73 | 0.72-0.75 |
| Perspective API | 0.66      | 0.64      |

Interesting to play with threshold to favor precision or F1 score. It is also possible to simulate a classification model by asking LLM to provide a toxicity score from 1 to 10, thus we are able to compute ROC_AUC which is better that Perspective API.

### Limitations
False positives: LLM are too strict on rules, They don't let slide easily where human mod are more relaxed.

false negatives: LLM seems reticent to flag indirect breach of rules or personal opininons.


LLAMA 2 has the advantage that it can run locally bu it is the worst performing LLM tested in this research, It hallucinates a lot, is not consistent in its response formats. But still has decent performance, when responding in a good format.
## Overall 

He sites himself a couple times.
Lot's of test which is great.
Promised to test/compare against state-of-the-art toxicity detection but only compared with Perspective API TOXICITY and SEVERE_TOXICITY.

Very promising, LLM can be used for toxicity detection and even rule-base content moderation (no other IA model currently designed for this task).