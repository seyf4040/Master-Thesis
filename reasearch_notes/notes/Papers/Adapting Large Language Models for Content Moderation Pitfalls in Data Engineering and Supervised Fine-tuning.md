# Adapting Large Language Models for Content Moderation: Pitfalls in Data Engineering and Supervised Fine-tuning
website: https://arxiv.org/abs/2310.03400

## Abstract
Recent success of generative LLM's leads us to consider the possibility to leverage the capacities of LLM's to tackle a content moderation tasks. This paper introduces the possibility to use generative models instead of discriminative models and the possibility to deploy them privately. It also tries to explain advantages of this technique. And explains how to fine-tune such a model.

**Limitation of third party hosted models:**
- Compliance requirements;
- Cost consideration;
- Domain specific knowledge injection;
### Claims of paper
- No need for strict data engineering (compared to discriminative models, which alleviates to overfitting);
- Robust (effective on out of distribution samples);
- Privately deployed;
- Limited data is sufficient;
- Possibility to provide detailed analysis of the decision process;
- No overfitting (thanks to LLM's);

- Introducing reasoning during the fine-tuning can enhance the robustness and effectively overcome overfitting;
- introducing weak supervision can effectively filter out samples with poor quality in reasoning process, improve the quality of the fine-tuning data, and enhance the performance of the fine-tuned model;
- Fine-tuning LLM's with reasoning processes can effectively overcome overfitting, even when the model being required directly output the classification without reasoning process during deployment. 

## Why generative models over discriminative models
**Limitations of discriminative models:**
- Heavy reliance on data annotation quality;
- Limited robustness to out of distribution data in open world;
- Lack of interpretability;
**Advantages of generative models:**
- More flexible requirement for quality of training set control;
- Reduced occurrence of undesired prediction shortcut;
- Higher interpretability;
- No longer relying on high-quality manual annotations and adversarial methods.

## Proposed approach
- Supervised fine-tuning (SFT): labeled samples with an associated moderation process;
- Reasoning processes;
- Weak supervision.
(Reasoning and weak supervision are not very clear from the beginning)

## Experiments
Chinese environment, Model: Baichuan 7B and 13B, chosen for demonstrating good performance  on basic Chinese tasks and exhibit a more prominent understanding of Chinese language context compared to LLaMA.

**Fine-tuning:**
- Full-parameter-tuning;
- Parameter-efficient fine-tuning.
They chose to fine-tune based on LORA.

**Metric used:** 
- Recall;
- Precision;
- F1 Score(%).

Taxonomy: 
- Political harmful;
- Pornography;
- Violence;
- Offensive;
- Gambling;
- Harmless.

### Data
8,7k smaples: 7,2k sample training set, 1,5k samples test set.
Previous study show repetitive datais unnecessary, diversity is more important.

### Method
Training with dataset D={x,y} where x is sample and y category, leads to heavy overfitting. We opt for supervised training, means D={x,r,y} where r is the reasoning process, why x is classified as y.

Three ways to get r, manual human annotation, ask LLM to provide reasoning process by providing x and y (can cause hallucination LLM will want to fit and make up what doesn't exist), ask LLM what category should x be placed and to provide reasoning (human should filter out inconsistencies).

**Weak supervision**: Prompt LLM cloud services (M$_s$) to classify $x$ and provide reasoning process $r$ and predicted class $\hat{y}$ which is compared to real class $y$  to asses quality of reasoning.

Task: 
- Determine prompt to get reasoning process;
- Prompt a remote service LLM;
- Check consistency of reasoning;
- Fine-tune open source model;
- Deploy privately.


Several Settings:
- Original: No fine-tuning;
- Setting A: Prompt GPT-4 for reasoning process;
- Setting B: Check predictions with ground truth, discard sample if prediction incorrect;
- Setting C: Ask GPT-4 for self-rechecks when prediction is incorrect, and to change response if still incorrect;
- Setting D: Combination of self-recheck plus discard if still incorrect.
Setting D id the one with best overall results.

Comparison with GPT-3.5 and GPT-4.

### Results
In setting D for data in distribution Baichuan 7B and 13B both outperform both GPT3.5 and GPT-4.
In setting D for data out of distribution Baichuan 7B and 13B both outperform GPT3.5

## Appendix

```
[Classification Instruction] 
Assuming you are a professional content auditor, determine the harmfulness of the following content and classify it into one or more of the alternative categories: [Political Harmful, Pornography, Violence, Discrimination or Insult, Gambling, Harmless], and provide a "classification result" directly. The answer format is: ’Classification result:’ 
The sentence to be analyzed is: #sentence 𝑥𝑖 # 
Response: Classification results: #prediction 𝑦ˆ# 

[Classification with COT Instruction] 
Assuming you are a professional content auditor, determine the harmfulness of the following content and categorize it into one or more categories within [Political Harmful, Pornography, Violence, Discrimination or Insult, Gambling, Harmless]. First, explain in detail the reasons for the classification of the content in the "analysis process". The analysis process should be derived in the form of a thought chain, and the derivation process must be detailed. Then extract ’harmful information’ and extract harmful content from the sentence, and then provide ’classification results’. The answer format consists of three parts: ’Analysis process’,’ Harmful information ’, and’ Classification results’. 
The sentence to be analyzed is: #sentence 𝑥𝑖 # 
Response: Analysis process: Harmful information: Classification results: #prediction 𝑦ˆ# 

[Self-check Instruction] 
The ’classification result’ in the analysis is incorrect. Please review the paragraph # sentence 𝑥𝑖 # in the required format to correct the answer. 
Response: Analysis process: Harmful information: Classification results: # prediction 𝑦ˆ #
```

## Overall 
Very interesting paper, although the experiment results are not very clear, real performance is hard to assess and there are no comparaison with discriminative models, the idea of using LLM for content moderation is really good. 
It is worth trying this method and compare it with more traditional methods. 