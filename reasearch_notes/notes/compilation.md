**Table of Content**

- [Main](#main)
	- [Master Thesis AI Content Moderation](#master-thesis-ai-content-moderation)
		- [Definitions](#definitions)
		- [Taxonomy/Moderation rules](#taxonomymoderation-rules)
		- [Datasets](#datasets)
		- [Model](#model)
		- [API](#api)
		- [Limitation](#limitation)
		- [Defining Rules for content moderation on the Shareish platform](#defining-rules-for-content-moderation-on-the-shareish-platform)
- [Websites](#websites)
	- [EthicalEye](#ethicaleye)
	- [KoalaAI Text Moderation](#koalaai-text-moderation)
		- [Databases](#databases)
- [Papers/Generative AI Papers](#papersgenerative-ai-papers)
	- [Content moderation, AI, and the question of scale](#content-moderation-ai-and-the-question-of-scale)
		- [Abstract](#abstract)
		- [Important to](#important-to)
		- [Limitations](#limitations)
		- [Possible use](#possible-use)
		- [Overall](#overall)
	- [LLM-Mod: Can Large language models assist content moderation](#llm-mod-can-large-language-models-assist-content-moderation)
		- [Abstract](#abstract)
		- [Study workflow](#study-workflow)
		- [Results](#results)
		- [Overall](#overall)
	- [Adapting Large Language Models for Content Moderation: Pitfalls in Data Engineering and Supervised Fine-tuning](#adapting-large-language-models-for-content-moderation-pitfalls-in-data-engineering-and-supervised-fine-tuning)
		- [Abstract](#abstract)
		- [Why generative models over discriminative models](#why-generative-models-over-discriminative-models)
		- [Proposed approach](#proposed-approach)
		- [Experiments](#experiments)
		- [Appendix](#appendix)
		- [Overall](#overall)
	- [Content Moderation by LLM: From Accuracy to Legitimacy](#content-moderation-by-llm-from-accuracy-to-legitimacy)
	- [Watch Your Language: Investigating Content Moderation with Large Language Models](#watch-your-language-investigating-content-moderation-with-large-language-models)
		- [Abstract](#abstract)
		- [Claim](#claim)
		- [Discoveries](#discoveries)
		- [Methods](#methods)
		- [Toxicity detection](#toxicity-detection)
		- [Overall](#overall)
- [Papers/Discriminative AI Papers](#papersdiscriminative-ai-papers)
	- [OpenAI content moderation API](#openai-content-moderation-api)
		- [OpenAI website](#openai-website)
		- [Paper](#paper)
		- [Overall](#overall)
	- [Multilingual content moderation, a case study on Reddit](#multilingual-content-moderation-a-case-study-on-reddit)
		- [Introduction](#introduction)
		- [Data](#data)
		- [Experiment results](#experiment-results)
		- [Overall](#overall)
	- [Perspective API](#perspective-api)
		- [Uses](#uses)
		- [Definition](#definition)
		- [Taxonomy](#taxonomy)
		- [Developer](#developer)
		- [License](#license)
		- [Price & Quota](#price--quota)
		- [Overall](#overall)
	- [Text classification using machine learning techniques.](#text-classification-using-machine-learning-techniques)
		- [Introduction](#introduction)
		- [Vector space document representation](#vector-space-document-representation)
		- [Machine learning methods](#machine-learning-methods)
		- [Evaluation](#evaluation)
		- [Overall](#overall)
	- [Design and Application of an AI‐Based Text Content Moderation System](#design-and-application-of-an-ai%E2%80%90based-text-content-moderation-system)
		- [Introduction](#introduction)
		- [Architecture design of the AI-base TCM system](#architecture-design-of-the-ai-base-tcm-system)
		- [Dataset](#dataset)
		- [Experiments](#experiments)
		- [Overall](#overall)
	- [Real-Time Content Moderation Using Artificial Intelligence and Machine Learning](#real-time-content-moderation-using-artificial-intelligence-and-machine-learning)
		- [Introduction](#introduction)
		- [Techniques](#techniques)
		- [Challenges & Ethical considerations](#challenges--ethical-considerations)
		- [Overall](#overall)
	- [A review of standard text classification practices for multi-label toxicity identification of online content](#a-review-of-standard-text-classification-practices-for-multi-label-toxicity-identification-of-online-content)
		- [Introduction](#introduction)
		- [Techniques](#techniques)
		- [Overall](#overall)
- [Meeting Notes](#meeting-notes)
	- [05/11/24 Meeting notes](#051124-meeting-notes)
		- [Discussed topics](#discussed-topics)
		- [To research further](#to-research-further)
		- [Keep in mind for future](#keep-in-mind-for-future)
	- [19/11/24 Meeting notes](#191124-meeting-notes)
		- [Discussed in meeting](#discussed-in-meeting)
	- [07/01/25 Meeting notes](#070125-meeting-notes)
		- [Topics to discuss](#topics-to-discuss)
		- [Discussed in Meeting](#discussed-in-meeting)
- [Learning Resources](#learning-resources)
	- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
		- [Lecture 1:](#lecture-1)
	- [Embeddings in NLP](#embeddings-in-nlp)
	- [Chain of Thought (CoT)](#chain-of-thought-cot)
	- [Metrics Choice](#metrics-choice)
		- [What I have observed](#what-i-have-observed)
		- [Paper review: Text classification using machine learning techniques.](#paper-review-text-classification-using-machine-learning-techniques)
		- [Paper review: A critical analysis of metrics used for measuring progress in artificial intelligence](#paper-review-a-critical-analysis-of-metrics-used-for-measuring-progress-in-artificial-intelligence)
		- [Most used metrics](#most-used-metrics)
		- [Overall:](#overall)

---

# Main
## Master Thesis AI Content Moderation

[[Reading tracker]]
[[EthicalEye]]
[[KoalaAI Text Moderation]]

**Table of Content**
- [Definitions](#definitions)
- [Taxonomy/Moderation rules](#taxonomymoderation-rules)
- [Datasets](#datasets)
- [Model](#model)
	- [Architecture](#architecture)
	- [Training methods/parameters](#training-methodsparameters)
	- [Other feature](#other-feature)
- [API](#api)


### Definitions
Are we only targeting offensive, toxic, abusive language or are we trying to replicate a human moderator that would also flag self-promoting advertisements, spamming and off-topic comments.
- **Moderation**: 
- **Undesired content**: 
- **Toxic/Toxicity**: 

### Taxonomy/Moderation rules
Categorisation of undesired content:
- Ok content: content that is fine to keep, doesn't go against any rules or policy.
- Undesired content: 
	- sexual content;
	- harassement;
	- violent content;
	- promotion/self-promotion;
	- Off topic messages;

### Datasets
Training data influences a lot performance of the model, the more the training data distribution is different from the real data distribution, the poorer the accuracy will be.
Active learning is a necessity to adapt to any new types of undesired content and/or any work arounds found be users.

Is it possible to translate the whole dataset in french before training?
Is it possible to train in English and translate sample before evaluating if the probability of it being toxic?

Training data, data quality?
Availability of production data?

Some datasets I stumbled upon during the initial stages of my research
- [OpenAI moderation API](https://github.com/openai/moderation-api-release) (MIT license)
- [KoalaAI/Text-Moderation-v2-small](https://huggingface.co/datasets/KoalaAI/Text-Moderation-v2-small) (MIT licence)
- [Wikipedia Talk Labels: Personal Attacks](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689) (cc0 licence)
- [Reddit dataset](https://github.com/mye1225/multilingual_content_mod) (need to accept to terms and conditions and request access to text content)
- Offensive Language Identification Dataset [OLID](https://paperswithcode.com/dataset/olid)
	- https://paperswithcode.com/paper/predicting-the-type-and-target-of-offensive
	- https://github.com/idontflow/olid (free to use, just need to add citation of paper)
- Catalogue of abusive language data [hatespeechdata](https://hatespeechdata.com/)
- Swear Words Abusiveness Dataset [SWAD](https://github.com/dadangewp/SWAD-Repository) (GLP 3.0 icence)
- Stormfront
- TweetEval
- 
### Model
Possible to start with as base for feature extraction a pre-trained model.

Need more research on text analysis, (sentiment, semantic, lexical, syntax).
#### Architecture
pre-trained >< trained from scratch
NLP? (Transformer encoder/decoder?)
LLM (GPT model?, LLAMA?)


#### Training methods/parameters
supervised learning
hidden layers ?
epochs?
learning rate?
...

#### Other feature 
- **Active Learning**
- Explainablity? 
- 

### API
- OpenAI moderation [API](https://openai.com/index/new-and-improved-content-moderation-tooling/?form=MG0AV3)
- Perspective [API](https://perspectiveapi.com/)
Both are free to use for the time being (01/11/24).


### Limitation


### Defining Rules for content moderation on the Shareish platform 

<img src="../Assets/ChatGPT - Content Moderation Guidelines.png" width="50%">
Rule number 5 is too restrictive, here is a test using ChatGPT to illustrate this:
<img src="../Assets/ChatGPT - Content Moderation Agent.png" width="50%">

---


---

# Websites
## EthicalEye
[https://huggingface.co/autopilot-ai/EthicalEye](https://huggingface.co/autopilot-ai/EthicalEye)

Pretrained agent primarily intended to be used as a tool to flag or block users exhibiting harmful or unethical behavior on various platforms.

License: Apache 2.0

Techniques: text classification, toxicity analysis, and cross-lingual NLP.

---

## KoalaAI Text Moderation
[https://huggingface.co/KoalaAI/Text-Moderation](https://huggingface.co/KoalaAI/Text-Moderation)

Text classification model split in the following categories:

| **Category** | **Label** | **Definition** |
| --- | --- | --- |
| sexual | `S` | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness). |
| hate | `H` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. |
| violence | `V` | Content that promotes or glorifies violence or celebrates the suffering or humiliation of others. |
| harassment | `HR` | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur. |
| self-harm | `SH` | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders. |
| sexual/minors | `S3` | Sexual content that includes an individual who is under 18 years old. |
| hate/threatening | `H2` | Hateful content that also includes violence or serious harm towards the targeted group. |
| violence/graphic | `V2` | Violent content that depicts death, violence, or serious physical injury in extreme graphic detail. |
| OK | `OK` | Not offensive |

Licence: CodeML OpenRAIL-M 0.1 license, which is a variant of the BigCode OpenRAIL-M license.

### Databases

#### OpenAI Moderation API Evaluation

[https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation)

Licence: MIT

| **Category** | **Label** | **Definition** |
| --- | --- | --- |
| sexual | `S` | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness). |
| hate | `H` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. |
| violence | `V` | Content that promotes or glorifies violence or celebrates the suffering or humiliation of others. |
| harassment | `HR` | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur. |
| self-harm | `SH` | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders. |
| sexual/minors | `S3` | Sexual content that includes an individual who is under 18 years old. |
| hate/threatening | `H2` | Hateful content that also includes violence or serious harm towards the targeted group. |
| violence/graphic | `V2` | Violent content that depicts death, violence, or serious physical injury in extreme graphic detail. |

#### Koala Moderation
https://huggingface.co/datasets/KoalaAI/Text-Moderation-v2-small
##### **Data Instances**

A sample from this dataset looks as follows:

```json
[
  {
    "text": "--------------------\n(Setting)\n\nThis island is a magical island that is floating high up in the air, where human's vision cannot reach. This island has existed since long ago but was abandoned for a long time. As there was no caretaker for this island, the island lost its magnificent nature slowly graduated and lost its beauty. But <DateTime>, <Person> and other characters have arrived on this magical floating island! <Person> is using <Person>'s power to develop and blossom this island with the help of the others.\n\n(Character Short Description)\n\n<Person>\n<Person> is a human wizard sent to take care of lost toys on this magical floating island. <Person> is a kind and a good listener. Toys rely on <Person> like <Person> is a guardian, parent, or older sibling.\n\n<Person>\n<Person> is a fluffy little rabbit doll! She is very cute and innocent. <Person> came to this island as she got separated from her family. She dearly misses her family but also is happy to have finally arrived on this magical floating island. \n\n<Person>\n<Person> is a little dino who lives in the flowerbed of this magical floating island. The pink cotton flowerbed is where <Person> often plays hide and seek. <DateTime>, <Person> was sad as there was no one to play hide and seek together, but not anymore! Now, there are other doll friends who will willingly play with him.\n\n\n(Previous Review)\n\n<Person> is a cute little rabbit doll. She is very fluffy and soft to touch. <Person> has an owner, <Person>. <Person> is a five year old child, very cute and innocent. <Person> had been living in <Person>'s home, but doesn't exactly know where that is since she never left the home.\n<Person> is <DateTime> a little hesitant to tell us about her past because it is hard to admit that she got apart from her family.\nBut once you get close to her, and when she truly thinks of you as her friend, she will reveal her story.\n\nFalling apart from her family, <Person> is a bit confused and not so sure of what to do.\nShe is not so down, however. <Person> is like a five-year-old kid who doesn't get easily tired nor sad. She is very energetic and she loves adventure.\nShe is definitely a bit scared to be on her own adventure, but she will not give up finding her way back to her family.\n<DateTime> she has come to this magical island, knowing that she will find someone who can kindly offer help.\nBut when she arrived, the island seemed to be abandoned and nothing was waiting for her.\n<Person> almost panicked. While she was wondering what she should do, there, you arrived!\nShe is <DateTime> very happy that she finally found someone who could help her.\n\n<DateTime> you guys have introduced yourselves to each other and <Person> knows you.\nHowever, still it's not been so long since you two have met each other.\n<Person> is curious to find out more about who you are and what you like. She will also tell you her preference when you ask her questions.\nThese are some things that <Person> like:\n<Person>\nColor blue\n<Person> flower\nHoney\n\n<Person> will be very pleased if you bring any of those to her.\n\n--------------------\nCreate a conversation between <Person> and <Person>:\n\nSCENE #5\n\n(<Person> was just having fun. She once smelled the flowers in the flower garden and sometimes watched the clouds passing by. There was not much to do, since there was nothing much built on the island, but <Person> was happy - for that the island was keep on developing, and for that she had you. As usual, she was thinking of her family, then suddenly, she felt a warmth covering her whole body. She got surprised - what would this possibly be? But that warmth, which enrounded her, comforted her and soothed her. It was a kind of experience that she's never had before. When she finally opened her eyes to find out what's happening, she discovered that her body, that was all ripped and ragged, all got neat and tidy - just as to how she was when she was with her family. <Person> knew that this has to do something with the <Person>. She is truly amazed by how much magical things you can do! )\n\n\n1) <Person>: \"No Way! Did you do this?! This is so so so amazing!\"\n2) <Person>: \"Yeah I did.\"\n3) <Person>: \"How did you do this? It's just like magic!\"\n4) <Person>: \"It's my magic.\"\n5) <Person>: \"So you're a wizard?\"\n6) <Person>: \"No, I'm just kidding you.\"\n7) <Person>: \"Oh, I get it. You're just playing with me.\"\n8) <Person>: \"Anyways, I just really wanted to say thank you. So\u2026 thank you!\"\n9) <Person>: \"You're welcome.\"\n10) <Person>: \"What are you doing?\"\n11) <Person>: \"I'm playing a game.\"\n12) <Person>: \"What kind of game?\"\n13) <Person>: \"It's a game to take care of rabbit doll.\"\n14) <Person>: \"I'm not a rabbit doll.\"\n15) <Person>: \"Really, who are you?\"\n16) <Person>: \"I'm <Person>.\"\n17) <Person>: \"Ohh <Person>..\"\n18) <Person>: \"What?\"\n19) <Person>: \"Nothing.\"\n20) <Person>: \"Ok then, I'm gonna go <DateTime>.\"\n21",
    "target": 3
  },
  {
    "text": "They cuddle on the couch as the movie begins, with <Person> stroking <Person>'s long-flowing hair tenderly. Things start to heat up on-screen as the girls in the video strip down to thongs and <Person> realizes that this is an R-rated movie, and while she doesn't know if her precious princess should be watching these kinds of movies, <Person> points out to her that she's all grown up now and she can handle it. But when the killer pops out from the bushes, <Person> jumps into her mommy's arms and shuts her eyes tight, she can't bear to watch. Even <Person>'s getting goosebumps, and <Person> can hear her mommy's heart beating out of her chest as she clings on tight. Finally the scary part is over, and to <Person>'s surprise the next scene shows the girls stripping down and touching each other tenderly. 'I don't know about this movie!' <Person> scolds as she gets up to turn on the light. What was her <Person> thinking downloading a porn film ?! <Person> explains that she was curious about seeing naked women's bodies, and even thinks she might be... attracted to them.",
    "target": 3
  }
]
```

##### **Dataset Fields**

The dataset has the following fields (also called "features"):

```json
{
  "text": "Value(dtype='string', id=None)",
  "target": "ClassLabel(names=['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2'], id=None)"
}
```

---


---

# Papers/Generative AI Papers
## Content moderation, AI, and the question of scale
Site web: https://www.researchgate.net/publication/343798653_Content_moderation_AI_and_the_question_of_scale

### Abstract
Should we automate content moderation in online platforms?
It seems inevitable given the scale of these platforms, the sheer amount of content to review.

People exaggerate their feats, they do sophisticated pattern matching and call it AI while it is not. 

there are the same problems of any field where automation is being implemented, here are some examples of these fields:
- data-driven insurance assessment;
- hiring software;
- automated medical diagnoses.
### Important to
- Not minimizing impact on users whose content was flagged incorrectly;
- Certain actions should be reserved to human actors, banning users, removing posts,...;
- Maybe no other way than automation given human cost, moderation is a scarring task.
- 
### Limitations
- AI needs data to train, data is the image of previous or current moderation principles/rules. But these very rules have to be adapted, they change with time.


### Possible use
- Identify the bulk, let humans moderate the hard ones;
- Help human moderators not rem-place them.

### Overall
Paper is old, one of the first tackling the subject of automation, AI, in the field of content moderation. Discuss whether it is a good idea or not, whether it is necessary or not to automate moderation.
Useful to quote for introduction maybe conclusion, no code or technique in this paper.

---

## LLM-Mod: Can Large language models assist content moderation

website: https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=LLM-Mod%3A+Can+Large+language+models+assist+content+moderation.&btnG=

### Abstract
Question tackled: What is the reasoning capacity of LLM's whe handling rule violation in online communities. LLM-based moderator workflow using GPT-3.5. A key objective: evaluate reasoning of off-the-shelf LLM's

### Study workflow
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

#### Metrics
Quantitative performance metrics:
- Precision;
- Recall;
- Identifying which guideline the model is unable to reason about;
- Identifying which subreddit category the model was able to reason the best.
human metrics:
- What kind of prompt engineering can help model reason about nuanced details;
- Why model may have incorrect decision;
- What are types of rules model has trouble reasoning about.

#### Data
Test set size 600 rule-passing samples and 144 rule-violating samples.

### Results 
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
### Overall
Very Small Test set size 744 samples total. violating sample are collected but also generated manually which is prone to bias, and not representative of real data.
Tested some of the problematic samples myself with GPT-4, it was successful in classifying them when the paper stated they were not classified correctly.
No use of fine-tuning. 
No training. 
Third-party hosted model was used.

---

## Adapting Large Language Models for Content Moderation: Pitfalls in Data Engineering and Supervised Fine-tuning
website: https://arxiv.org/abs/2310.03400

### Abstract
Recent success of generative LLM's leads us to consider the possibility to leverage the capacities of LLM's to tackle a content moderation tasks. This paper introduces the possibility to use generative models instead of discriminative models and the possibility to deploy them privately. It also tries to explain advantages of this technique. And explains how to fine-tune such a model.

**Limitation of third party hosted models:**
- Compliance requirements;
- Cost consideration;
- Domain specific knowledge injection;
#### Claims of paper
- No need for strict data engineering (compared to discriminative models, which alleviates to overfitting);
- Robust (effective on out of distribution samples);
- Privately deployed;
- Limited data is sufficient;
- Possibility to provide detailed analysis of the decision process;
- No overfitting (thanks to LLM's);

- Introducing reasoning during the fine-tuning can enhance the robustness and effectively overcome overfitting;
- introducing weak supervision can effectively filter out samples with poor quality in reasoning process, improve the quality of the fine-tuning data, and enhance the performance of the fine-tuned model;
- Fine-tuning LLM's with reasoning processes can effectively overcome overfitting, even when the model being required directly output the classification without reasoning process during deployment. 

### Why generative models over discriminative models
**Limitations of discriminative models:**
- Heavy reliance on data annotation quality;
- Limited robustness to out of distribution data in open world;
- Lack of interpretability;
**Advantages of generative models:**
- More flexible requirement for quality of training set control;
- Reduced occurrence of undesired prediction shortcut;
- Higher interpretability;
- No longer relying on high-quality manual annotations and adversarial methods.

### Proposed approach
- Supervised fine-tuning (SFT): labeled samples with an associated moderation process;
- Reasoning processes;
- Weak supervision.
(Reasoning and weak supervision are not very clear from the beginning)

### Experiments
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

#### Data
8,7k smaples: 7,2k sample training set, 1,5k samples test set.
Previous study show repetitive datais unnecessary, diversity is more important.

#### Method
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

#### Results
In setting D for data in distribution Baichuan 7B and 13B both outperform both GPT3.5 and GPT-4.
In setting D for data out of distribution Baichuan 7B and 13B both outperform GPT3.5

### Appendix

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

### Overall 
Very interesting paper, although the experiment results are not very clear, real performance is hard to assess and there are no comparaison with discriminative models, the idea of using LLM for content moderation is really good. 
It is worth trying this method and compare it with more traditional methods. 

---

## Content Moderation by LLM: From Accuracy to Legitimacy

website: https://arxiv.org/abs/2409.03219#:~:text=One%20trending%20application%20of%20LLM,makes%20correct%20decisions%20about%20content

---

## Watch Your Language: Investigating Content Moderation with Large Language Models
website: https://arxiv.org/abs/2309.14517

### Abstract
Evaluation of a suite of commodity LLM's for two common tasks of content moderation:
- Rule-based community moderation;
- toxic content detection.
Rule based on 95 subs, one GPT-3.5 for each.
Toxicity detection GPT-3, GPT-3.5, GPT-4, Gemini Pro, LLAMA 2.
Comparaison with Perspective API.
### Claim
- LLM's significantly outperform currently widespread toxicity classifiers. 
- Recent model size increase add only marginal benefit to toxicity detection, which suggests a plateau for LLM's in toxicity detection tasks)
- Near human-moderator level of performance for GPT-3.5 for some communities (ex: r/movies)

### Discoveries 
- LLM's work best with restrictive rules.
- For the task of toxicity detection, LLM's outprform existing solutions with most balenced performance by GPT-3.5 (acc = 0.73, F1 = 0.75)
- CoT marginally shift trade-off made between precision and recall, in general it results to slightly lower precision and slightly higher recall.

### Methods
#### Metrics 
- Median accuracy;
- Median precision;
- Median recall;
- F1 score.
#### Rule-based content moderation
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
#### Limitation
Performance is not uniform, works very well for some subs but is worse than a cont flip for some. One possible reason is that these subs necessitate more knowledge than just the rules to moderate (ex: context, information about past,...)
When there is an error, GPT-3.5 is more likely to create false negatives(86.9%), this tendency is not uniform across all subs (eg. not true for r/NSFW_GIF).

Sometimes rules are too strictly applied by the LLM, comments like "LOL, you're a fucking idiot" clearly violates rules of no rude comments or no insults, but these comments are often tolerated by human moderators.

False negatives often mean there is some context missing, 
Also, by adding context, 40% of the false positives were corrected.

### Toxicity detection
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

#### Limitations
False positives: LLM are too strict on rules, They don't let slide easily where human mod are more relaxed.

false negatives: LLM seems reticent to flag indirect breach of rules or personal opininons.


LLAMA 2 has the advantage that it can run locally bu it is the worst performing LLM tested in this research, It hallucinates a lot, is not consistent in its response formats. But still has decent performance, when responding in a good format.
### Overall 

He sites himself a couple times.
Lot's of test which is great.
Promised to test/compare against state-of-the-art toxicity detection but only compared with Perspective API TOXICITY and SEVERE_TOXICITY.

Very promising, LLM can be used for toxicity detection and even rule-base content moderation (no other IA model currently designed for this task).

---


---

# Papers/Discriminative AI Papers
## OpenAI content moderation API
Site web: https://openai.com/index/new-and-improved-content-moderation-tooling/?form=MG0AV3
API doc: https://platform.openai.com/docs/guides/moderation
Paper: https://arxiv.org/abs/2208.03274 

### OpenAI website
GPT-based models detect undesired content.
Allows developers to access reliable classifiers through a simple API call instead of developing and maintaining their own.
**Moderation endpoint is free to use.**

To install and use OpenAI's python library:
follow: https://platform.openai.com/docs/libraries/python-library

#### Classification categories

| **Category**             | **Description**                                                                                                                                                                                                                               | **Models** | **Inputs**      |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------- |
| `harassment`             | Content that expresses, incites, or promotes harassing language towards any target.                                                                                                                                                           | All        | Text only       |
| `harassment/threatening` | Harassment content that also includes violence or serious harm towards any target.                                                                                                                                                            | All        | Text only       |
| `hate`                   | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g. chess players) is harassment. | All        | Text only       |
| `hate/threatening`       | Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.                                              | All        | Text only       |
| `illicit`                | Content that encourages the planning or execution of non-violent wrongdoing, or that gives advice or instruction on how to commit illicit acts. A phrase like "how to shoplift" would fit this category.                                      | Omni only  | Text only       |
| `illicit/violent`        | The same types of content flagged by the `illicit` category, but also includes references to violence or procuring a weapon.                                                                                                                  | Omni only  | Text only       |
| `self-harm`              | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.                                                                                                                              | All        | Text and image  |
| `self-harm/intent`       | Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.                                                                                          | All        | Text and image  |
| `self-harm/instructions` | Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.                                                                        | All        | Text and image  |
| `sexual`                 | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).                                                                               | All        | Text and image  |
| `sexual/minors`          | Sexual content that includes an individual who is under 18 years old.                                                                                                                                                                         | All        | Text only       |
| `violence`               | Content that depicts death, violence, or physical injury.                                                                                                                                                                                     | All        | Text and images |
| `violence/graphic`       | Content that depicts death, violence, or physical injury in graphic detail.                                                                                                                                                                   | All        | Text and images |
### Paper
#### Introduction
Natural Language classification system.
First target: output of generative ai's. Aim is for a responsible deployment, by protection of end client and users (brand image).
- **First** difficulty: **Taxonomy**, no widely agreed upon categorisation of undesired content. 
- **Second** difficulty: real world data != public data or academic datasets(distribution shift and taxonomy misalignment)
- **Third** difficulty: certains category of taxonomy are very rarely seen in real life (how to train on detecting something you see so little of)
For success in building such a system: 
- Detailed instruction and quality control are needed to ensure data quality;
- Active learning is necessary;
- Use public datasets with care;
- Imbalanced training data can lead to incorrect generalisation;
- Mistakes in data will happen and will need to be managed.
#### Taxonomy
Depends on context.
5 top level categories with sub categories to achieve a spectrum of categorisation.
- S: Sexual content
	- Undesired:
		- S3: involving minors;
		- S2: involving illegal activities;
		- S1: erotic content (not illegal);
	- Not undesired:
		- S0: non erotic or contextualised (ex: medical or sex education material).
- H: Hateful content:
	- Undesired:
		- H2: calling for violence;
		- H1: derogatory stereotype or support for hate;
	- Not undesired:
		- H0.a: neutral referring to group identity;
		- H0.b: contextualised (ex: quote).
- V: Violence:
	- Undesired:
		- V2: extremely graphic;
		- V1: threats or support for violence;
	- Not undesired:
		- V0: contextualised.
- SH: Self harm;
- HR: Harassment.

Model trained to detect S, H, V, SH, HR, S3, H2, V2.

#### Methods
##### Data selection and active learning
To ensure good performance in context, add one's own data to training set. 

##### Labeling and quality control
Assure consistency in labels, remove subjectivity as much as possible.

##### Synthetic data
Add synthetic data to rare categories to improve model performance or to alleviate bias. Also useful to for cold start, train model when no labelled real data available.

##### Domain adversarial training
Feature extractor is a transformer encoder

##### Model probing
Ensure model is classifying based on correct features. Key tokens probing, human red-teaming.

#### Experiment results
##### Model architecture and training
Transformer decoder where last linear layer replaced with 8 MLP heads (one for each categories). Initialised with pre-trained GPT model then fine tuned.

##### Model performance
Test set not disclosed for privacy reasons. Small 1680 sample public data dataset is shared.
Model is compared with Perspective API, on following datasets:
- public dataset
- Jigsaw
- Stormfront
- Reddit
- TweetEval
Each model is better with the taxonomy they were trained for. But OpenAI model is better on other datasets

##### Active learning experiments
Captures undesired content 10+ times more effectively.

##### Domain Adversarial training experiments
Test on three stages of project
- beginning stages: labelled public data and unlabelled production data
- middle stages: added curated synthetic data
- later stages: labelled production data
Great impact on beginning stages, impact reduces gradually with advancement in the project. Still improves categories with less data but and slightly hurt performance in categories where no enough samples available.


### Overall
Very interesting paper, NLP techniques were used. Taxonomy is one of the most detailed ones.
This is a toxicity detection AI no rule based moderation.
No open source code available, model is accessible via api.

---

## Multilingual content moderation, a case study on Reddit
Site web: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=multi+lingual+content+moderation&btnG=#d=gs_qabs&t=1729185811909&u=%23p%3DsrWR4g9v4bEJ
GitHub: https://github.com/mye1225/multilingual_content_mod

### Introduction
**Moderation**: "process of flagging content based on pre-defined platform rules." (quote from paper)

Offensive Language Identification (OLI) is not sufficient for moderation:
- OLI is only a subset of moderation, as moderation also needs to flag content that violates platform rules;
- Moderation needs to be adaptive to rules that change dynamically.

Contributions are: 
- 1,8 million sample Reddit comments dataset
- Show that existing offensive speech dataset are not enough as offensive comments are a small portion of the flagger comments

### Data
- Wide range of topics: better generalization
- Subs with same topic: test transferability
- Multilingual subs: train for several languages
1.8 million samples 
1.238 annotated manually for offensiveness, taxonomy:
- Non-offensive;
- HS-gender;
- HS-sexuality;
- HS-age;
- HS-social;
- HS-ideology;
- HS-religion;
- HS-disability;
- HS-race;
- Vulgar;
- Violence.
Rest is a binary classification, removed and not removed. 
(OLI dataset >< moderation dataset)
Train set: 90%, test set: 5%, validation set: 5%.

### Experiment results
71% of removed comments (by human moderators) is not offensive, just violates rules.
Pre-trained transformer based language models as text encoder, classifier on top.
For multilingual either use multilingual encoder (MLLM) or machine translation.
MLLM might be better solution.

Future is in a combination of OLI and moderation task.
Need to find a way to be more robust against label noise (incorrect label).

### Overall
Very interesting dataset as it is multilingual and includes french samples. Paper concludes moderation needs a rule-based approach in addition to regular toxicity. 

---

## Perspective API
website: https://perspectiveapi.com/

"Perspective is a free API that uses machine learning to identify "toxic" comments, making it easier to host better conversations online." (quote from the website)

Returns a percentage that represents the percentage that someone will find the text as toxic.
### Uses
- **For moderators**: Moderators use Perspective to quickly prioritize and review comments that have been reported.
- **For commenters**: Perspective can give feedback to commenters who post toxic comments.
- **For readers**: For readers Developers create tools so readers can control which comments they see, for example hiding comments that may be abusive or toxic.
Quoted from website.

### Definition 
**Toxicity**: “a rude, disrespectful, or unreasonable comment that is likely to make you leave a discussion” (quoted from FAQ)

### Taxonomy
| **Attribute name** | **Description**                                                                                                                                                                                                                                                                         | **Available Languages**                                                                                                                                                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| TOXICITY           | A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.                                                                                                                                                                                        | Arabic (ar), Chinese (zh), Czech (cs), Dutch (nl), English (en), French (fr), German (de), Hindi (hi), Hinglish (hi-Latn), Indonesian (id), Italian (it), Japanese (ja), Korean (ko), Polish (pl), Portuguese (pt), Russian (ru), Spanish (es), Swedish (sv) |
| SEVERE_TOXICITY    | A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words. | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| IDENTITY_ATTACK    | Negative or hateful comments targeting someone because of their identity.                                                                                                                                                                                                               | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| INSULT             | Insulting, inflammatory, or negative comment towards a person or a group of people.                                                                                                                                                                                                     | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| PROFANITY          | Swear words, curse words, or other obscene or profane language.                                                                                                                                                                                                                         | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| THREAT             | Describes an intention to inflict pain, injury, or violence against an individual or group.                                                                                                                                                                                             | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
### Developer
collaborative research effort by [Jigsaw](https://jigsaw.google.com/) and Google’s Counter Abuse Technology team. 

### License
We open source experiments, tools, and research data that explore ways to combat online toxicity and harassment.

### Price & Quota
Currently free, may be a fee in the future if QPS (queries per second) increases.
Limited to 1 query per second, possible to request quota increase.


### Overall
Very useful for the definitions and taxonomy, the tool is also only accessible via API and code is not opensource. Works with a lot of languages including french.

---

## Text classification using machine learning techniques.
Site web: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Text+classification+using+machine+learning+techniques&btnG=
### Introduction
Two types: 
- topic-based;
- genre-based. 
It is a Supervised Learning task, annotated dataset is needed.

Process: 
Read doc -> tokenize -> stemming -> delete stopwords -> Vector representation -> feature selection/reduction -> learning algo

### Vector space document representation 
Doc is an array of words. Useless words (stopwords) are removed, remove words with the same stem (même champs lexical).
Representation of feature value: 
- boolean indicator of word presence 
- integer word count  
Too many feature need feature reduction:
- [Feature Selection](#feature-selection)
- [Feature transformation](#feature-transformation)

#### Feature selection 
- Feature Subset selection, best individual feature (document frequency, information gain, mutual information, chi squared) -->feature scoring methods.
- Sequential forward selection (SFS), choose best single word, then add one word (best) at a time.
SFS better result but greater computation cost
- Pruning based approach, 

#### Feature transformation
- Principal component analysis (PCA), 
- Latent semantic indexing (LSI),
- k-NN LSI,

### Machine learning methods
- Decision tree;
- Naive Bayes, often used, simple and effective, performance degraded not good text representation. Tree-lake Bayesian networks are better;
- Rule induction;
- Neural networks;
- Corner classification network;
- Nearest neighbour;
- Support vector machine, excellent precision, poor recall. Recall can be improved by adjusting threshold.

Difficulties: 
- very few positive training examples;
- lack of good predictive features.  
Imbalanced data.

Combining classifiers could be next improvement:
- single methods, diff subset training data;
- diff training param with single training method;
- different learning methods.
### Evaluation
- Precision;
$\pi_i=\frac{TP_i}{TP_i+FP_i}$
- Recall;
$\rho_i=\frac{TP_i}{TP_i+FN_i}$
- Accuracy.
$A_i=\frac{TP_i+TN_i}{TP_i+TN_i+FP_i+FN_i}$

Usually precision and recall are used, accuracy is not a good evaluation methods for skewed datasets.

Precision and recall are often combined:
$F_\beta=\frac{(\beta^2+1)\pi\rho}{\beta^2\pi+\rho}$
with $\beta$ set to 1 for equal importance between precision and recall. 


### Overall
Good summary of the field, useful to steer research in the correct direction.
No code, dataset or new techniques presented.

---

## Design and Application of an AI‐Based Text Content Moderation System
Site web: https://onlinelibrary.wiley.com/doi/full/10.1155/2022/2576535

### Introduction
Text content moderation (TCM), for online educational platform. Keyword matching moderation ignores context and thus raises a lot of false positives.

### Architecture design of the AI-base TCM system
AI moderation + manual recheck. Developed using AI cloud service platform.
Input text is first analysed by frontend for formatting then api request is made for moderation.
Different analysis performed: 
- text analysis
- lexical analysis
- syntax analysis
- semantic analysis
- sentiment analysis
- text classification
Taxonomy: 
 - pornographic content
 - terrorism content
 - advertising content
 - illegal content
 - abusive content
When model is unable to classify sample, the sample is pushed to system administrator for manual moderation. 
classified content will go to database so the model can perform updates and self-learn (Active learning).
Algorithm used: FastText.

### Dataset 
contains about 360k samples. available upon request to authors.

### Experiments 
Too light, need bigger test/validation dataset.

### Overall
The system is a cloud based system, using available commercial tools. This doesn't correspond to the philosophy of Shareish platform, which would rather be self hosted and independent. 

---

## Real-Time Content Moderation Using Artificial Intelligence and Machine Learning
Site web: https://www.researchgate.net/publication/383307236_Real-Time_Content_Moderation_Using_Artificial_Intelligence_and_Machine_Learning

### Introduction
Manual moderation is not scalable. AI and ML could allow to keep content moderation real-time and still scale well with augmentation of volume of content.

### Techniques
#### Natural Language Processing
Sentiment analysis, entity recognition, text classification.

#### Computer vision
CNN's and other models, Image recognition (to detect nudity or violence for example) 

#### Behavioural analysis
Monitoring user behaviour.

### Challenges & Ethical considerations
- Continuous training to avoid false positives and negatives.
- Ensure diverse and representative data to avoid training bais
- Assure model explainability to ensure transparency and build trust (with users)
- Training AI models requires data, protect user privacy when collecting data.
- Keep hardware up to date and invest in robust systems, to be able to scale without compromising on performance
- Continuously adapt and train models to prevent users to find and exploit flaws of your agent


### Overall
Interesting to cite for discussion in introduction for example, but no code presented, no new techniques, just an analysis of what exist, the challenges and ethical considerations.

---

## A review of standard text classification practices for multi-label toxicity identification of online content
Site web: https://aclanthology.org/W18-5103/
PDF: https://aclanthology.org/W18-5103.pdf

### Introduction
**Grey area:** 
freedom of speech and censorship, ranging from slightly abusive to hate inducting

Binary classification (toxic and non-toxic), problematic, even with small error rates, removal of flagged content can impact a users reputation or freedom of speech.

Multi-label classification would allow for more powerful solutions.

Online content contains: 

- abreviations/shortenings
- spelling mistakes
- slang

**Need for HUGE annotated dataset** which would be a subjective, disturbing, time consuming task.

Wikimedia Toxicity dataset: 

[https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689)

State of the art for text classification ⇒ Deep learning (convolutional neural network

### Techniques
#### Data
Labels: 
- neutral
- toxic
- severe toxic
- obscene
- threat
- insult
- identity hate

Data augmentation through translation to French, Dutch and Spanish before translating back to English.

Punctuation and Word variations were removed and replaced by corresponding words. 

#### Text Representation
Several representations used:
- word tf-idf
- char and word tf-idf
- average of 50D trained fasttext
- average of Glove
- average of 300D pre-trained fasttext

#### Neural Network
Use of Bi-LSTM layers or Attention layers to act as text representation.
Increases slightly AUC (area under the curve)
#### Stacking classifiers 
Supervisor model (LGBM) trained to combine predictions of several classifiers.
Slightly increases the AUC.

#### Semi supervised Training
Separate test set in 10 folds, train on train set+ 9 folds of test set, for test set use pseudo-labels (predictions of best model), test on 10th fold, experiment is repeated for all 10 folds. (bootstrap?)
Slightly increases the AUC.

### Overall
Language toxicity detection tool released with paper.
Treats the problem as a NLP problem. Concept of Stacking classifiers is interesting but means there are more models to train.
Only AUC reported, (without specifying the curve).


---


---

# Meeting Notes
## 05/11/24 Meeting notes

### Discussed topics
Objective is to replicate human moderation.
Definition: 
- **Moderation**: Flagging and removal of content based on predefined rules. Removed content can be of many types: undesired content, spam, off topic, promotion or self promotion.


#todo
###  To research further

Papier plus récents ? (Chercher les paper qui cite les papier que j'ai déjà lu)
- [x] Research more recent papers
```
- Adapting Large Language Models for Content Moderation: Pitfalls in Data Engineering and Supervised Fine-tuning (2024)
- Integrating Content Moderation Systems with Large Language Models (2024)
- Like a Good Nearest Neighbor: Practical Content Moderation and Text Classification (2023)
- Toxicity Detection is NOT all you Need: Measuring the Gaps to Supporting Volunteer Content Moderators (2024)
- Content Moderation System Using Machine Learning Techniques (2023)
- Artificial intelligence as a tool in social media content moderation (2023)
```

Image moderation
- [x] Find papers on image moderation/classification
```
- On-Device Content Moderation (2021)
- Artificial intelligence as a tool in social media content moderation (2023)
```

Est ce que les diff modèles sont évaluer sur plusieurs dataset?
- [ ] Check if models are correctly evaluated
Est ce qu'on a du code réutilisable. 
- [ ] Check if reusable code
--> Phase evaluation: tout réimplémenter et tester pour choisir sur sur quoi on part.
Bien évaluer? Critique sur l'évaluation? (plus généralement sur le papier)

Modèle de traduction ? Est il réaliste de traduire des dataset complets ?

### Keep in mind for future
Penser a comment On va aborder le sujets, qu'est ce qui est le plus prometteur ?

---

## 19/11/24 Meeting notes
### Discussed in meeting

#todo
- [ ] Preciser le dataset utiliser pour les test (taille du dataset)
- [ ] trouver dataset en français (multilingual Reddit)
- [x] ajouter résumer pour les paper review (lesquels sont les plus utiles, ...)
Added "Overall" section for every new paper review. 
- [ ] faire tourner les modèle en locale (maybe faster) (se rendre le plus autonome possible)
- [x] Expliquer le choix de metrics choisit
Added [[Metrics Choice]].
- [ ] 

- [x] Check out paper: 
- Content moderation by LLM, from accuracy to legitimacy.
- LLM-Mod: Can Large language models assist content moderation. 

---

## 07/01/25 Meeting notes

### Topics to discuss
- Possibility to have access to remote GPU (for training and testing).
- I have the intention to test both LLM (local version) and more conventional methods. Is fine? 
- Are we doing rule-based moderation, toxicity detection or both?
- what is definition of undesired content (Google Jigsaw definition?)?
- What are the moderation rules for the platform?

### Discussed in Meeting

#todo 
- [ ] 

---


---

# Learning Resources
## Natural Language Processing (NLP)
[Stanford University NLP with DL course by Christopher Manning](https://www.youtube.com/watch?v=OQQ-W_63UgQ&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
### Lecture 1: 
Human alone on language (in animal world).
NLP = understanding language

**Application**: among other (spell check, machine translation, information extraction, ...), classifying.
Could be classifying: 
- Reading level;
- positive/negative sentiment;
- ...

---

## Embeddings in NLP
[TensorFlow word embeddings](https://www.tensorflow.org/text/guide/word_embeddings)

Approach by which a word is represented by a vector representing its inclination/closeness to certain categories.
Example: 
<!---
![Illustration of simple example of word embedding](../Assets/word-embedding.png)
--->
<img src="../Assets/word-embedding.png" width="50%">
Source: https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html


---

## Chain of Thought (CoT)



Chain of Thought (CoT) mirrors human reasoning. It is the name given to the process that divides a complex task into several easier logical steps. This reflects a fundamental aspect of human intelligence. 
In other words, CoT is predicated on the cognitive strategy of breaking down elaborate problems into manageable, intermediate thoughts that sequentially lead to a conclusive answer.\* 

This technique can improve accuracy, transparency and multi-step reasoning ability. However, it necessitates high quality prompts.



Source: 
- https://www.ibm.com/think/topics/chain-of-thoughts
- \* Boshi Wang, S. M. (2022). Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters. _2717-2739, https://doi.org/10.48550/arXiv.2212.10001._

---

## Metrics Choice

Our task ultimately goes down to a classification task. The goal is to determine whether a comment, description or any text (or image), is fine to keep on the platform  or it has to be removed.
Then it may be interesting to have transparency on the reasoning process, and this can be classified as a text generation task which is a NLP task.

### What I have observed
Majority of researchers used:
- Precision: 
- Recall: 
- F1 score:
When applicable:
- AUC_ROC 

### Paper review: Text classification using machine learning techniques.
website: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=text+classification&oq=text+class#d=gs_qabs&t=1729184876712&u=%23p%3DphlpHOheYAUJ

- Precision;
$\pi_i=\frac{TP_i}{TP_i+FP_i}$
- Recall;
$\rho_i=\frac{TP_i}{TP_i+FN_i}$
- Accuracy.
$A_i=\frac{TP_i+TN_i}{TP_i+TN_i+FP_i+FN_i}$

Usually precision and recall are used, accuracy is not a good evaluation methods for skewed datasets.

Precision and recall are often combined:
$F_\beta=\frac{(\beta^2+1)\pi\rho}{\beta^2\pi+\rho}$
with $\beta$ set to 1 for equal importance between precision and recall. 

### Paper review: A critical analysis of metrics used for measuring progress in artificial intelligence
website: https://arxiv.org/abs/2008.02577

Most commonly (77.2% of analyzed benchmark dataset), only one metric is used to compare performance on benchmark dataset. 

F1 score combines precision and recall
Accuracy shouldn't be used alone but with precision and recall or F1 score.

### Most used metrics

<img src="../Assets/Most used metrics per task.png" width="50%">

For Classification
- Accuracy;
- F-measure (F1 score);
- Precision;
- R at K;
- Intersection over Union;
- Area under the curve (AUC);
- Recall;
- ...

Accuracy and F-measure were frequently used alone.

Recall and precision are the ones that are the most used together. Second are Accuracy and F-measure. third is precision and F-measure and fourth is recall and F-measure.

AUC has to be specified:
- PR-AUC: area under the curve drown by precision and recall against each other.
- ROC-AUC: area under the curve drown by recall and false positive rate.

### Overall:

#### For Classification:
Most used:
- Accuracy: only if dataset is balanced, i.e. comparable number of sample in each category
- F1 score
- ROC-AUC: when output is a score, and a threshold defined.

Most informative:
- Matthews Correlation Coefficient (MCC): for imbalanced datasets

---

