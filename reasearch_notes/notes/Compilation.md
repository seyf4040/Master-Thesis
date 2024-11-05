# Master Thesis AI Content Moderation

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


## Definitions
Are we only targeting offensive, toxic, abusive language or are we trying to replicate a human moderator that would also flag self-promoting advertisements, spamming and off-topic comments.
- **Moderation**: 
- **Undesired content**: 
- **Toxic/Toxicity**: 

## Taxonomy/Moderation rules
Categorisation of undesired content:
- 

## Datasets
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
## Model
Need more research on text analysis, (sentiment, semantic, lexical, syntax).
### Architecture
NLP?
Transformer encoder/decoder?
pre-trained? 
GPT model?

### Training methods/parameters
supervised learning
hidden layers ?
epochs?
learning rate?
...

### Other feature 
- **Active Learning**
- Explainablity? 
- 

## API
- OpenAI moderation [API](https://openai.com/index/new-and-improved-content-moderation-tooling/?form=MG0AV3)
- Perspective [API](https://perspectiveapi.com/)
Both are free to use for the time being (01/11/24).

---

# EthicalEye
[https://huggingface.co/autopilot-ai/EthicalEye](https://huggingface.co/autopilot-ai/EthicalEye)

Pretrained agent primarily intended to be used as a tool to flag or block users exhibiting harmful or unethical behavior on various platforms.

License: Apache 2.0

Techniques: text classification, toxicity analysis, and cross-lingual NLP
---

# KoalaAI Text Moderation
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

# Databases

## OpenAI Moderation API Evaluation

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

## Koala Moderation
https://huggingface.co/datasets/KoalaAI/Text-Moderation-v2-small
### **Data Instances**

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

### **Dataset Fields**

The dataset has the following fields (also called "features"):

```json
{
  "text": "Value(dtype='string', id=None)",
  "target": "ClassLabel(names=['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2'], id=None)"
}
```
---

# OpenAI content moderation API
Site web: https://openai.com/index/new-and-improved-content-moderation-tooling/?form=MG0AV3
API doc: https://platform.openai.com/docs/guides/moderation
Paper: https://arxiv.org/abs/2208.03274 

## OpenAI website
GPT-based models detect undesired content.
Allows developers to access reliable classifiers through a simple API call instead of developing and maintaining their own.
**Moderation endpoint is free to use.**

To install and use OpenAI's python library:
follow: https://platform.openai.com/docs/libraries/python-library

### Classification categories

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
## Paper
### Introduction
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
## Taxonomy
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

## Methods
### Data selection and active learning
To ensure good performance in context, add one's own data to training set. 

### Labeling and quality control
Assure consistency in labels, remove subjectivity as much as possible.

### Synthetic data
Add synthetic data to rare categories to improve model performance or to alleviate bias. Also useful to for cold start, train model when no labelled real data available.

### Domain adversarial training
Feature extractor is a transformer encoder

### Model probing
Ensure model is classifying based on correct features. Key tokens probing, human red-teaming.

## Experiment results
### Model architecture and training
Transformer decoder where last linear layer replaced with 8 MLP heads (one for each categories). Initialised with pre-trained GPT model then fine tuned.

### Model performance
Test set not disclosed for privacy reasons. Small 1680 sample public data dataset is shared.
Model is compared with Perspective API, on following datasets:
- public dataset
- Jigsaw
- Stormfront
- Reddit
- TweetEval
Each model is better with the taxonomy they were trained for. But OpenAI model is better on other datasets

### Active learning experiments
Captures undesired content 10+ times more effectively.

### Domain Adversarial training experiments
Test on three stages of project
- beginning stages: labelled public data and unlabelled production data
- middle stages: added curated synthetic data
- later stages: labelled production data
Great impact on beginning stages, impact reduces gradually with advancement in the project. Still improves categories with less data but and slightly hurt performance in categories where no enough samples available.

## ChatGPT's review
### Summary
This paper, authored by a team from OpenAI, proposes a comprehensive approach to developing a robust content moderation system for detecting undesired content in real-world applications. It addresses challenges like imbalanced data, subjective labeling, and cold start problems by employing techniques such as active learning, taxonomy design, and domain adversarial training. The paper demonstrates significant improvements in identifying undesired content, including hateful, violent, and self-harm-related text, outperforming off-the-shelf models in multiple categories.

### Strengths
- **Comprehensive Approach**: The paper presents a holistic framework combining taxonomy design, active learning, synthetic data generation, and domain adversarial training to create a robust content moderation system.
- **Real-world Relevance**: It tackles the practical challenges of moderating real-world content by accounting for the complexities of taxonomy mismatches and distribution shifts between public datasets and production traffic.
- **Active Learning Efficacy**: The use of active learning to capture rare categories significantly improves model performance by efficiently selecting samples, improving the system’s detection capabilities.
- **Synthetic Data Usage**: The authors introduce creative approaches to generate synthetic data for cold-start problems and rare categories, which is critical in handling scenarios where real-world data is limited.
- **Ethical Considerations**: The paper thoughtfully addresses the ethical implications of content moderation, including bias in training data, transparency, and the mental health of annotators.

### Weaknesses
- **Bias Mitigation**: Although the paper outlines strategies for bias mitigation, the results show that the issue of demographic bias, especially around racial and gender terms, persists. The model continues to show disproportionate results for specific demographic groups.
- **Limited Multilingual Support**: The system currently focuses primarily on English content, with limited exploration into handling non-English languages, which may reduce its global applicability.
- **Generalizability Concerns**: The domain adaptation strategy helps align the model to production data, but the results suggest that it struggles when domain discrepancies are large, as seen in categories with few public data samples.
- **Dependence on Synthetic Data**: While synthetic data is useful in cold start and rare category scenarios, its over-reliance can introduce noise, which the authors acknowledge may hurt performance in some cases.
- **Lack of Comparative Results**: The paper compares the proposed model to Perspective API but does not provide a detailed comparison with other state-of-the-art models in content moderation, leaving questions about its standing against leading alternatives.

### Minor Comments
- **Figures and Tables**: Some figures, such as Figure 1 (model training framework), lack sufficient explanation and context, which can make it difficult for readers to understand the flow and structure of the proposed system.
- **Terminology and Clarity**: Certain sections, particularly those discussing the taxonomy structure, could benefit from clearer explanations of the classification system used for undesired content. More concrete examples would help clarify abstract concepts.
- **Evaluation Dataset**: The paper reports on model performance using production traffic, but since this dataset is proprietary, there is limited transparency in evaluating the model against external benchmarks.

### Recommendation
The paper presents a well-rounded solution to a pressing real-world problem—content moderation. The techniques used, especially the combination of active learning and domain adversarial training, showcase a sophisticated understanding of the challenges in detecting undesired content. However, improvements are needed to address persistent bias issues and to enhance multilingual capabilities. I recommend **acceptance with minor revisions**, focusing on further refining the bias mitigation strategies and expanding support for non-English content.
---

# Multilingual content moderation, a case study on Reddit
Site web: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=multi+lingual+content+moderation&btnG=#d=gs_qabs&t=1729185811909&u=%23p%3DsrWR4g9v4bEJ
GitHub: https://github.com/mye1225/multilingual_content_mod

## Introduction
**Moderation**: "process of flagging content based on pre-defined platform rules." (quote from paper)

Offensive Language Identification (OLI) is not sufficient for moderation:
- OLI is only a subset of moderation, as moderation also needs to flag content that violates platform rules;
- Moderation needs to be adaptive to rules that change dynamically.

Contributions are: 
- 1,8 million sample Reddit comments dataset
- Show that existing offensive speech dataset are not enough as offensive comments are a small portion of the flagger comments

## Data
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

## Experiment results
71% of removed comments (by human moderators) is not offensive, just violates rules.
Pre-trained transformer based language models as text encoder, classifier on top.
For multilingual either use multilingual encoder (MLLM) or machine translation.
MLLM might be better solution.

Future is in a combination of OLI and moderation task.
Need to find a way to be more robust against label noise (incorrect label).
---

# Perspective API
website: https://perspectiveapi.com/

"Perspective is a free API that uses machine learning to identify "toxic" comments, making it easier to host better conversations online." (quote from the website)

Returns a percentage that represents the percentage that someone will find the text as toxic.
## Uses
- **For moderators**: Moderators use Perspective to quickly prioritize and review comments that have been reported.
- **For commenters**: Perspective can give feedback to commenters who post toxic comments.
- **For readers**: For readers Developers create tools so readers can control which comments they see, for example hiding comments that may be abusive or toxic.
Quoted from website.

## Definition 
**Toxicity**: “a rude, disrespectful, or unreasonable comment that is likely to make you leave a discussion” (quoted from FAQ)

## Taxonomy
| **Attribute name** | **Description**                                                                                                                                                                                                                                                                         | **Available Languages**                                                                                                                                                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| TOXICITY           | A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.                                                                                                                                                                                        | Arabic (ar), Chinese (zh), Czech (cs), Dutch (nl), English (en), French (fr), German (de), Hindi (hi), Hinglish (hi-Latn), Indonesian (id), Italian (it), Japanese (ja), Korean (ko), Polish (pl), Portuguese (pt), Russian (ru), Spanish (es), Swedish (sv) |
| SEVERE_TOXICITY    | A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words. | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| IDENTITY_ATTACK    | Negative or hateful comments targeting someone because of their identity.                                                                                                                                                                                                               | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| INSULT             | Insulting, inflammatory, or negative comment towards a person or a group of people.                                                                                                                                                                                                     | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| PROFANITY          | Swear words, curse words, or other obscene or profane language.                                                                                                                                                                                                                         | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| THREAT             | Describes an intention to inflict pain, injury, or violence against an individual or group.                                                                                                                                                                                             | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
## Developer
collaborative research effort by [Jigsaw](https://jigsaw.google.com/) and Google’s Counter Abuse Technology team. 

## License
We open source experiments, tools, and research data that explore ways to combat online toxicity and harassment.

## Price & Quota
Currently free, may be a fee in the future if QPS (queries per second) increases.
Limited to 1 query per second, possible to request quota increase.
---

# Text classification using machine learning techniques.
Site web: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=text+classification&oq=text+class#d=gs_qabs&t=1729184876712&u=%23p%3DphlpHOheYAUJ
## Introduction
Two types: 
- topic-based;
- genre-based. 
It is a Supervised Learning task, annotated dataset is needed.

Process: 
Read doc -> tokenize -> stemming -> delete stopwords -> Vector representation -> feature selection/reduction -> learning algo

## Vector space document representation 
Doc is an array of words. Useless words (stopwords) are removed, remove words with the same stem (même champs lexical).
Representation of feature value: 
- boolean indicator of word presence 
- integer word count  
Too many feature need feature reduction:
- [Feature Selection](#feature-selection)
- [Feature transformation](#feature-transformation)

## Feature selection 
- Feature Subset selection, best individual feature (document frequency, information gain, mutual information, chi squared) -->feature scoring methods.
- Sequential forward selection (SFS), choose best single word, then add one word (best) at a time.
SFS better result but greater computation cost
- Pruning based approach, 

## Feature transformation
- Principal component analysis (PCA), 
- Latent semantic indexing (LSI),
- k-NN LSI,

## Machine learning methods
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
## Evaluation
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

---

# Design and Application of an AI‐Based Text Content Moderation System
Site web: https://onlinelibrary.wiley.com/doi/full/10.1155/2022/2576535

# Introduction
Text content moderation (TCM), for online educational platform. Keyword matching moderation ignores context and thus raises a lot of false positives.

## Architecture design of the AI-base TCM system
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

## Dataset 
contains about 360k samples. available upon request to authors.

## Experiments 
Too light, need bigger test/validation dataset.

## ChatGPT's review
# Detailed Review of "Design and Application of an AI-Based Text Content Moderation System"

## Summary
The paper explores the development and deployment of an AI-based Text Content Moderation (TCM) system. The system leverages cloud computing and machine learning models to moderate text in real-time, particularly for an educational platform. It combines rule-based and AI-driven techniques to improve the accuracy and efficiency of content filtering, reducing reliance on manual moderation. The authors provide a technical breakdown of the system's architecture, including its modular layers, reliance on Tencent Cloud infrastructure, and implementation of the FastText model for text classification.

## Strengths
- **Relevance and Application**: The paper addresses a pressing need for effective moderation in online educational platforms, focusing on reducing harmful or disruptive content.
- **Technical Detail**: A thorough explanation of the system’s architecture and modular breakdown (hardware, middleware, algorithm service, and application service layers) supports a clear understanding of its functionality.
- **Efficient Use of AI**: The integration of FastText and cloud-based resources demonstrates the system’s ability to handle high text volumes with good accuracy (98.3% in testing).
- **Cloud-Based Scalability**: The use of Tencent Cloud services enables flexible scaling to meet various computational demands, suggesting that the system can handle increased workloads.
- **Continuous Learning and Adaptation**: The authors implemented mechanisms for the model to update its sample library based on flagged or moderated content, which will improve its adaptability to emerging patterns in harmful content.

## Weaknesses
- **Bias in Text Classification**: Although FastText performs well with Chinese text, the study does not address bias in handling nuanced or sensitive content, which can lead to inappropriate moderation.
- **Lack of Comparative Analysis**: There is no comparative analysis with other TCM systems, such as Google’s Perspective API, to benchmark the model’s performance across various content moderation needs.
- **Limited Contextualization of Results**: While the accuracy rate is noted, the paper lacks a breakdown of performance across content types (e.g., spam, harassment, misinformation) or false positive/negative rates for each.
- **User Experience Consideration**: The paper does not address user feedback or experience after moderation interventions, which is vital in educational contexts where censorship can affect open dialogue.
- **Limited Discussion of Multilingual Support**: Since the system operates primarily in Chinese, its broader applicability to multilingual educational environments is unclear.

## Minor Comments
- **Figures and Diagrams**: Figures (e.g., Figures 1 and 2) detailing the system’s architecture would benefit from more descriptive captions to assist readers in following complex processes.
- **Terminology Consistency**: Terms like “AI-based moderation” and “intelligent moderation” are used interchangeably but could be defined to clarify specific methods or approaches within the system.
- **Future Directions**: While the conclusion mentions the potential of the TCM system in education, a more detailed roadmap for future improvements or upgrades would strengthen the paper’s relevance.

## Recommendation
The paper presents a well-designed and technically sound AI-based content moderation system. It offers practical applications and a strong use of cloud-based resources, making it suitable for educational settings. However, the paper requires some minor revisions, specifically to address bias issues, expand on multilingual capabilities, and provide a comparative analysis. My recommendation is **acceptance with minor revisions** to enhance the study’s applicability and depth.
---

# Real-Time Content Moderation Using Artificial Intelligence and Machine Learning
Site web: https://www.researchgate.net/publication/383307236_Real-Time_Content_Moderation_Using_Artificial_Intelligence_and_Machine_Learning

## Introduction
Manual moderation is not scalable. AI and ML could allow to keep content moderation real-time and still scale well with augmentation of volume of content.

## Techniques
### Natural Language Processing
Sentiment analysis, entity recognition, text classification.

### Computer vision
CNN's and other models, Image recognition (to detect nudity or violence for example) 

### Behavioural analysis
Monitoring user behaviour.

## Challenges & Ethical considerations
- Continuous training to avoid false positives and negatives.
- Ensure diverse and representative data to avoid training bais
- Assure model explainability to ensure transparency and build trust (with users)
- Training AI models requires data, protect user privacy when collecting data.
- Keep hardware up to date and invest in robust systems, to be able to scale without compromising on performance
- Continuously adapt and train models to prevent users to find and exploit flaws of your agent


## ChatGPT's review
### **Summary**  
This paper investigates real-time content moderation using AI and machine learning techniques, focusing on methods like Natural Language Processing (NLP), computer vision, audio analysis, and behavioral analysis. The author explores challenges such as false positives, bias, and privacy concerns while emphasizing the importance of continuous improvement in these systems.

### **Strengths**
- The paper tackles a critical issue in today's digital ecosystem: moderating online content efficiently while scaling solutions.
- A comprehensive overview of the techniques used in content moderation (NLP, computer vision, audio analysis) offers a holistic view of the tools available.
- The author highlights key ethical concerns (false positives, bias, privacy), which are essential to the discourse on AI-based moderation.
- The inclusion of real-world applications and references to platforms like YouTube, Facebook, and Reddit provides relevance and context.

### **Weaknesses**
- The discussion on the limitations of manual moderation and AI-driven systems could have been further deepened with empirical data or specific case studies.
- The paper does not delve into the comparative effectiveness of the various techniques (NLP, computer vision, etc.) across different types of content.
- Ethical challenges such as bias are mentioned but not explored in enough depth. The impact of biased training data could be better supported by quantitative analysis or referenced studies.
- The paper could have discussed user-centric perspectives in more detail, especially in terms of how moderation affects user experience and freedom of speech.
- Limited explanation is provided regarding the trade-offs between accuracy and computational costs in scaling AI for real-time moderation.

### **Minor Comments**
- The transitions between sections are somewhat abrupt, and further explanation could be provided when moving from one technique (e.g., NLP) to another (e.g., computer vision).
- Some figures lack detailed captions, which would aid readers unfamiliar with the technical details.
- Citations are extensive, but a clearer discussion on the selection of certain models and techniques would strengthen the practical relevance of the research.

### **Recommendation**  
While this paper provides a strong overview of the tools available for real-time content moderation, it needs to address the ethical and operational limitations of these tools more robustly. I recommend **minor revisions** with an emphasis on strengthening the discussion on the trade-offs between different techniques and the practical implications for platforms at scale.
---

# A review of standard text classification practices for multi-label toxicity identification of online content
Site web: https://aclanthology.org/W18-5103/
PDF: https://aclanthology.org/W18-5103.pdf

## Introduction
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

## Dataset
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

## Text Representation
Several representations used:
- word tf-idf
- char and word tf-idf
- average of 50D trained fasttext
- average of Glove
- average of 300D pre-trained fasttext

## Neural Network
Use of Bi-LSTM layers or Attention layers to act as text representation.
Increases slightly AUC (area under the curve)
## Stacking classifiers 
Supervisor model (LGBM) trained to combine predictions of several classifiers.
Slightly increases the AUC.

## Semi supervised Training
Separate test set in 10 folds, train on train set+ 9 folds of test set, for test set use pseudo-labels (predictions of best model), test on 10th fold, experiment is repeated for all 10 folds. (bootstrap?)
Slightly increases the AUC.

## Conclusion
see sections above

## Interesting to note
Language toxicity detection tool released with paper.


## Chat GPT review

### **Summary**  
The paper examines various text classification methods for detecting multi-label toxicity in online content. By comparing traditional and advanced techniques, including SVMs and neural networks with embeddings, the authors demonstrate that stacking classifiers and using semi-supervised learning improves performance, achieving an ROC AUC score of 0.9862.

### **Strengths**
- The paper covers a relevant and important topic: identifying toxic online behavior while balancing free speech and platform safety.
- The comparison of multiple classification techniques, from traditional SVMs to advanced neural networks, provides comprehensive insights.
- The use of real-world data (Wikimedia Toxicity Dataset) adds to the study's applicability.
- The augmentation techniques and semi-supervised learning approach add value to the paper's contribution.

### **Weaknesses**
- The paper does not sufficiently explore the ethical implications of using AI for toxicity detection, particularly in terms of bias in datasets or over-censorship.
- There is limited discussion about the interpretability of the models, which is crucial in toxic content identification.
- The study relies heavily on AUC as a performance metric, neglecting potential practical issues like false positives, which are critical in toxicity detection.
- The dataset is noted to be unbalanced, but the methods for dealing with this imbalance are not robustly addressed beyond augmentation.
- The paper lacks sufficient contextualization in terms of how these models compare to industry-standard systems already in use by social media platforms.

### **Minor Comments**
- Some parts of the methodology section could benefit from clearer descriptions, particularly the neural network architectures (Figures 2a and 2b).
- There is a minor inconsistency in the citation style for some references (e.g., formatting of URLs vs. other references).
- Figures 1 and 3 would benefit from a more detailed explanation in the captions to enhance clarity for readers unfamiliar with the technical terms.
- The conclusion section could be expanded to discuss more future directions beyond the immediate results of the paper.

### **Recommendation**  
The paper provides valuable insights into toxicity detection, with solid methodological comparisons. However, it requires revisions to address ethical considerations, model interpretability, and real-world applicability before it can be recommended for publication.
---
