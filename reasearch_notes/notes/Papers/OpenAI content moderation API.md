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
### Taxonomy
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

### Methods
#### Data selection and active learning
To ensure good performance in context, add one's own data to training set. 

#### Labeling and quality control
Assure consistency in labels, remove subjectivity as much as possible.

#### Synthetic data
Add synthetic data to rare categories to improve model performance or to alleviate bias. Also useful to for cold start, train model when no labelled real data available.

#### Domain adversarial training
Feature extractor is a transformer encoder

#### Model probing
Ensure model is classifying based on correct features. Key tokens probing, human red-teaming.

### Experiment results
#### Model architecture and training
Transformer decoder where last linear layer replaced with 8 MLP heads (one for each categories). Initialised with pre-trained GPT model then fine tuned.

#### Model performance
Test set not disclosed for privacy reasons. Small 1680 sample public data dataset is shared.
Model is compared with Perspective API, on following datasets:
- public dataset
- Jigsaw
- Stormfront
- Reddit
- TweetEval
Each model is better with the taxonomy they were trained for. But OpenAI model is better on other datasets

#### Active learning experiments
Captures undesired content 10+ times more effectively.

#### Domain Adversarial training experiments
Test on three stages of project
- beginning stages: labelled public data and unlabelled production data
- middle stages: added curated synthetic data
- later stages: labelled production data
Great impact on beginning stages, impact reduces gradually with advancement in the project. Still improves categories with less data but and slightly hurt performance in categories where no enough samples available.


## Overall
Very interesting paper, NLP techniques were used. Taxonomy is one of the most detailed ones.
This is a toxicity detection AI no rule based moderation.
No open source code available, model is accessible via api.