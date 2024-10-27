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