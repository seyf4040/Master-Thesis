# Definitions of Toxic Content - Compilation from Literature

This document compiles toxicity and harmful content definitions from reviewed papers and APIs relevant to the Shareish content moderation thesis.

---

```table-of-contents
title: ## 📋 Table of contents
minLevel:2
maxLevel:2
```
---
## Google Jigsaw / Perspective API

**Source:** Perspective API (https://perspectiveapi.com/)  
**Used in papers:** "Watch Your Language: Investigating Content Moderation with Large Language Models"

### Toxicity

"A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."

### Taxonomy

|Attribute|Definition|
|---|---|
|**TOXICITY**|A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.|
|**SEVERE_TOXICITY**|A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words.|
|**IDENTITY_ATTACK**|Negative or hateful comments targeting someone because of their identity.|
|**INSULT**|Insulting, inflammatory, or negative comment towards a person or a group of people.|
|**PROFANITY**|Swear words, curse words, or other obscene or profane language.|
|**THREAT**|Describes an intention to inflict pain, injury, or violence against an individual or group.|

**Languages supported:** Arabic, Chinese, Czech, Dutch, English, French, German, Hindi, Hinglish, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish

---

## OpenAI Moderation API

**Source:** OpenAI Content Moderation API (https://openai.com/index/new-and-improved-content-moderation-tooling/)  
**Paper:** https://arxiv.org/abs/2208.03274

### Taxonomy

|Category|Description|
|---|---|
|**harassment**|Content that expresses, incites, or promotes harassing language towards any target.|
|**harassment/threatening**|Harassment content that also includes violence or serious harm towards any target.|
|**hate**|Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g. chess players) is harassment.|
|**hate/threatening**|Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.|
|**illicit**|Content that encourages the planning or execution of non-violent wrongdoing, or that gives advice or instruction on how to commit illicit acts. A phrase like "how to shoplift" would fit this category.|
|**illicit/violent**|The same types of content flagged by the illicit category, but also includes references to violence or procuring a weapon.|
|**self-harm**|Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.|
|**self-harm/intent**|Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.|
|**self-harm/instructions**|Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.|
|**sexual**|Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).|
|**sexual/minors**|Sexual content that includes an individual who is under 18 years old.|
|**violence**|Content that depicts death, violence, or physical injury.|
|**violence/graphic**|Content that depicts death, violence, or physical injury in graphic detail.|

---

## KoalaAI Text Moderation

**Source:** https://huggingface.co/KoalaAI/Text-Moderation  
**License:** CodeML OpenRAIL-M 0.1 license

### Taxonomy

|Category|Label|Definition|
|---|---|---|
|**sexual**|S|Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).|
|**hate**|H|Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.|
|**violence**|V|Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.|
|**harassment**|HR|Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.|
|**self-harm**|SH|Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.|
|**sexual/minors**|S3|Sexual content that includes an individual who is under 18 years old.|
|**hate/threatening**|H2|Hateful content that also includes violence or serious harm towards the targeted group.|
|**violence/graphic**|V2|Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.|
|**OK**|OK|Not offensive|

---

## Multilingual Content Moderation (Reddit Study)

**Source:** "Multilingual content moderation, a case study on Reddit" (https://github.com/mye1225/multilingual_content_mod)

### Moderation Definition

"Process of flagging content based on pre-defined platform rules."

### Key Finding

71% of removed comments (by human moderators) were not offensive, but violated platform-specific rules (e.g., off-topic, spam, self-promotion).

### Offensive Language Identification (OLI) Taxonomy

Manual annotation of 1,238 samples with categories:

- Non-offensive
- HS-gender (Hate speech based on gender)
- HS-sexuality (Hate speech based on sexuality)
- HS-age (Hate speech based on age)
- HS-social (Hate speech based on social class)
- HS-ideology (Hate speech based on ideology)
- HS-religion (Hate speech based on religion)
- HS-disability (Hate speech based on disability)
- HS-race (Hate speech based on race)
- Vulgar
- Violence

**Important distinction:** OLI (Offensive Language Identification) ≠ Moderation. Moderation requires rule-based approach in addition to toxicity detection.

---

## Jigsaw Toxic Comment Classification Challenge

**Source:** "A review of standard text classification practices for multi-label toxicity identification of online content" (https://aclanthology.org/W18-5103/)  
**Dataset:** Wikimedia Toxicity dataset (https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689)

### Multi-Label Categories

- **neutral** - Not toxic
- **toxic** - General toxicity
- **severe toxic** - Extremely toxic content
- **obscene** - Sexually explicit or profane
- **threat** - Threats of violence
- **insult** - Insulting language
- **identity hate** - Hate based on identity

### Challenges Noted

**Grey area between:**
- Freedom of speech and censorship
- Slightly abusive to hate-inducing content

**Online content characteristics:**
- Abbreviations/shortenings
- Spelling mistakes
- Slang

---

## ToxiGen Dataset

**Source:** "ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection" (https://arxiv.org/abs/2203.09509)  
**Published:** ACL 2022

### Implicit vs. Explicit Toxicity

**Key Innovation:** ~95% implicit toxicity (vs. ~30% in traditional datasets)

### Implicit Toxicity Characteristics

Content that is toxic but:
- Contains no slurs or profanity
- Uses stereotypes or generalizations
- Employs coded language
- Relies on context for harm
- Appears neutral on surface reading

### Examples of Implicit Toxic Content

- "They're naturally better at physical tasks than intellectual ones"
- "Women are too emotional to hold positions of power."
- "Muslims don't share our values and don't belong here."
- "People with mental illness are dangerous and unpredictable."

### Target Groups (13 minority groups)

**Race/Ethnicity:** Black/African Americans, Asian people, Latino/Hispanic people, Native Americans, Middle Eastern people  
**Religion:** Muslims, Jews  
**Gender/Sexuality:** Women, LGBTQ+ folks  
**Disability:** Mental disabilities, Physical disabilities  
**Other:** Mexican people, Chinese people

---

## Detoxify Models

**Source:** https://github.com/unitaryai/detoxify  
**License:** Apache 2.0 (code), MIT-equivalent (models)  
**Based on:** Jigsaw Challenges (2018-2020)

### Original Model Categories (6)

- `toxic` - General toxicity
- `severe_toxic` - Extremely toxic
- `obscene` - Sexually explicit or profane
- `threat` - Threats of violence
- `insult` - Insulting language
- `identity_hate` - Hate based on identity

### Unbiased Model Categories (7)

All of the above, plus:

- `identity_attack` - Attacks based on identity (renamed from identity_hate)
- `sexual_explicit` - Sexual content

**Key Feature:** Domain adversarial training to reduce false positives on benign identity mentions (-60% false positive rate)

---

## HateCheck Functional Testing

**Source:** "HateCheck: Functional Tests for Hate Speech Detection Models" (https://arxiv.org/abs/2012.15606)  
**Published:** ACL 2021

### 29 Functionalities Across 7 Categories

#### 1. Hateful Content (7 functionalities)

- Hate with slurs
- Hate without slurs
- Hate with negated positive statements
- Hate with profanity
- Hateful questions
- Hate expressed as opinion
- Hate using historical references

#### 2. Non-Hateful Slurs (4 functionalities)

- Slur homonyms (multiple meanings)
- Reclaimed slurs (in-group usage)
- Slurs in non-hateful contexts (metalinguistic)
- Slurs in counter-speech

#### 3. Positive/Neutral Phrases (5 functionalities)

- Positive statements about target groups
- Neutral statements about target groups
- Neutral self-identification
- Positive statements with slur homonyms
- Discussions of hate without endorsement

#### 4. Target Group References (5 functionalities)

- Direct references
- Indirect references ("those people", "them")
- Spelling variations/obfuscation
- Slang/coded language
- Multiple target groups (intersectional)

#### 5. Phrasing & Grammar (4 functionalities)

- Different grammatical persons
- Declarative vs. imperative
- Slang/informal language
- Length variations

#### 6. Negations & Contrasts (2 functionalities)

- Negated hate
- Hate with contrasting positive statements

#### 7. Comparisons (2 functionalities)

- Implicit comparisons
- Explicit comparisons

### Target Groups Covered (7)

Women, Trans people, Gay people, Black people, Disabled people, Muslims, Immigrants

---

## Meeting Notes - Shareish Context

**Source:** Meeting notes (05/11/24, 07/01/25)

### Moderation Definition for Shareish

"Flagging and removal of content based on predefined rules. Removed content can be of many types: undesired content, spam, off-topic, promotion or self-promotion."

### Proposed Shareish Taxonomy (Draft)

**OK content:** Content that is fine to keep, doesn't violate any rules or policy.

**Undesired content:**
- Sexual content
- Harassment
- Violent content
- Promotion/self-promotion
- Off-topic messages
- Spam

### Key Consideration

**Question:** Are we only targeting offensive, toxic, abusive language OR replicating human moderator behavior (including spam, ads, off-topic)?

---

## Summary & Recommendations for Shareish

### Most Comprehensive Taxonomies

1. **OpenAI Moderation API** - 12 categories with clear hierarchical structure
2. **KoalaAI** - 9 categories, similar to OpenAI
3. **Perspective API** - 6 core attributes with multilingual support

### Recommended Definition for Shareish

**Toxic Content:** Content that expresses, incites, or promotes harm through hate speech, harassment, threats, or violence based on protected characteristics (race, gender, religion, sexuality, disability) or that promotes self-harm, contains explicit sexual material, or uses language that is rude, disrespectful, or unreasonable in a way likely to make users leave the platform.

**Undesired Content (broader):** Toxic content PLUS platform rule violations including spam, off-topic posts, promotional content, and any content that degrades the solidarity-focused community experience.

---

## Sources Referenced

1. Perspective API (Google Jigsaw) - https://perspectiveapi.com/
2. OpenAI Moderation API - https://arxiv.org/abs/2208.03274
3. KoalaAI Text Moderation - https://huggingface.co/KoalaAI/Text-Moderation
4. Multilingual Reddit Study - https://github.com/mye1225/multilingual_content_mod
5. Jigsaw Toxic Comment Classification - https://aclanthology.org/W18-5103/
6. ToxiGen Dataset - https://arxiv.org/abs/2203.09509
7. Detoxify - https://github.com/unitaryai/detoxify
8. HateCheck - https://arxiv.org/abs/2012.15606
9. Watch Your Language (LLM Moderation) - https://arxiv.org/abs/2309.14517
10. Wikipedia Talk Labels: Personal Attacks - https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689

---

**Compiled for:** Master's Thesis - "Deep Learning for Content Moderation on the Shareish Solidarity Platform"  
**Date:** 2025  
**Note:** All sources use open licenses compatible with academic research requirements.