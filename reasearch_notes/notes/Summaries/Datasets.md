# Comprehensive Dataset Inventory for Content Moderation Research

**Compiled for:** Deep Learning for Content Moderation on the Shareish Solidarity Platform  
**Date:** January 2025  
**Purpose:** Complete reference of all datasets mentioned in paper reviews and research notes

---

```table-of-contents
title: ## 📋 Table of contents
minLevel:2
maxLevel:2
```
---

## Dataset Comparison Table

| Dataset                        | Size          | Languages      | Type        | Implicit % | License      | Availability | Relevance      |
| ------------------------------ | ------------- | -------------- | ----------- | ---------- | ------------ | ------------ | -------------- |
| **ToxiGen**                    | 274K          | EN             | Training    | 95%        | MIT          | Open         | ⭐⭐⭐ Very High  |
| **OpenAI Mod**                 | 1.6K public   | EN             | Training    | Mixed      | MIT          | Partial      | ⭐⭐⭐ Very High  |
| **Multilingual Reddit**        | 1.8M          | Multi          | Training    | N/A        | Restricted   | Restricted   | ⭐⭐⭐ Very High  |
| **Jigsaw Challenges**          | 160K-2M       | EN, 7 langs    | Training    | 30%        | Open         | Open         | ⭐⭐ Medium-High |
| **Civil Comments**             | 2M            | EN             | Training    | 30%        | Open         | Open         | ⭐⭐ Medium      |
| **OLID**                       | 14.1K         | EN             | Training    | 35%        | Free w/ cite | Open         | ⭐⭐ Medium-High |
| **Wikipedia Attacks**          | Moderate      | EN             | Training    | N/A        | CC0          | Open         | ⭐ Low-Medium   |
| **Stormfront**                 | 10K           | EN             | Training    | 50%        | Research     | Open         | ⭐ Low          |
| **SWAD**                       | Corpus        | EN?            | Training    | N/A        | GPL 3.0      | Open         | ⭐⭐ Medium      |
| **TweetEval**                  | Subset        | EN             | Evaluation  | N/A        | Open         | Open         | ⭐ Low-Medium   |
| **French Hate Superset**       | Moderate      | **FR** ✅       | Training    | N/A        | Check        | Open         | ⭐⭐⭐ Very High  |
| **Multilingual Hate (Kaggle)** | Varies        | Multi+FR       | Training    | N/A        | Check        | Open         | ⭐⭐⭐ Very High  |
| **HateCheck**                  | 3.7K          | EN, **FR**✅ +8 | **Testing** | Varies     | CC BY 4.0    | Open         | ⭐⭐⭐ Very High  |
| **HateSpeechData.com**         | N/A (catalog) | Many           | Meta        | N/A        | Varies       | Catalog      | ⭐⭐⭐ Very High  |

---

## Training & Evaluation Datasets

### 1. ToxiGen

**Purpose:** Large-scale implicit hate speech training data  
**Type:** Machine-generated, adversarial

#### Key Information

- **Size:** 274,186 statements (250K training, 8K human-annotated evaluation)
- **Languages:** English only
- **Target Groups:** 13 minority groups (Race/Ethnicity: Black, Asian, Latino, Native American, Middle Eastern; Religion: Muslims, Jews; Gender/Sexuality: Women, LGBTQ+; Disability: Mental, Physical; Other: Mexican, Chinese)
- **Label Distribution:** 50% toxic, 50% benign
- **Implicit Toxicity:** ~95% of toxic examples are implicit (no slurs)
- **Annotation Quality:** 94.5% agreement, Cohen's κ = 0.72

#### Access

- **HuggingFace:** `toxigen/toxigen-data`
- **GitHub:** https://github.com/microsoft/TOXIGEN
- **License:** MIT (code), data license in repository
- **Citation Required:** Yes

#### Characteristics

- Generated using GPT-3 with ALICE (Adversarially Learned Implicit Hate Speech) technique
- Adversarially designed to fool existing classifiers
- Average statement length: 68 characters
- High vocabulary diversity (12K unique tokens)

#### Performance Impact

Training on ToxiGen improves:

- Civil Comments: +9% F1 (0.72 → 0.81)
- Twitter Hate Speech: +8% F1 (0.68 → 0.76)
- Stormfront: +8% F1 (0.74 → 0.82)
- False positive reduction: -9 percentage points
- Implicit toxicity recall: +23 percentage points

#### Relevance to Shareish

⭐⭐⭐ **Very High** - Excellent for pre-training and data augmentation, especially for implicit hate detection

**Limitations:**

- English only (needs translation for French)
- Machine-generated (may have artifacts)
- Limited to 13 groups
- Static dataset

---

### 2. OpenAI Moderation API Dataset

**Purpose:** Training data for hierarchical toxicity classification  
**Type:** Human-annotated content moderation dataset

#### Key Information

- **Size:** 1,680 samples (public subset)
- **Languages:** English (primarily)
- **Categories:** 8 hierarchical categories
    - Sexual (S0-S3: non-erotic → minors)
    - Hate (H0-H2: neutral → violence incitement)
    - Violence (V0-V2: contextual → graphic)
    - Self-harm (SH)
    - Harassment (HR)
- **Training Set:** Undisclosed size (production data)

#### Access

- **GitHub:** https://github.com/openai/moderation-api-release
- **License:** MIT
- **API:** Free (as of Jan 2025)
- **Public Dataset:** 1,680 labeled samples

#### Characteristics

- Hierarchical taxonomy (spectrum-based, not binary)
- Mix of public data + proprietary production data
- Includes synthetic data for rare categories
- Domain adversarial training applied

#### Performance

Compared against Perspective API, Jigsaw, Stormfront, Reddit, TweetEval

- Better performance on own taxonomy
- Outperforms others on cross-dataset evaluation

#### Relevance to Shareish

⭐⭐⭐ **Very High** - Excellent taxonomy design, hierarchical approach useful for nuanced moderation

**Limitations:**

- Full training set not public (only 1,680 samples)
- Model only accessible via API (no open-source model)
- English-centric

---

### 3. Multilingual Reddit Dataset

**Purpose:** Multilingual content moderation research  
**Type:** Real-world platform data

#### Key Information

- **Size:** 1.8 million Reddit comments
- **Annotated:** 1,238 manually labeled for offensiveness
- **Languages:** Multilingual (specific languages not fully detailed)
- **Split:** 90% train, 5% validation, 5% test
- **Annotation Taxonomy:**
    - Non-offensive
    - HS-gender, HS-sexuality, HS-age, HS-social
    - HS-ideology, HS-religion, HS-disability, HS-race
    - Vulgar, Violence
- **Moderation Labels:** Binary (removed/not removed by human moderators)

#### Access

- **GitHub:** https://github.com/mye1225/multilingual_content_mod
- **License:** Requires accepting terms and conditions + request access
- **Availability:** Restricted access

#### Characteristics

- Real moderation decisions (not just toxicity labels)
- 71% of removed comments are not offensive (violate other rules)
- Demonstrates gap between offensive language detection and moderation
- Wide range of topics for better generalization

#### Key Finding

**Critical insight:** Offensive Language Identification (OLI) ≠ Content Moderation  
Only 29% of flagged content is offensive; rest violates platform rules (spam, off-topic, self-promotion)

#### Relevance to Shareish

⭐⭐⭐ **Very High** - Real moderation data, demonstrates full scope of moderation beyond toxicity

**Limitations:**

- Restricted access (requires approval)
- Dataset composition/language distribution unclear
- May have Reddit-specific biases

---

### 4. Jigsaw Challenges Datasets

**Purpose:** Toxicity detection and bias mitigation  
**Type:** Wikipedia comments, human-annotated

#### Key Information

**Challenge 1 (2018): Toxic Comment Classification**

- **Size:** 160,000 comments
- **Source:** Wikipedia talk pages
- **Categories:** 6 (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Use:** Training Detoxify "original" model

**Challenge 2 (2019): Unintended Bias in Toxicity**

- **Size:** 2 million+ comments
- **Focus:** Reduce false positives on identity mentions
- **Innovation:** Domain adversarial training
- **Use:** Training Detoxify "unbiased" model

**Challenge 3 (2020): Multilingual Toxic Comment Classification**

- **Size:** Translated data
- **Languages:** 7 (including French ❌ - limited quality)
- **Use:** Training Detoxify "multilingual" model

#### Access

- **Kaggle:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **License:** Open for research
- **Pre-trained Models:** Via Detoxify library (Apache 2.0)

#### Characteristics

- Wikipedia context (may not generalize to social platforms)
- Progressive dataset improvement (2018 → 2019 → 2020)
- Focus on explicit toxicity (less implicit)

#### Performance

Detoxify models trained on Jigsaw data:

- Original: 98.64% AUC (6 categories)
- Unbiased: 93.64% AUC (-60% false positives on identity mentions)
- Multilingual: 90% AUC (non-English), 98% AUC (English)

#### Relevance to Shareish

⭐⭐ **Medium-High** - Good for baseline, but Wikipedia context differs from solidarity platform

**Limitations:**

- Wikipedia-specific (formal language, encyclopedia context)
- Multilingual versions have quality issues
- Less implicit toxicity than ToxiGen

---

### 5. Civil Comments

**Purpose:** Toxicity detection with demographic annotations  
**Type:** Human-annotated news comments

#### Key Information

- **Size:** ~2 million comments
- **Source:** News article comment sections
- **Labels:** Toxicity scores + demographic identity mentions
- **Split:** Public train/test sets available

#### Access

- **TensorFlow Datasets:** Available
- **Kaggle:** Various competitions
- **License:** Open for research

#### Characteristics

- Real-world news comments
- Continuous toxicity scores (not binary)
- Identity attribute annotations (race, religion, gender, etc.)
- ~30% implicit toxicity (vs. 95% in ToxiGen)

#### Performance Baseline

Used extensively for benchmarking:

- Baseline models: 0.72 F1
- After ToxiGen training: 0.81 F1 (+9%)

#### Relevance to Shareish

⭐⭐ **Medium** - Good for evaluation, but news comments may differ from solidarity platform discourse

**Limitations:**

- English only
- News comment context
- More explicit than implicit toxicity

---

### 6. OLID (Offensive Language Identification Dataset)

**Purpose:** Hierarchical offensive language classification  
**Type:** Twitter data, human-annotated

#### Key Information

- **Size:** 14,100 tweets
- **Languages:** English
- **Hierarchical Taxonomy:**
    - **Level A:** Offensive (OFF) vs. Not Offensive (NOT)
    - **Level B:** Targeted (TIN) vs. Untargeted (UNT)
    - **Level C:** Target = Individual (IND), Group (GRP), Other (OTH)

#### Access

- **GitHub:** https://github.com/idontflow/olid
- **Papers with Code:** https://paperswithcode.com/dataset/olid
- **License:** Free to use with citation
- **Related Paper:** "Predicting the Type and Target of Offensive Posts in Social Media"

#### Characteristics

- Twitter-specific (short text, informal)
- Hierarchical classification allows nuanced labeling
- Enables priority-based moderation (targeted > untargeted)

#### Performance

- Level A (Offensive detection): F1 = 0.80
- Level B (Type classification): F1 = 0.68
- Level C (Target identification): F1 = 0.47 (most challenging)

#### Relevance to Shareish

⭐⭐ **Medium-High** - Hierarchical approach useful for severity-based moderation

**Limitations:**

- Twitter-specific
- English only
- Relatively small size (14K)
- Level C has low performance

---

### 7. Wikipedia Talk Labels: Personal Attacks

**Purpose:** Personal attack detection  
**Type:** Wikipedia talk page comments

#### Key Information

- **Size:** Moderate (exact size not specified in notes)
- **Languages:** English
- **Focus:** Personal attacks and aggression
- **Source:** Wikipedia talk pages

#### Access

- **Figshare:** https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689
- **License:** CC0 (public domain)
- **Citation:** Recommended

#### Characteristics

- Specific to personal attacks (subset of toxicity)
- Wikipedia editing context
- Public domain (no restrictions)

#### Relevance to Shareish

⭐ **Low-Medium** - Narrow focus on personal attacks, limited to Wikipedia context

---

### 8. Stormfront Dataset

**Purpose:** Hate speech from white supremacist forum  
**Type:** Real hate speech data

#### Key Information

- **Size:** ~10,000 posts
- **Source:** Stormfront forum (white supremacist)
- **Content:** Explicit hate speech
- **Implicit Toxicity:** ~50%

#### Access

- **Availability:** Available for research
- **License:** Research use (check restrictions)
- **Ethical Concerns:** ⚠️ Contains extreme hate speech

#### Characteristics

- Real hate speech from extremist community
- High concentration of explicit hate
- Used for evaluating model robustness

#### Performance Baseline

- ToxiGen training improves F1: 0.74 → 0.82 (+8%)

#### Relevance to Shareish

⭐ **Low** - Extreme content unlikely on solidarity platform; useful for robustness testing only

**Ethical Warning:** Contains extremely offensive content; use with caution

---

### 9. Swear Words Abusiveness Dataset (SWAD)

**Purpose:** Nuanced swearing classification  
**Type:** Swear words categorized by abusiveness

#### Key Information

- **Size:** Corpus of swear words with abusiveness ratings
- **Languages:** Unclear (likely English-focused)
- **Focus:** Distinguishing abusive vs. non-abusive swearing

#### Access

- **GitHub:** https://github.com/dadangewp/SWAD-Repository
- **License:** GPL 3.0

#### Characteristics

- Context-aware swearing classification
- Recognizes profanity ≠ toxicity
- Useful for reducing false positives

#### Relevance to Shareish

⭐⭐ **Medium** - Useful for handling edge cases where profanity isn't abusive

**Limitations:**

- GPL 3.0 (copyleft license, may complicate deployment)
- Swear word focus (narrow scope)

---

### 10. KoalaAI Text-Moderation-v2-small

**Purpose:** Small-scale toxicity dataset  
**Type:** Community-contributed moderation data

#### Key Information

- **Size:** Small (specific size not detailed)
- **Format:** Text moderation examples

#### Access

- **HuggingFace:** https://huggingface.co/datasets/KoalaAI/Text-Moderation-v2-small
- **License:** MIT

#### Characteristics

- Open-licensed
- Community-driven
- Smaller scale

#### Relevance to Shareish

⭐ **Low** - Small size limits usefulness for training

---

### 11. TweetEval (Hate Speech Subset)

**Purpose:** Tweet-based hate speech detection  
**Type:** Twitter benchmark

#### Key Information

- **Size:** Subset of larger TweetEval benchmark
- **Source:** Twitter
- **Focus:** Hate speech detection

#### Access

- **HuggingFace:** https://huggingface.co/datasets/ought/raft/viewer/tweet_eval_hate
- **License:** Open for research

#### Characteristics

- Part of broader tweet evaluation suite
- Short-form text (Twitter)
- Standardized benchmark

#### Relevance to Shareish

⭐ **Low-Medium** - Twitter context may not transfer well to solidarity platform

---

### 12. French Hate Speech Datasets

#### 12a. French Hate Speech Superset

**Purpose:** French-language hate speech collection  
**Type:** Aggregated French datasets

##### Key Information

- **Size:** Moderate (compilation of multiple sources)
- **Languages:** **French** ✅
- **Content:** Hate speech in French

##### Access

- **HuggingFace:** https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset
- **License:** Check dataset card
- **Availability:** Public

##### Relevance to Shareish

⭐⭐⭐ **Very High** - Native French data, directly applicable

---

#### 12b. Multilingual Hatespeech Dataset (Kaggle)

**Purpose:** Hate speech in multiple languages  
**Type:** Multilingual corpus

##### Key Information

- **Size:** Varies by language
- **Languages:** Multiple (including French)
- **Content:** Hate speech across languages

##### Access

- **Kaggle:** https://www.kaggle.com/datasets/wajidhassanmoosa/multilingual-hatespeech-dataset
- **License:** Check Kaggle page
- **Availability:** Public via Kaggle

##### Relevance to Shareish

⭐⭐⭐ **Very High** - Includes French, multilingual coverage

---

### 13. HateSpeechData.com Catalogue

**Purpose:** Comprehensive catalog of abusive language datasets  
**Type:** Meta-resource (links to many datasets)

#### Key Information

- **Content:** Links and information about 100+ hate speech datasets
- **Coverage:** Multiple languages, platforms, types
- **Purpose:** Dataset discovery

#### Access

- **Website:** https://hatespeechdata.com/
- **License:** Varies by dataset
- **Availability:** Public catalog

#### Characteristics

- Centralized resource for finding datasets
- Metadata about each dataset
- Regularly updated

#### Relevance to Shareish

⭐⭐⭐ **Very High** - Essential resource for finding additional datasets

---

## Benchmark & Test Datasets

### 14. HateCheck

**Purpose:** Functional testing for hate speech models  
**Type:** Synthetic test suite (template-based)

#### Key Information

- **Size:** 3,728 test cases
- **Languages:** Originally English; **Multilingual HateCheck includes French** ✅
- **Test Cases per Language:** ~3,500-4,000
- **Functionalities Tested:** 29 (English), 34 (multilingual)
- **Target Groups:** 7 (Women, Trans, Gay, Black, Disabled, Muslims, Immigrants)
- **Label Distribution:** 51.6% hateful, 48.4% non-hateful

#### Access

- **HuggingFace:** `hatecheckhq/hatecheck`
- **Specific:** `hatecheck-french` for French ✅
- **GitHub:** https://github.com/paul-rottger/hatecheck-data
- **License:** CC BY 4.0 (open, permissive)
- **Interactive Website:** Available for easy testing

#### Functional Categories (29 tests)

1. **Hateful Content (7):** Various hate expressions
2. **Non-Hateful Slurs (4):** Context-dependent slur usage
3. **Positive/Neutral (5):** Benign mentions of groups
4. **Target References (5):** Direct/indirect/obfuscated references
5. **Phrasing/Grammar (4):** Linguistic variations
6. **Negations (2):** Negated statements
7. **Comparisons (2):** Comparative statements

#### Critical Functionalities (Common Failures)

Models typically fail on:

- **F9: Reclaimed slurs** (in-group usage) - 32-45% accuracy
- **F10: Slurs in discussion** (metalinguistic) - 38-52% accuracy
- **F19: Spelling variations** - 41-54% accuracy
- **F20: Coded language** (dog whistles) - 35-48% accuracy

#### Performance Benchmarks

State-of-the-art models:

- Overall accuracy: 64-72%
- Pass rate (>70% per functionality): 15-19 out of 29

#### Relevance to Shareish

⭐⭐⭐ **Very High** - Essential evaluation tool, French version available

**Use Cases:**

1. **Baseline Evaluation:** Test Llama Guard 3 out-of-the-box
2. **Weakness Identification:** Discover specific failure modes
3. **Targeted Improvement:** Generate augmentation data for weak functionalities
4. **Progress Tracking:** Measure improvement after fine-tuning
5. **Continuous Monitoring:** Re-evaluate after each model update

**Limitations:**

- Template-based (limited linguistic diversity)
- Binary labels (no severity gradation)
- Static (doesn't evolve with new hate tactics)
- No conversation context
- Training on HateCheck can cause overfitting (-3% on i.i.d. data)

---

## Licensing Summary

### ✅ Fully Open Licenses (Safe for Use)

- **MIT:** ToxiGen, OpenAI Mod, KoalaAI
- **CC0 (Public Domain):** Wikipedia Talk Labels
- **CC BY 4.0:** HateCheck (attribution required)
- **Apache 2.0:** Detoxify models

### ⚠️ Restricted/Conditional Licenses

- **GPL 3.0:** SWAD (copyleft - derivatives must be GPL)
- **Research Use:** Jigsaw, Stormfront (check terms)
- **Terms & Conditions:** Multilingual Reddit (requires approval)
- **Varies:** Kaggle datasets, HateSpeechData catalog

### 📋 Citation Requirements

Most datasets require academic citation even when openly licensed. Always cite:

- ToxiGen (Hartvigsen et al., 2022)
- HateCheck (Röttger et al., 2021, 2022 multilingual)
- OLID (Zampieri et al., 2019)
- Jigsaw Challenges (Kaggle/Google)

---

## French Language Datasets

### Priority Datasets for Shareish

#### 1. **HateCheck (French)** ✅

- **Purpose:** Evaluation/testing
- **Size:** ~3,500-4,000 test cases
- **Availability:** Open (CC BY 4.0)
- **Quality:** High (expert-crafted)
- **Use:** Essential for functional testing

#### 2. **French Hate Speech Superset** ✅

- **Purpose:** Training
- **Size:** Moderate (aggregated sources)
- **Availability:** Open (HuggingFace)
- **Quality:** Check dataset card
- **Use:** Direct training data in French

#### 3. **Multilingual Hatespeech Dataset (Kaggle)** ✅

- **Purpose:** Training
- **Size:** Varies
- **Availability:** Open (Kaggle)
- **Quality:** Check reviews
- **Use:** Additional French training data

#### 4. **ToxiGen (Translated)** ⚠️

- **Purpose:** Training augmentation
- **Size:** 274K (if fully translated)
- **Availability:** Would require translation
- **Quality:** Machine translation quality varies
- **Use:** Consider translating subset (e.g., 50K examples) for data augmentation

---

## Recommendations for Shareish

### Phase 1: Baseline (Weeks 1-2)

#### Evaluation
1. **HateCheck French** - Evaluate Llama Guard 3 baseline
2. **French Hate Speech Superset** - Small validation set
3. Document baseline performance and failure modes

### Phase 2: Data Acquisition (Weeks 3-4)

#### Training Data Collection
1. **French Hate Speech Superset** (primary French data)
2. **Multilingual Hatespeech Dataset** (supplementary French)
3. **ToxiGen English** (consider translating 50K examples)
4. **Multilingual Reddit** (request access if possible)

#### Evaluation Data
1. **HateCheck French** (functional testing)
2. Hold out portion of French datasets for validation

### Phase 3: Fine-Tuning Strategy (Weeks 5-8)

#### Recommended Approach

```
1. Pre-train on ToxiGen English (optional, if translating)
   → Reduces identity bias, improves implicit detection
   
2. Fine-tune on French Hate Speech Superset
   → Domain adaptation to French language
   
3. Augment with Multilingual Hatespeech (Kaggle)
   → Increase diversity
   
4. Incorporate Shareish production data (gradual)
   → Active learning loop
```

### Phase 4: Evaluation & Iteration (Ongoing)

#### Testing Protocol

1. **HateCheck French** - Functional testing (29-34 functionalities)
2. **Hold-out validation** - French datasets
3. **Shareish production data** - Real-world performance
4. **Re-evaluate after each model update**

### Critical Considerations

#### Data Quality

- **Verify French dataset quality** before investing in training
- **Check for label noise** and inconsistencies
- **Assess domain relevance** (Twitter vs. news vs. social platform)

#### Translation Approach

- **Option A:** Translate ToxiGen subset (50K examples)
    - Pro: Large-scale implicit hate examples
    - Con: Translation quality concerns
- **Option B:** Use native French data only
    - Pro: No translation artifacts
    - Con: Smaller dataset size

#### Feedback Loop Design

- **Collect Shareish moderation decisions** from day 1
- **Prioritize high-confidence disagreements** for human review
- **Retrain regularly** (e.g., monthly) with new data
- **Track performance drift** on HateCheck over time

---

## Missing Datasets to Investigate

Based on project needs, consider searching for:

1. **French solidarity platform data** (if any public datasets exist)
2. **Multilingual implicit hate** (beyond ToxiGen)
3. **Context-aware conversation datasets** (thread-level moderation)
4. **Rule-based moderation examples** (spam, off-topic, self-promotion in French)
5. **Low-resource hate speech** (rare/emerging hate tactics)

Use **HateSpeechData.com** catalog to discover additional datasets.

---

## Dataset Usage Best Practices

### Academic Integrity
1. **Always cite sources** in thesis and publications
2. **Check licenses** before using data
3. **Document data provenance** in thesis methodology
4. **Avoid plagiarism** in data descriptions

### Ethical Considerations
1. **Handle offensive content responsibly**
2. **Use content warnings** when sharing examples
3. **Protect annotator well-being** if doing manual labeling
4. **Consider bias** in dataset construction
5. **Never use data for generating hate speech**

### Technical Best Practices
1. **Verify data integrity** (checksums, missing values)
2. **Document preprocessing steps** (tokenization, cleaning)
3. **Split data properly** (stratified train/val/test)
4. **Avoid data leakage** between sets
5. **Version control datasets** (track changes over time)

---

## References

### Primary Papers Citing Datasets

1. **Hartvigsen et al. (2022)** - ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection. _ACL 2022_.
    
2. **Röttger et al. (2021)** - HateCheck: Functional Tests for Hate Speech Detection Models. _ACL 2021_.
    
3. **Röttger et al. (2022)** - Multilingual HateCheck: Functional Tests for Multilingual Hate Speech Detection Models. _EMNLP 2022_.
    
4. **Zampieri et al. (2019)** - Predicting the Type and Target of Offensive Posts in Social Media. _NAACL 2019_.
    
5. **Markov et al. (2023)** - Holistic Evaluation of Language Models. _arXiv 2023_ (references multiple datasets).
    
6. **OpenAI (2022)** - New and Improved Content Moderation Tooling. _OpenAI Blog_.
    

### Dataset Repositories

- **HuggingFace Datasets:** https://huggingface.co/datasets
- **Papers with Code:** https://paperswithcode.com/datasets
- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **HateSpeechData.com:** https://hatespeechdata.com/

---

## Conclusion

This inventory provides a comprehensive overview of available datasets for training and evaluating content moderation systems. For the Shareish platform:

### Highest Priority Datasets:
1. **HateCheck French** (evaluation) ✅
2. **French Hate Speech Superset** (training) ✅
3. **ToxiGen** (augmentation via translation)
4. **Multilingual Reddit** (if access granted)

### Recommended Strategy:
- Start with native French datasets
- Use HateCheck French for systematic evaluation
- Consider translating ToxiGen subset for data augmentation
- Implement active learning with Shareish production data
- Continuously re-evaluate and iterate

All datasets listed are either open-licensed or available with academic citation, ensuring compliance with the thesis requirement for open-source materials.