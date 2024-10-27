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