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

## Techniques
### Data
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

### Text Representation
Several representations used:
- word tf-idf
- char and word tf-idf
- average of 50D trained fasttext
- average of Glove
- average of 300D pre-trained fasttext

### Neural Network
Use of Bi-LSTM layers or Attention layers to act as text representation.
Increases slightly AUC (area under the curve)
### Stacking classifiers 
Supervisor model (LGBM) trained to combine predictions of several classifiers.
Slightly increases the AUC.

### Semi supervised Training
Separate test set in 10 folds, train on train set+ 9 folds of test set, for test set use pseudo-labels (predictions of best model), test on 10th fold, experiment is repeated for all 10 folds. (bootstrap?)
Slightly increases the AUC.

## Overall
Language toxicity detection tool released with paper.
Treats the problem as a NLP problem. Concept of Stacking classifiers is interesting but means there are more models to train.
Only AUC reported, (without specifying the curve).

