# Metrics Choice

Our task ultimately goes down to a classification task. The goal is to determine whether a comment, description or any text (or image), is fine to keep on the platform  or it has to be removed.
Then it may be interesting to have transparency on the reasoning process, and this can be classified as a text generation task which is a NLP task.

## What I have observed
Majority of researchers used:
- Precision: 
- Recall: 
- F1 score:
When applicable:
- AUC_ROC 

## Paper review: Text classification using machine learning techniques.
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

## Paper review: A critical analysis of metrics used for measuring progress in artificial intelligence
website: https://arxiv.org/abs/2008.02577

Most commonly (77.2% of analyzed benchmark dataset), only one metric is used to compare performance on benchmark dataset. 

F1 score combines precision and recall
Accuracy shouldn't be used alone but with precision and recall or F1 score.

## Most used metrics

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

## Overall:

### For Classification:
Most used:
- Accuracy: only if dataset is balanced, i.e. comparable number of sample in each category
- F1 score
- ROC-AUC: when output is a score, and a threshold defined.

Most informative:
- Matthews Correlation Coefficient (MCC): for imbalanced datasets