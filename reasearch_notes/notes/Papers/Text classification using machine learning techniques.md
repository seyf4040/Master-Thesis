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
