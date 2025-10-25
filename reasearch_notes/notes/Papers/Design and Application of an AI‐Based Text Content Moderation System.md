# Design and Application of an AI‐Based Text Content Moderation System
Site web: https://onlinelibrary.wiley.com/doi/full/10.1155/2022/2576535

## Introduction
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

## Overall
The system is a cloud based system, using available commercial tools. This doesn't correspond to the philosophy of Shareish platform, which would rather be self hosted and independent. 