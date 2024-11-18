# HuggingFace model performance
HuggingFace: Platform where one can find among others ai models

Here I will compare performance of different models available on the HuggingFace platform on different datasets.

## Testing procedure
- All models will be tested on all available datasets;
- In the initial phase, no fine-tuning will be used for any models;
- The outputs of all models will be standardised to a binary decision: 'fine' or 'removed'; 
- Sample is removed if 'removed' category probability higher than 50%, threshold = 0.5; 
- AUC and accuracy are the main metrics used to asses performance.

## Datasets
- Kaggle toxic comment challenge
- 