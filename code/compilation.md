**Table of Content**

- [Files from current directory](#files-from-current-directory)
	- [Personal development environment](#personal-development-environment)
	- [HuggingFace model performance](#huggingface-model-performance)
		- [Testing procedure](#testing-procedure)
		- [Datasets](#datasets)

---


---

# Files from current directory
## Personal development environment
IDE: VS code
ecosystem: Anaconda
environment name: thesis  
python --version: 3.10.13 
(problems with versions 3.13 and 3.12)
 
packages to install:
- tensorflow
- pandas
- matplotlib
- scikit-learn
- seaborn

- gensim

- pytorch
- transformers

- tiktoken
- sentencepiece

---

## HuggingFace model performance
HuggingFace: Platform where one can find among others ai models

Here I will compare performance of different models available on the HuggingFace platform on different datasets.

### Testing procedure
- All models will be tested on all available datasets;
- In the initial phase, no fine-tuning will be used for any models;
- The outputs of all models will be standardised to a binary decision: 'fine' or 'removed'; 
- Sample is removed if 'removed' category probability higher than 50%, threshold = 0.5; 
- AUC and accuracy are the main metrics used to asses performance.

### Datasets
- Kaggle toxic comment challenge
- 

---

