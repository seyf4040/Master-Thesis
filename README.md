# Master's Thesis: AI-Powered Content Moderation

## Overview

This repository contains all code, documentation, and resources for my master's thesis, titled **"Deep Learning for Content Moderation on the Shareish Solidarity Platform"**. The thesis focuses on designing and evaluating AI-based techniques to automate content moderation for images and text on digital platforms, with an implementation aimed at enhancing moderation capabilities on the **Shareish** platform by ULiège.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
%%-  [Installation](#installation)
- [Usage](#usage) %%
- [Project Stages](#project-stages)
- [Branching Strategy](#branching-strategy)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Repository Structure

The repository is organized to manage various phases of the thesis work:

```bash
├── reasearch_notes/        # Papers, notes, summaries, and Obsidian files for the literature review 
├── code/                     # Source code for AI model development and training 
│   ├── image_moderation/     # Image moderation model code 
│   ├── text_moderation/      # Text moderation model code 
│   ├── utils/                # Utility functions for data processing 
│   └── tests/                # Unit tests for code components 
├── experiments/              # Notebooks and logs for testing and evaluating models 
├── datasets/                 # Placeholder for datasets (not included in the repo) 
├── thesis_document/          # LaTeX files and documents for thesis writing 
└── requirements.txt          # Dependencies for the project`
```
%%
## Installation

Clone the repository and install required packages.

```bash
git clone https://github.com/yourusername/thesis-ai-content-moderation.git 
cd thesis-ai-content-moderation 
pip install -r requirements.txt`
```
## Usage

### Running Model Training Scripts

To train a model, navigate to the `code/` directory and run the corresponding script for the model:

```bash
python code/image_moderation/train_image_moderation.py --dataset_path datasets/image_dataset`
```
 
### Jupyter Notebooks

Notebooks for experiments are in `experiments/`. Open them directly in Jupyter for detailed testing and analysis.
%%
## Project Stages

This project includes the following stages:

1. **Literature Review**: Review of current AI content moderation research and documentation of findings.
2. **Model Development**: Code and train models for image and text moderation using state-of-the-art machine learning techniques.
3. **Experimentation and Evaluation**: Evaluate and compare models based on metrics like accuracy, recall, and ethical implications.
4. **Implementation on Shareish**: Integrate models into the Shareish platform to test real-world feasibility.
5. **Thesis Writing**: Draft, revise, and finalize the thesis document.

## Branching Strategy

The repository follows a **Git Flow** branching strategy:

- `main`: Stable branch containing finalized code and documents.
- `reasearch`: Store literature review notes, links, and references here.
- `development`: Branch for ongoing development and integration testing.
- `writing`: Draft your thesis, manage LaTeX files, and keep updates on this branch.

## Contributing

This repository is dedicated to my master's thesis work, so direct contributions are not expected. However, if you have suggestions, feedback, or notice any issues, please feel free to contact me by email to share your insights.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or further information, please contact:  
**Ural Seyfullah**  
Master's Student, Engineering  
Université de Liège  
seyfullah.ural@student.uliege.be

---