# Like a Good Nearest Neighbor: Practical Content Moderation and Text Classification
website: https://arxiv.org/abs/2302.08957

## Introduction
Modification of SetFit (Thunstall et al., 2022),

- Pre-trained Language Models (PLM): state-of-the-art
Modern research:
- In context learning;
- pattern exploiting training;
- adapter based fine tuning;
- parameter efficient fine-tuning.
These depend on billion-parameter PLMs, pay-to-use APIs, and/or prompting.


## Overall
The paper presents a modification of a transformer based model -SetFit- called LaGoNN. The main contribution lies in added nearest neighbor information to samples during training. No comparison with other classifiers than their own or SetFit. 
Idea of adding nearest neighbor to training sample could be used but for the model, better candidates on other papers, conclusions of the paper are not very enthusiastic and imply that there is still a lot to do to achieve an inexpensive, reliable, robust content moderation model.