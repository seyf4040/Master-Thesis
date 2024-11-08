# Multilingual content moderation, a case study on Reddit
Site web: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=multi+lingual+content+moderation&btnG=#d=gs_qabs&t=1729185811909&u=%23p%3DsrWR4g9v4bEJ
GitHub: https://github.com/mye1225/multilingual_content_mod

## Introduction
**Moderation**: "process of flagging content based on pre-defined platform rules." (quote from paper)

Offensive Language Identification (OLI) is not sufficient for moderation:
- OLI is only a subset of moderation, as moderation also needs to flag content that violates platform rules;
- Moderation needs to be adaptive to rules that change dynamically.

Contributions are: 
- 1,8 million sample Reddit comments dataset
- Show that existing offensive speech dataset are not enough as offensive comments are a small portion of the flagger comments

## Data
- Wide range of topics: better generalization
- Subs with same topic: test transferability
- Multilingual subs: train for several languages
1.8 million samples 
1.238 annotated manually for offensiveness, taxonomy:
- Non-offensive;
- HS-gender;
- HS-sexuality;
- HS-age;
- HS-social;
- HS-ideology;
- HS-religion;
- HS-disability;
- HS-race;
- Vulgar;
- Violence.
Rest is a binary classification, removed and not removed. 
(OLI dataset >< moderation dataset)
Train set: 90%, test set: 5%, validation set: 5%.

## Experiment results
71% of removed comments (by human moderators) is not offensive, just violates rules.
Pre-trained transformer based language models as text encoder, classifier on top.
For multilingual either use multilingual encoder (MLLM) or machine translation.
MLLM might be better solution.

Future is in a combination of OLI and moderation task.
Need to find a way to be more robust against label noise (incorrect label).
