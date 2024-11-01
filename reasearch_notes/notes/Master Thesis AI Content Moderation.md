# Master Thesis AI Content Moderation

[[Reading tracker]]
[[EthicalEye]]
[[KoalaAI Text Moderation]]

## Table Of Content
- [[#Definitions|Definitions]]
- [[#Taxonomy/Moderation rules|Taxonomy/Moderation rules]]
- [[#Datasets|Datasets]]
- [[#Model|Model]]
	- [[#Model#Architecture|Architecture]]
	- [[#Model#Training methods/parameters|Training methods/parameters]]
	- [[#Model#Other feature|Other feature]]

## Definitions
Are we only targeting offensive, toxic, abusive language or are we trying to replicate a human moderator that would also flag self-promoting advertisements, spamming and off-topic comments.
- **Moderation**: 
- **Undesired content**: 
- **Toxic/Toxicity**: 

## Taxonomy/Moderation rules
Categorisation of undesired content:
- 

## Datasets
Training data, data quality?
Availability of production data?

Some datasets I stumbled upon during the initial stages of my research
- [OpenAI moderation API](https://github.com/openai/moderation-api-release) (MIT license)
- [KoalaAI/Text-Moderation-v2-small](https://huggingface.co/datasets/KoalaAI/Text-Moderation-v2-small) (MIT licence)
- [Wikipedia Talk Labels: Personal Attacks](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689) (cc0 licence)
- [Reddit dataset](https://github.com/mye1225/multilingual_content_mod) (need to accept to terms and conditions and request access to text content)
- Offensive Language Identification Dataset [OLID](https://paperswithcode.com/dataset/olid)
	- https://paperswithcode.com/paper/predicting-the-type-and-target-of-offensive
	- https://github.com/idontflow/olid (free to use, just need to add citation of paper)
- Catalogue of abusive language data [hatespeechdata](https://hatespeechdata.com/)
- Swear Words Abusiveness Dataset [SWAD](https://github.com/dadangewp/SWAD-Repository) (GLP 3.0 icence)
- Stormfront
- TweetEval
- 
## Model
Need more research on text analysis, (sentiment, semantic, lexical, syntax).
### Architecture
NLP?
Transformer encoder/decoder?
pre-trained? 
GPT model?

### Training methods/parameters
supervised learning
hidden layers ?
epochs?
learning rate?
...

### Other feature 
- **Active Learning**
- Explainablity?
- 