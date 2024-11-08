# Master Thesis AI Content Moderation

[[Reading tracker]]
[[EthicalEye]]
[[KoalaAI Text Moderation]]

**Table of Content**
- [Definitions](#definitions)
- [Taxonomy/Moderation rules](#taxonomymoderation-rules)
- [Datasets](#datasets)
- [Model](#model)
	- [Architecture](#architecture)
	- [Training methods/parameters](#training-methodsparameters)
	- [Other feature](#other-feature)
- [API](#api)


## Definitions
Are we only targeting offensive, toxic, abusive language or are we trying to replicate a human moderator that would also flag self-promoting advertisements, spamming and off-topic comments.
- **Moderation**: 
- **Undesired content**: 
- **Toxic/Toxicity**: 

## Taxonomy/Moderation rules
Categorisation of undesired content:
- Ok content: content that is fine to keep, doesn't go against any rules or policy.
- Undesired content: 
	- sexual content;
	- harassement;
	- violent content;
	- promotion/self-promotion;
	- Off topic messages;

## Datasets
Training data influences a lot performance of the model, the more the training data distribution is different from the real data distribution, the poorer the accuracy will be.
Active learning is a necessity to adapt to any new types of undesired content and/or any work arounds found be users.

Is it possible to translate the whole dataset in french before training?
Is it possible to train in English and translate sample before evaluating if the probability of it being toxic?

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
Possible to start with as base for feature extraction a pre-trained model.

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

## API
- OpenAI moderation [API](https://openai.com/index/new-and-improved-content-moderation-tooling/?form=MG0AV3)
- Perspective [API](https://perspectiveapi.com/)
Both are free to use for the time being (01/11/24).
