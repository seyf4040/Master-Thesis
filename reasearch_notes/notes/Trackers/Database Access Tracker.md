# Dataset Access Tracker

**Shareish Content Moderation - Master's Thesis**  
**Last Updated:** October 2025

---

## 📊 Quick Access Table

| Dataset                  | Size        | Language | License                                                                                                                           | Access Link                                                                                      | Status            |
| ------------------------ | ----------- | -------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------- |
| **HateCheck French**     | 3,72K cases | FR ✅     | CC BY 4.0                                                                                                                         | [HuggingFace](https://huggingface.co/datasets/Paul/hatecheck-french)                             | ✅ Accessed        |
| **HateCheck English**    | 2K cases    | EN       | CC BY 4.0                                                                                                                         | [HuggingFace](https://huggingface.co/datasets/Paul/hatecheck)                                    | ✅ Accessed        |
| **French Hate Superset** | Moderate    | FR ✅     | Open                                                                                                                              | [HuggingFace](https://huggingface.co/datasets/manueltonneau/french-hate-speech-superset)         | ⬜ Not downloaded  |
| **ToxiGen**              | 274K        | EN       | MIT                                                                                                                               | [HuggingFace](https://huggingface.co/datasets/toxigen/toxigen-data)                              | ✅ Downloaded      |
| **Multilingual Reddit**  | 1.8M        | Multi    | [Restricted](obsidian://open?vault=Master-Thesis&file=reasearch_notes%2Fprivate%2FReddit%20-%20DATA%20USE%20AGREEMENT_signed.pdf) | [GitHub](https://github.com/mye1225/multilingual_content_mod)                                    | ✅  Access aquired |
| **OpenAI Moderation**    | 1.6K        | EN       | MIT                                                                                                                               | [GitHub](https://github.com/openai/moderation-api-release)                                       | ✅ Downloaded      |
| **OLID**                 | 14K         | EN       | Free w/ cite                                                                                                                      | [GitHub](https://github.com/idontflow/olid)                                                      | ⬜ Not downloaded  |
| **Wikipedia Attacks**    | Moderate    | EN       | CC0                                                                                                                               | [Figshare](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Personal_Attacks/4054689) | ⬜ Not downloaded  |
| **Civil Comments**       | 2M          | EN       | Apache 2.0                                                                                                                        | Kaggle/TensorFlow                                                                                | ✅ Accessed        |
| **Jigsaw Challenges**    | 160K-2M     | Multi    | Open                                                                                                                              | Kaggle                                                                                           | ⬜ Not downloaded  |
| **SWAD**                 | Corpus      | EN       | GPL 3.0                                                                                                                           | [GitHub](https://github.com/dadangewp/SWAD-Repository)                                           | ⬜ Not downloaded  |
| **HateSpeechData.com**   | Catalog     | Various  | Varies                                                                                                                            | [Website](https://hatespeechdata.com/)                                                           | N/A - Catalog     |

---

## 🎯 Priority Actions

- [x] Download **HateCheck French** (primary evaluation)
- [x] Download **HateCheck English** (secondary evaluation)
- [x] Download **French Hate Superset** (training data)
- [x] Download **ToxiGen** (cold-start solution)
- [x] Request access to **Multilingual Reddit** (takes time!)

- [ ] Download **OLID** (supplementary)
- [x] Download **OpenAI Moderation** (taxonomy reference)
- [ ] Browse **HateSpeechData.com** (discovery)

---

## 📝 Access Notes

### Immediate Download (No Restrictions)
- HateCheck French
- HateCheck English
- French Hate Superset
- ToxiGen
- OpenAI Moderation
- OLID
- Wikipedia Attacks
- SWAD

### Requires Request/Account
- **Multilingual Reddit:** Submit request via GitHub (requires ToS acceptance)
- **Civil Comments:** Kaggle account required
- **Jigsaw Challenges:** Kaggle account required

---
## Dataset Exploration Summary
Model: Detoxify Multilingual
**Toxicity Threshold:** 0.5  
**Device:** CUDA  
**Date:** 2025-11-11 14:08:06

### Dataset Overview

| Dataset Name           | Size (MB) | Num Samples | Num Safe | Num Unsafe | Language       |
|------------------------|-----------|-------------|----------|------------|----------------|
| HateCheck French       | 1.97      | 3,718       | 1,118    | 2,600      | FR             |
| HateCheck English      | 1.39      | 3,728       | 1,165    | 2,563      | EN             |
| French Hate Superset   | 5.62      | 5,000       | 3,816    | 1,184      | FR             |
| ToxiGen                | 184.38    | 5,000       | 3,680    | 1,320      | EN             |
| OpenAI Moderation      | 1.18      | 1,680       | 1,158    | 522        | EN             |
| Civil Comments         | 1,032.63  | 5,000       | 4,603    | 397        | EN             |
| Multilingual Reddit    | 13.04     | 5,000       | 2,724    | 2,276      | Multi (EN, FR) |

### Performance Summary

| Dataset Name           | Accuracy | Precision | Recall | F1     | Avg Time (ms) | GPU Mem (MB) | Errors |
|------------------------|----------|-----------|--------|--------|---------------|--------------|--------|
| HateCheck French       | 0.7023   | 0.7872    | 0.7869 | 0.7871 | 10.53         | 1.89         | 0      |
| HateCheck English      | 0.7143   | 0.7635    | 0.8467 | 0.8030 | 10.61         | 1.68         | 0      |
| French Hate Superset   | 0.7166   | 0.3574    | 0.2466 | 0.2919 | 10.99         | 5.18         | 0      |
| ToxiGen                | 0.6496   | 0.2702    | 0.1924 | 0.2248 | 10.94         | 2.45         | 0      |
| OpenAI Moderation      | 0.8196   | 0.7439    | 0.6398 | 0.6880 | 13.27         | 18.01        | 0      |
| Civil Comments         | 0.9490   | 0.6360    | 0.8363 | 0.7225 | 11.50         | 10.42        | 0      |
| Multilingual Reddit    | 0.6096   | 0.6875    | 0.2610 | 0.3783 | 11.27         | 18.01        | 0      |

### Confusion Matrix (Rates)

| Dataset Name           | TPR (Recall) | FPR    | TNR (Specificity) | FNR    |
|------------------------|--------------|--------|-------------------|--------|
| HateCheck French       | 0.7869       | 0.4946 | 0.5054            | 0.2131 |
| HateCheck English      | 0.8467       | 0.5768 | 0.4232            | 0.1533 |
| French Hate Superset   | 0.2466       | 0.1376 | 0.8624            | 0.7534 |
| ToxiGen                | 0.1924       | 0.1864 | 0.8136            | 0.8076 |
| OpenAI Moderation      | 0.6398       | 0.0993 | 0.9007            | 0.3602 |
| Civil Comments         | 0.8363       | 0.0413 | 0.9587            | 0.1637 |
| Multilingual Reddit    | 0.2610       | 0.0991 | 0.9009            | 0.7390 |


---
## ✅ Completion Checklist

- [x] All priority datasets downloaded
- [x] Reddit access requested
- [x] Storage structure created
- [x] Basic exploration completed

- [ ] Datasets preprocessed
- [ ] Train/val/test splits created
- [ ] Statistics documented

- [ ] Ready for model training
