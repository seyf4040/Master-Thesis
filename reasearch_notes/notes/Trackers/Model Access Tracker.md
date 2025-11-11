# Model Access Tracker

**Shareish Content Moderation - Master's Thesis**  
**Last Updated:** October 2025

---

## 🤖 LLM Models - Quick Access

| Model                 | French | License    | Access Link                                                            | GPU Required | Status          | Comments                                                                                    |
| --------------------- | ------ | ---------- | ---------------------------------------------------------------------- | ------------ | --------------- | ------------------------------------------------------------------------------------------- |
| **Llama Guard 4-12B** | ✅      | LLama 4    | [HuggingFace](https://huggingface.co/meta-llama/Llama-Guard-4-12B)     | 24Gb         | ❌ Abandoned     | Not same architechture / usage as LLama 3, did not manage to run it properly within 2 weeks |
| **Llama Guard 3-8B**  | ✅      | Llama 3    | [HuggingFace](https://huggingface.co/meta-llama/Llama-Guard-3-8B)      | 16GB         | ✅ Installed     |                                                                                             |
| **Llama Guard 3-1B**  | ✅      | Llama 3    | [HuggingFace](https://huggingface.co/meta-llama/Llama-Guard-3-1B-INT4) | 4GB/CPU      | ✅ Installed     |                                                                                             |
| **ShieldGemma 9B**    | ✅      | Gemma      | [HuggingFace](https://huggingface.co/google/shieldgemma-9b)            | 18GB         | ✅ Installed     |                                                                                             |
| **ShieldGemma 2B**    | ✅      | Gemma      | [HuggingFace](https://huggingface.co/google/shieldgemma-2b)            | 8GB          | ✅ Installed     |                                                                                             |
| **Mistral 7B**        | ✅      | Apache 2.0 | [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1)        | 14GB         | ✅ Installed     |                                                                                             |
| **WildGuard** (7B)    | ❌      | Apache 2.0 | [Github](https://github.com/allenai/wildguard)                         | 14GB         | ⬜ Not installed |                                                                                             |

---

## 🔧 ML Models - Quick Access

| Model                 | Type        | French | License    | Installation           | Status          |
| --------------------- | ----------- | ------ | ---------- | ---------------------- | --------------- |
| **Detoxify Multi**    | XLM-RoBERTa | ✅      | Apache 2.0 | `pip install detoxify` | ✅ Installed     |
| **Detoxify Unbiased** | RoBERTa     | ❌      | Apache 2.0 | `pip install detoxify` | ✅ Installed     |
| **Detoxify Original** | BERT        | ❌      | Apache 2.0 | `pip install detoxify` | ⬜ Not installed |

---

## 🎯 Priority Installation


```bash
# 1. Install Detoxify 
pip install detoxify

# Test immediately
python -c "from detoxify import Detoxify; print(Detoxify('multilingual').predict('test'))"
```
✅ Done

```bash
# 2. Install transformers
pip install transformers torch accelerate

# 3. Test Llama Guard 3-8B (requires HuggingFace login)
huggingface-cli login
# Then download model in Python
```
✅ Done

```bash
# 4. Test ShieldGemma 7B
# Same process as Llama Guard
```
✅ Done

---

## 🔐 Access Requirements

### HuggingFace Account

**Required for:** All LLM models

- [x] **Setup:**
1. Create account at https://huggingface.co/
2. Generate access token (Settings > Access Tokens)
3. Login via CLI: `huggingface-cli login`

- [x] **Meta Llama Models:**
- Additional step: Accept license at model page
- Click "Agree and access repository"
- Wait for approval (usually instant)
### No Account Needed
- [x] **Detoxify:** Direct pip install

---

## 📊 Performance Tracking

Simple baseline test on OpenAI moderation evaluation dataset.
code location: code/model_experiments/test_hf_models.py
GPU used: a5000
### Results Summary

| Model                 | Accuracy | Precision | Recall | F1     | Avg Time (ms) | GPU Mem (MB) | Errors |
| --------------------- | -------- | --------- | ------ | ------ | ------------- | ------------ | ------ |
| detoxify-multilingual | 0.8196   | 0.7439    | 0.6398 | 0.6880 | 13.38         | 1086.9       | 0      |
| detoxify-unbiased     | 0.8161   | 0.7530    | 0.6073 | 0.6723 | 11.58         | 503.0        | 0      |
| Mistral-7B-v0.1       | 0.3107   | 0.3107    | 1.0000 | 0.4741 | 407.09        | 14312.8      | 0      |
| Llama-Guard-3-1B      | 0.6893   | 0.0000    | 0.0000 | 0.0000 | 91.33         | 2886.5       | 0      |
| shieldgemma-2b        | 0.6893   | 0.0000    | 0.0000 | 0.0000 | 1021.36       | 5302.6       | 0      |

### Confusion Matrix Details

|Model|TP|FP|TN|FN|
|---|---|---|---|---|
|detoxify-multilingual|334|115|1043|188|
|detoxify-unbiased|317|104|1054|205|
|Mistral-7B-v0.1|522|1158|0|0|
|Llama-Guard-3-1B|0|0|1158|522|
|shieldgemma-2b|0|0|1158|522|

---

## 📝 Notes Section

### Detoxify

```
- Performance on openai eval: Seems great! [accuracy: 0.8] 
- Performance on French:
- Speed: Very fast, on GPU(a5000): 20ms
- Issues encountered:
```

### LLM models

```
- either always safe or always unsafe, 
-> Need to check and debug input format and return value parsing
```

