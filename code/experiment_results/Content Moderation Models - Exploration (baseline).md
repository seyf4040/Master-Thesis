# Content Moderation Models - Exploration (baseline)

> [!info] **Experiment Metadata**
> - **Date**: 2025-11-23 18:05:09
> - **Dataset**: OpenAI samples (1,680 samples)
> - **Device**: CUDA (a5000 - GPU)
> - **Toxicity Threshold**: 0.5
> - **Total Experiment Time**: 1:12:51

---

## Models Tested

| Model Name               | Type        | Model ID                                                 | Parameters | French Support | Architecture |
| ------------------------ | ----------- | -------------------------------------------------------- | ---------- | -------------- | ------------ |
| Detoxify Multilingual    | detoxify    | unitary/multilingual-toxic-xlm-roberta                   | ~550M      | ✓              | XLM-RoBERTa  |
| Detoxify Unbiased        | detoxify    | unitary/unbiased-toxic-roberta                           | ~125M      | ✗              | RoBERTa      |
| ShieldGemma 2B           | shieldgemma | google/shieldgemma-2b                                    | 2B         | ✗              | Gemma 2      |
| ShieldGemma 9B           | shieldgemma | google/shieldgemma-9b                                    | 9B         | ✗              | Gemma 2      |
| Mistral 7B Instruct v0.3 | mistral     | mistralai/Mistral-7B-Instruct-v0.3                       | 7B         | ✓              | Mistral      |
| EthicalEye               | specialized | autopilot-ai/EthicalEye                                  | ~125M      | ✗              | RoBERTa      |
| KoalaAI Text-Moderation  | specialized | KoalaAI/Text-Moderation                                  | ~125M      | ✗              | RoBERTa      |
| CitizenLab XLM-RoBERTa   | specialized | citizenlab/twitter-xlm-roberta-base-sentiment-finetunned | ~270M      | ✓              | XLM-RoBERTa  |
| Llama-Guard-3-1B         | llama-guard | meta-llama/Llama-Guard-3-1B                              | 1B         | ✗              | Llama        |
| Llama-Guard-3-8B         | llama-guard | meta-llama/Llama-Guard-3-8B                              | 8B         | ✗              | Llama        |

---

## Classification Performance

### Classification Metrics
| Model                          | Accuracy  | Precision | Recall    | F1-Score  | French |
| ------------------------------ | --------- | --------- | --------- | --------- | ------ |
| **KoalaAI Text-Moderation** 🏆 | **0.910** | **0.945** | **0.753** | **0.838** | ✗      |
| **Llama-Guard-3-8B** ⭐         | **0.874** | **0.886** | **0.682** | **0.771** | ✗      |
| **ShieldGemma 9B**             | **0.820** | **0.668** | **0.833** | **0.742** | ✗      |
| **Detoxify Multilingual**      | **0.820** | **0.744** | **0.640** | **0.688** | ✓      |
| **Detoxify Unbiased**          | **0.816** | **0.753** | **0.607** | **0.672** | ✗      |
| EthicalEye                     | 0.772     | 0.612     | 0.728     | 0.665     | ✗      |
| ShieldGemma 2B                 | 0.715     | 0.522     | 0.973     | 0.680     | ✗      |
| Mistral 7B Instruct            | 0.704     | 0.833     | 0.057     | 0.108     | ✓      |
| CitizenLab XLM                 | 0.671     | 0.444     | 0.230     | 0.303     | ✓      |
| Llama-Guard-3-1B               | 0.542     | 0.399     | 0.929     | 0.558     | ✗      |

---

## Confusion Matrix Results

| Model                    | TP  | FP  | TN   | FN  | TPR   | FPR   | TNR   | FNR   |
| ------------------------ | --- | --- | ---- | --- | ----- | ----- | ----- | ----- |
| KoalaAI Text-Moderation  | 393 | 23  | 1135 | 129 | 0.753 | 0.020 | 0.980 | 0.247 |
| Llama-Guard-3-8B         | 356 | 46  | 1112 | 166 | 0.682 | 0.040 | 0.960 | 0.318 |
| ShieldGemma 9B           | 435 | 216 | 942  | 87  | 0.833 | 0.187 | 0.813 | 0.167 |
| Detoxify Multilingual    | 334 | 115 | 1043 | 188 | 0.640 | 0.099 | 0.901 | 0.360 |
| Detoxify Unbiased        | 317 | 104 | 1054 | 205 | 0.607 | 0.090 | 0.910 | 0.393 |
| EthicalEye               | 380 | 241 | 917  | 142 | 0.728 | 0.208 | 0.792 | 0.272 |
| ShieldGemma 2B           | 508 | 465 | 693  | 14  | 0.973 | 0.402 | 0.598 | 0.027 |
| Mistral 7B Instruct v0.3 | 30  | 6   | 1152 | 492 | 0.057 | 0.005 | 0.995 | 0.943 |
| CitizenLab XLM-RoBERTa   | 120 | 150 | 1008 | 402 | 0.230 | 0.130 | 0.870 | 0.770 |
| Llama-Guard-3-1B         | 485 | 732 | 426  | 37  | 0.929 | 0.632 | 0.368 | 0.071 |

---

## Performance & Resource Usage

| Model                    | Avg Inference (ms) | Total Time | Samples/sec | GPU Memory (MB) | CPU Util (%) |
| ------------------------ | ------------------ | ---------- | ----------- | --------------- | ------------ |
| CitizenLab XLM-RoBERTa   | 8.96               | 0:00:46    | 36.10       | 18,612          | 0.19         |
| EthicalEye               | 9.03               | 0:00:47    | 35.05       | 18,612          | 0.15         |
| Detoxify Unbiased        | 12.06              | 0:00:43    | 38.72       | 1,087           | 0.16         |
| Detoxify Multilingual    | 13.80              | 0:01:06    | 25.13       | 1,087           | 0.16         |
| KoalaAI Text-Moderation  | 24.24              | 0:01:05    | 25.48       | 18,612          | 0.14         |
| Llama-Guard-3-1B         | 206.58             | 0:06:35    | 4.25        | 3,041           | 0.15         |
| Llama-Guard-3-8B         | 371.12             | 0:13:09    | 2.13        | 15,753          | 0.15         |
| ShieldGemma 2B           | 396.87             | 0:12:15    | 2.28        | 15,753          | 0.10         |
| Mistral 7B Instruct v0.3 | 506.65             | 0:16:44    | 1.67        | 18,612          | 0.16         |
| ShieldGemma 9B           | 587.57             | 0:19:34    | 1.43        | 18,612          | 0.13         |

---

## Energy Consumption & Environmental Impact

| Model | Energy (kWh) | CO₂ Emissions (kg) |
|-------|-------------|-------------------|
| Detoxify Unbiased | 0.000604 | 0.000287 |
| Detoxify Multilingual | 0.000931 | 0.000442 |
| CitizenLab XLM-RoBERTa | 0.003234 | 0.001536 |
| EthicalEye | 0.003330 | 0.001582 |
| KoalaAI Text-Moderation | 0.004581 | 0.002176 |
| Llama-Guard-3-1B | 0.005505 | 0.002615 |
| ShieldGemma 2B | 0.051126 | 0.024285 |
| Llama-Guard-3-8B | 0.054859 | 0.026058 |
| Mistral 7B Instruct v0.3 | 0.069789 | 0.033150 |
| ShieldGemma 9B | 0.081621 | 0.038770 |
| **Total** | **0.275581** | **0.130901** |

---

## Charts 

### Classification Performance Comparison
```chart
type: bar
labels: [KoalaAI, Llama-8B, ShieldGemma-9B, Detoxify Multi, Detoxify Unbiased, EthicalEye, ShieldGemma-2B, Mistral-7B]
series:
  - title: Accuracy
    data: [0.910, 0.874, 0.820, 0.820, 0.816, 0.772, 0.715, 0.704]
    color: '#4CAF50'
  - title: F1-Score
    data: [0.838, 0.771, 0.742, 0.688, 0.672, 0.665, 0.680, 0.108]
    color: '#2196F3'
width: 100%
beginAtZero: true
```

### Performance vs Efficiency Trade-off
```chart
type: scatter
labels: [Models]
series:
  - title: Performance-Efficiency Plot
    data: [{x: 13.80, y: 0.820}, {x: 12.06, y: 0.816}, {x: 9.03, y: 0.772}, {x: 206.58, y: 0.542}, {x: 371.12, y: 0.874}, {x: 396.87, y: 0.715}, {x: 587.57, y: 0.820}, {x: 506.65, y: 0.704}, {x: 8.96, y: 0.671}, {x: 24.24, y: 0.910}]
xTitle: Inference Time (ms)
yTitle: Accuracy
```

### Precision-Recall Trade-off
```chart
type: scatter
labels: [Models]
series:
  - title: Precision vs Recall
    data: [{x: 0.640, y: 0.744}, {x: 0.607, y: 0.753}, {x: 0.728, y: 0.612}, {x: 0.929, y: 0.399}, {x: 0.682, y: 0.886}, {x: 0.973, y: 0.522}, {x: 0.833, y: 0.668}, {x: 0.057, y: 0.833}, {x: 0.230, y: 0.444}, {x: 0.753, y: 0.945}]
xTitle: Recall
yTitle: Precision
width: 100%
```

### Energy Consumption
```chart
type: bar
labels: [Detoxify-U, Detoxify-M, CitizenLab, EthicalEye, KoalaAI, Llama-1B, Shield-2B, Llama-8B, Mistral-7B, Shield-9B]
series:
  - title: Energy (kWh)
    data: [0.0006, 0.0009, 0.0032, 0.0033, 0.0046, 0.0055, 0.0511, 0.0549, 0.0698, 0.0816]
    color: '#9C27B0'
  - title: CO₂ (kg × 100)
    data: [0.029, 0.044, 0.154, 0.158, 0.218, 0.262, 2.429, 2.606, 3.315, 3.877]
    color: '#F44336'
width: 100%
beginAtZero: true
```

---

## 💡 Key Insights

> [!success]+ NEW LEADER: **KoalaAI Text-Moderation** 🎉
> **Performance**:
> - 🏆 **Best Overall Accuracy**: 91.0%
> - 🏆 **Best Precision**: 94.5%
> - 🏆 **Best F1-Score**: 0.838
> - ✅ Excellent balance: High precision (94.5%) with good recall (75.3%)
> - ✅ **Extremely low false positive rate**: 2.0% (best of all models)
> - ✅ Moderate false negative rate: 24.7%
> 
> **Efficiency**:
> - ⚡ Fast inference: 24.24ms per sample (25.5 samples/sec)
> - ✅ Low energy consumption: 0.0046 kWh
> - ⚠️ Higher GPU memory: 18.6 GB
> 
> **Why it wins**:
> - Only 23 false positives (vs 115 for Detoxify Multi, 46 for Llama-8B)
> - Minimizes user frustration from incorrect flagging
> - Fast enough for real-time moderation
> 
> **Trade-off**:
> - ❌ No French support (critical limitation for Shareish)
> - ⚠️ Higher GPU memory than Detoxify models

> [!abstract]+ Best for Production (French Required): **Detoxify Multilingual**
> **Performance**:
> - ⭐ Strong accuracy: 82.0%
> - ⭐ Good F1-score: 0.688
> - ✅ Balanced precision (74.4%) and recall (64.0%)
> 
> **Advantages**:
> - ✅ **Native French support** (critical for Shareish)
> - ✅ **27× faster** than Llama-Guard-8B (13.8ms vs 371ms)
> - ✅ **59× lower energy** consumption (0.0009 vs 0.0549 kWh)
> - ✅ **14.5× smaller** GPU memory footprint (1.1 GB vs 15.8 GB)
> - ✅ Reasonable false positive rate (9.9%)
> 
> **Comparison to KoalaAI**:
> - Lower accuracy (82% vs 91%)
> - More false positives (115 vs 23)
> - But: **Native French support**
> - Faster inference (13.8ms vs 24.2ms)
> - Lower GPU memory (1.1GB vs 18.6GB)

> [!tip]+ Architecture Recommendation: **Hybrid System**
> 
> **Option 1: KoalaAI + French Translation**
> - Translate French → English → Moderate → Results
> - Pros: Best accuracy (91%)
> - Cons: Translation overhead, potential loss of nuance
> 
> **Option 2: Detoxify Multi**
> - Fast, French ✓
> - Expected: ~85% accuracy, ~5% FPR, full French support

> [!warning]+ Model Reliability Issues
> **Highly Variable Performance**:
> - **Llama-Guard-3-1B**: Ranges from 54-72% accuracy across runs
> - **Mistral 7B**: Varies wildly (30% → 52% → 70% accuracy), very bad F1 score 
> 

> [!danger]+ Models to Avoid
> **Mistral 7B Instruct**:
> - ❌ Only 5.7% recall (misses 94.3% of toxic content)
> - ❌ Completely unreliable despite French support
> - ❌ Inconsistent across runs
> 
> **Llama-Guard-3-1B**:
> - ❌ 63.2% false positive rate
> - ❌ Would incorrectly flag 2 out of 3 safe posts
> - ❌ Unstable performance
> 
> **ShieldGemma 2B**:
> - ⚠️ 40.2% false positive rate
> - ⚠️ Too many false alarms for production

---

## 🔬 Next Steps

> [!todo]+ Immediate Actions
> - [ ] **Verify reproducibility** - Run KoalaAI 3 more times to confirm stability
> - [ ] **Evaluate on French dataset** - Test all models on HateCheck FR

> [!question]+ Research Questions
> 1. Can we achieve >90% accuracy with French support? (KoalaAI)
> 2. What's the optimal threshold for each model?
> 3. How do these models perform on Shareish-specific policies?
> 4. What's the cost/benefit of translation vs native French models?

