# Complete Papers Analysis: Updated Comparison Table & Flowchart (34 Papers)

## 📊 Comprehensive Comparison Table

### Legend

- ✅ Available / Confirmed
- ❌ Not available
- ⚠️ Partial / Limited
- 🔒 Requires access/payment
- 📝 Need to verify from full paper

---

### Part 1: LLM-Based Approaches

|Paper Title|Year|Authors|Model/Approach|Dataset Size|Performance (F1)|Precision|Recall|Language Support|Code Available|Key Contribution|Relevance to Shareish|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Watch Your Language**|2024|Kumar et al.|GPT-3.5, GPT-4, Gemini, LLaMA 2|95 subreddits<br>~5,000 posts|Rule-based: varies<br>Toxicity: 0.72-0.75|83% (median, rule-based)|📝|English|❌|First comprehensive LLM moderation eval|⭐⭐⭐ Architecture & benchmarking|
|**Adapting LLMs for Content Moderation**|2024|Chinese researchers|Baichuan 7B/13B<br>+ LoRA + CoT|8.7K samples<br>(7.2K train, 1.5K test)|📝 (not reported)|Outperforms GPT-4 (Setting D)|📝|Chinese<br>(English via GPT-4)|❌|Weak supervision + CoT fine-tuning|⭐⭐⭐ Fine-tuning methodology|
|**Content Moderation by LLM: Accuracy to Legitimacy**|2024|Policy researchers|Conceptual framework|N/A (theoretical)|N/A|N/A|N/A|Language-agnostic|N/A|Legitimacy > Accuracy argument|⭐⭐⭐ Evaluation philosophy|
|**LLM-Mod**|2024|Kolla et al.|GPT-3.5|744 samples<br>(9 subreddits)|Low (not competitive)|Low|43.1% (TPR)|English|❌|Identifies LLM limitations in rule-based|⭐⭐ Negative results (what doesn't work)|
|**Integrating Content Moderation with LLMs**|2024|Franco et al.|GPT-3.5, LLaMA 2|📝|📝|📝|📝|Multilingual (claimed)|❌|Policy-as-prompt framework|⭐⭐⭐ System architecture design|
|**ShieldGemma**|2024|Google DeepMind|Gemma 2B/7B|📝|0.75-0.85 (estimated)|📝|📝|Multilingual<br>(FR included)|✅ Open weights|Production-ready open model|⭐⭐⭐ Practical deployment option|
|**Llama Guard 3**|2024|Meta AI|Llama 3.1-8B<br>+ safety fine-tuning|Large (undisclosed)|~0.80 (estimated)<br>Matches/exceeds OpenAI|📝|📝|8 languages<br>**French ✅**|✅ Open weights<br>(1B INT4 available)|Customizable taxonomy<br>Input+Output classification|⭐⭐⭐ **Primary recommendation**|

---

### Part 2: Traditional ML/Discriminative Approaches

|Paper Title|Year|Authors|Model/Approach|Dataset Size|Performance (F1)|Precision|Recall|Language Support|Code Available|Key Contribution|Relevance to Shareish|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**OpenAI Moderation API**|2022|Markov et al.|GPT-based transformer<br>8 MLP heads|1,680 public samples<br>Large private dataset|Better than Perspective on most datasets|📝|📝|English primarily|❌ (API only)|Detailed taxonomy (S/H/V/SH/HR)|⭐⭐⭐ Taxonomy reference|
|**Multilingual Content Moderation (Reddit)**|2023|Ye et al.|Transformer encoder<br>(XLM-RoBERTa)|1.8M samples<br>(FR, EN, ES, etc.)|📝|📝|📝|Multilingual<br>French included|✅ Dataset available|71% violations non-toxic<br>Need for rule-based|⭐⭐⭐ Dataset + findings|
|**Perspective API**|2018-ongoing|Google Jigsaw|Proprietary classifier|Large (undisclosed)|Toxicity: 0.64 (F1)|📝|📝|18+ languages<br>French included|❌ (API only)|Industry standard baseline|⭐⭐ Baseline comparison|
|**Detoxify**|2020|Unitary AI|BERT multilingual<br>+ toxic-bert|Various datasets<br>(combined)|0.92 AUC<br>(multilingual)|📝|📝|7 languages<br>**French ✅**|✅ Open source<br>(Apache 2.0)|Fast inference (50ms)<br>Production-ready|⭐⭐ **First-pass filter**|
|**Text Classification Using ML**|2004|Sebastiani|Review: NB, SVM, NN|Survey paper|N/A|N/A|N/A|Language-agnostic|N/A|Comprehensive ML overview|⭐ Background only|
|**Design & Application AI-Based TCM**|2022|Chinese researchers|FastText|360K samples|⚠️ Insufficient eval|⚠️|⚠️|Chinese|⚠️ Upon request|Cloud-based system|❌ Not aligned with Shareish|
|**Real-Time Content Moderation**|2024|Various|Review: NLP, CV, Behavioral|Survey paper|N/A|N/A|N/A|Language-agnostic|N/A|Challenges & ethical considerations|⭐ Introduction/discussion|
|**A Review of Standard Text Classification**|2018|Various|Review: CNN, LSTM, etc.|Kaggle Toxic (159K)|AUC improvements with ensembles|📝|📝|English|✅ Tool released|Multi-label classification<br>Stacking classifiers|⭐ Technical background|
|**Comparison of DL Models & Preprocessing**|2020|Various|CNN, LSTM, Bi-LSTM, GRU, BERT|Kaggle Toxic (159K)|BERT best<br>Others with preprocessing|📝|📝|English|📝|Preprocessing vs. model trade-offs|⭐ If building discriminative|

---

### Part 3: Specialized Classification Approaches

|Paper Title|Year|Authors|Model/Approach|Dataset Size|Performance (F1)|Precision|Recall|Language Support|Code Available|Key Contribution|Relevance to Shareish|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**On-Device Content Moderation**|2021|Apple (?)|SSD + MobileNetV3|OpenYahoo|0.91|95%|88%|📝|❌ No code/data|Image moderation<br>On-device deployment|⭐ If doing image moderation|
|**Do You Really Want to Hurt Me**|2020|Various|Context-aware classifier|SWAD dataset|📝|Significantly better than keyword|📝|English|⚠️ Dataset (GPL 3.0)|Abusive vs. casual swearing|⭐⭐ If allowing some swearing|
|**Predicting Type and Target**|2019|Zampieri et al.|Hierarchical BERT|OLID (14.1K tweets)|Level A: 0.80<br>Level B: 0.68<br>Level C: 0.47|📝|📝|English|✅ Dataset on GitHub|3-level hierarchical classification|⭐⭐ For fine-grained moderation|

---

### Part 4: Datasets & Evaluation Benchmarks

|Paper Title|Year|Authors|Type|Dataset Size|Key Features|Code/Data Available|Relevance to Shareish|
|---|---|---|---|---|---|---|---|
|**ToxiGen**|2022|Hartvigsen et al. (Microsoft)|Adversarial dataset|274K examples<br>(13 minority groups)|**95% implicit toxicity**<br>ALICE generation method|✅ HuggingFace<br>`toxigen/toxigen-data`|⭐⭐⭐ **Training augmentation**|
|**HateCheck**|2021-2022|Röttger et al.|Functional test suite|3,728 test cases<br>(29 functionalities)|**French version ✅**<br>Template-based<br>Pass threshold: 70%|✅ HuggingFace<br>`hatecheckhq/hatecheck`|⭐⭐⭐ **Essential evaluation**|
|**WildGuard**|2024|Han et al. (AI2)|Multi-task safety model + dataset|92K training examples<br>(WildGuardMix)|3 tasks: prompt harm, response harm, refusal<br>Adversarial robustness|✅ HuggingFace<br>`allenai/wildguard`|⭐⭐ **Adversarial testing**|

---

### Part 5: Theoretical & Policy Papers

|Paper Title|Year|Authors|Type|Key Concepts|Empirical Data|Code/Models|Relevance to Shareish|
|---|---|---|---|---|---|---|---|
|**Content Moderation, AI, and Scale**|2020|Policy paper|Conceptual|Automation necessity vs. risks|No|N/A|⭐ Introduction context|
|**Like a Good Nearest Neighbor**|2023|Academic|LaGoNN (SetFit modification)|k-NN + transformer|Small datasets|❌|⭐ Alternative approach (not compelling)|
|**Learning to Defer in Content Moderation**|2024|Academic|Learning to Defer framework|When AI should escalate to humans|Theoretical + experiments|⚠️ Framework|⭐⭐⭐ AI-human collaboration strategy|
|**Artificial Intelligence as a Tool**|2023|Bachelor thesis|Literature review|Benefits/limitations of AI moderation|Survey|N/A|⭐ Background/overview|
|**Online Content Moderation (Regulatory)**|2024|Master thesis|Legal analysis|DSA, GDPR implications|Legal frameworks|N/A|⭐⭐ Legal compliance|
|**Transfer Learning for Text Classification**|2005|Academic|Foundational concept|Pre-training + fine-tuning|Historical|N/A|⭐⭐ Conceptual foundation|
|**From ML to Explainable AI**|2018|Academic|XAI techniques|LIME, SHAP, attention viz|Methods|⚠️ Libraries|⭐⭐⭐ Explanation generation|
|**The Use of AI in Online CM**|2022|Policy report|Industry practices|Platform approaches, regulations|Industry survey|N/A|⭐⭐ Context & best practices|
|**GPTFUZZER**|2023|Security research|Adversarial testing|Jailbreak generation|Attack methods|✅ Code|⭐ Security considerations|
|**Oversight of CM by AI**|📝|Academic|Impact assessment|Evaluation frameworks beyond accuracy|Frameworks|N/A|⭐⭐ Evaluation methodology|

---

### Part 6: Additional Papers (Pending/In Progress)

|Paper Title|Year|Status|Key Info|Relevance|
|---|---|---|---|---|
|**Deeper Attention to Abusive User CM**|2017|Priority to read|Early attention mechanisms for abuse detection|⭐⭐ Historical context|
|**Toxicity Detection is NOT All You Need**|2024|In progress|Gaps in supporting volunteer moderators|⭐⭐⭐ System design beyond detection|
|**Content Moderation System Using ML**|2023|Read (to-do)|General ML techniques survey|⭐ Background|
|**The oversight of CM by AI**|📝|To read|Impact assessments|⭐⭐ Evaluation|

---

## 📈 Performance Comparison (Known Metrics)

### Toxicity Detection F1 Scores

```
Llama Guard 3:                   ~0.80 (estimated, multilingual)
ShieldGemma:                     0.75-0.85 (estimated)
GPT-4 (Watch Your Language):     0.75
GPT-3.5 (Watch Your Language):   0.72-0.75
Baichuan-13B (Setting D):        > GPT-4 (on Chinese data)
Detoxify (multilingual):         0.92 AUC
Perspective API:                 0.64
Traditional ML (best):           ~0.70
```

### Specialized Benchmarks

**ToxiGen (Implicit Hate Detection):**

```
With ToxiGen pre-training:       +8% F1 on implicit toxicity
                                 -9% false positives on identity mentions
```

**HateCheck (Functional Testing):**

```
Typical model:                   15-20/29 functionalities pass (70% threshold)
Target for Shareish:             25+/29 functionalities pass
Weak areas:                      Negations (F13-F16), Spelling variations (F17-F20)
```

**WildGuard (Adversarial Robustness):**

```
WildGuard:                       State-of-the-art on jailbreak defense
                                 Exceeds GPT-4 on adversarial inputs
Llama Guard 3:                   Good adversarial robustness
```

---

## 🗺️ Updated Visual Relationship Flowchart

```mermaid
graph TB
    subgraph "FOUNDATIONAL CONCEPTS"
        A[Transfer Learning<br/>2005<br/>⭐⭐ Conceptual]
        B[Text Classification ML<br/>2004<br/>⭐ Background]
        C[Content Moderation AI & Scale<br/>2020<br/>⭐ Philosophy]
    end

    subgraph "TRADITIONAL ML APPROACHES"
        D[Comparison of DL Models<br/>2020<br/>⭐]
        E[Review of Text Classification<br/>2018<br/>⭐]
        G[Real-Time CM AI/ML<br/>2024<br/>⭐ Overview]
    end

    subgraph "DISCRIMINATIVE MODELS"
        H[OpenAI Moderation API<br/>2022<br/>⭐⭐⭐ Taxonomy]
        I[Perspective API<br/>2018+<br/>⭐⭐ Baseline]
        J[Multilingual Reddit<br/>2023<br/>⭐⭐⭐ Dataset]
        DET[Detoxify<br/>2020<br/>⭐⭐ Fast Filter]
    end

    subgraph "SPECIALIZED CLASSIFICATION"
        K[Predicting Type & Target<br/>2019<br/>⭐⭐ Hierarchical]
        L[Abusive Swearing<br/>2020<br/>⭐⭐ Nuance]
        M[On-Device Image Mod<br/>2021<br/>⭐ Images]
    end

    subgraph "DATASETS & BENCHMARKS"
        TOX[ToxiGen<br/>2022<br/>⭐⭐⭐ Adversarial Data]
        HC[HateCheck<br/>2021-22<br/>⭐⭐⭐ Evaluation]
        WG[WildGuard<br/>2024<br/>⭐⭐ Adversarial]
    end

    subgraph "LLM REVOLUTION"
        N[Watch Your Language<br/>2024<br/>⭐⭐⭐ Benchmark]
        O[Adapting LLMs for CM<br/>2024<br/>⭐⭐⭐ Fine-tuning]
        P[ShieldGemma<br/>2024<br/>⭐⭐⭐ Open Model]
        LG[Llama Guard 3<br/>2024<br/>⭐⭐⭐ Safety Model]
        Q[LLM-Mod<br/>2024<br/>⭐⭐ Limitations]
    end

    subgraph "LLM INTEGRATION"
        R[Integrating CM with LLMs<br/>2024<br/>⭐⭐⭐ Architecture]
        S[CM by LLM: Accuracy→Legitimacy<br/>2024<br/>⭐⭐⭐ Philosophy]
    end

    subgraph "HUMAN-AI COLLABORATION"
        T[Learning to Defer<br/>2024<br/>⭐⭐⭐ Hybrid Strategy]
    end

    subgraph "EXPLAINABILITY & EVALUATION"
        U[From ML to XAI<br/>2018<br/>⭐⭐⭐ Explanations]
        V[Oversight of CM by AI<br/>📝<br/>⭐⭐ Assessment]
    end

    subgraph "LEGAL & POLICY"
        W[Online CM Regulatory<br/>2024<br/>⭐⭐ Legal]
        X[The Use of AI in CM<br/>2022<br/>⭐⭐ Policy]
    end

    subgraph "SECURITY"
        Z[GPTFUZZER<br/>2023<br/>⭐ Adversarial]
    end

    %% Foundational influences
    A --> D
    A --> E
    A --> O
    A --> P
    A --> LG
    B --> D
    B --> E
    C --> X

    %% Traditional ML evolution
    D --> H
    E --> H
    D --> I
    E --> I
    H --> DET
    I --> DET

    %% Specialized from traditional
    H --> K
    I --> K
    H --> L
    D --> M

    %% Dataset creation and influence
    J --> TOX
    TOX --> N
    TOX --> O
    TOX --> HC
    HC --> N
    HC --> O
    WG --> N

    %% LLM developments
    H --> N
    I --> N
    DET --> N
    N --> O
    N --> Q
    N --> R
    O --> P
    O --> LG
    P --> R
    LG --> R

    %% Integration & philosophy
    N --> S
    O --> S
    R --> S
    S --> T
    LG --> T

    %% Explainability
    U --> N
    U --> O
    U --> R
    V --> S
    V --> T

    %% Legal/Policy awareness
    W --> R
    W --> T
    X --> W

    %% Security considerations
    Z --> P
    Z --> LG
    Z --> WG

    %% Influence on Shareish decision
    N -.-> SHAREISH[🎯 SHAREISH<br/>ARCHITECTURE<br/>DECISION]
    O -.-> SHAREISH
    LG -.-> SHAREISH
    DET -.-> SHAREISH
    R -.-> SHAREISH
    S -.-> SHAREISH
    T -.-> SHAREISH
    U -.-> SHAREISH
    TOX -.-> SHAREISH
    HC -.-> SHAREISH
    J -.-> SHAREISH
    H -.-> SHAREISH

    style SHAREISH fill:#ff6b6b,stroke:#c92a2a,stroke-width:4px,color:#fff
    style LG fill:#51cf66,stroke:#2f9e44,stroke-width:4px
    style TOX fill:#ffd43b,stroke:#fab005,stroke-width:3px
    style HC fill:#ffd43b,stroke:#fab005,stroke-width:3px
    style DET fill:#74c0fc,stroke:#1971c2,stroke-width:3px
    style N fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style O fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style P fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style R fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style S fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style T fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style U fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style H fill:#ffd43b,stroke:#fab005,stroke-width:2px
    style J fill:#ffd43b,stroke:#fab005,stroke-width:2px
```

**Color Legend:**

- 🟢 **Green (thick border)**: Core papers for Shareish (Llama Guard 3, LLM approaches)
- 🟡 **Yellow**: Important datasets and baselines (ToxiGen, HateCheck, OpenAI, Reddit)
- 🔵 **Blue**: Complementary tools (Detoxify)
- 🔴 **Red**: Final Shareish architecture decision point

---

## 🎯 Updated Decision Tree: Which Papers for What Purpose?

```mermaid
graph TD
    START[What do you need?]
    
    START --> Q1{Primary Model<br/>Selection?}
    START --> Q2{Training Data?}
    START --> Q3{Evaluation?}
    START --> Q4{Architecture<br/>Design?}
    
    Q1 --> LG[Llama Guard 3<br/>⭐⭐⭐<br/>Customizable, FR support]
    Q1 --> SG[ShieldGemma<br/>⭐⭐⭐<br/>Alternative option]
    Q1 --> DET2[Detoxify<br/>⭐⭐<br/>Fast pre-filter]
    
    Q2 --> Q2A{Data Type?}
    Q2A --> IMPLICIT[Implicit Toxicity]
    Q2A --> GENERAL[General Training]
    Q2A --> FRENCH[French-Specific]
    
    IMPLICIT --> TOX2[ToxiGen<br/>⭐⭐⭐<br/>274K adversarial examples]
    GENERAL --> REDDIT[Multilingual Reddit<br/>⭐⭐⭐<br/>1.8M samples]
    FRENCH --> REDDIT2[Reddit FR subset<br/>⭐⭐⭐]
    
    Q3 --> Q3A{Evaluation Type?}
    Q3A --> FUNC[Functional Testing]
    Q3A --> ADV[Adversarial]
    Q3A --> BASELINE[Baseline Comparison]
    
    FUNC --> HC2[HateCheck<br/>⭐⭐⭐<br/>29 functionalities, FR]
    ADV --> WG2[WildGuard<br/>⭐⭐<br/>Jailbreak testing]
    BASELINE --> PERSP[Perspective API<br/>⭐⭐<br/>Industry standard]
    
    Q4 --> Q4A{Focus Area?}
    Q4A --> POLICY[Policy-as-Prompt]
    Q4A --> COLLAB[Human-AI Collab]
    Q4A --> FINETUNE[Fine-Tuning Method]
    
    POLICY --> INT[Integrating with LLMs<br/>⭐⭐⭐]
    COLLAB --> DEF[Learning to Defer<br/>⭐⭐⭐]
    FINETUNE --> ADAPT[Adapting LLMs<br/>⭐⭐⭐<br/>CoT + weak supervision]
    
    style LG fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style TOX2 fill:#ffd43b,stroke:#fab005,stroke-width:3px
    style HC2 fill:#ffd43b,stroke:#fab005,stroke-width:3px
    style DET2 fill:#74c0fc,stroke:#1971c2,stroke-width:2px
    style INT fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style DEF fill:#51cf66,stroke:#2f9e44,stroke-width:3px
    style ADAPT fill:#51cf66,stroke:#2f9e44,stroke-width:3px
```

---

## 📊  Gap Analysis Matrix

|Requirement|Papers Addressing|Coverage Quality|Missing Elements|**New Papers Help?**|
|---|---|---|---|---|
|**Cold-start problem**|Transfer Learning (concept only)|⚠️ Partial|Specific strategies for <1000 samples|✅ **ToxiGen** provides 274K augmentation|
|**French language**|Reddit, ShieldGemma, Perspective, **Llama Guard 3**, **HateCheck FR**, **Detoxify**|✅ **Good**|Cultural nuance analysis|✅ **Llama Guard 3 + HateCheck FR**|
|**Implicit toxicity**|Limited coverage|⚠️ Partial|Training data for implicit hate|✅ **ToxiGen** (95% implicit)|
|**Functional testing**|Limited|⚠️ Partial|Systematic weakness identification|✅ **HateCheck** (29 functionalities)|
|**Adversarial robustness**|GPTFUZZER only|⚠️ Partial|Jailbreak defense evaluation|✅ **WildGuard** dataset|
|**Fast pre-filtering**|None|❌ Poor|Cost-effective first pass|✅ **Detoxify** (50ms inference)|
|**Rule-based moderation**|Watch Your Language, LLM-Mod, Integrating|✅ Good|Production implementation|✅ **Llama Guard 3** (customizable)|
|**Explainability**|From ML to XAI, Adapting LLMs|✅ Good|User-facing formats|✅ **Llama Guard 3** (can generate)|
|**Small platform scale**|❌ None|❌ Poor|Cost-benefit for <100K users|⚠️ **Detoxify helps** (lower cost)|

**Key Improvement:** The 5 new papers significantly strengthen coverage of French language support, implicit toxicity detection, systematic evaluation, and cost-effective deployment.

---

## 💡 Quick Reference: Top 10 Papers by Use Case

### **For System Architecture:**

1. **Llama Guard 3** ⭐⭐⭐ (Primary model)
2. Integrating Content Moderation with LLMs ⭐⭐⭐
3. Learning to Defer ⭐⭐⭐
4. **Detoxify** ⭐⭐ (Fast pre-filter)

### **For Training & Fine-Tuning:**

5. **ToxiGen** ⭐⭐⭐ (274K adversarial examples)
6. Adapting LLMs for Content Moderation ⭐⭐⭐
7. Multilingual Reddit Dataset ⭐⭐⭐

### **For Evaluation:**

8. **HateCheck** ⭐⭐⭐ (Functional testing, French version)
9. Content Moderation by LLM: Accuracy→Legitimacy ⭐⭐⭐
10. **WildGuard** ⭐⭐ (Adversarial robustness)