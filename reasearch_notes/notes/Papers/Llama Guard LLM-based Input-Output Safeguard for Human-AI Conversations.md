# Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations

**Website**: https://arxiv.org/abs/2312.06674  
**Published**: December 2023, Meta AI  
**Authors**: Hakan Inan, Kartikeya Upasani, et al.

### Introduction

Llama Guard is a specialized LLM-based safeguard model designed to classify safety risks in both user prompts (input classification) and AI-generated responses (output classification) for conversational AI systems. It addresses a critical gap in content moderation by leveraging the capabilities of large language models as the moderation backbone, rather than relying on smaller transformer models.

**Key Innovation**: Unlike traditional moderation tools that use fixed policies and separate models for inputs vs outputs, Llama Guard:

- Uses a single LLM for both prompt and response classification
- Incorporates the safety taxonomy directly into the model input (policy-as-prompt)
- Can adapt to new policies via zero-shot and few-shot learning
- Open-source and can be fine-tuned for specific use cases

### Safety Risk Taxonomy

Llama Guard introduces a comprehensive safety risk taxonomy covering potential legal and policy risks:

**Main Categories**:

1. **Violence & Hate**: Content promoting violence or hatred against people/groups
2. **Sexual Content**: Sexually explicit or suggestive content
3. **Criminal Planning**: Content facilitating criminal activity
4. **Guns & Illegal Weapons**: Content related to weapons, drugs
5. **Regulated or Controlled Substances**: Drug-related content
6. **Self-Harm**: Content promoting self-injury or suicide
7. **Additional categories**: Can be customized per use case

The taxonomy can be included in the prompt, allowing the model to classify according to specified guidelines.

### Technical Approach

#### Base Model

- **Architecture**: Llama 2-7B (instruction-tuned)
- **Training**: Supervised fine-tuning on curated safety classification dataset
- **Dataset**: High-quality labeled examples (low volume but high quality)

#### Input Format

```
Task: Classify if prompt/response is safe or unsafe according to policy.

<BEGIN UNSAFE CONTENT CATEGORIES>
[List of categories with descriptions]
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: [prompt text]
Agent: [response text if classifying output]
<END CONVERSATION>

Provide your safety assessment
```

#### Output Format

The model generates text indicating:

- **Safe** or **Unsafe**
- If unsafe: which categories are violated
- Classification reasoning (optional)

**Example Output**:

```
unsafe
Violence & Hate
```

### Performance

#### Benchmarks Evaluated

**1. OpenAI Moderation Evaluation Dataset**:

- Performance **matches or exceeds** OpenAI's proprietary moderation API
- Strong performance across multiple safety categories

**2. ToxicChat Dataset**:

- Dataset specifically for LLM-generated content toxicity
- Llama Guard demonstrates robust detection of toxic AI outputs

#### Comparison with Baselines

- **vs. Perspective API**: Superior performance, especially on implicit toxicity
- **vs. Smaller transformers**: Significantly better, leveraging LLM capabilities
- **vs. GPT-4 (zero-shot)**: Competitive performance with much lower cost

**Key Strength**: Better adaptability to new policies through zero-shot/few-shot learning compared to fixed-policy tools.

### Adaptability

#### Zero-Shot and Few-Shot Learning

Llama Guard can adapt to new taxonomies without retraining:

**Zero-Shot**:

- Provide new category definitions in the prompt
- Model generalizes to unseen categories

**Few-Shot**:

- Include 2-5 examples per new category
- Significantly improves classification accuracy

**Fine-Tuning**:

- For production deployment with custom policies
- Requires small labeled dataset (hundreds to thousands of examples)
- Further improves performance on domain-specific content

### Architecture Variants

**Llama Guard Versions**:

1. **Llama Guard 1** (Dec 2023): Original Llama 2-7B based model
2. **Llama Guard 2** (2024): Improved version with better multilingual support
3. **Llama Guard 3** (2024): Based on Llama 3.1-8B, supports 8 languages including French
4. **Llama Guard 3-1B-INT4** (Nov 2024): Compressed version (1B parameters, INT4 quantization) for on-device deployment

**Llama Guard 3 Improvements**:

- Aligned with **MLCommons** standardized hazards taxonomy
- Multilingual: English, French, German, Italian, Portuguese, Hindi, Spanish, Thai
- Tool use safety: Search queries and code interpreter abuse detection
- Outperforms Llama Guard 2 and GPT-4 across benchmarks

### Limitations

**Acknowledged in Paper**:

1. **Context Limitations**: May struggle with highly context-dependent cases
2. **Cultural Nuances**: Taxonomy may not capture all cultural variations of harm
3. **False Positives**: Can be overly cautious, flagging benign content (especially identity mentions)
4. **False Negatives**: May miss subtle, implicit violations
5. **No Visual Content**: Text-only (though Llama Guard 3 Vision addresses this)

**From Deployment Experience**:

- Requires careful prompt engineering for optimal performance
- Inference latency higher than smaller specialized models
- Cost considerations for high-volume applications (though open weights help)

### Deployment Considerations

#### For Shareish Platform

**Advantages** ⭐⭐⭐:

1. **Open Source**: No API costs, full control, GDPR-compliant
2. **Adaptable**: Can customize taxonomy to Shareish's specific policies
3. **Dual Classification**: Handles both user input and system output
4. **Multilingual**: Llama Guard 3 supports French natively
5. **Self-Hosted**: Privacy-preserving, no data leaves infrastructure

**Practical Implementation**:

```python
# Pseudocode for Shareish integration
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/LlamaGuard-7b")

def classify_content(text, content_type="prompt"):
    prompt = f"""Task: Classify if the {content_type} is safe or unsafe.

<BEGIN UNSAFE CONTENT CATEGORIES>
1. Violence & Hate
2. Sexual Content  
3. Harassment & Bullying
4. Self-Harm Promotion
5. Spam & Scams
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {text}
<END CONVERSATION>

Provide your safety assessment:"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return parse_output(result)
```

**Resource Requirements**:
- **GPU**: Requires GPU for reasonable inference speed
    - Llama Guard 3-8B: A100/V100 recommended
    - Llama Guard 3-1B-INT4: Can run on CPU or smaller GPUs
- **Memory**: ~16GB GPU RAM for 7B model, ~4GB for 1B INT4
- **Latency**: ~200-500ms per classification (depending on hardware)

#forShareish
**Possible Approach for Shareish**:
1. **Start**: Llama Guard 3-1B-INT4 for lower resource requirements
2. **Evaluate**: Test on French Shareish content
3. **Fine-Tune**: On small Shareish-specific dataset (~500-1000 examples)
4. **Upgrade**: To Llama Guard 3-8B if needed for better accuracy

### Dataset and Training

**Training Data**:
- High-quality human-annotated examples
- Low volume but carefully curated (~10K-50K examples estimated) #tocheck 
- Covers diverse safety scenarios across taxonomy categories

**Fine-Tuning Process**:
1. Start with Llama 2-7B base model (instruction-tuned)
2. Supervised fine-tuning on safety classification data
3. Format: Prompt with taxonomy + conversation → Safe/Unsafe classification
4. Optimization: Standard cross-entropy loss on next-token prediction

**Data Quality Over Quantity**:
- Emphasis on diverse, unambiguous examples
- Multiple annotators for quality control
- Structured annotation process with guidelines

### Citations

**Primary Paper**:

```bibtex
@article{inan2023llamaguard,
  title={Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations},
  author={Inan, Hakan and Upasani, Kartikeya and Chi, Jianfeng and Rungta, Rashi and Iyer, Krithika and Mao, Yuning and Tontchev, Michael and Hu, Qing and Fuller, Brian and Testuggine, Davide and Khabsa, Madian},
  journal={arXiv preprint arXiv:2312.06674},
  year={2023}
}
```

### Overall Assessment

**Relevance to Shareish**: ⭐⭐⭐ **Very High**

**Strengths**:

- Open-source with permissive license (Llama 2/3 license)
- Customizable taxonomy for Shareish-specific rules
- Strong performance on benchmarks
- Dual input/output classification
- Active development (Llama Guard 3 released 2024)
- Good multilingual support including French

**Weaknesses**:

- Higher computational requirements than specialized models
- May need fine-tuning for optimal Shareish performance
- Inference latency may be issue for real-time moderation

**Recommendation**:

- **Good choice** for Shareish's LLM-based moderation system
- Start with Llama Guard 3-1B-INT4 for feasibility testing
- Fine-tune on Shareish-specific data for production
- Implement with learning-to-defer for edge cases