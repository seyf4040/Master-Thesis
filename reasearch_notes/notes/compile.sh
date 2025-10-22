#!/bin/bash

# Path to store the combined file
OUTPUT_FILE="Compilation.md"

# Empty the output file if it exists
> "$OUTPUT_FILE"

# File lists defined as separate arrays
FILES=(
    "Master Thesis AI Content Moderation.md"
)

WEBSITES=(
    "Websites/EthicalEye.md"
    "Websites/KoalaAI Text Moderation.md"
)

# LLM-Based Approaches (7 papers)
PAPERS_LLM=(
    "Papers/Watch Your Language Investigating Content Moderation with Large Language Models.md"
    "Papers/Adapting Large Language Models for Content Moderation Pitfalls in Data Engineering and Supervised Fine-tuning.md"
    "Papers/Content Moderation by LLM From Accuracy to Legitimacy.md"
    "Papers/LLM-Mod Can Large language models assist content moderation.md"
    "Papers/Integrating Content Moderation Systems with Large Language Models.md"
    "Papers/Shieldgemma, Generative ai content moderation based on gemma.md"
    "Papers/Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations.md"
)

# Traditional ML & Discriminative Approaches (9 papers)
PAPERS_TRADITIONAL_ML=(
    "Papers/OpenAI content moderation API.md"
    "Papers/Multilingual content moderation, a case study on Reddit.md"
    "Papers/Perspective API.md"
    "Papers/Text classification using machine learning techniques.md"
    "Papers/Design and Application of an AI‐Based Text Content Moderation System.md"
    "Papers/Real-Time Content Moderation Using Artificial Intelligence and Machine Learning.md"
    "Papers/A review of standard text classification practices for multi-label toxicity identification of online content.md"
    "Papers/Detoxify.md"
    "Papers/Comparison of deep learning models and various text pre-processing techniques for the toxic comments classification.md"
)

# Specialized Classification & Techniques (4 papers)
PAPERS_SPECIALIZED=(
    "Papers/Like a Good Nearest Neighbor Practical Content Moderation and Text Classification.md"
    "Papers/Do You Really Want to Hurt Me Predicting Abusive Swearing in Social Media.md"
    "Papers/Predicting the Type and Target of Offensive Posts in Social Media.md"
    "Papers/Deeper Attention to Abusive User Content Moderation.md"
)

# Datasets & Evaluation Benchmarks (3 papers)
PAPERS_DATASETS=(
    "Papers/ToxiGen A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection.md"
    "Papers/HateCheck Functional Tests for Hate Speech Detection Models.md"
    "Papers/WildGuard Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.md"
)

# Image & Multimodal Moderation (1 paper)
PAPERS_IMAGE=(
    "Papers/On-Device Content Moderation.md"
)

# Theoretical, Policy & Framework Papers (8 papers)
PAPERS_THEORETICAL=(
    "Papers/Content moderation, AI, and the question of scale.md"
    "Papers/Learning to Defer in Content Moderation, The Human-AI Interplay.md"
    "Papers/Artificial intelligence as a tool in social media content moderation.md"
    "Papers/Online content moderation, regulatory challenges and the unique status of media content.md"
    "Papers/Transfer learning for text classification.md"
    "Papers/From Machine Learning to Explainable AI.md"
    "Papers/The Use of AI in Online Content Moderation.md"
    "Papers/The oversight of content moderation by AI, impact assessments and their limitations.md"
)

# Security & Adversarial Testing (1 paper)
PAPERS_SECURITY=(
    "Papers/GPTFUZZER Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.md"
)

# Additional/Pending Papers (2 papers)
PAPERS_ADDITIONAL=(
    "Papers/Toxicity Detection is NOT all you Need Measuring the Gaps to Supporting Volunteer Content Moderators.md"
    "Papers/Content Moderation System Using Machine Learning Techniques.md"
)

# Summaries folder (4 documents)
SUMMARIES=(
    "Summaries/Complete Papers Analysis comparison Table & Flowchart.md"
    "Summaries/Datasets.md"
    "Summaries/Evaluation Framework.md"
    "Summaries/LLM Models.md"
)

# Meeting notes (3 documents)
MEETINGS=(
    "Meeting notes/05-11-2024.md"
    "Meeting notes/26-11-2024.md"
    "Meeting notes/07-01-2025.md"
)

# Learning Resources (5 documents)
LEARNING=(
    "Learning Resources/Natural Language Processing (NLP).md"
    "Learning Resources/Embeddings.md"
    "Learning Resources/Chain of Thought (CoT).md"
    "Learning Resources/Metrics Choice.md"
    "Learning Resources/Few-Shots Learning Concepts and Methods.md"
)

# Thesis Progress Tracker (2 documents)
PROGRESS=(
    "Thesis progression.md"
    "Reading tracker.md"
)

# Categories and their corresponding arrays
declare -A CATEGORIES
CATEGORIES["Main"]="FILES"
CATEGORIES["Websites"]="WEBSITES"
CATEGORIES["Papers_LLM"]="PAPERS_LLM"
CATEGORIES["Papers_Traditional_ML"]="PAPERS_TRADITIONAL_ML"
CATEGORIES["Papers_Specialized"]="PAPERS_SPECIALIZED"
CATEGORIES["Papers_Datasets"]="PAPERS_DATASETS"
CATEGORIES["Papers_Image"]="PAPERS_IMAGE"
CATEGORIES["Papers_Theoretical"]="PAPERS_THEORETICAL"
CATEGORIES["Papers_Security"]="PAPERS_SECURITY"
CATEGORIES["Papers_Additional"]="PAPERS_ADDITIONAL"
CATEGORIES["Summaries"]="SUMMARIES"
CATEGORIES["Meetings"]="MEETINGS"
CATEGORIES["Learning"]="LEARNING"
CATEGORIES["Progress"]="PROGRESS"

# Define the desired order of categories with display names
declare -A CATEGORY_NAMES
CATEGORY_NAMES["Main"]="Main"
CATEGORY_NAMES["Progress"]="Progress Tracking"
CATEGORY_NAMES["Websites"]="Websites"
CATEGORY_NAMES["Summaries"]="Summaries"
CATEGORY_NAMES["Papers_LLM"]="Papers/LLM-Based Approaches"
CATEGORY_NAMES["Papers_Traditional_ML"]="Papers/Traditional ML & Discriminative"
CATEGORY_NAMES["Papers_Specialized"]="Papers/Specialized Classification"
CATEGORY_NAMES["Papers_Datasets"]="Papers/Datasets & Evaluation"
CATEGORY_NAMES["Papers_Image"]="Papers/Image & Multimodal"
CATEGORY_NAMES["Papers_Theoretical"]="Papers/Theoretical & Policy"
CATEGORY_NAMES["Papers_Security"]="Papers/Security & Adversarial"
CATEGORY_NAMES["Papers_Additional"]="Papers/Additional & Pending"
CATEGORY_NAMES["Meetings"]="Meeting Notes"
CATEGORY_NAMES["Learning"]="Learning Resources"

# Define the desired order of categories
CATEGORY_ORDER=(
    "Main"
    "Progress"
    "Summaries"
    "Websites"
    "Papers_LLM"
    "Papers_Traditional_ML"
    "Papers_Specialized"
    "Papers_Datasets"
    "Papers_Image"
    "Papers_Theoretical"
    "Papers_Security"
    "Papers_Additional"
    "Meetings"
    "Learning"
)

# Function to add an extra # for all heading levels
add_layer_to_headings() {
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ $line =~ ^\#+\  ]]; then
            echo "#$line"
        else
            echo "$line"
        fi
    done < "$1"
}

# Function to process a category
process_category() {
    local category_key="$1"
    local category_display="${CATEGORY_NAMES[$category_key]}"
    local file_array_name="${CATEGORIES[$category_key]}"
    
    eval "local files=(\"\${${file_array_name}[@]}\")"

    echo -e "\n---\n" >> "$OUTPUT_FILE"
    echo -e "# $category_display" >> "$OUTPUT_FILE"

    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            echo "Processing $file..."
            add_layer_to_headings "$file" >> "$OUTPUT_FILE"
            echo -e "\n---\n" >> "$OUTPUT_FILE"
        else
            echo "Warning: $file not found."
        fi
    done
}

# Process categories in the specified order
for category in "${CATEGORY_ORDER[@]}"; do
    if [[ -n "${CATEGORIES[$category]}" ]]; then
        process_category "$category"
    else
        echo "Warning: Category '$category' is not defined."
    fi
done

echo "Combined specified Markdown files into $OUTPUT_FILE"
echo ""
echo "=== Compilation Statistics ==="
echo "Total categories: ${#CATEGORY_ORDER[@]}"
echo "Files by category:"
echo "  - LLM-Based: ${#PAPERS_LLM[@]}"
echo "  - Traditional ML: ${#PAPERS_TRADITIONAL_ML[@]}"
echo "  - Specialized: ${#PAPERS_SPECIALIZED[@]}"
echo "  - Datasets: ${#PAPERS_DATASETS[@]}"
echo "  - Image: ${#PAPERS_IMAGE[@]}"
echo "  - Theoretical: ${#PAPERS_THEORETICAL[@]}"
echo "  - Security: ${#PAPERS_SECURITY[@]}"
echo "  - Additional: ${#PAPERS_ADDITIONAL[@]}"
echo "  - Summaries: ${#SUMMARIES[@]}"
echo "  - Learning: ${#LEARNING[@]}"
echo "  - Meetings: ${#MEETINGS[@]}"
echo "  - Progress: ${#PROGRESS[@]}"
echo ""
total_papers=$((${#PAPERS_LLM[@]} + ${#PAPERS_TRADITIONAL_ML[@]} + ${#PAPERS_SPECIALIZED[@]} + ${#PAPERS_DATASETS[@]} + ${#PAPERS_IMAGE[@]} + ${#PAPERS_THEORETICAL[@]} + ${#PAPERS_SECURITY[@]} + ${#PAPERS_ADDITIONAL[@]}))
echo "Total papers: $total_papers"