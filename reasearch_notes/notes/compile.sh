#!/bin/bash

# Path to store the combined file
OUTPUT_FILE="compilation.md"

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
PAPERS_GENERATIVE_AI=(
    "Papers/Content moderation, AI, and the question of scale.md"
    "Papers/LLM-Mod Can Large language models assist content moderation.md"
    "Papers/Adapting Large Language Models for Content Moderation Pitfalls in Data Engineering and Supervised Fine-tuning.md"
    "Papers/Content Moderation by LLM From Accuracy to Legitimacy.md"
    "Papers/Watch Your Language Investigating Content Moderation with Large Language Models.md"
    "Papers/Like a Good Nearest Neighbor Practical Content Moderation and Text Classification.md"
)
PAPERS_DISCRIMINATIVE_AI=(
    "Papers/OpenAI content moderation API.md"
    "Papers/Multilingual content moderation, a case study on Reddit.md"
    "Papers/Perspective API.md"
    "Papers/Text classification using machine learning techniques.md"
    "Papers/Design and Application of an AI‐Based Text Content Moderation System.md"
    "Papers/Real-Time Content Moderation Using Artificial Intelligence and Machine Learning.md"
    "Papers/A review of standard text classification practices for multi-label toxicity identification of online content.md"
)
PAPERS_IMAGE=(
    "Papers/On-Device Content Moderation.md"
)
MEETINGS=(
    "Meeting notes/05-11-2024.md"
    "Meeting notes/26-11-2024.md"
    "Meeting notes/07-01-2025.md"
)
LEARNING=(
    "Learning Resources/Natural Language Processing (NLP).md"
    # "Learning Resources/NLP TensorFlow.md"
    "Learning Resources/Embeddings.md"
    "Learning Resources/Chain of Thought (CoT).md"
    "Learning Resources/Metrics Choice.md"
)

# Categories and their corresponding arrays
declare -A CATEGORIES
CATEGORIES["Main"]="FILES"
CATEGORIES["Websites"]="WEBSITES"
CATEGORIES["Papers/Generative AI Papers"]="PAPERS_GENERATIVE_AI"
CATEGORIES["Papers/Discriminative AI Papers"]="PAPERS_DISCRIMINATIVE_AI"
CATEGORIES["Papers/Image moderation Papers"]="PAPERS_IMAGE"
CATEGORIES["Meeting Notes"]="MEETINGS"
CATEGORIES["Learning Resources"]="LEARNING"

# Define the desired order of categories
CATEGORY_ORDER=(
    "Main"
    "Websites"
    "Papers/Generative AI Papers"
    "Papers/Discriminative AI Papers"
    "Papers/Image moderation Papers"
    "Meeting Notes"
    "Learning Resources"
)

# Function to add an extra # for all heading levels
add_layer_to_headings() {
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ $line =~ ^\#+\  ]]; then
            echo "#$line"  # Add an extra # to all headings
        else
            echo "$line"   # Keep non-headings as they are
        fi
    done < "$1"
}

# Function to process a category
process_category() {
    local category_name="$1"
    local file_array_name="$2"
    
    # Use eval to dereference the array dynamically
    eval "local files=(\"\${${file_array_name}[@]}\")"

    echo -e "\n---\n" >> "$OUTPUT_FILE"
    echo -e "# $category_name" >> "$OUTPUT_FILE"

    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            echo "Processing $file..."
            add_layer_to_headings "$file" >> "$OUTPUT_FILE"
            echo -e "\n---\n" >> "$OUTPUT_FILE"  # Add separator
        else
            echo "Warning: $file not found."
        fi
    done
}

# Process categories in the specified order
for category in "${CATEGORY_ORDER[@]}"; do
    if [[ -n "${CATEGORIES[$category]}" ]]; then
        process_category "$category" "${CATEGORIES[$category]}"
    else
        echo "Warning: Category '$category' is not defined."
    fi
done

echo "Combined specified Markdown files into $OUTPUT_FILE"
