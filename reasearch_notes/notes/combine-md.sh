#!/bin/bash

# Path to store the combined file
OUTPUT_FILE="compilation.md"

# Empty the output file if it exists
> "$OUTPUT_FILE"

# List of specific Markdown files in the current directory (notes)
FILES=(
    "Master Thesis AI Content Moderation.md"
    "EthicalEye.md"
    "KoalaAI Text Moderation.md"
)

# List of specific Markdown files in the Papers directory (notes\Papers)
PAPERS=(
    "OpenAI content moderation API.md"
    "Multilingual content moderation, a case study on Reddit.md"
    "Perspective API.md"
    "Text classification using machine learning techniques.md"
    "Design and Application of an AI‐Based Text Content Moderation System.md"
    "Real-Time Content Moderation Using Artificial Intelligence and Machine Learning.md"
    "A review of standard text classification practices for multi-label toxicity identification of online content.md"
)

# List of specific Markdown files in the Meeting notes directory (notes\Meeting notes)
MEETINGS=(
    "05-11-24.md"
    "19-11-24.md"
)

echo -e "\n---\n" >> "$OUTPUT_FILE"             # Add initial separator for Table of contents

# Process files in the current directory
for filename in "${FILES[@]}"; do
    if [[ -f "$filename" ]]; then
        echo "Processing $filename..."
        # echo -e "\n# ${filename%.md}\n" >> "$OUTPUT_FILE"  # Add filename as a header
        cat "$filename" >> "$OUTPUT_FILE"                   # Append file content
        echo -e "\n---\n" >> "$OUTPUT_FILE"                 # Add separator
    else
        echo "Warning: $filename not found in the current directory."
    fi
done

# Process files in the Papers directory
for filename in "${PAPERS[@]}"; do
    file="Papers/$filename"
    if [[ -f "$file" ]]; then
        echo "Processing $file..."
        # echo -e "\n# ${filename%.md}\n" >> "$OUTPUT_FILE"  # Add filename as a header
        cat "$file" >> "$OUTPUT_FILE"                       # Append file content
        echo -e "\n---\n" >> "$OUTPUT_FILE"                 # Add separator
    else
        echo "Warning: $file not found in Papers directory."
    fi
done

# Process files in the Meeting notes directory
for filename in "${MEETINGS[@]}"; do
    file="Meeting notes/$filename"
    if [[ -f "$file" ]]; then
        echo "Processing $file..."
        # echo -e "\n# ${filename%.md}\n" >> "$OUTPUT_FILE"  # Add filename as a header
        cat "$file" >> "$OUTPUT_FILE"                       # Append file content
        echo -e "\n---\n" >> "$OUTPUT_FILE"                 # Add separator
    else
        echo "Warning: $file not found in Papers directory."
    fi
done

echo "Combined specified Markdown files into $OUTPUT_FILE"
