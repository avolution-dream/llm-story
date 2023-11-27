#!/bin/bash

# Define the story root directory
STORY_ROOT="./scraping/flash-fiction-library/romance"

# Other fixed arguments
STORY_TYPE="txt"
RESULT_ROOT="./results/flash-fiction-library/romance"
INSTRUCTION_ROOT="./prompts/instructions"
MODEL_NAME="gpt-4-1106-preview"
CHUNK_SIZE=1500
CHUNK_OVERLAP=0
LANGUAGE="Chinese"
SAVE_TIME="--save_time"

# Find all .txt files in the story root directory and process them
find "$STORY_ROOT" -maxdepth 1 -name "*.txt" | while read file; do
    # Extract the file name without the path and extension
    story_name=$(basename "$file" .txt)

    # Skip specific stories
    if [[ "$story_name" == "my-tail" ]] || [[ "$story_name" == "those-that-live-longest" ]]; then
        echo "Skipping story: $story_name"
        continue
    fi

    # Echo the story name
    echo "Processing story: $story_name"

    # Run the Python script with the extracted story name
    python run_infinite.py -sn "$story_name" -st $STORY_TYPE -sp $STORY_ROOT -rp $RESULT_ROOT -is $INSTRUCTION_ROOT -mn $MODEL_NAME -mt $CHUNK_SIZE -co $CHUNK_OVERLAP -la $LANGUAGE $SAVE_TIME
done
