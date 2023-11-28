#!/bin/bash

# Define the story root directory
STORY_ROOT="./scraping/fairy-tales-japanese"

# Other fixed arguments
STORY_TYPE="txt"
RESULT_ROOT="./results/fairy-tales-japanese"
INSTRUCTION_ROOT="./prompts/instructions"
MODEL_NAME="gpt-4-1106-preview"
CHUNK_SIZE=1500
CHUNK_OVERLAP=0
LANGUAGE="Chinese"

# Initialize a counter
count=0

# Find all .txt files in the story root directory and process them
find "$STORY_ROOT" -maxdepth 1 -name "*.txt" | while read file; do
    # Extract the file name without the path and extension
    story_name=$(basename "$file" .txt)

    # Increment the counter
    ((count++))

    # Break the loop if five files have been processed
    if [ $count -gt 5 ]; then
        break
    fi

    # Echo the story name
    echo "Starting parallel processing for: $story_name"

    # Run the Python script with the extracted story name in the background
    python3.10 run_infinite.py -sn "$story_name" -st $STORY_TYPE -sp $STORY_ROOT -rp $RESULT_ROOT -is $INSTRUCTION_ROOT -mn $MODEL_NAME -mt $CHUNK_SIZE -co $CHUNK_OVERLAP -la $LANGUAGE 2>&1 &

done

# Wait for all background processes to finish
wait
