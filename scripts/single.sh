#!/bin/bash

# Define the story root directory
STORY_NAME="48-fox-fire"
STORY_ROOT="./scraping/fairy-tales-chinese"

# Other fixed arguments
STORY_TYPE="txt"
RESULT_ROOT="./results/fairy-tales-chinese"
INSTRUCTION_ROOT="./prompts/instructions"
MODEL_NAME="gpt-4-1106-preview"
CHUNK_SIZE=1500
CHUNK_OVERLAP=20
LANGUAGE="Chinese"


# Run the Python script with the extracted story name in the background
python3.10 run_infinite.py -sn $STORY_NAME -st $STORY_TYPE -sp $STORY_ROOT -rp $RESULT_ROOT -is $INSTRUCTION_ROOT -mn $MODEL_NAME -mt $CHUNK_SIZE -co $CHUNK_OVERLAP -la $LANGUAGE
