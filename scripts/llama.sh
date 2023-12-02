#!/bin/bash

# Start the server in the background
python3.10 -m vllm.entrypoints.openai.api_server \
--model meta-llama/Llama-2-7b-chat-hf \
--port 8000 \
--trust-remote-code \
--tensor-parallel-size 4 &
echo $! > server.pid

# Initialize a counter for the timeout
counter=0
max_wait=60  # Maximum wait time in seconds

echo "Waiting for the server to start..."

# Wait for the server to start with a timeout
while ! nc -z localhost 8000; do
sleep 1
counter=$((counter+1))
if [ "$counter" -ge "$max_wait" ]; then
  echo "Timeout reached. Server is not up. Please run the first command to troubleshoot."
  echo "It may just be the case that model is downloading"
  exit 1
fi
done

echo "Server is up and running."

# Run the second command
python3.10 run_infinite.py \
--story_name the-dream-eater \
--story_root ./scraping/flash-fiction-library/horror \
--result_root ./results/flash-fiction-library/horror \
--model_name meta-llama/Llama-2-7b-chat-hf \
--chunk_size 1000 \
-sv
