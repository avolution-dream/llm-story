"""
Generating the storyboard with a single agent (with context limit).
"""

import json
import openai
import argparse
import nest_asyncio
from typing import Dict
from pathlib import Path
from instructor import patch
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
patch()


# #########################
# Arguments
# #########################
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-sn', '--story_name', type=str, default='nightingale-and-rose')
parser.add_argument('-sp', '--story_path', type=str, default='./stories')
parser.add_argument('-rp', '--result_path', type=str, default='./results')
parser.add_argument('-is', '--instruction_path', type=str, default='./prompts/instructions')
parser.add_argument('-mn', '--model_name', type=str, default='gpt-3.5-turbo-16k')
parser.add_argument('-mt', '--max_tokens', type=int, default=2048)
parser.add_argument('-mts', '--max_tokens_storyboard', type=int, default=8192)
parser.add_argument('-s', '--stop', type=str, default=None)
parser.add_argument('-t', '--temperature', type=float, default=0.5)
parser.add_argument('-l', '--language', type=str, default='Chinese')
parser.add_argument('-sv', '--save_time', action='store_true',
                    help='Flag to add current time in filename.')
parser.add_argument('-sm', '--summary_filename', type=str,
                    default='summary_test.txt')
parser.add_argument('-st', '--storyboard_filename', type=str,
                    default='storyboard_test.txt')

# Parse the arguments
p = parser.parse_args()
story_name = p.story_name
story_path = Path(p.story_path)
result_path = Path(p.result_path)
instruction_path = Path(p.instruction_path)
model_name = p.model_name
max_tokens = p.max_tokens
max_tokens_storyboard = p.max_tokens_storyboard
stop = p.stop
temperature = p.temperature
language = p.language

# Configure the time to be added in the resulted filename
if p.save_time:
    save_time = '-' + datetime.now().strftime('%Y-%m-%d-%H-%M')
else:
    save_time = ''

# Configure the path
story_path = story_path / f'{story_name}.txt'
character_prompt_path = instruction_path / p.summary_filename
storyboard_prompt_path = instruction_path / p.storyboard_filename
character_result_path = result_path / f'{story_name}{save_time}_test_summary.txt'
storyboard_result_path = result_path / f'{story_name}{save_time}_test_storyboard.txt'

# #########################
# Helper functions
# #########################
# Text Processeing
def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def save_text(response, output_file):
    with open(output_file, 'w') as file:
        file.write(response + '\n')


def split_into_chunks(text, tokens=500):
    chunks = []

    for i in range(0, len(text), tokens):
        chunks.append(text[i:i + tokens])

    return chunks

# OpenAI API Call
def call_openai_api(history,
                    content,
                    model_name: str='gpt-3.5-turbo-16k',
                    max_tokens: int=2048,
                    stop: str=None,
                    temperature: float=0.5):

    history.append({'role': 'user',
                    'content': content})

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=history,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
    )

    return response.choices[0]['message']['content'].strip()


# Story Processing
def process_story(story_path,
                  character_prompt_path,
                  storyboard_prompt_path,
                  model_name: str='gpt-3.5-turbo-16k',
                  max_tokens: int=4096,
                  max_tokens_storyboard: int=4096,
                  stop: str=None,
                  temperature: float=0.5,
                  language: str='Chinese'):

    story = load_text(story_path)
    chunks = split_into_chunks(story)

    # Initialize the conversation history
    history = [{'role': 'system',
                'content': 'You are a helpful assistant.'},
               {'role': 'system',
                'content': 'Below is the long text provided by the user.'}]

    if language != 'English':
        history.append({'role': 'user',
                        'content': f'Please translate all your responses in {language}'})

    # Add each chunk as a user message to history
    for chunk in chunks:
        history.append({'role': 'user',
                        'content': chunk})

    character_prompt = load_text(character_prompt_path)
    storyboard_prompt = load_text(storyboard_prompt_path)

    print('The model is getting the summary.')

    # Task 1: Summarize characters
    character_summary = call_openai_api(history, character_prompt, model_name,
                                        max_tokens, stop, temperature)

    # Update the history to include the character summary
    history.append({'role': 'assistant',
                    'content': character_summary})

    print('The model is getting the storyboard script.')

    # Task 2: Generate storyboard script
    storyboard_script = call_openai_api(history, storyboard_prompt, model_name,
                                        max_tokens_storyboard, stop, temperature)

    # Save to output file
    return character_summary, storyboard_script


# #########################
# Helper functions
# #########################
if __name__ == '__main__':
    character_summary, storyboard_script = process_story(story_path,
                                                         character_prompt_path,
                                                         storyboard_prompt_path,
                                                         model_name,
                                                         max_tokens,
                                                         max_tokens_storyboard,
                                                         stop,
                                                         temperature,
                                                         language)
    save_text(storyboard_script, storyboard_result_path)
    save_text(character_summary, character_result_path)
