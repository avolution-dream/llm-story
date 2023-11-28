"""
Generating the storyboard for stories with infinite lengths.
"""

import os
import re
import json
import openai
import argparse

from typing import Dict
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from concurrent.futures import ThreadPoolExecutor

from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# #########################
# Arguments
# #########################
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-sn', '--story_name', type=str, default='my-tail')
parser.add_argument('-st', '--story_type', type=str, default='txt')
parser.add_argument('-sp', '--story_root', type=str, default='./scraping/flash-fiction-library/romance')
parser.add_argument('-rp', '--result_root', type=str, default='./results/flash-fiction-library/romance')
parser.add_argument('-is', '--instruction_root', type=str, default='./prompts/instructions')
parser.add_argument('-mn', '--model_name', type=str, default='gpt-4-1106-preview')
parser.add_argument('-mt', '--chunk_size', type=int, default=1500)
parser.add_argument('-co', '--chunk_overlap', type=int, default=0)
parser.add_argument('-la', '--language', type=str, default='Chinese')
parser.add_argument('-sv', '--save_time', action='store_true',
                    help='Flag to add current time in filename.')


# #########################
# Helper functions
# #########################
# Text Processeing
def load_text(file_path):

    with open(file_path, 'r') as file:
        return file.read()


def save_text(text, file_path):

    with open(file_path, 'w') as file:
        file.write(text + '\n')


def format_scene(storyboard_result):
    """
    This is a temporary sol and would
    be replace by Pydantic modules later.

    storyboard_result (dict): a chain map reduce dict.
    """
    # Initialize a string
    text = ''

    # Format the divide line
    for i in storyboard_result['intermediate_steps']:
        text += '\n---\n\n' + i + '\n'

    # Format the number which was discarded in map reduce
    scenes = text.split('[Scene]')
    text = scenes[0]
    for j, scene in enumerate(scenes[1:], start=1):
        text += f'[Scene {j}]{scene}'

    return text


# #########################
# Split the docs
# #########################
def get_split_docs(story_path: str='./story.txt',
                   chunk_size: int=1500,
                   chunk_overlap: int=0):
    """
    Make stories to be splitted.
    """

    # Load with text loader
    loader = TextLoader(story_path)
    doc = loader.load()

    # Split the story into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(doc)

    return split_docs


# #########################
# Get the results
# #########################
def get_summary_storyboard(model_name: str='gpt-4-1106-preview',
                           summary_question_prompt_path: str='',
                           summary_refine_prompt_path: str='',
                           storyboard_map_prompt_path: str='',
                           summary_save_path: str='',
                           storyboard_save_path: str='',
                           split_docs: tuple=None):

    ##### Get the summary #####
    # Set the chat model
    chat_model = ChatOpenAI(model_name=model_name)

    # Load the prompt
    summary_question_prompt = load_text(summary_question_prompt_path)
    summary_refine_prompt = load_text(summary_refine_prompt_path)

    # Set the prompts
    question_prompt = PromptTemplate.from_template(summary_question_prompt)
    refine_prompt = PromptTemplate.from_template(summary_refine_prompt)

    # Run the chain
    summary_chain = load_summarize_chain(
        llm=chat_model,
        chain_type='refine',
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key='input_documents',
        output_key='output_text',
    )

    # Get the results
    print('Getting the summary for characters and environments.')
    print('This may take a while as i am constantly refining results.')
    summary_result = summary_chain({'input_documents': split_docs})
    summary = summary_result['output_text']


    ##### Get the storyboad #####
    # Load the prompt
    storyboard_map_prompt = load_text(storyboard_map_prompt_path)

    # Set the prompts
    map_prompt = PromptTemplate.from_template(storyboard_map_prompt)
    combine_prompt = PromptTemplate.from_template('Provide an overall summary of {text}.')

    # Run the chain
    storyboard_chain = load_summarize_chain(
        llm=chat_model,
        chain_type='map_reduce',
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
        input_key='input_documents'
    )

    # Get the results
    print('Getting the storyboard script.')
    print('i am faster this time since it is fine to be parallel here.')
    storyboard_result = storyboard_chain(
        {'input_documents': split_docs,
         'summary': summary})
    storyboard = format_scene(storyboard_result)

    # Save them
    print('All done!')
    save_text(summary, summary_save_path)
    save_text(storyboard, storyboard_save_path)

    return summary, storyboard


# #########################
# Translate things
# #########################
def translate_summary_storyboard(model_name: str='gpt-4-1106-preview',
                                 language: str='Chinese',
                                 summary: str='',
                                 storyboard: str=''):
    """
    Translate the summary into the designated language.
    """

    # Set the chat model
    chat_model = ChatOpenAI(model_name=model_name)

    # Translate for SUMMARY
    summary_translated = chat_model.predict(
        f'Translate the following text in {language}: {summary}')

    # Translate for the STORYBOARD
    # Below is a bit nasty
    # We will update that with Pydantic to keep clean
    def extract_all_scenes(text):
        # Split the text by '---' to get the scenes
        scenes = text.split('---')

        # Remove empty strings and strip leading/trailing white spaces from each scene
        scenes = [scene.strip() for scene in scenes if scene.strip()]

        return scenes

    # Extract all scenes using the revised function
    scenes = extract_all_scenes(storyboard)

    # The helper function for scene translation
    def translate_scene(scene):
        translation_prompt = (f'Translate the following text in {language}: \n'
                              f'{scene}\n\n'
                              'YOUR RESPONSE GOES HERE: \n')
        scene_translated = chat_model.predict(translation_prompt)
        return scene_translated

    # List to hold translated scenes
    storyboard_translated_list = []

    # Using ThreadPoolExecutor to translate scenes in parallel
    with ThreadPoolExecutor() as executor:
        storyboard_translated_list = list(executor.map(translate_scene, scenes))

    # Initialize a string
    storyboard_translated = ''

    # Format the divide line
    for i in storyboard_translated_list:
        storyboard_translated += '\n---\n\n' + i + '\n'

    # Save them
    save_text(summary_translated,
              result_root / f'{story_name}-{save_time}-summary-translated.txt')
    save_text(storyboard_translated,
              result_root / f'{story_name}-{save_time}-storyboard-translated.txt')

    # Notify
    print('All done!')

    return None


# #########################
# Put everything together
# #########################
def parse_single_story(story_path: str='./story.txt',
                       chunk_size: int=1500,
                       chunk_overlap: int=0,
                       model_name: str='gpt-4-1106-preview',
                       summary_question_prompt_path: str='',
                       summary_refine_prompt_path: str='',
                       storyboard_map_prompt_path: str='',
                       summary_save_path: str='',
                       storyboard_save_path: str='',
                       language: str='Chinese'):

    # Split stories into chunks
    split_docs = get_split_docs(story_path,
                                chunk_size,
                                chunk_overlap)

    # Get and save the summary and the storyboard
    summary, storyboard = get_summary_storyboard(model_name,
                                                 summary_question_prompt_path,
                                                 summary_refine_prompt_path,
                                                 storyboard_map_prompt_path,
                                                 summary_save_path,
                                                 storyboard_save_path,
                                                 split_docs)

    # Translate and save the resulted content
    if language:
        translate_summary_storyboard(model_name,
                                     language,
                                     summary,
                                     storyboard)

    return summary, storyboard


# #########################
# >>> Set the paths
# #########################
# Parse the arguments
p = parser.parse_args()

# Set the argument
for key, value in vars(p).items():
    globals()[key] = value

# Configure the time to be added in the resulted filename
save_time = datetime.now().strftime('%Y-%m-%d-%H-%M') if p.save_time else ''

# Set the result root
result_root = Path(result_root)
os.makedirs(result_root, exist_ok=True)

# Set the story path
story_path = Path(story_root) / f'{story_name}.{story_type}'

# Set the prompt paths for summary and storyboard
summary_question_prompt_path = Path(instruction_root) / 'summary_question.txt'
summary_refine_prompt_path = Path(instruction_root) / 'summary_refine.txt'
storyboard_map_prompt_path = Path(instruction_root) / 'storyboard_map.txt'

# Set the saving paths for summary and storyboard
summary_save_path = result_root / f'{story_name}-{save_time}-summary.txt'
storyboard_save_path = result_root / f'{story_name}-{save_time}-storyboard.txt'


# ##########################
# >>> Run the main function
# ##########################
if __name__ == '__main__':
    summary, storyboard = parse_single_story(story_path,
                                             chunk_size,
                                             chunk_overlap,
                                             model_name,
                                             summary_question_prompt_path,
                                             summary_refine_prompt_path,
                                             storyboard_map_prompt_path,
                                             summary_save_path,
                                             storyboard_save_path,
                                             language)
