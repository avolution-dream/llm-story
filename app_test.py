"""
Visualizing the storyboard for stories with infinite lengths.
The input should be a single story file.
"""

import io
import os
import re
import json
import torch
import openai
import argparse
import gradio as gr
from PIL import Image

from typing import Dict
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from diffusers import StableDiffusionXLPipeline
from concurrent.futures import ThreadPoolExecutor

from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# #########################
# Set the paths
# #########################
# Setting the default values for prompts
# To update them later in the gradio webpage
model_config_root = './configs/model.yaml'
model_name = 'gpt-4-1106-preview'
chunk_size = 1500
chunk_overlap = 0

# Set the prompt paths for summary and storyboard
instruction_root = Path('./prompts/instructions')
summary_question_prompt_path = instruction_root / 'summary_question.txt'
summary_refine_prompt_path = instruction_root / 'summary_refine.txt'
storyboard_map_prompt_path = instruction_root / 'storyboard_map.txt'


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
def get_summary_storyboard(model_config_root: str='./configs/model.yaml',
                           model_name: str='gpt-4-1106-preview',
                           summary_question_prompt_path: str='',
                           summary_refine_prompt_path: str='',
                           storyboard_map_prompt_path: str='',
                           # summary_save_path: str='',
                           # storyboard_save_path: str='',
                           split_docs: tuple=None):

    ##### Get the summary #####
    # Set the chat model
    model_config = yaml.safe_load(open(model_config_root))
    chat_model = ChatOpenAI(**model_config[model_name])

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
    # save_text(summary, summary_save_path)
    # save_text(storyboard, storyboard_save_path)

    return summary, storyboard


# ################################
# Helper Functions
# ################################
def extract_prompts_from_content(content: str,
                                 keyword: str='Text-to-Image Prompt'):
    """
    Get the text to image prompts from a string,
    allowing for various formats of the keyword.
    """
    prompts = []
    pattern = re.compile(rf'-? \[?\s*{re.escape(keyword)}\s*\]?:\s*(.*)',
                         re.IGNORECASE)

    # Split the file content into lines and process each line
    for line in content.splitlines():
        match = pattern.search(line)
        if match:
            prompt = match.group(1).strip()
            prompts.append(prompt)

    return prompts


# ################################
# Generate the images
# ################################
def generate_images(story_file):
    """
    Generating images with stable diffusions.
    """
    try:
        # Split stories into chunks
        print('Spliting the documents.')
        split_docs = get_split_docs(story_file.name,
                                    chunk_size,
                                    chunk_overlap)

        # Get and save the summary and the storyboard
        summary, storyboard = get_summary_storyboard(model_config_root,
                                                     model_name,
                                                     summary_question_prompt_path,
                                                     summary_refine_prompt_path,
                                                     storyboard_map_prompt_path,
                                                     # summary_save_path,
                                                     # storyboard_save_path,
                                                     split_docs)

        prompts = extract_prompts_from_content(
            storyboard,
            'Text-to-Image Prompt')

        # ####################################################################
        # This is temporary as it needs to be refined from the summary
        # We need to retrieve characters and envrionments from the summary
        # So as to maintain the consistency
        prompt_2 = 'Ghibli style. 80s early 90s aesthetic anime. studio anime, highly detailed. Japanese anime.'
        negative_prompt = 'photo, deformed, black and white, realism, disfigured, low contrast'
        # ####################################################################

        print("Loading pipeline...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
            custom_pipeline='lpw_stable_diffusion_xl').to('cuda')

        print("Pipeline loaded. Generating images...")
        images = []
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}")
            print(prompt)
            image = pipeline(prompt=prompt,
                             prompt_2=prompt_2).images[0]
            images.append(image)
            torch.cuda.empty_cache()

        print("Image generation completed.")
        return images

    except Exception as e:
        print(f'An error occurred: {e}')
        return []

    return None


# ################################
# The Gradio Interface
# ################################
with gr.Blocks() as demo:
    # Add the notation
    gr.Markdown('# Turn your stories into comics!')
    gr.Markdown('(The movie option will soon be updated.)')

    # Add the results
    with gr.Row():
        with gr.Column():
            storyboard_file = gr.File(label='Upload your story here.')
            generate_btn = gr.Button('Visualize your stories now!')
        with gr.Column():
            gallery = gr.Gallery(label='Generated Images',
                                 show_label=False,
                                 elem_id='gallery',
                                 columns=[4],
                                 rows=[4],
                                 object_fit='contain',
                                 height='auto')

    generate_btn.click(generate_images,
                       inputs=storyboard_file,
                       outputs=gallery)


if __name__ == '__main__':
    demo.queue().launch(share=True, debug=True)
