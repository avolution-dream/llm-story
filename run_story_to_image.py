"""
Take the parsed stories and make it into images.
Next step is to connect with stable videos.
"""

from diffusers import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline
from PIL import Image

import os
import torch
import warnings
import argparse
import torch


# #########################
# Arguments
# #########################
# Set the parser
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-sbp', '--storyboard_path', type=str,
                    default='./results/flash-fiction-library/scifi/those-that-live-longest-2023-11-26-19-32-storyboard.txt')
parser.add_argument('-smp', '--summary_path', type=str,
                    default='results/flash-fiction-library/scifi/those-that-live-longest-2023-11-26-19-32-summary.txt')
parser.add_argument('-imp', '--image_path', type=str,
                    default='images/flash-fiction-library/scifi/those-that-live-longest')

# The warning is sourced from the SDXL tokenizer, which is already fixed by lpw.
warnings.filterwarnings('ignore', message='sequence length is longer than the specified maximum')

# Parse the arguments
p = parser.parse_args('')

# Set the argument
for key, value in vars(p).items():
    globals()[key] = value

# Set the path
os.makedirs(image_path, exist_ok=True)


# #########################
# Helper functions
# #########################
def extract_prompts(file_path: str='',
                    keyword: str='[Text-to-Image Prompt]: '):
    """
    Get the text to image prompts from the file.

    This is a temporary solution as we need pydantic to
    do exact match.
    """
    prompts = []

    with open(file_path, 'r') as file:
        content = file.readlines()
        for line in content:
            if keyword in line:
                prompt = line.split(keyword)[-1].strip()
                prompts.append(prompt)

    return prompts


# #########################
# Generate images
# #########################
# Get the prompts
prompts = extract_prompts(storyboard_path, '[Text-to-Image Prompt]: ')
voiceovers = extract_prompts(storyboard_path, '[Voiceover]: ')

# This is temporary as it needs to be refined from the summary
# We need to retrieve characters and envrionments from the summary
# So as to maintain the consistency
prompt_2 = 'Ghibli style. 80s early 90s aesthetic anime. studio anime, highly detailed. Japanese anime.'
negative_prompt = 'photo, deformed, black and white, realism, disfigured, low contrast'

# Set the pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype      = torch.float16,
    variant          = 'fp16',
    use_safetensors  = True,
    custom_pipeline  = 'lpw_stable_diffusion_xl').to('cuda')

# Generate images
for i, prompt in enumerate(prompts):
    # Get the image
    image = pipeline(prompt=prompt,
                     prompt_2=prompt_2,
                     negative_prompt=negative_prompt).images[0]

    # Add to list (if needed)
    images.append(image)

    # Print the results
    print(voiceovers[i])

    # Save the image
    image_file_path = os.path.join(image_path, f'image_{i}.png')
    image.save(image_file_path)

    # Empty the cache
    torch.cuda.empty_cache()

print('All done!')
print(f'Images saved in {image_path}.')
