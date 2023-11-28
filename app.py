from diffusers import StableDiffusionXLPipeline
from PIL import Image
import gradio as gr
import torch
import io
import re


# ################################
# Helper Functions
# ################################
def extract_prompts(file_path: str='',
                    keyword: str='Text-to-Image Prompt'):
    """
    Get the text to image prompts from the file,
    allowing for various formats of the keyword.
    """
    prompts = []

    with open(file_path, 'r') as file:
        content = file.readlines()
        pattern = re.compile(rf'-? \[?\s*{re.escape(keyword)}\s*\]?:\s*(.*)',
                             re.IGNORECASE)

        for line in content:
            match = pattern.search(line)
            if match:
                prompt = match.group(1).strip()
                prompts.append(prompt)

    return prompts


def generate_images(storyboard_file, summary_file):
    """
    Generating images with stable diffusions.
    """
    try:
        print("Reading files...")
        print(storyboard_file.name)
        print("Extracting prompts...")
        prompts = extract_prompts(storyboard_file.name, 'Text-to-Image Prompt')

        print("Loading pipeline...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
            custom_pipeline='lpw_stable_diffusion_xl').to('cuda')

        # ####################################################################
        # This is temporary as it needs to be refined from the summary
        # We need to retrieve characters and envrionments from the summary
        # So as to maintain the consistency
        prompt_2 = 'Ghibli style. 80s early 90s aesthetic anime. studio anime, highly detailed. Japanese anime.'
        negative_prompt = 'photo, deformed, black and white, realism, disfigured, low contrast'
        # ####################################################################

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
        print(f"An error occurred: {e}")
        return []


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
            storyboard_file = gr.File(label="Upload Storyboard File")
            summary_file = gr.File(label="Upload Summary File")
            generate_btn = gr.Button("Generate Images")
        with gr.Column():
            gallery = gr.Gallery(label='Generated Images',
                                 show_label=False,
                                 elem_id='gallery',
                                 columns=[4],
                                 rows=[4],
                                 object_fit='contain',
                                 height="auto")

    generate_btn.click(generate_images,
                       inputs=[storyboard_file, summary_file],
                       outputs=gallery)


if __name__ == '__main__':
    demo.queue().launch(share=True, debug=True)
