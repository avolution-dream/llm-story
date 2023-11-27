import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import io

# Function to extract prompts from the text file
def extract_prompts(content, keyword='[Text-to-Image Prompt]:'):
    prompts = []
    keyword_lower = keyword.lower()

    for line in content:
        line_lower = line.lower()
        if keyword_lower in line_lower:
            # Find the starting index of the actual prompt in the original line (not the lowercased one)
            start_index = line_lower.find(keyword_lower) + len(keyword)
            # Extract the actual prompt text
            prompt = line[start_index:].strip()
            prompts.append(prompt)

    return prompts

# Image generation function for Gradio
def generate_images(storyboard_file, summary_file):
    try:
        print("Reading files...")
        storyboard_content = storyboard_file.read()
        summary_content = summary_file.read()

        print("Extracting prompts...")
        prompts = extract_prompts(storyboard_content, '[Text-to-Image Prompt]: ')
        print('i am printing prompts!')
        for i in prompts:
            print(i)

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
            image = pipeline(prompt=prompt).images[0]

            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')
            image_bytes = byte_arr.getvalue()
            images.append(image_bytes)

            torch.cuda.empty_cache()

        print("Image generation completed.")
        return images
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Define Gradio Blocks Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            storyboard_file = gr.File(label="Upload Storyboard File")
            summary_file = gr.File(label="Upload Summary File")
            generate_btn = gr.Button("Generate Images")
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images", show_label=False, columns=3, rows=1, object_fit="contain", height="auto")

    generate_btn.click(generate_images, inputs=[storyboard_file, summary_file], outputs=gallery)

if __name__ == "__main__":
    demo.launch(share=True)

