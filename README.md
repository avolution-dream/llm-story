# LLM-Story
<p align="center">
  <img src="./gradio.png" />
</p>


## 🕹 Setup
```bash
# (Optional) Create a conda virtual environment and activate it
conda create --name story python=3.10
conda activate story

# Install the packages
pip install -r requirements.txt
```
### Running with Open Source Models (LLAMA/Mosaic/...)
The current code base supports most models from Hugging Face and we use vllm for fast inference. Below is an example command to start a server in the background.
```bash
python3.10 -m vllm.entrypoints.openai.api_server \
  --model mosaicml/mpt-7b-8k-chat \
  --dtype float \
  --trust-remote-code \
  --tensor-parallel-size 4
```
By default, the above command line starts the server at `http://localhost:8000`, and you may also specify the address with `--host` and `--post`. Check the [vllm doc](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) for reference.

We have tested `llama`, `vicuna`, `mpt`, `zephyr`. Other models should work as well but not heavily tested yet.

Next, modify the `configs/model.yaml` to match your setting, and be sure the adjust the arguments later on when you work on `run_infinite.py`.

<!-- Currently, open-source models may fail in certain cases; we are fine -->

### Running with Blackbox Models (GPT/Anthropic/...)
You can set the api key in the environment, or directly modify the `configs/model.yaml`.
```bash
export OPENAI_API_KEY=<insert your OpenAI API key>
```

## 🕹 Usage
### Gradio Interface
Directly uploading your story file and get the generated images/videos:
```bash
gradio app_test.py
```
If you have obtained parsed storyboard and summary script (by running `run_infinite.py` prior to this, see the next section for details), you can run them with:
```bash
gradio app.py
```



### Getting Storyboard Scripts from Stories
#### Command Lines
<!-- A stable version is at `run-gpt.py`. Below is an example command, and you may upload your own story (defaul path is `./stories`) and check the Python file for customized arguments.
```bash
python3.10 run_gpt.py
```
An advanced yet in development version is at `run_infinite.py`. -->
Get the text files for storyboard and summary! We support infinite length of stories. Modify `-mn` to match the model you are using with `./configs/model.yaml`.
```bash
python3.10 run_infinite.py \
    -sp ./scraping/flash-fiction-library/horror \
    -rp ./results/flash-fiction-library/horror \
    -mn gpt-4-1106-preview \
    -sn the-dream-eater \
    -sv
```
Check the bash template as well.
```bash
chmod +x ./scripts/flashlib-fantasy.sh
./scripts/flashlib-fantasy.sh
```
Generating images/videos using SDXL.
```bash
python3.10 run_story_to_image.py
```
#### Collected Stories and Results
Parsed storyboard scripts locate in `./results`.

Copyright-free stories (special thanks to the Gutenberg Project) locate in `./scraping`, we have collected hundreds of different-length stories ranging from: fairy tales, romance, fantasy, literary, horror, detective fiction, science fiction.

(Notice the flash-fiction are only for test use.) This effort is ongoing and we will focusing on generating best-quality scripts and the derived images.



<!-- ### (Images/Videos) Comics Generation -->
<!-- If you have obtained parsed storyboard and summary script, you can run:
```bash
gradio app.py
```
We will soon updating this so that you can run with a single story txt file. -->

<!-- For now, we directly use the huggingface tools for image/video generation.
```bash
python3.10 run_story_to_image.py
``` -->


## 🕹 Introduction
This repository is for automatically generating high-quality summary (for characters and environments) and storyboard script for any-length story files, ultimately for the adaptations across multiple media formats, such as films and animation. The codes are structured as follows:
- `./run.py`: a compact file to generate the summary and storyboard script for any given file.
- `./prompts/`: the directory for crafted prompts serving different purposes, including instructions and demonstrations.
- `./results`: the directory for saving results.
- `./scraping`: the directory for stories that scraped from different flash fictions.
    - The categories include fantasy, horror, romance, sci-fi, classic, humor, literary.

Currently, the methodology is:
- Use the `refine chain` to generate the summary for characters and environments, which works by looping over story chunks and iteratively update the results as new contexts become available.
- Use the `map reduce chain` to generate the storyboard; in this case, each chunk of story is independently visited along with the previously generated summary by the agent; a vector database may be connected for smoother generation.


## 🕹 Tasks (to be refined as a detailed doc)
Major and mid-term goal: Improving the quality of the generated scripts and the derived prompts.

- [ ] Add the story generation function.
- [ ] Make this to be a streamlit app.
- [ ] Scraping for more stories with different varieties.
- [ ] Refining prompt engineering.
    - Pre-prompt engineering: few-shot demonstrations, instructions, reasoning techniques
    - Post-prompt engineering: designing critiques.
    - The consistency issue.
- [ ] Improving the parallel part with memory modules.
- [ ] Refining the gradio with stable video diffusion.
- [ ] Finetuning/Distilling Open-Source llm models.
    - Currently the open-source model cannot match that of the GPT-4 models. A simple approach is to use the GPT-4-generated results to finetune that of other open-source models.
    <!-- - [ ] Demonstrations for few-shot illustration with expert exemplars.
        - This is essential to improve the story performance. -->
    <!-- - [PromptBase](https://promptbase.com/)
    - Reddit
    - [PromptHero](https://prompthero.com/) -->
<!-- - [ ] Making this as an agents; adding critics to improve writing -->
