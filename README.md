# LLM-Story

## 🕹 Introduction
This repository is for automatically generating high-quality summary (for characters and environments) and storyboard script for any-length story files, ultimately for the adaptations across multiple media formats, such as films and animation. The codes are structured as follows:
- `./run.py`: a compact file to generate the summary and storyboard script for any given file.
- `./prompts/`: the directory for crafted prompts serving different purposes, including instructions and demonstrations.
- `./results`: the directory for saving results.
- `./scraping`: the directory for stories that scraped from different flash fictions.
    - The categories include fantasy, horror, romance, sci-fi, classic, humor, literary.

You may check an example output at `./results/one-day-prisoner-storyboard.txt`.

Currently, the methodology is:
- Use the `refine chain` to generate the summary for characters and environments, which works by looping over story chunks and iteratively update the results as new contexts become available.
- Use the `map reduce chain` to generate the storyboard; in this case, each chunk of story is independently visited along with the previously generated summary by the agent; a vector database may be connected for smoother generation.

## 🕹 Setup
```bash
# Export your OpenAI API key
export OPENAI_API_KEY=<insert your OpenAI API key, something like sk-...bsdY>

# (Optional) Create a conda virtual environment and activate it
conda create --name story python=3.10
conda activate story

# Install the packages
pip install -r requirements.txt
```

## 🕹 Usage
### (Text) Storyboard Generation
A stable version is at `run-gpt.py`. Below is an example command, and you may upload your own story (defaul path is `./stories`) and check the Python file for customized arguments.
```bash
python3.10 run_gpt.py
```
An advanced yet in development version is at `run_infinite.py`.
```bash
python3.10 run_infinite.py -sp ./scraping/flash-fiction-library/horror -rp ./results/flash-fiction-library/horror -sv -sn the-dream-eater
```
We also provide a simple bash file to help parallel running.
```bash
chmod +x ./scripts/flashlib-fantasy.sh
./scripts/flashlib-fantasy.sh
```
You may check `./demo-infinite.ipynb` for an high-level overview.

### (Images/Videos) Comics Generation
For now, we directly use the huggingface tools for image/video generation.
```bash
python3.10 run_story_to_image.py
```



## 🕹 Tasks (to be refined to the production level)
- [ ] Scraping for more stories with different varieties.
- [ ] Making this as an agents; adding critics to improve writing
- [ ] Improve the parallel part with memory modules.
- [ ] Demonstrations for few-shot illustration with expert exemplars.
    - This is essential to improve the story performance.
- [ ] Connect with diffusers and get a gradio website for this.
- [ ] Tune open-source models (using the blackbox adaptation technique.).
