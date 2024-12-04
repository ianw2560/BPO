# BPO

Implementation of Black-Box Prompt Optimization (BPO) for CAP6614.

## Prerequisites

This project uses Poetry. You will need to download the Python dependencies with:

```
pip install poetry
poetry install
```

This project also requires API keys for the GPT and Claude models. You must set the following variables with appropriate API key:

| Model | Variable |
| ----- | -------- |
| GPT | `OPENAI_API_KEY` |
| Claude| `ANTHROPIC_API_KEY` |

To utlize Google cloud models like Text-Bison and Gemini, it requires getting set up with Google CLI. Start by downloading the SDK found here: https://cloud.google.com/sdk/docs/install

After following all steps in that guide, including initializing Google CLI with ``` gcloud init    ```, a user must log in and choose their unique project ID.

Additionally, a user must specify their project name in the ```generate_response_textbison()``` function in generate_responses.py.

Example:
```vertexai.init(project='bpo111', location="us-central1")```

## Training

You will need access to the `meta-llama/Llama-2-7b-chat-hf` model. 
To access this model, you will first need to login to Hugging Face.

```
huggingface-cli login --token <hugging-face-access-token>
```

## Evaluation

### Generate Optimized Prompts

To evaluate the BPO model, you will first need to generate optimized versions of the prompts in the evaluation datasets. These are located in `data/eval_datasets`.

You can specify the datasets you want to generate optimized prompts for using the `-d` flag.

This can be done using the following command:

```bash
poetry run python evaluation/generate_responses.py opt -d dolly vicuna self_instruct bpo_test
```

This will require loading in the BPO model, which requires running on a system with more RAM and GPU resources. It is recommended to use the Newton cluster if you are student at UCF. See the [section on Newton for how to do this.](#newton)

### Generate Prompt Responses

Next you will need to generate responses for both the original and optimized prompts.

You can specify the datasets and LLM models you want to use to generate responses with using the `-d` and `-m` flags respectively.

```bash
poetry run python evaluation/generate_responses.py eval -d dolly vicuna self_instruct bpo_test
```

### Evaluate Responses

Next, you can evaluate which responses are preferred by GPT-4, the original or BPO responses, using the following command:

```
poetry run python evaluation/evaluate.py eval -d dolly vicuna self_instruct bpo_test
```

A set of tables with the evaluation data will be printed.

### Newton

If you are running on Newton, you will need to run the job using SLURM. This command has 

```bash
module load python/python-3.11.4-gcc-12.2.0
python3 install -r requirements.txt
srun -N 1 --gres=gpu:1 --gres-flags=enforce-binding --time=2:00:00 --mem=70G --constraint="gpu32|gpu80" python3 evaluation/generate_responses.py opt -d dolly vicuna self_instruct bpo_test
```

## Results

The follow section contains our results

**Model: GPT-4o**
Dataset        | Original | Tie | BPO | Original (%) | Tie (%) | BPO (%) |
|:-------------- | --------:| ---:| ---:| ------------:| -------:| -------:|
bpo_test       | 90.0  | 19.0  | 91.0  |   45.0  |  9.5  |  45.5
dolly          | 69.0  | 36.0  | 95.0  |   34.5  |  18.0 |  47.5
vicuna         | 33.0  | 6.0   | 41.0  |  41.25  |  7.5  | 51.25
self_instruct  | 57.0  | 82.0  | 113.0 |  22.62  | 32.54 | 44.84

**Model: GPT-3.5-Turbo**


Dataset        | Original | Tie | BPO | Original (%) | Tie (%) | BPO (%) |
|:-------------- | --------:| ---:| ---:| ------------:| -------:| -------:|
bpo_test       | 98.0  | 23.0  | 79.0  | 49.0 | 11.5 | 39.5
dolly          | 84.0  | 28.0  | 88.0  | 42.0 | 14.0 | 44.0
vicuna         | 30.0  |  3.0  | 47.0  | 37.5 | 3.75 | 58.75
self_instruct  | 101.0 | 27.0  | 124.0 | 40.08| 10.71| 49.21


