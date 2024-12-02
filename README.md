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
