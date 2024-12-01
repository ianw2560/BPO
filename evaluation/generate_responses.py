import argparse
import json
import torch
import pandas

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import anthropic
import vertexai

from openai import OpenAI
from vertexai.generative_models import GenerativeModel

def generate_response_gpt3_5_turbo(prompt: str):

    model = "gpt-3.5-turbo"

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response = response.choices[0].message.content

    return response


def generate_response_gpt4o(prompt: str):

    model = "gpt-4o"

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response = response.choices[0].message.content

    return response

def generate_response_claude_haiku(prompt: str):
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(message.content)
    llm_response = message.choices[0].message.content

    return llm_response

def generate_response_claude2(prompt: str):
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(message.content)
    llm_response = message.choices[0].message.content

    return llm_response

def generate_response_textbison(prompt: str):
    vertexai.init(project='bpo111', location="us-central1")

    model = GenerativeModel("gemini-1.5-flash-002")

    response = model.generate_content(prompt)

    print(response.text)
    llm_response = response.text

    return llm_response

def generate_optimized_prompt_bpo(prompt: str, context: str, device, tokenizer, model):
    """Calls our Seq2Seq model and returns an optimized version of the input prompt."""

    optimize_prompt_template = open("evaluation/optimize_prompt_template.txt")
    optimize_prompt_template = optimize_prompt_template.read()

    prompt = optimize_prompt_template.replace("{prompt}", prompt) #.replace("{context}", context)
    # prompt = f"[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response. Output the improved prompt by surround it with [BEGIN] and [END] tags.\n\n Here is the prompt to improve:\n{prompt} [/INST]"

    num_attempts = 5
    for i in range(num_attempts):

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
        optimized_prompt = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()

        print("Raw Model Output:")
        print(optimized_prompt)
        print()

        if ("Improved Prompt:" not in optimized_prompt) or ("END" not in optimized_prompt):
            if i == num_attempts - 1:
                optimized_prompt = prompt
            else:
                continue

        try:
            # optimized_prompt = optimized_prompt.split("Optimized Prompt:")[1].strip().split("[END]")[0].strip().strip("\"")
            # optimized_prompt = optimized_prompt.strip().split("\n")[1:-1].strip().strip("\"")
            optimized_prompt = optimized_prompt.strip().split("Improved Prompt:")[1].strip().split("END")[0].strip().strip("\"")
            break
        except Exception as ex:
            print("Regenerate and try again...")


    return optimized_prompt

def generate_bpo_optimized_prompts(dataset: str, device, tokenizer, bpo_model):

    if dataset != "vicuna":
        with open(f"data/eval_datasets/{dataset}_eval.json", "r") as file:
            eval_prompts = json.load(file)
    else:
        eval_prompts = []
        with open(f"data/eval_datasets/{dataset}_eval.jsonl", "r") as file:
            for line in file:
                eval_prompts.append(json.loads(line))

    bpo_opt_prompts = []
    for i, data in enumerate(eval_prompts):

        if dataset == "dolly":
            prompt = data['instruction'] #+ "\n" + data['context']
            #context = data['context']
            context = ""
        elif dataset == "self_instruct":
            prompt = data['instruction'] #+ "\n" + data['context']
            # context = data['context']
            context = ""
        elif dataset == "vicuna":
            prompt = data['text']
            context = ""
        elif dataset == "bpo_test":
            prompt = data['prompt']
            context = ""
        else:
            print("Invalid dataset specified!")
            exit(1)

        # Remove leading whitespace
        original_prompt = prompt.strip()

        print(f"GENERATING OPTIMIZED PROMPT - {dataset.upper()} DATASET - PROMPT {i+1}")
        print("Original Prompt:")
        print(original_prompt)
        print()

        optimized_prompt = generate_optimized_prompt_bpo(original_prompt, context, device, tokenizer, bpo_model)

        # print(f"Generating optimized prompt for prompt {i}...")
        # num_attempts = 5
        # for i in range(num_attempts):
        #     try:
        #         optimized_prompt = generate_optimized_prompt_bpo(original_prompt, context, device, tokenizer, bpo_model)
        #         break
        #     except Exception as ex:
        #         print("Regenerate and try again...")

        print("Optimized Prompt:")
        print(optimized_prompt)
        print("========================================================")

        current_output = {}

        # Build output JSON
        current_output["original_prompt"] = original_prompt
        current_output["optimized_prompt"] = optimized_prompt

        bpo_opt_prompts.append(current_output)

        # if i == 20:
        #     break

    with open(f"data/evaluation/bpo_optimized_prompts_{dataset}.json", "w") as json_file:
        json.dump(bpo_opt_prompts, json_file, indent=4)

def generate_responses(dataset: str, model: str):
    """Generate the responses for the original and optimized versions of the same prompt from the evaluation dataset.
        The optimized prompt is generated using our Seq2Seq model.
    """

    with open(f"data/evaluation/bpo_optimized_prompts_{dataset}.json", "r") as file:
        bpo_optimized_prompts = json.load(file)

    optimized_responses = []
    for i, data in enumerate(bpo_optimized_prompts):

        original_prompt = data["original_prompt"]
        optimized_prompt = data["optimized_prompt"]

        print(f"GENERATING RESPONSES - {dataset.upper()} DATASET - PROMPT {i+1}")
        print("Original Prompt:")
        print(original_prompt)
        print()
        print("Optimized Prompt:")
        print(optimized_prompt)

        if model == "gpt_4o":
            original_response = generate_response_gpt4o(original_prompt)
            optimized_response = generate_response_gpt4o(optimized_prompt)
        elif model == "gpt_3.5_turbo":
            original_response = generate_response_gpt3_5_turbo(original_prompt)
            optimized_response = generate_response_gpt3_5_turbo(optimized_prompt)
        elif model == "claude_haiku":
            original_response = generate_response_claude_haiku(original_prompt)
            optimized_response = generate_response_claude_haiku(optimized_prompt)
        elif model == "claude2":
            original_response = generate_response_claude2(original_prompt)
            optimized_response = generate_response_claude2(optimized_prompt)
        elif model == "text_bison":
            original_response = generate_response_textbison(original_prompt)
            optimized_response = generate_response_textbison(optimized_prompt)
        else:
            print("Invalid LLM model specified!")
            exit(1)

        print("Original Response:")
        print(original_response[:70] + "...")
        print()
        print("Optimized Response:")
        print(optimized_response[:70] + "...")
        print("=========================================================================")

        current_output = {}

        # Build output JSON
        current_output["original_prompt"] = original_prompt
        current_output["optimized_prompt"] = optimized_prompt
        current_output["original_response"] = original_response
        current_output["optimized_response"] = optimized_response

        optimized_responses.append(current_output)

        # if i == 20:
        #     break

    with open(f"data/evaluation/{dataset}_{model}_opt_responses.json", "w") as json_file:
        json.dump(optimized_responses, json_file, indent=4)

def main():

    # Specify datasets
    dataset_options = ["bpo_test", "dolly", "vicuna", "self_instruct"]
    model_options = ["gpt_4o", "claude_haiku"]

    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=["opt", "eval"])
    parser.add_argument('-d', '--datasets', choices=dataset_options, nargs="*", default="dolly", dest="datasets")
    parser.add_argument('-m', '--models', choices=model_options, nargs="*", default="gpt_4o", dest="models")
    args = parser.parse_args()

    if args.mode == "opt":

        # Check for GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("GPU with cuda support detected!")
        else:
            device = torch.device('cpu')
            print("No GPU detected. Falling back to using CPU!")

        # Load pretrained BPO model
        model_checkpoint = "./infer/bpo_model/"
        llama_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
        bpo_model = AutoModelForCausalLM.from_pretrained(model_checkpoint).half().eval().to(device)
        text_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # Loop through all datasets and models
        for ds in args.datasets:
            generate_bpo_optimized_prompts(ds, device, text_tokenizer, bpo_model)

    elif args.mode == "eval":
        for ds in args.datasets:
            for model in args.models:
                generate_responses(ds, model)
    else:
        print("Invalid mode selected!")
        exit(1)

if __name__ == "__main__":
    main()
