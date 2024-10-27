from datasets import load_dataset
import json
import tiktoken

from llm_prompts import *


def calculate_llm_cost():

    MODELS = ["gpt-4o", "gpt-4"]

    # Dict to store token count for each model
    model_tokens = {}

    for model in MODELS:
        model_tokens[model] = {}
        model_tokens[model]['input'] = 0
        model_tokens[model]['output'] = 0

    ds = load_dataset("THUDM/BPO")
    train_ds = ds['train']

    response_data = []
    total_tokens = 0

    for i in range(len(train_ds)):

        instruction = train_ds['prompt'][i]
        bad_res = train_ds['bad_res'][i]
        good_res = train_ds['good_res'][i]

        data = {
            "instruction": instruction,
            "good_res": good_res,
            "bad_res": bad_res
        }

        response_data.append(data)

        llm_prompt = get_optimize_prompt(instruction, bad_res, good_res)

        print("Prompt", i, end=' ')
        for model in MODELS:

            enc = tiktoken.encoding_for_model(model)

            num_input_tokens = len(enc.encode(llm_prompt))

            # Estimate the number of output tokens to be the same as the original instruction
            # Add an additional 100 tokens for the explanation
            num_output_tokens = len(enc.encode(instruction)) + 100

            model_tokens[model]['input'] += num_input_tokens
            model_tokens[model]['output'] += num_output_tokens

            print(f"{model} input tokens: {model_tokens[model]['input']} {model} output tokens: {model_tokens[model]['output']}", end=' ')
        print()
        
    # Output calculations
    for model in MODELS:
        print("==========================================================================")
        input_token_cost = 2.5 * (model_tokens[model]['input'] / 1000000)
        output_token_cost = 10 * ( model_tokens[model]['input'] / 1000000)

        print("Model:", model)
        print(f"Total Number of Input Tokens: {model_tokens[model]['input']}")
        print(f"Total Number of Output Tokens: {model_tokens[model]['output']}")
        print(f"Input token cost: ${input_token_cost}")
        print(f"Output token cost: ${output_token_cost}")



calculate_llm_cost()

# total_cost = 2.50 * (total_tokens/1000000)


# Output the instruction/response groups to a JSON file
# with open('data/unoptimized_prompts.json', 'w') as json_file:
#     json.dump(response_data, json_file, indent=4)


