from datasets import load_dataset
import json
import tiktoken
import os

from openai import OpenAI

def avg_output_tokens(model: str) -> int:

    # Read in the test set of optimized prompts to get an idea of how large the outputs are
    with open('data/raw_llm_output/optimized_prompts.json', 'r') as file:
        llm_responses = json.load(file)

    enc = tiktoken.encoding_for_model(model)

    output_tokens = []
    for response in llm_responses:
        token_num = enc.encode(response['llm_response'])
        print(token_num)
        output_tokens.append(token_num)

    print(output_tokens)


    avg_token_num = sum(output_tokens) / len(output_tokens)

    print(avg_token_num)

    return avg_token_num


def calculate_llm_costs():

    MODELS = ["gpt-4o", "gpt-4"]

    f = open("generate_data/prompts/gpt4_generate_prompt_no_ctx.txt")
    GPT_PROMPT_NO_CONTEXT = f.read()

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

        # Replace prompt variables with actual data
        llm_prompt = GPT_PROMPT_NO_CONTEXT.replace("{instruction}", instruction)
        llm_prompt = llm_prompt.replace("{bad_res}", bad_res)
        llm_prompt = llm_prompt.replace("{good_res}", good_res)

        print("Prompt", i, end=' ')
        for model in MODELS:
            
            # Encode the input using the tokenizer for the given LLM
            enc = tiktoken.encoding_for_model(model)

            num_input_tokens = len(enc.encode(llm_prompt))

            # Estimate the number of output tokens by taking the average number of output tokens
            # in a sample of generated responses
            num_output_tokens = avg_output_tokens(model)

            model_tokens[model]['input'] += num_input_tokens
            model_tokens[model]['output'] += avg_output_tokens

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

if __name__=="__main__":
    
    calculate_llm_costs()
