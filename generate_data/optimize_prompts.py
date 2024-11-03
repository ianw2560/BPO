#!/bin/python3

from datasets import load_dataset
import json
import tiktoken

from openai import OpenAI


def calculate_llm_cost():

    MODELS = ["gpt-4o", "gpt-4"]

    f = open("prompts/gpt4_generate_prompt_no_ctx.txt")
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

def optimize_prompts():

    f = open("./gpt4_generate_prompt_no_ctx.txt")
    GPT_PROMPT_NO_CONTEXT = f.read()

    # Dict to store token count for each model
    model_tokens = {}

    ds = load_dataset("THUDM/BPO")
    train_ds = ds['train']

    optimized_prompts_training_data = []
    raw_llm_output = []

    # num_prompts = len(train_ds)
    num_prompts = 5

    print(f"Begin optimizing {num_prompts} prompts...")
    for i in range(num_prompts):

        instruction = train_ds['prompt'][i]
        bad_res = train_ds['bad_res'][i]
        good_res = train_ds['good_res'][i]

        print(f"Optimizing Prompt #{i+1}: \"{instruction}\"")

        # Replace prompt variables with actual data
        llm_prompt = GPT_PROMPT_NO_CONTEXT.replace("{instruction}", instruction)
        llm_prompt = llm_prompt.replace("{bad_res}", bad_res)
        llm_prompt = llm_prompt.replace("{good_res}", good_res)

        # Get raw response and optimized prompt
        llm_response, optimized_prompt = generate_optimized_prompt(instruction, llm_prompt)

        output_data = {
            "prompt": instruction,
            "optimized_prompt": optimized_prompt,
            "good_res": good_res,
            "bad_res": bad_res
        }

        raw_output = {
            "instruction": instruction,
            "llm_response": llm_response
        }

        optimized_prompts_training_data.append(output_data)
        raw_llm_output.append(raw_output)

    # Output the instruction/response groups to a JSON file
    with open('data/gpt4o_optimized_prompts.json', 'w') as json_file:
        json.dump(optimized_prompts_training_data, json_file, indent=4)

    # Output the entire raw LLM response
    with open("data/raw_llm_output/optimized_prompts.json", "w") as json_file:
        json.dump(raw_llm_output, json_file, indent=4)

def generate_optimized_prompt(instruction: str, prompt: str):

    model = "gpt-4o-mini"

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

    llm_response = response.choices[0].message.content

    # Extract optimized prompt from LLM response
    optimized_prompt = llm_response.split("Optimized Instruction:")[1]
    optimized_prompt = optimized_prompt.split("[END]")[0]
    optimized_prompt = optimized_prompt.strip().strip("\"")

    return llm_response, optimized_prompt

if __name__=="__main__":
    main()
# calculate_llm_cost()
optimize_prompts()

# Output the instruction/response groups to a JSON file
# with open('data/unoptimized_prompts.json', 'w') as json_file:
#     json.dump(response_data, json_file, indent=4)


