#!/bin/python3

from datasets import load_dataset
import json
import tiktoken
import os

from openai import OpenAI


def optimize_prompts():

    f = open("generate_data/prompts/gpt4_generate_prompt_no_ctx.txt")
    GPT_PROMPT_NO_CONTEXT = f.read()

    # Dict to store token count for each model
    model_tokens = {}

    ds = load_dataset("THUDM/BPO")
    train_ds = ds['train']

    optimized_prompts_training_data = []
    raw_llm_output = []

    num_prompts = 10

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
    os.makedirs("data/raw_llm_output", exist_ok=True)
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
    optimize_prompts()
