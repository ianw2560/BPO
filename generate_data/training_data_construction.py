import pandas as pd
import numpy as numpy
import json
import re
import os
from datasets import load_dataset

PATH = "data"

def get_response_pair(instruction, good_res, bad_res, context=""):
    return {"instruction": instruction, "context": context, "good_res": good_res, "bad_res": bad_res}

def generate_oasst1():

    os.makedirs(PATH, exist_ok=True)

    # Import data from HuggingFace
    splits = {'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}
    df = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + splits["train"])

    response_dataset = []

    # Get all initial prompts
    parents = df[df['parent_id'].isnull()]
    parents = parents[parents['lang'] == 'en']['message_id']

    for message_id in parents:

        message = df[df['message_id'] == message_id].iloc[0]

        # Get the instruction
        instruction = message['text']

        # Get all responses that are children of the current instruction
        responses = df[df['parent_id'] == message_id]

        rank_sorted_responses = responses.sort_values(by='rank', ascending=True)
        
        # The good response is the top rated and the bad response is the lowest rated
        good_res = rank_sorted_responses.iloc[0]['text']
        bad_res = rank_sorted_responses.iloc[-1]['text']
        
        # Build the instruction response group
        instruction_pair = get_response_pair(instruction, good_res, bad_res)

        response_dataset.append(instruction_pair)

    # Output the instruction/response groups to a JSON file
    with open(f'{PATH}/oasst1_prompt_pairs.json', 'w') as json_file:
        json.dump(response_dataset, json_file, indent=4)

def get_hh_rlhf_responses(prompt):

    # Create regex to get the instruction and first response
    pattern = r'Human:(.*?)Assistant:(.*?)(?=Human:|$)'
    match = re.search(pattern, prompt, re.DOTALL)

    instruction = match.group(1).strip()
    response = match.group(2).strip()

    return instruction, response

def generate_hh_rlhf():

    os.makedirs(PATH, exist_ok=True)

    # Import data from HuggingFace
    dataset = load_dataset("Anthropic/hh-rlhf")

    chosen_dataset = dataset["train"]["chosen"]
    rejected_dataset = dataset["train"]["rejected"]

    responses = []

    # Get good responses
    for i, data in enumerate(chosen_dataset):

        instruction, res = get_hh_rlhf_responses(chosen_dataset[i])

        # Append instruction and good response
        responses.append([instruction, res])

    # Get bad responses
    for i, data in enumerate(rejected_dataset):
        instruction, res = get_hh_rlhf_responses(rejected_dataset[i])

        if instruction != responses[i][0]:
            print(f"Error: Something happened. The instructions do not match:\n\"{[instruction]}\\n\"{[responses[i][0]]}\"")
            exit(1)
        
        # Append bad response
        responses[i].append(res)

    same_response_count = 0

    response_dataset = []
    for res in responses:

        if res[1] == res[2]:
            same_response_count += 1
        
        instruction_pair = get_response_pair(res[0], res[1], res[2])
        response_dataset.append(instruction_pair)

    # Output the instruction/response groups to a JSON file
    with open(f'{PATH}/hh_rlhf_prompt_pairs.json', 'w') as json_file:
        json.dump(response_dataset, json_file, indent=4)

    print("Responses that are the same:", same_response_count)
    print("Responses that are different:", len(responses) - same_response_count)

if __name__ == "__main__":
    
    generate_oasst1()
    generate_hh_rlhf()
