import pandas as pd
import numpy as numpy
import json


# Import data from HuggingFace
splits = {'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'}
df = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + splits["train"])

#print(df.iloc[1])
#print(df.columns)
# print(df.loc[df['parent_id'] == None])


response_dataset = []

# Get all initial prompts
parents = df[df['parent_id'].isnull()]
parents = parents[parents['lang'] == 'en']['message_id']
print(parents)

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
    instruction_pair = {"instruction": instruction, "context": "", "good_res": good_res, "bad_res": bad_res}

    response_dataset.append(instruction_pair)

# Output the instruction/response groups to a JSON file
with open('data/prompt_pairs.json', 'w') as json_file:
    json.dump(response_dataset, json_file, indent=4)

