from datasets import load_dataset
import json
import tiktoken

MODELS = ["gpt-4o", "gpt-4"]

# Dict to store token count for each model
model_tokens = {}

for model in MODELS:
    model_tokens[model] = 0

print(model_tokens)

ds = load_dataset("THUDM/BPO")

response_data = []

train_ds = ds['train']

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

    llm_prompt = f"""instruction: "{instruction}"

bad response:
"{bad_res}"

good response:
"{good_res}"

Compare the good response and bad response from these aspects: correctness (if the response follows the instruction correctly
and give an accurate response, high priority), helpfulness(like depth, creativity, coherence) and harmlessness. Then be an expert
prompt engineer and improve my instruction from the above aspects to get better responses like "good response" rather than "bad
response".

Pay attention to:
1.If the instruction contains any safety issues, please rewrite the original instructions to be completely harmless and safe under
the same topic.
2.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.
3.Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction,
unless in form of examples.
4.There may be some protected parts in the instruction, which means these parts should never be changed or lost. Please carefully
protect these parts.
5.You should never generate a response to the original instruction!
6.Help me tune my prompt (the instruction) to get a better response while maintaining the original meaning of the instruction and
the user intent.

Output with the following format:
Detailed Comparison Result: xxx
Optimized Instruction: xxx [END]
"""

    
    print("Prompt", i, end=' ')
    for model in MODELS:

        enc = tiktoken.encoding_for_model(model)


        num_tokens = len(enc.encode(llm_prompt))

        model_tokens[model] += num_tokens

        print(f"Current count for {model}: {model_tokens[model]}", end=' ')
    print()
    


for model in MODELS:
    print(f"Total Number of Tokens for {model}: {model_tokens[model]}")

# total_cost = 2.50 * (total_tokens/1000000)


# Output the instruction/response groups to a JSON file
# with open('data/unoptimized_prompts.json', 'w') as json_file:
#     json.dump(response_data, json_file, indent=4)


