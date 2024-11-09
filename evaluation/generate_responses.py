import json

from openai import OpenAI

def generate_response_gpt3_5_turbo(prompt: str):
    pass

def generate_response_gpt4o(prompt: str):

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

    return llm_response

def generate_response_claude_instant(prompt: str):
    pass

def generate_response_claude2(prompt: str):
    pass

def generate_response_textbison(prompt: str):
    pass

def generate_optimized_prompt_seq2seq(prompt: str):
    """Calls our Seq2Seq model and returns an optimized version of the input prompt."""

    return prompt + "THIS IS OPTIMIZED FROM SEQ2SEQ"


def generate_responses(dataset: str, model: str):
    """Generate the responses for the original and optimized versions of the same prompt from the evaluation dataset.
        The optimized prompt is generated using our Seq2Seq model.
    """

    with open(f"data/eval_datasets/{dataset}_eval.json", "r") as file:
        eval_prompts = json.load(file)

    optimized_responses = []
    for i, data in enumerate(eval_prompts):

        if dataset == "dolly":
            prompt = data['instruction'] + "\n" + data['context']
        elif dataset == "self_instruct":
            prompt = data['instruction'] + "\n" + data['context']
        elif dataset == "vicuna":
            pass
        elif dataset == "bpo_test":
            prompt = data['prompt']
        else:
            print("Invalid dataset specified!")
            exit(1)

        # Remove leading whitespace
        original_prompt = prompt.strip()

        # print("Original Prompt:")
        # print(original_prompt)

        print(f"Generating optimized prompt for prompt {i}...")
        optimized_prompt = generate_optimized_prompt_seq2seq(original_prompt)

        # print("Optimized Prompt:")
        # print(optimized_prompt)

        print(f"Generating original and optimized responses for prompt {i}...")
        if model == "gpt_4o":
            original_response = generate_response_gpt4o(original_prompt)
            optimized_response = generate_response_gpt4o(optimized_prompt)
        elif model == "gpt_3.5_turbo":
            original_response = generate_response_gpt3_5_turbo(original_prompt)
            optimized_response = generate_response_gpt3_5_turbo(optimized_prompt)
        elif model == "claude_instant":
            original_response = generate_response_claude_instant(original_prompt)
            optimized_response = generate_response_claude_instant(optimized_prompt)
        elif model == "claude2":
            original_response = generate_response_claude2(original_prompt)
            optimized_response = generate_response_claude2(optimized_prompt)
        elif model == "text_bison":
            original_response = generate_response_textbison(original_prompt)
            optimized_response = generate_response_textbison(optimized_prompt)
        else:
            print("Invalid LLM model specified!")
            exit(1)

        # print("Original Response:")
        # print(original_response)
        # print()
        # print("Optimized Response:")
        # print(optimized_response)
        # print()

        current_output = {}

        # Build output JSON
        current_output["original_prompt"] = original_prompt
        current_output["optimized_prompt"] = optimized_prompt
        current_output["original_response"] = original_response
        current_output["optimized_response"] = optimized_response

        optimized_responses.append(current_output)

        if i == 10:
            break

    with open(f"evaluation/{dataset}_{model}_opt_responses.json", "w") as json_file:
        json.dump(optimized_responses, json_file, indent=4)

def main():

    datasets = ["bpo_test", "dolly", "vicuna", "self_instruct"]
    models = ["gpt-4o"]

    # TODO: Add for loop to loop through all datasets and models

    generate_responses("dolly", "gpt_4o")

if __name__ == "__main__":
    main()
