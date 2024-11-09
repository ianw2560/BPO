

def generate_responses_gpt3_5_turbo():
    pass

def generate_responses_gpt4o(prompt: str):

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

def generate_responses_claude_instant():
    pass

def generate_responses_claude2():
    pass

def generate_responses_textbison():
    pass

def generate_optimized_prompt_seq2seq(prompt: str):
    """Calls our Seq2Seq model and returns an optimized version of the input prompt."""

    return prompt + "THIS IS OPTIMIZED FROM SEQ2SEQ"


def optimize_prompts_seq2seq(dataset: str):
    """Generate the optimized version of a prompt"""

    with open(f"data/eval_datasets/{dataset}_eval.json", "r") as file:
        eval_prompts = json.load(file)

    seq2seq_output = []

    for data in eval_prompts:

        if dataset is "dolly":
            prompt = data['instruction'] + "\n" + data['context']
        elif: dataset is "self_instruct":
            prompt = data['instruction'] + "\n" + data['context']
        elif: dataset is "vicuna":
            pass
        elif: dataset is "bpo_test":
            prompt = data['prompt']
        else:
            print("Invalid dataset!")
            exit(1)

        # Remove leading whitespace
        prompt = prompt.strip()

        opt_prompt = generate_optimized_prompt_seq2seq(prompt)

        current_output = {}







def main():

    datasets = ["bpo_test", "dolly", "vicuna", "self_instruct"]





    generate_responses("dolly")

    print(scores)

if __name__ == "__main__":
    main()
