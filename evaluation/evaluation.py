import os
import random

from openai import OpenAI


errors = 0
scores = {
    "vicuna":        {"ori":0, "bpo":0 ,"tie":0},
    "self_instruct": {"ori":0, "bpo":0 ,"tie":0},
    "dolly":         {"ori":0, "bpo":0 ,"tie":0},
    "bpo_test":      {"ori":0, "bpo":0 ,"tie":0}, 
}

def evaluate_response(instruction: str, original_response: str, bpo_response: str, system_prompt: str, prompt_template: str, eval_dataset: str):

    model = "gpt-4o-mini"

    # Randomly shuffle original and BPO responses
    rand_order = random.choice([True, False])
    if rand_order:
        resp_A = (original_response, "ori")
        resp_B = (bpo_response, "bpo")
    else:
        resp_A = (bpo_response, "bpo")
        resp_B = (original_response, "ori")

    # Build user message
    user_message = prompt_template.replace("{question}", instruction)
    user_message = user_message.replace("{answer_a}", resp_A[0])
    user_message = user_message.replace("{answer_b}", resp_B[0])

    # Perform evaluation with GPT-4o
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    llm_response = response.choices[0].message.content

    print("LLM Response:")
    print(llm_response)

    if "[[A]]" in llm_response:
        method = resp_A[1]
        scores[eval_dataset][method] += 1
    elif "[[B]]" in llm_response:
        method = resp_B[1]
        scores[eval_dataset][method] += 1
    elif "[[C]]" in llm_response:
        method = "tie"
        scores[eval_dataset][method] += 1
    else:
        print("Unknown response!")
        errors += 1

def evaluate(dataset):
    """
    This function reads in a dataset containing an original prompt, an optimized prompt, and the respective responses
    and outputs the preferred response scoring.
    """

    system_prompt = open("evaluation/system_prompt.txt")
    system_prompt = system_prompt.read()

    prompt_template = open("evaluation/prompt_template.txt")
    prompt_template = prompt_template.read()



    evaluate_responses(instruction, original_response, bpo_response, system_prompt, prompt_template, "dolly")




def main():

    instruction = "What is Urho3D engine?"
    original_response = "Urho3D is a MIT licensed open-source lightweight, cross-platform 2D and 3D game engine implemented in C++. \n\nThe name \"Urho\" means \"hero\" in Finnish."
    bpo_response = "Urho3D is a free cross-platform, lightweight 3d and 2d game engine released under the MIT license. Due to drama between the lead and an active developer the project has been made Russian only. The last English language version was released on github on November 21, 2022 with the previous project leader starting a new project called Turso3D."
    


    datasets = ["bpo_test", "dolly", "vicuna", "self_instruct"]


    evaluate("dolly")

    print(scores)

if __name__ == "__main__":
    main()
