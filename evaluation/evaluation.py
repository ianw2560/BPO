import os
import random
import json
import pandas as pd

from openai import OpenAI


class Evaluation():

    datasets = ["bpo_test", "dolly", "vicuna", "self_instruct"]
    models = ["gpt_4o"]

    def __init__(self):

        self.errors = 0

        # Initialize evaluation scores
        self.scores = {}
        for model in Evaluation.models:
            self.scores[model] = {}
            for ds in Evaluation.datasets:
                self.scores[model][ds] = {"ori":0, "bpo":0 ,"tie":0}


        self.system_prompt = open("evaluation/system_prompt.txt")
        self.system_prompt = self.system_prompt.read()

        self.prompt_template = open("evaluation/prompt_template.txt")
        self.prompt_template = self.prompt_template.read()

    def evaluate_response(self, dataset: str, base_llm: str, instruction: str, original_response: str, bpo_response: str):

        # Randomly shuffle original and BPO responses
        rand_order = random.choice([True, False])
        if rand_order:
            resp_A = (original_response, "ori")
            resp_B = (bpo_response, "bpo")
        else:
            resp_A = (bpo_response, "bpo")
            resp_B = (original_response, "ori")

        # Build user message
        user_message = self.prompt_template.replace("{question}", instruction)
        user_message = user_message.replace("{answer_a}", resp_A[0])
        user_message = user_message.replace("{answer_b}", resp_B[0])

        # Perform evaluation with GPT-4o
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        llm_response = response.choices[0].message.content

        # print("LLM Response:")
        # print(llm_response)

        if "[[A]]" in llm_response:
            method = resp_A[1]
            self.scores[base_llm][dataset][method] += 1
        elif "[[B]]" in llm_response:
            method = resp_B[1]
            self.scores[base_llm][dataset][method] += 1
        elif "[[C]]" in llm_response:
            method = "tie"
            self.scores[base_llm][dataset][method] += 1
        else:
            print("Unknown response!")
            self.errors += 1

    def evaluate(self, dataset: str, model: str):
        """
        This function reads in a dataset containing an original prompt, an optimized prompt, and the respective responses
        and outputs the preferred response scoring.
        """

        with open(f"evaluation/{dataset}_{model}_opt_responses.json", "r") as file:
            opt_responses = json.load(file)

        for i, resp in enumerate(opt_responses):

            instruction = resp["original_prompt"]
            original_response = resp["optimized_prompt"]
            bpo_response = resp["optimized_response"]

            print(f"Evaluating Response {i+1}") 
            self.evaluate_response(dataset, model, instruction, original_response, bpo_response)

    def print_scores(self):

        for model in self.models:
            print("Model:", model)
            print("===================")

            df = pd.DataFrame(columns=["Orig.", "Tie", "BPO"])

            for ds in self.datasets:
                original = self.scores[model][ds]["ori"]
                tie = self.scores[model][ds]["tie"]
                bpo = self.scores[model][ds]["bpo"]

                df.loc[ds] = [original, tie, bpo]

            print(df)

if __name__ == "__main__":

    e = Evaluation()
    e.evaluate("dolly", "gpt_4o")
    e.print_scores()
