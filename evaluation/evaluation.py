import argparse
import os
import random
import json
import pandas as pd

from openai import OpenAI


class Evaluation():

    dataset_options = ["bpo_test", "dolly", "vicuna", "self_instruct"]
    model_options = ["gpt_4o", "gpt_3.5_turbo", "claude3_haiku", "claude3.5_haiku", "gemini"]

    def __init__(self):

        self.errors = 0

        # Initialize evaluation scores
        self.scores = {}
        for model in Evaluation.model_options:
            self.scores[model] = {}
            for ds in Evaluation.dataset_options:
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

        # print("USER MESSAGE:")
        # print(user_message)

        # Perform evaluation with GPT-4o
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
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
            print("LLM Response: [[A]]")
            method = resp_A[1]
            self.scores[base_llm][dataset][method] += 1
        elif "[[B]]" in llm_response:
            print("LLM Response: [[B]]")
            method = resp_B[1]
            self.scores[base_llm][dataset][method] += 1
        elif "[[C]]" in llm_response:
            print("LLM Response: [[C]]")
            method = "tie"
            self.scores[base_llm][dataset][method] += 1
        else:
            print("Unknown response!")
            self.errors += 1

        # Print current results
        ori = self.scores[model][dataset]["ori"]
        tie = self.scores[model][dataset]["tie"]
        bpo = self.scores[model][dataset]["bpo"]
        print(f"Current Score: Original = {ori}, Tie = {tie}, BPO = {bpo}")

    def evaluate(self, dataset: str, model: str):
        """
        This function reads in a dataset containing an original prompt, an optimized prompt, and the respective responses
        and outputs the preferred response scoring.
        """

        with open(f"data/evaluation/{dataset}_{model}_opt_responses.json", "r") as file:
            opt_responses = json.load(file)

        for i, resp in enumerate(opt_responses):

            instruction = resp["original_prompt"]
            original_response = resp["original_response"]
            bpo_response = resp["optimized_response"]

            print(f"EVALUATING {model.upper()} RESPONSES FOR {dataset.upper()} DATASET - PROMPT {i+1}")
            self.evaluate_response(dataset, model, instruction, original_response, bpo_response)

    def print_scores(self):

        for model in self.model_options:
            print("Model:", model)
            print("===================")

            df = pd.DataFrame(columns=["Orig.", "Tie", "BPO", "Orig.(%)", "Tie(%)", "BPO(%)"])

            for ds in self.dataset_options:
                original = self.scores[model][ds]["ori"]
                tie = self.scores[model][ds]["tie"]
                bpo = self.scores[model][ds]["bpo"]

                # Calculate percentages
                total = (original + tie + bpo)

                if total > 0:
                    original_per = float(original / total) * 100
                    tie_per = float(tie / total) * 100
                    bpo_per = float(bpo / total) * 100
                else:
                    original_per = "N/A"
                    tie_per = "N/A"
                    bpo_per = "N/A"

                df.loc[ds] = [original, tie, bpo, original_per, tie_per, bpo_per]

            print(df)

if __name__ == "__main__":

    e = Evaluation()

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--datasets', choices=e.dataset_options, nargs="*", default="dolly", dest="datasets")
    parser.add_argument('-m', '--models', choices=e.model_options, nargs="*", default="gpt_4o", dest="models")
    args = parser.parse_args()

    # Loop through all specified datasets and models
    for ds in args.datasets:
        for model in args.models:
            e.evaluate(ds, model)

    print(e.scores)
    e.print_scores()
