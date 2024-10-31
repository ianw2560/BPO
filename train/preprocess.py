import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
"""Creating dataset by loading the json data containing the training information. Currently it is a list of dictionary entries with the annotation keys being: good_res, bad_res    """

class OptimizedDataset(Dataset):
    def __init__(self, prompt_json_file):
        self.opt = []
        self.org = []

        with open(prompt_json_file, 'r') as file:
            data = json.load(file)
        self.prompts = data

        for entry in tqdm(self.prompts):
            self.opt.append(entry['good_res'])
            self.org.append(entry['bad_res'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.opt[idx], self.org[idx]

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
        
    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.data.items()}

# # """Quick Test Dataset"""
# if __name__ == '__main__':
#     prompt_file = "/home/cap6614.student1/Rafeeq/BPO_/prompt_pairs.json"
#     train_dataset = OptimizedDataset(prompt_file)
# # #     print(train_dataset[0].keys())
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# #     for org, opt in train_dataloader:
# #         print(opt)
#     it = iter(train_dataloader)
#     while it:
#         print(next(it[0]))
    #for _, bad in train_dataloader:
        #print(bad)
