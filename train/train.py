import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainer
from preprocess import OptimizedDataset, TokenizedDataset

def tokenizeInputs(tokenizer, dataset):    
    # opt_tokens = tokenizer(dataset.opt, padding=True, truncation=True, return_tensors="pt")#) #, padding=True, truncation=True)
    # org_tokens = tokenizer(dataset.org, padding=True, truncation=True, return_tensors="pt")#, return_tensors="pt")
    # org_tokens['labels'] = opt_tokens['input_ids']
    # breakpoint() 

    # Concatenate each original prompt with its corresponding optimized prompt.
    concatenated_prompts = [f"{org} {tokenizer.eos_token} {opt}" for org, opt in zip(dataset.org, dataset.opt)]

    # breakpoint() 

    # Tokenize and pad the concatenated sequences to ensure consistent length.
    tokenized_data = tokenizer(concatenated_prompts, padding=True, truncation=True, return_tensors="pt")

    # Shift the labels for causal language modeling by masking input prompt tokens (original) from labels.
    labels = tokenized_data["input_ids"].clone()
    prompt_lengths = [len(tokenizer(org)["input_ids"]) for org in dataset.org]
    # breakpoint()

    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100  # Mask original tokens in the labels with -100 to ignore them in loss calculation.
        # breakpoint()

    tokenized_data["labels"] = labels

    return tokenized_data #org_tokens

def main():
    ##############################################
    # SELECT DEVICE: GPU OR CPU
    ##############################################
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Device:         {device}")
    print("")

    ####################################
    # LOAD THAT TRAIN AND TEST DATASET
    ####################################
    prompt_file = "/home/cap6614.student1/Rafeeq/BPO_/prompt_pairs.json"
    train_dataset = OptimizedDataset(prompt_file)

    ######################################################
    # Load pretrained Llama2 model and Training arguments
    ######################################################
    modelCheckpoint = "meta-llama/Llama-2-7b-chat-hf"
    output_path = "/home/cap6614.student1/Rafeeq/test_trainer"
    model = AutoModelForCausalLM.from_pretrained(modelCheckpoint)
    # model = LlamaForCausalLM.from_pretrained("path_to_llama2")

    ################################################################
    # Pre-process tokenizer to set the data for the Llama-2-7b model
    ################################################################
    textTokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)
    textTokenizer.pad_token = textTokenizer.eos_token                       # SET PADDING AS EOS </s>
    org_tokenized_data = tokenizeInputs(textTokenizer, train_dataset)
    tokenized_train_dataset = TokenizedDataset(org_tokenized_data)
    # print(type(org_tokenized_data))
    # print(type(tokenized_train_dataset))
    # breakpoint()

    data_collator = DataCollatorForSeq2Seq(textTokenizer, model=model)
    # print(data_collator)
    # breakpoint()

    training_args = TrainingArguments(output_dir=output_path, 
                                    per_device_train_batch_size=1,
                                    per_device_eval_batch_size=1,
                                    num_train_epochs=3,
                                    logging_dir="./logs",
                                    logging_steps=100,
                                    evaluation_strategy="epoch",
                                    save_strategy="epoch",
                                    learning_rate=2e-5,)

    ################################################################
    # Defining trainer with Huggingface Trainer Class
    ################################################################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        #eval_dataset=small_eval_dataset,               TO:DO
        #compute_metrics=compute_metrics,               TO:DO
    )

    # # finetune the model
    trainer.train()


if __name__ == '__main__':
    main()
