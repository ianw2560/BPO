import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from preprocess import OptimizedDataset

def tokenizeInputs(tokenizer, dataset):    
    opt_tokens = tokenizer(dataset.opt, padding=True)#, return_tensors="pt") #, padding=True, truncation=True)
    org_tokens = tokenizer(dataset.org, padding=True)#, return_tensors="pt")
    org_tokens['labels'] = opt_tokens['input_ids']
    # breakpoint() 
    
    return org_tokens

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

    ##############################################
    # LOAD THAT TRAIN AND TEST DATASET
    ##############################################
    prompt_file = "/home/cap6614.student1/Rafeeq/BPO_/prompt_pairs.json"
    train_dataset = OptimizedDataset(prompt_file)

    ################################################################
    # Pre-process tokenizer to set the data for the Llama-2-7b model
    ################################################################
    modelCheckpoint = "meta-llama/Llama-2-7b-chat-hf"
    textTokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)
    textTokenizer.pad_token = textTokenizer.eos_token                       # SET PADDING AS EOS </s>
    org_tokenized_data = tokenizeInputs(textTokenizer, train_dataset)
    print(org_tokenized_data.keys())
    
    ######################################################
    # Load pretrained Llama2 model and Training arguments
    ######################################################
    output_path = "/home/cap6614.student1/Rafeeq/test_trainer"
    model = AutoModelForCausalLM.from_pretrained(modelCheckpoint)
    # model = LlamaForCausalLM.from_pretrained("path_to_llama2")

    training_args = TrainingArguments(output_dir=output_path, 
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    num_train_epochs=3,
                                    logging_dir="./logs",
                                    logging_steps=100,
                                    evaluation_strategy="epoch",
                                    save_strategy="epoch",
                                    learning_rate=5e-5,)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=org_tokenized_data,
        #eval_dataset=small_eval_dataset,               TO:DO
        #compute_metrics=compute_metrics,               TO:DO
    )

    # finetune the model
    # trainer.train()

    #decode
    #decoder = tokenizer.decode(encoded_input["input_ids"])
    #print(decoder)


if __name__ == '__main__':
    main()
