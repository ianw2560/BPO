import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig, Seq2SeqTrainer
from preprocess import OptimizedDataset, TokenizedDataset

from trl import SFTTrainer
from peft import LoraConfig, PeftModel


#QLoRA parameter

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1


def tokenizeInputs(tokenizer, dataset):    

    # Concatenate each original prompt with its corresponding optimized prompt.
    concatenated_prompts = [f"{org} {tokenizer.eos_token} {opt}" for org, opt in zip(dataset.org, dataset.opt)]

    # Tokenize and pad the concatenated sequences to ensure consistent length.
    tokenized_data = tokenizer(concatenated_prompts, padding=True, truncation=True, return_tensors="pt")

    # Shift the labels for causal language modeling by masking input prompt tokens (original) from labels.
    labels = tokenized_data["input_ids"].clone()
    prompt_lengths = [len(tokenizer(org)["input_ids"]) for org in dataset.org]

    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100  # Mask original tokens in the labels with -100 to ignore them in loss calculation.

        # Mask the optimized prompt in `input_ids` to prevent it from being "seen" by the model during training.
        tokenized_data["input_ids"][i, prompt_len:] = 2 #<EOS>

    tokenized_data["labels"] = labels

    return tokenized_data

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
    train_file = "/home/cap6614.student1/Rafeeq/BPO_/train.json"
    val_file = "/home/cap6614.student1/Rafeeq/BPO_/val.json"
    train_dataset = OptimizedDataset(train_file)
    val_dataset = OptimizedDataset(val_file)

    #Bitsand bytes
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
    )

    ######################################################
    # Load pretrained Llama2 model and Training arguments
    ######################################################
    modelCheckpoint = "meta-llama/Llama-2-7b-chat-hf"
    output_path = "/home/cap6614.student1/Rafeeq/test_trainer"
    model = AutoModelForCausalLM.from_pretrained(modelCheckpoint,
                                                quantization_config=bnb_config,
                                                # load_in_4bit=True,
                                                device_map="auto")
    # model = LlamaForCausalLM.from_pretrained("path_to_llama2")

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ################################################################
    # Pre-process tokenizer to set the data for the Llama-2-7b model
    ################################################################
    textTokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)
    textTokenizer.pad_token = textTokenizer.eos_token                       # SET PADDING AS EOS </s>
    
    org_tokenized_data = tokenizeInputs(textTokenizer, train_dataset)
    tokenized_train_dataset = TokenizedDataset(org_tokenized_data)

    val_tokenized_data = tokenizeInputs(textTokenizer, val_dataset)
    tokenized_val_dataset = TokenizedDataset(val_tokenized_data)

    # print(type(org_tokenized_data))
    # print(type(tokenized_train_dataset))
    # breakpoint()

    data_collator = DataCollatorForSeq2Seq(textTokenizer, model=model)
    # print(data_collator)
    # breakpoint()

    training_args = TrainingArguments(output_dir=output_path,
                                    report_to="none",
                                    num_train_epochs=3,
                                    logging_dir="./logs",
                                    logging_steps=100,
                                    learning_rate=2e-5,
                                    save_total_limit=3,
                                    save_strategy="epoch",
                                    evaluation_strategy="epoch",
                                    gradient_checkpointing=True,
                                    load_best_model_at_end=True,
                                    gradient_accumulation_steps=1,
                                    per_device_train_batch_size=10,
                                    per_device_eval_batch_size=10,
                                    metric_for_best_model="loss",
                                    optim="paged_adamw_8bit",
                                    greater_is_better=False,
                                    # deepspeed="/content/Zero2_config.json",
                                    # torch_empty_cache_steps=100,
                                    # bf16=True,
                                    # torch_compile=True,
                                    # fp16=True,
                                    )

    ################################################################
    # Defining trainer with Huggingface Trainer Class
    ################################################################

    trainer = SFTTrainer(
      model=model,
      args=training_args,
      tokenizer=textTokenizer,
      peft_config=peft_config,
      dataset_text_field="text",
      data_collator=data_collator,
      train_dataset=tokenized_train_dataset,
      eval_dataset=tokenized_val_dataset,
      max_seq_length=None,
      packing=False,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_train_dataset,
    #     data_collator=data_collator,
    #     #eval_dataset=small_eval_dataset,               TO:DO
    #     #compute_metrics=compute_metrics,               TO:DO
    # )

    # # finetune the model
    trainer.train()


if __name__ == '__main__':
    main()
