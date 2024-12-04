import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#Model checkpoint

def promptOptimize(device, inputText, template, tokenizer, model):
    prompt = template.format(inputText)

    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
    resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
    return resp

def aggPromptOptimize(device, inputText, template, tokenizer, model):
    texts = [inputText] * 5 
    responses = []
    for text in texts:
        seed = torch.seed()
        torch.manual_seed(seed)
        prompt = template.format(text)
        min_length = len(tokenizer(prompt)['input_ids']) + len(tokenizer(text)['input_ids']) + 5
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        bad_words_ids = [tokenizer(bad_word, add_special_tokens=False).input_ids for bad_word in ["[PROTECT]", "\n\n[PROTECT]", "[KEEP", "[INSTRUCTION]"]]
        # eos and \n
        eos_token_ids = [tokenizer.eos_token_id, 13]
        output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.9, bad_words_ids=bad_words_ids, num_beams=1, eos_token_id=eos_token_ids, min_length=min_length)
        resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].split('[KE')[0].split('[INS')[0].split('[PRO')[0].strip()
        responses.append(resp)
    return None


def main():
    ##############################################
    # SELECT DEVICE: GPU OR CPU
    ##############################################
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU with cuda support detected!")
    else:
        device = torch.device('cpu')
        print("No GPU detected. Falling back to using CPU!")

    prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"

    ############################
    # Load pretrained BPO model
    ############################
    modelCheckpoint = "./infer/bpo_model/"
    llamaCheckpoint = "meta-llama/Llama-2-7b-chat-hf"
    bpo_model = AutoModelForCausalLM.from_pretrained(modelCheckpoint,
                                                # quantization_config=bnb_config,
                                                # load_in_4bit=True,
                                                # device_map="auto"
                                                ).half().eval().to(device)
    

    textTokenizer = AutoTokenizer.from_pretrained(modelCheckpoint)

    text = 'Tell me about Harry Potter?'

    # Stable optimization, this will sometimes maintain the original prompt
    response = promptOptimize(device, text, prompt_template, textTokenizer, bpo_model)
    print(" PRINTING BPO SIMPLE RESPONSE::::: \n\n\n")
    print(response)

    # Agressive optimization, this will refine the original prompt with a higher possibility
    # but there may be inappropriate changes
    aggResp = aggPromptOptimize(device, text, prompt_template, textTokenizer, bpo_model)
    print(" PRINTING BPO AGG RESPONSE::::: \n\n\n")
    print(aggResp)

    del(bpo_model)

    llamaModel = AutoModelForCausalLM.from_pretrained(llamaCheckpoint).half().eval().to(device)
    llamaResponse = promptOptimize(device, text, prompt_template, textTokenizer, llamaModel)
    print(" PRINTING LLAMA 2 SIMPLE RESPONSE:::: \n\n\n")
    print(llamaResponse)

    

if __name__ == '__main__':
    main()
