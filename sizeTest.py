import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################
# SELECT DEVICE: GPU OR CPU
#############################################
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device:         {device}")
print("")

modelCheckpoint = ".infer/bpo_model/"

BPOmodel = AutoModelForCausalLM.from_pretrained(modelCheckpoint).half().eval().to(device)
print("parameters for model is::", BPOmodel.num_parameters())


def get_layer_sizes(model):
    layer_sizes = {}
    total_size = 0

    for name, param in model.named_parameters():
        layer_size = param.numel() * param.element_size()  # numel() returns the number of elements, element_size() returns the size in bytes of each element
        total_size += layer_size
        layer_sizes[name] = (param.numel(), layer_size, param.dtype)

    return layer_sizes, total_size

layer_sizes, total_size = get_layer_sizes(BPOmodel)

for name, size in layer_sizes.items():
    print(f"Layer: {name}; Number of parameters: {size[0]:,} ({size[2]}); Size: {size[1] / (1024 ** 2):.2f} MiB")

print(f"Total Model Size: {total_size / (1024 ** 2):.2f} MiB")
