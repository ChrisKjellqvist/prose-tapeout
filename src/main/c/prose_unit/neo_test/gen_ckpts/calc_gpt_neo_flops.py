import torch
from transformers import GPTNeoForCausalLM, GPTNeoConfig
from calflops import calculate_flops_hf

# sample code:
#batch_size, max_seq_length = 1, 128
# model_name = "https://huggingface.co/THUDM/glm-4-9b-chat" # THUDM/glm-4-9b-chat
# flops, macs, params = calculate_flops_hf(model_name=model_name, input_shape=(batch_size, max_seq_length))
# print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))

# Load the GPT-Neo-125M model
model_name = "EleutherAI/gpt-neo-125M"
batch_size = 1
max_seq_length = 128

# Calculate FLOPs, MACs, and parameters
flops, macs, params = calculate_flops_hf(model_name=model_name, input_shape=(batch_size, max_seq_length))

# Print the results
print(f"{model_name} FLOPs: {flops}  MACs: {macs}  Params: {params}")

# Optional: Load the model and perform inference to verify output shape
model = GPTNeoForCausalLM.from_pretrained(model_name)
model.eval()

input_ids = torch.randint(0, model.config.vocab_size, (batch_size, max_seq_length))

with torch.no_grad():
    outputs = model(input_ids)

print(f"Output shape: {outputs.logits.shape}")