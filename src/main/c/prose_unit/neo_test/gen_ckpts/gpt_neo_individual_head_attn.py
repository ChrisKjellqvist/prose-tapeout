import torch
import torch.nn as nn
from transformers import GPTNeoForCausalLM
import os

def attn_layer(input_tensor, key_w, query_w, value_w, out_w, out_b):
    pass

if __name__ == "__main__":
    # read everything from the model_ckpts/attn.*.pt files
    state_dict = {}
    for f in os.listdir("model_ckpts"):
        if f.endswith(".pt") and f.startswith("attn."):
            k = f.replace("attn.", "").replace(".pt", "")
            state_dict[k] = torch.load(f"model_ckpts/{f}")
    for k, v in state_dict.items():
        print(k, v.shape)
    
    # slice the first head weight
    for h in range(12):
        key_w = state_dict['k_proj.weight']
        query_w = state_dict['q_proj.weight']
        value_w = state_dict['v_proj.weight']
        out_w = state_dict['out_proj.weight']
        out_b = state_dict['out_proj.bias']
    
        attn_layer(state_dict['input'], state_dict['k_proj.weight'], state_dict['q_proj.weight'], state_dict['v_proj.weight'], state_dict['out_proj.weight'], state_dict['out_proj.bias'])
    