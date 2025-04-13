# Load model directly
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from torch.profiler.profiler import *
from torch.profiler import *


if __name__ == "__main__":
    home = os.path.expanduser("~")
    tokenizer = LlamaTokenizer.from_pretrained(f"{home}/Code/llama")
    model = LlamaForCausalLM.from_pretrained(f"{home}/Code/llama/llama-2-7b/")
    inputSeq = ["My name is Chris. The names of my cats are "]
    # make this into a batch size of 1000
    inputSeq = inputSeq * 7
    input_ids = tokenizer(inputSeq, return_tensors="pt").input_ids
    print(model)

    with profile(activities=[ProfilerActivity.CPU, ProfilerAc], record_shapes=True, profile_memory=True,
                 with_flops=True, with_modules=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/llama'),
                 use_cuda=False) as prof:  #  if you want a tensorboard log
        with record_function("model_inference"):
            output = model.generate(input_ids, do_sample=True, max_length=100, pad_token_id=tokenizer.eos_token_id)
        # print out the profile
