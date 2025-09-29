from unsloth import FastLanguageModel
import torch
import wandb
import json
import pandas as pd
from calflops import calculate_flops

max_seq_length = 4096
dtype = None
load_in_4bit = True
batch_size = 1

model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B/outputsnewnewnew/checkpoint-48",
    model_name = "/your route",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print(model)
print(tokenizer)
FastLanguageModel.for_inference(model)

flops, macs, params = calculate_flops(model=model, input_shape=(batch_size,max_seq_length), transformer_tokenizer=tokenizer)
print('DB-LLM FLOPs: %s  MACs:%s   Params:%s \n' %(flops, macs, params))


