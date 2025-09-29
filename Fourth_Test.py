from unsloth import FastLanguageModel
import torch
import wandb
import json
import pandas as pd
import argparse
import os

max_seq_length = 4096
dtype = None
load_in_4bit = True

#Select the saved model in your repository.
model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B/codes/outputsrepository/checkpoint-72",
    model_name = "your model repository",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print(model)
print(tokenizer)
FastLanguageModel.for_inference(model)


def extract_values(json_data, key):
    # use values to save
    values = []

    # check data type
    if isinstance(json_data, dict):
        for k, v in json_data.items():
            if k == key:
                values.append(v)
            # recursive
            values.extend(extract_values(v, key))
    elif isinstance(json_data, list):
        for item in json_data:
            values.extend(extract_values(item, key))

    return values


def main():
    parser = argparse.ArgumentParser(description="unify the datasets")
    

    parser.add_argument("--dataset", type=str, default='loandata',required=True, 
                        help="name (e.g., 'japanese', 'german, 'simulated','loandata')")    
    # read json data, select your test data
    args = parser.parse_args()
    if args.dataset == 'loandata':
        with open('./datasets/testloan.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'japanese':
        with open('./datsets/testjapanese.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'german':
        with open('./datasets/testgerman.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'simulated':
        with open('./datasets/testsimulate.json', 'r') as f:
            data = json.load(f)         
    test = extract_values(data, 'Question')

    print(type(test))
    print('testdata are', test)

    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and remove chain of thoughts to ensure briefness.
    ### Instruction:You are a financial expert with advanced knowledge in credit risk prediction. Please output the final risk classification of the above client. Output: 0/1 (0 represents normal, 1 represents risky).
    ### Question:{}
    ### Response:{}"""

    result=[]
    for i in range(len(test)):
        inputs = tokenizer([prompt_style.format(test[i], "")], return_tensors="pt").to("cuda")
        outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
       )
        response = tokenizer.batch_decode(outputs)
        print(response[0])
        result.append(response[0])

    print('results are', result)
    result=pd.DataFrame(result)
    result.to_csv('results1.csv')
    print("Writing completed!")

if __name__ == "__main__":
    main()    


