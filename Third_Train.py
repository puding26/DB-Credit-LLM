from unsloth import FastLanguageModel
import torch
import wandb
import numpy as np
from datasets import load_dataset
import argparse 
import os

def main():
    parser = argparse.ArgumentParser(description="unify the datasets")
    

    parser.add_argument("--dataset", type=str, default='loandata',required=True, 
                        help="name (e.g., 'japanese', 'german, 'simulated','loandata')")
    args = parser.parse_args()
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    #Download the 'deepseek-r1-distill-qwen-7b' model into your local repository. If you prefer online mode, run the Hugging Face version.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    print(model)
    print(tokenizer)
    FastLanguageModel.for_inference(model)

    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and remove chain of thoughts to ensure briefness.
    ### Instruction:You are a financial expert with advanced knowledge in credit risk prediction. Please output the final risk classification of the above client. Output: 0/1 (0 represents normal, 1 represents risky).
    ### Question:{}
    ### Response:{}"""

    question = 'There is a client with demographical score 0, and financial behavior score 0.'

    inputs1 = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

    outputs1 = model.generate(input_ids=inputs1.input_ids, max_new_tokens=1200, use_cache=True,)
    response1 = tokenizer.batch_decode(outputs1)
    print(response1[0].split("### Response:")[1])

    #import pandas as pd
    #ds = pd.read_json('clean_data_train.json')
    #ds = ds.values.tolist()

    from datasets import load_dataset

    if args.dataset == 'loandata':
        dataset = load_dataset(
             "json", # JSON format data
            data_files="./datasets/trainloan.json",
            trust_remote_code=True # trust remote codes
        )
    elif args.dataset == 'japanese':
        dataset = load_dataset(
             "json", # JSON format data
            data_files="./datasets/trainjapanese.json",
            trust_remote_code=True # trust remote codes
        )
    elif args.dataset == 'german':
        dataset = load_dataset(
             "json", # JSON format data
            data_files="./datasets/traingerman.json",
            trust_remote_code=True # trust remote codes
        )
    elif args.dataset == 'simulated':
        dataset = load_dataset(
             "json", # JSON format data
            data_files="./datasets/trainsimulate.json",
            trust_remote_code=True # trust remote codes
        )
    else:
        print('Your dataset does not exist.')
    if isinstance(dataset, dict): 
        dataset = dataset["train"]
        
    EOS_TOKEN = tokenizer.eos_token
    #EOS_TOKEN = []

    def formatting_prompts_func(examples):
        #print(type(examples))
        #print(examples)
        inputs = examples["Question"]
        #cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, output in zip(inputs, outputs):
            text = prompt_style.format(input, output) + EOS_TOKEN
            texts.append(text)
            
        return {
                "text": texts,
            }

    if args.dataset == 'loandata':
        datasetval = load_dataset(
             "json", 
            data_files="./datasets/valloan.json",
            trust_remote_code=True 
        )
    elif args.dataset == 'japanese':
        datasetval = load_dataset(
             "json", 
            data_files="./datasets/valjapanese.json",
            trust_remote_code=True 
        )
    elif args.dataset == 'german':
        datasetval = load_dataset(
             "json", 
            data_files="./datasets/valgerman.json",
            trust_remote_code=True 
        )
    elif args.dataset == 'simulated':
        datasetval = load_dataset(
             "json", 
            data_files="./datasets/valsimulate.json",
            trust_remote_code=True 
        )    
    if isinstance(datasetval, dict): 
        datasetval = datasetval["train"]
        
    EOS_TOKEN = tokenizer.eos_token
    #EOS_TOKEN = []

    def formatting_prompts_func(examples):
        #print(type(examples))
        #print(examples)
        inputs = examples["Question"]
        #cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, output in zip(inputs, outputs):
            text = prompt_style.format(input, output) + EOS_TOKEN
            texts.append(text)
            
        return {
                "text": texts,
            }


    #dataset = formatting_prompts_func(ds)
    dataset = dataset.map(formatting_prompts_func,batched=True,)
    print(dataset)
    datasetval = datasetval.map(formatting_prompts_func,batched=True,)
    print(datasetval)
    #dataset = dataset.map(formatting_prompts_func, batched = True,)
    #dataset_list = list(dataset.items())
    #print(type(dataset))
    #print(dataset_list)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj",        "k_proj",        "v_proj",        "o_proj",        "gate_proj",        "up_proj",        "down_proj",    ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    from trl import SFTTrainer
    from transformers import TrainerCallback, TrainingArguments
    from unsloth import is_bfloat16_supported
                
    trainer = SFTTrainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=dataset,
       eval_dataset=datasetval, 
       dataset_text_field="text",
       max_seq_length=max_seq_length,
       #truncation=True,
       #callbacks=[ValidationCallback(datasetval, tokenizer)],
       dataset_num_proc=4,
       args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=3,
            #max_step=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputsrepository",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            save_total_limit=1,
        ),
    )


    wandb.init()

    trainer_stats = trainer.train()

    FastLanguageModel.for_inference(model)

    new_model_local = "DeepSeek-R1-Credit"
    tokenizer.save_pretrained(new_model_local)

if __name__ == "__main__":
    main()
    



