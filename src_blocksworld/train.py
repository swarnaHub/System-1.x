import os
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
        get_peft_model,
        LoraConfig
    )
from trl import SFTTrainer
from datasets import Dataset

import data_utils

def formatting_func(prompt):
    output = []
    for instruction, plan in zip(prompt['prompt'], prompt['plan']):
        output.append(instruction + ' ' + plan + ' ' + '</s>')

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/blocksworld', type=str)
    parser.add_argument('--output_dir', default='models', type=str)

    parser.add_argument('--task', default='blocksworld', type=str)
    parser.add_argument('--n_train_start', default=0, type=int)
    parser.add_argument('--n_train_end', default=-1, type=int)
    parser.add_argument('--n_val', default=-1, type=int)
    parser.add_argument('--n_test', default=-1, type=int)

    parser.add_argument('--base_model', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--cache_dir', default='cache', type=str)

    parser.add_argument('--method', choices=['heuristic'], default='heuristic', type=str)
    parser.add_argument('--level', choices=['task', 'sample'], default='sample', type=str)
    parser.add_argument('--system', choices=[0.0, 0.5, 1.0], default=0.5, type=float)
    parser.add_argument('--search_algo', choices=['a_star'], default='a_star', type=str)
    parser.add_argument('--decomp_style', choices=['sliding'], default='sliding', type=str)

    parser.add_argument('--max_seq_length', default=5000, type=int)
    parser.add_argument('--per_device_train_batch_size', default=1, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=2, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--eval_accumulation_steps', default=1, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--warmup_ratio', default=0.03, type=float)
    parser.add_argument('--max_grad_norm', default=0.3, type=float)
    parser.add_argument('--logging_steps', default=100, type=int)
    

    args = parser.parse_args()

    data_train, data_val, data_test = data_utils.get_finetuning_data(args)
    
    data_train = Dataset.from_list(data_train)
    data_val = Dataset.from_list(data_val)
    data_test = Dataset.from_list(data_test)

    print(data_train[0])

    access_token = ""

    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 cache_dir=args.cache_dir if args.cache_dir else 'cache', 
                                                 device_map = 'auto',
                                                 token=access_token
                                                 )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, 
                                              cache_dir=args.cache_dir if args.cache_dir else 'cache',
                                              padding_side='left', 
                                              add_eos_token=True, 
                                              token=access_token)

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    data_folder = args.data_dir.split("/")[-1]

    finetuned_model_name = f"{args.task}/{args.search_algo}_{args.method}_{args.decomp_style}_{args.level}_system_{args.system}_{len(data_train)}_epoch_{args.num_epochs}_lr_{args.learning_rate}_bs_{args.per_device_train_batch_size}"
    output_dir = os.path.join(args.output_dir, finetuned_model_name)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            evaluation_strategy="epoch",
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            group_by_length=False,
            lr_scheduler_type="constant",
            ddp_find_unused_parameters=False,
            eval_accumulation_steps=args.eval_accumulation_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            save_strategy="epoch",
            save_total_limit = 1,
            load_best_model_at_end = True
        )
    
    print(finetuned_model_name)

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    trainer.save_model(output_dir)







