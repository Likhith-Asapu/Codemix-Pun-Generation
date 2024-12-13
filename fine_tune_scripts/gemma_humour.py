# Load the dataset
from huggingface_hub import login

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, BloomForCausalLM, AutoModelForCausalLM
from datasets import load_metric
import numpy as np
import torch
import time
import pandas as pd

data = load_dataset("likhithasapu/context-situated-puns")
# data = data.train_test_split(test_size=0.2)
filtered_dataset = data["train"].filter(lambda x: x["user_pun"] != None)
filtered_dataset = filtered_dataset.filter(lambda x: len(x["user_pun"]) != 0)
filtered_dataset = filtered_dataset.filter(lambda x: x["user_pun"] != "{}")

print(filtered_dataset)

model_name='google/gemma-2b-it'

original_model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

"""# Zero-Shot Inference"""

index = 20

alternate_word = filtered_dataset[index]['alter_word']
user_pun = filtered_dataset[index]['user_pun']


prompt = f"""
Generate a humorous sentence containing the word {alternate_word}

Sentence:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN PUN:\n{user_pun}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

"""# Fine-tune model

"""

def tokenize_function(example):
    start_prompt = 'Generate a humorous sentence containing the word '
    end_prompt = '\n\nSentence: '
    alter_word = example["alter_word"]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=100).input_ids.squeeze(0)
    example['labels'] = tokenizer(example["user_pun"], padding="max_length", truncation=True, return_tensors="pt", max_length=100).input_ids

    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = filtered_dataset.map(tokenize_function)
tokenized_datasets = tokenized_datasets.remove_columns(['context', 'pun_word', 'alter_word', 'pun_word_sense', 'alter_word_sense', 'new_pun', 'user_pun'])

import wandb

wandb.login()
wandb.init(entity="teamxy")

output_dir = f'./gemma-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="humour-gen-gemma",
    load_best_model_at_end=True,
    gradient_accumulation_steps = 16
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
    )

trainer.train()
tokenizer.push_to_hub("humour-gen-gemma")
original_model.push_to_hub("humour-gen-gemma")

model_name = "likhithasapu/humour-gen-gemma"
instruct_model = AutoModelForCausalLM.from_pretrained(model_name)

index = 400

alternate_word = filtered_dataset[index]['alter_word']
user_pun = filtered_dataset[index]['user_pun']


prompt = f"""
Generate a humorous sentence containing the word {alternate_word}

Sentence:
"""

original_inputs = tokenizer(prompt, return_tensors='pt')
original_output = tokenizer.decode(
    original_model.generate(
        original_inputs["input_ids"].to("cuda"),
        max_new_tokens=100,
        num_beams=5,
      early_stopping=True
    )[0],
    skip_special_tokens=True
)


inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    instruct_model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN PUN:\n{user_pun}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{original_output}')
print(dash_line)
print(f'MODEL GENERATION - FINE TUNE:\n{output}')
