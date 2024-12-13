import os
import sys
from datasets import Dataset, load_dataset, DatasetDict
# Trainer
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer, DataCollatorForLanguageModeling, GenerationConfig, DataCollatorForSeq2Seq
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# Parameter efficient Fine-tuning
from peft import LoraConfig, prepare_model_for_kbit_training
from huggingface_hub import login
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

wandb.login()

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))

model_id = "ai4bharat/IndicBART"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Load dataset
data = load_dataset("kapilrk04/codemix-en_enhi",split="train",use_auth_token=True,cache_dir="/scratch/likhithasapu/codemix-dataset")
# Select only 1000 samples 
data = data.train_test_split(test_size=0.2,shuffle=True)
print(data)

def tokenize_function(examples):
    # texts = [f"CONTEXT: {context} GENERATE A SUITABLE RESPONSE:\n" for contex$
    # text_outputs = [f"RESPONSE: {response} <eos>" for response in examples['r$

    # input = tokenizer(texts, padding="max_length", truncation=True, max_lengt$
    # output = tokenizer(text_outputs, padding="max_length", truncation=True, m$
    
    prompts = [f"Translate the English sentence to Hindi-English sentence: <s>: {example['en']} </s>" for example in examples['translation']]
    completions = [f"<s> {example['en-hi']} </s>" for example in examples['translation']]

    return {"prompt": prompts, "completion": completions}

batch_size = 128
data = data.map(tokenize_function, batched=True, remove_columns = data["train"].column_names, num_proc=2, batch_size=batch_size)
print(data)

generation_config = GenerationConfig(
    max_new_tokens=150, do_sample=True, num_beams=5, no_repeat_ngram_size=3, eos_token_id=model.config.eos_token_id, num_return_sequences=1,
    pad_token=model.config.pad_token_id,
)

args = TrainingArguments(
    output_dir="/scratch/likhithasapu/results-indicbart-sft",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=4,
    warmup_steps=2,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="codemix-indicbart-sft",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10000,
    load_best_model_at_end=True,
)

# from datasets import load_metric
# import evaluate
# metric = evaluate.load("sacrebleu")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     print(predictions.shape, labels.shape)
  
#     #replace all -100 with tokenizer.pad_token_id
#     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     print(decoded_preds, decoded_labels)
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     return result

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=args,
    max_seq_length=256,
    data_collator=collator,
)


trainer.train()
generation_config.save_pretrained("likhithasapu/codemix-indicbart-sft", push_to_hub=True)
tokenizer.push_to_hub("likhithasapu/codemix-indicbart-sft")
model.push_to_hub("likhithasapu/codemix-indicbart-sft")
