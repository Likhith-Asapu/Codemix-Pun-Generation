from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from huggingface_hub import login

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")
# tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

model = AutoModelForCausalLM.from_pretrained("ai4bharat/IndicBART", device_map="auto")
# model = torch.nn.DataParallel(model)

# model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")

# Some initial mapping
# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']


data = load_dataset("kapilrk04/codemix-en_enhi",split="train",use_auth_token=True,cache_dir="/scratch/likhithasapu/codemix-dataset").train_test_split(test_size=0.2,shuffle=True)
print(data)


def preprocess_function(examples):
    prompts = [f"Translate the English sentence to Hindi-English sentence: <s> {example['en']} </s>" for example in examples['translation']]
    responses = [f"<s> {example['en-hi']} </s>" for example in examples['translation']]
    
    model_inputs = tokenizer(prompts,truncation=True, padding='max_length',max_length=256, return_tensors="pt").to("cuda")
    model_inputs["labels"] = tokenizer(responses,truncation=True, padding='max_length',max_length=256, return_tensors="pt")["input_ids"].to("cuda")
    
    return model_inputs
    
# Apply the preprocess function to the dataset with batching
batch_size = 128
processed_data = data.map(
    preprocess_function,
    batched=True,
    batch_size=batch_size,
    remove_columns=data["train"].column_names
)


import wandb

wandb.login()

# use blue score as metric

from datasets import load_metric
import evaluate
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
  
    #replace all -100 with tokenizer.pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result


from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/scratch/likhithasapu/results-codemix-indicbart",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="codemix-indicbart-causal-lm",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data['train'],
    eval_dataset=processed_data['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

tokenizer.push_to_hub("codemix-indicbart-causal-lm")
model.push_to_hub("codemix-indicbart-causal-lm")