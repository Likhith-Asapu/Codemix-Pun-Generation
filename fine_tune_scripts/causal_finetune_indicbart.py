from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForCausalLM.from_pretrained("ai4bharat/IndicBART")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

from datasets import load_dataset
from huggingface_hub import login
import wandb
wandb.login()

# Load dataset
data = load_dataset("prakod/gcm_enhi_with_cmi_ratings_gt_4_cmi_gt_10_allperidx", split="train", use_auth_token=True)
data = data.train_test_split(test_size=0.1)
 

def preprocess_function(examples):
    # Here, we're preparing the input text for CLM.
    # No special formatting is needed other than tokenizing.
    inputs = [example for example in examples['CM_candidates']]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    
    # For CLM, the labels are the same as the input, shifted by one position
    model_inputs["labels"] = model_inputs["input_ids"]
    
    return {
        "input_ids": model_inputs.input_ids.to("cuda"),
        "attention_mask": model_inputs.attention_mask.to("cuda"),
        "labels": model_inputs.labels.to("cuda")
    }

# Apply the preprocess function to the dataset with batching
batch_size = 128
processed_data = data.map(
    preprocess_function,
    batched=True,
    batch_size=batch_size,
    remove_columns=data["train"].column_names
)

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/scratch/likhithasapu/results-codemix-indicbart-clm",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="codemix-indicbart-clm",
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=1000,
    dataloader_num_workers=2,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data['train'],
    eval_dataset=processed_data['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Push to hub
tokenizer.push_to_hub("gcm_codemix_indicbart_cmi_ratings_gt_4_cmi_gt_30_allperidx_causal_lm", private=True)
model.push_to_hub("gcm_codemix_indicbart_cmi_ratings_gt_4_cmi_gt_30_allperidx_causal_lm", private=True)
