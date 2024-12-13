# %%
import os
import sys
from datasets import Dataset, load_dataset, DatasetDict
# Trainer
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer, DataCollatorForLanguageModeling
# Parameter efficient Fine-tuning
from peft import LoraConfig, prepare_model_for_kbit_training
from huggingface_hub import login
import wandb
from peft import LoraConfig, get_peft_model, TaskType

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"Trainable model parameters: {trainable_model_params}\nAll model parameters: {all_model_params}\nPercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


# LoRA configuration
lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    target_modules = ["q_proj", "v_proj", "k_proj", "down_proj", "gate_proj", "up_proj"],
    task_type = "CAUSAL_LM", 
)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device))

# %%
model_id = "ai4bharat/Airavata"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
# model.gradient_checkpointing_enable()


# %%
# Apply PEFT with LoRA
peft_model = get_peft_model(model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

# %%
main_dataset = load_dataset("prakod/gcm_enhi_with_cmi_ratings_gt_4_cmi_gt_10_allperidx",split="train",use_auth_token=True, cache_dir='/scratch/codemix-data')


# %%
# Login to wandb and Hugging Face
wandb.login()

# %%
def tokenize_function(examples):
    samples = [example for example in examples['CM_candidates']]
    inputs = tokenizer(samples, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return inputs

dataset = main_dataset.select(range(10000)).train_test_split(test_size=0.1)
print(dataset)

args = TrainingArguments(
    output_dir="/scratch/results-airavata",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="codemixed-airavata",
    learning_rate=5e-4,
    fp16=True,
    logging_steps=1000,
    optim="adamw_torch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=2
)

# Define Trainer
trainer = Trainer(
    model=peft_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

tokenizer.push_to_hub("gcm_codemix_airavata_cmi_ratings_gt_4_cmi_gt_30_allperidx_causal_lm", private=True)
model.push_to_hub("gcm_codemix_airavata_cmi_ratings_gt_4_cmi_gt_30_allperidx_causal_lm", private=True)