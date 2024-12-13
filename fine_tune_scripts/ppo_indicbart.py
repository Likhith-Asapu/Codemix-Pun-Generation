# %%
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, GenerationConfig, DataCollatorWithPadding
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
from datasets import load_dataset
from huggingface_hub import login

data = load_dataset("kapilrk04/codemix-en_enhi",split="train",use_auth_token=True,cache_dir="codemix-dataset")
data = data.train_test_split(test_size=0.2,shuffle=True)
print(data)

# %%

# %%
# Load tokenizers and models
translation_model_name = "likhithasapu/codemix-indicbart"
acceptability_model_name = "likhithasapu/indic-bert-regression-v1"
tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
acceptability_model = AutoModelForSequenceClassification.from_pretrained(acceptability_model_name)

# %%
# Preprocess function
def preprocess_function(examples):
    prompts = [f"Translate the English sentence to Hindi-English sentence: <s> {example['en']} </s>" for example in examples['translation']]
    responses = [f"<s> {example['en-hi']} </s>" for example in examples['translation']]
    
    model_inputs = tokenizer(prompts, truncation=True, padding=True, max_length=256)
    labels = tokenizer(responses, truncation=True, padding=True, max_length=256)["input_ids"]
    
    model_inputs["labels"] = labels
    return model_inputs

# %%

# Apply the preprocess function to the dataset with batching
batch_size = 64
processed_data = data.map(
    preprocess_function,
    batched=True,
    batch_size=batch_size,
    remove_columns=data["train"].column_names
)


# %%
# Define PPO configuration
ppo_config = PPOConfig(
    model_name=translation_model_name,    
    learning_rate=1.41e-5,
    ppo_epochs=1,
    mini_batch_size=2,
    batch_size=8
)

# %%
# Define reward function
def compute_reward(predictions):
    inputs = tokenizer(predictions, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(acceptability_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = acceptability_model(**inputs).logits.squeeze().tolist()
        
    return scores

# %%
from torch.utils.data import DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(processed_data["train"], batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(processed_data["test"], batch_size=8, collate_fn=data_collator)

# %%
# Initialize PPO model
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(translation_model)
ref_model = create_reference_model(ppo_model)
ppo_trainer = PPOTrainer(config=ppo_config, model=ppo_model, ref_model=ref_model, tokenizer=tokenizer, dataset=processed_data["train"], data_collator=data_collator)

# %%
# Training loop
max_ppo_steps = 1000
# Set up length sampler and generation configurations
output_min_length = 10
output_max_length = 100


# Training loop
for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break   

    prompt_tensors = [tensor for tensor in batch["input_ids"]]
    attention_mask = [tensor for tensor in batch["attention_mask"]]

    # Get response from the model
    summary_tensors = []

    for prompt_tensor,attention_tensor in zip(prompt_tensors,attention_mask):        
        summary = ppo_model.generate(input_ids = prompt_tensor.unsqueeze(0).to("cuda"), attention_mask = attention_tensor.unsqueeze(0).to("cuda"), max_new_tokens=output_max_length, num_beams=5, no_repeat_ngram_size=3, do_sample=True, num_return_sequences=1)
        summary_tensors.append(summary.squeeze()[-output_max_length:])
        
    batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in summary_tensors]

    # Compute reward outputs.
    query_response_pairs = [tokenizer.decode(q, skip_special_tokens=True) + r for q, r in zip(batch["input_ids"], batch["response"])]    
    rewards = [torch.tensor(reward) for reward in compute_reward(query_response_pairs)]
    print(rewards)
    # Run PPO step.
    
    stats = ppo_trainer.step(prompt_tensors, summary_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# %%
# Save the model
tokenizer.push_to_hub("likhithasapu/codemix-indicbart-ppo-1000")
ppo_model.push_to_hub("likhithasapu/codemix-indicbart-ppo-1000")
