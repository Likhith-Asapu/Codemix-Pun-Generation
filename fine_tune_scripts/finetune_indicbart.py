from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

# tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
# model = torch.nn.DataParallel(model)

# model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']

from datasets import load_dataset
from huggingface_hub import login

data = load_dataset("prakod/gcm_enhi_with_cmi_ratings_gt_4_cmi_gt_10_allperidx",split="train",use_auth_token=True)
data = data.filter(lambda example: example['cmi'] > 30)
data = data.train_test_split(test_size=0.1)


# import matplotlib.pyplot as plt
# import seaborn as sns
# # Tokenize the sentences and get the number of tokens
# num_tokens = [len(tokenizer.tokenize(sentence)) for sentence in data['train']['CM_candidates']]

# # Plotting the distribution of the number of tokens
# sns.histplot(num_tokens, bins=50)
# plt.title("Distribution of Number of Tokens per Sentence")
# plt.xlabel("Number of Tokens")
# plt.ylabel("Frequency")
# plt.show()

# max_length = 0
# for sentence in data['train']['CM_candidates']:
#     tokenized_sentence = tokenizer.tokenize(sentence)
#     length = len(tokenized_sentence)
    
#     if length > max_length:
#         max_length = length
#         max_sentence = sentence

# # Output the sentence with the maximum length and the number of tokens
# print(f"Sentence with the maximum number of tokens ({max_length} tokens):")
# print(max_sentence)

def preprocess_function(examples):
    prompts = [f"Translate the English sentence to Hindi-English sentence: <s> {example} </s>" for example in examples['L2']]
    responses = [f"<s> {example} </s>" for example in examples['CM_candidates']]
    model_inputs = tokenizer(prompts,truncation=True, padding=True,max_length=128, return_tensors="pt").to("cuda")
    model_inputs["labels"] = tokenizer(responses,truncation=True, padding=True,max_length=128, return_tensors="pt")["input_ids"].to("cuda")
    
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

from datasets import load_metric
metric = load_metric("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
  
    # replace all -100 with tokenizer.pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq1Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="/scratch/likhithasapu/results-codemix-indicbart",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="codemix-indicbart-bleu-cmi-30",
    load_best_model_at_end=True,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=128,
    logging_steps=1000,
    dataloader_num_workers=2,
    gradient_accumulation_steps=4,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_data['train'],
    eval_dataset=processed_data['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

tokenizer.push_to_hub("gcm_codemix_indicbart_cmi_ratings_gt_4_cmi_gt_30_allperidx", private=True)
model.push_to_hub("gcm_codemix_indicbart_cmi_ratings_gt_4_cmi_gt_30_allperidx", private=True)
