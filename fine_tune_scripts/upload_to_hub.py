from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TrainingArguments
from huggingface_hub import login

# Load the locally saved model checkpoint
model_path = "/scratch/likhithasapu/results-codemix-indicbart/checkpoint-36915"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load the tokenizer corresponding to the model
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.push_to_hub("prakod/gcm_codemix_indicbart_cmi_ratings_gt_4_cmi_gt_30_allperidx", private=True)
model.push_to_hub("prakod/gcm_codemix_indicbart_cmi_ratings_gt_4_cmi_gt_30_allperidx", private=True)

