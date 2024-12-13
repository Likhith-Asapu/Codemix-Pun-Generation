import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("likhithasapu/codemix-indicbart")
model = AutoModelForSeq2SeqLM.from_pretrained("likhithasapu/codemix-indicbart").to("cuda")  # Ensure the model is on the GPU

# Open the JSON file
with open('questions.json') as file:
    data = json.load(file)

# Parameters
batch_size = 8
data = data
total_data = len(data)

def process_batch(batch):
    texts = [row['question'] for row in batch]
    input_texts = [f"Translate the English sentence to Hindi-English sentence: <s> {text} </s>" for text in texts]
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"], max_new_tokens=150, num_beams=5, no_repeat_ngram_size=3, do_sample=False, num_return_sequences=1, temperature=1, top_k=50, top_p=0.95, early_stopping=True)
    translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return translations

# Process data in batches
for i in tqdm(range(0, total_data, batch_size)):
    batch = data[i:i + batch_size]
    translations = process_batch(batch)
    for j, row in enumerate(batch):
        row['translated_question'] = translations[j]

# Write the translated questions to a new file
with open('translated_questions.json', 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)
