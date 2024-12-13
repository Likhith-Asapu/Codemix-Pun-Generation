import json
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# Load the pretrained model and tokenizer
model_name = "likhithasapu/xlmr-pun-detection"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Load data from data.json
with open("/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/puns_5.json", "r") as f:
    data_samples = json.load(f)  # Assuming data.json contains a list of samples

# Function to calculate confidence score for a sentence
def get_confidence_score(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    confidence_score = torch.softmax(logits, dim=1)[0, 1].item()  # Confidence for pun class (assuming label 1 = pun)
    return confidence_score

# Process each sample and determine the best sentence
best_sentences = []

for sample in data_samples:
    # Extract sentences for this sample
    sentences = {
        "task1": sample["task1"],
        "task2": sample["task2"],
        "task3": sample["task3"]
    }
    
    # Calculate confidence scores for each sentence
    confidence_scores = {key: get_confidence_score(sentence) for key, sentence in sentences.items()}
    
    # Find the sentence with the highest confidence score
    best_method = max(confidence_scores, key=confidence_scores.get)
    best_sentence = sentences[best_method]
    best_score = confidence_scores[best_method]
    
    # Append the result for this sample
    best_sentences.append({
        "sample_index": data_samples.index(sample),  # Optional: Track sample index
        "best_method": best_method,
        "best_sentence": best_sentence,
        "confidence_score": best_score,
        "Pun_word": sample["Pun_word"],
        "Alternate_word": sample["Alternate_word"],
        "Pun_word_translation": sample["Pun_word_translation"],
        "Pun_word_pos": sample["Pun_word_pos"],
        "Alternate_word_pos": sample["Alternate_word_pos"],
        "Compatibility_score": sample["Compatibility_score"],
        "baseline": sample["baseline"],
        "task1": sample["task1"],
        "task2": sample["task2"],
        "task3": sample["task3"]
    })

# Output results for each sample
for result in best_sentences:
    print(f"Sample {result['sample_index']}:")
    print(f"  Best method: {result['best_method']}")
    print(f"  Best sentence: {result['best_sentence']}")
    print(f"  Confidence score: {result['confidence_score']}\n")
    
# Save the results to a JSON file
with open("/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/best_sentences.json", "w") as f:
    json.dump(best_sentences, f, indent=4, ensure_ascii=False)
