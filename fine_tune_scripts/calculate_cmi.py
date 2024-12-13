from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm

data = load_dataset("prakod/gcm_enhi_with_cmi_ratings_gt_4_cmi_gt_10_allperidx",split="train",use_auth_token=True)
data = data.train_test_split(test_size=0.1)

import re
from collections import Counter

def detect_language(word):
    if re.match(r'^[\u0900-\u097F]+$', word):
        return 'hindi'
    elif re.match(r'^[a-zA-Z]+$', word):
        return 'english'
    return 'other'

def calculate_cmi(sentence):
    words = sentence.split()
    n = len(words)
    languages = [detect_language(word) for word in words]
    
    lang_counts = Counter(languages)
    u = lang_counts['other']  # 'u' is the count of language-independent tags (classified as 'other')

    if n == u:
        return 0  # If all tokens are language-independent, CMI is 0

    max_wi = max(lang_counts['hindi'], lang_counts['english'])
    
    # Calculate CMI
    cmi = 100 * (1 - (max_wi / (n - u)))
    return cmi

sentences = [sentence for sentence in data['train']['CM_candidates']]

cmi_scores = []

for sentence in tqdm(sentences,total = len(sentences)):
    cmi = calculate_cmi(sentence)
    cmi_scores.append(cmi)

# Plotting the histogram of CMI scores
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(cmi_scores, bins=20)
plt.xlabel('CMI')
plt.ylabel('Number of Sentences')
plt.title('Histogram of Code-Mixing Index (CMI) per Sentence')
plt.show()
plt.savefig("cmi_histogram.png")
