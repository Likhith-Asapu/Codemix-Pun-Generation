import json
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import spearmanr
import random
import numpy as np
from collections import Counter
from itertools import combinations
import krippendorff
from sklearn.metrics import cohen_kappa_score

data = []
files = ['final_data.json', 'final_data2.json', 'final_data3.json', 'final_data4.json', 'final_data5.json', 'final_data6.json']

for file in files:
    file_path = '/home2/likhithasapu/Codemixed-Pun-Generation/pun/classification/Data/' + file 
    with open(file_path) as f:
        data.append(json.load(f))
        

def add_data(data, data_combined, label):
    for entry in data:
        entry_id = entry['id']
        if entry_id in data_combined:
            data_combined[entry_id].append(entry[label])
        else:
            data_combined[entry_id] = [entry[label]]
            
    return data_combined

def get_max_label(labels):
    max_label = 0
    max_count = 0
    if len(labels) == 2:
        return labels[0]
    for label in labels:
        count = labels.count(label)
        if count > max_count:
            max_count = count
            max_label = label
    return max_label

data_combined = {}
for d in data:
    data_combined = add_data(d, data_combined, 'label')

for key in data_combined:
    data_combined[key] = get_max_label(data_combined[key])
    if data_combined[key] == 2:
        data_combined[key] = 0

def total_sum(data_combined):
    sum = 0
    total = 0
    for key in data_combined:
        data_combined[key] = get_max_label(data_combined[key])
        if data_combined[key] == 2:
            data_combined[key] = 0
        sum += data_combined[key]
        total += 1
        print("Key: ", key, " Value: ", data_combined[key])
        
    print("Sum: ", sum, " Total: ", total)
    
def calculate_compatibility(data_combined):
    compatibility_scores = {}
    for i in range(0, 500):
        score = data_combined[i] + data_combined[i+500] + data_combined[i+1000] + data_combined[i+1500]
        compatibility_scores[i] = score
    return compatibility_scores

# Write the compatibility scores to a file
compatability_scores = calculate_compatibility(data_combined)
words_data = []
with open('data.json', 'r') as f:
    words_data = json.load(f)
    
words_data = words_data[:500]

for i in range(0, 500):
    words_data[i]['compatibility_score'] = compatability_scores[i]
    words_data[i].pop('label')
    words_data[i].pop('text')
    
with open('compatibility_scores.json', 'w') as f:
    json.dump(words_data, f, indent=4, ensure_ascii=False)

    
# Plot a histogram of the compatibility scores
import matplotlib.pyplot as plt
import seaborn as sns

# Convert compatibility scores to a list for plotting
scores = list(compatability_scores.values())

# Create the histogram plot
plt.figure(figsize=(8, 6))
sns.histplot(scores, bins=5, kde=False, color='skyblue')

# Add labels and title
plt.xlabel("Compatibility Score")
plt.ylabel("Frequency")
plt.title("Histogram of Compatibility Scores")

# Add numbers on top of each bar
for p in plt.gca().patches:
    plt.gca().text(
        p.get_x() + p.get_width() / 2, 
        p.get_height() + 0.5, 
        int(p.get_height()), 
        ha="center"
    )

# Save the plot
plt.savefig('compatibility_scores.png', bbox_inches='tight')
        