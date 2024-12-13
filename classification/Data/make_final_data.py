import json
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import spearmanr
import random
import numpy as np

data = []
files = ['final_data.json', 'final_data2.json', 'final_data3.json', 'final_data4.json', 'final_data5.json', 'final_data6.json']

for file in files:
    with open(file) as f:
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

final_data = []
for key in sorted(data_combined.keys()):
    print(key, data_combined[key], data[0][key]['text'])
    final_data.append({
        'id': key,
        'text': data[0][key]['text'],
        'label': get_max_label(data_combined[key])
    })
    
with open('labels.json', 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)