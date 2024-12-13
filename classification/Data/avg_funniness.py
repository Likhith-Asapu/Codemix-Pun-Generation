import json
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import spearmanr
import random
import numpy as np

with open('final_data.json') as f:
    data1 = json.load(f)
    
with open('final_data2.json') as f:
    data2 = json.load(f)
    
with open('final_data3.json') as f:
    data3 = json.load(f)
    
with open('final_data4.json') as f:
    data4 = json.load(f)
    
with open('final_data5.json') as f:
    data5 = json.load(f)
    
with open('final_data6.json') as f:
    data6 = json.load(f)

data_combined = {}
funny_combined = {}

def add_data(data, data_combined):
    for entry in data:
        entry_id = entry['id']
        if entry_id in data_combined:
            data_combined[entry_id].append(entry['label'])
        else:
            data_combined[entry_id] = [entry['label']]
        
        if entry['label'] == 1 and entry_id in funny_combined:
            funny_combined[entry_id].append(entry['funniness'])
        elif entry['label'] == 1 and entry_id not in funny_combined:
            funny_combined[entry_id] = [entry['funniness']]
            
    return data_combined, funny_combined

data_combined, funny_combined = add_data(data1, data_combined)
data_combined, funny_combined = add_data(data2, data_combined)
data_combined, funny_combined = add_data(data3, data_combined)
data_combined, funny_combined = add_data(data4, data_combined)
data_combined, funny_combined = add_data(data5, data_combined)
data_combined, funny_combined = add_data(data6, data_combined)

# Set the number of raters (e.g., 2 raters in this case, from file1 and file2)
num_raters = 3

# Get all possible unique categories (labels)
categories = set()
for labels in data_combined.values():
    categories.update(labels)
categories = sorted(categories)  # Sorting to ensure consistent ordering

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

def get_avg_funniness(index):
    if index in funny_combined:
        funny = funny_combined[index]
    else:
        return 0
    return sum(funny) / len(funny)

# for i in range(0,500):
#     if get_max_label(data_combined[i]) == 1 and get_max_label(data_combined[i+500]) == 1 and get_max_label(data_combined[i+1000]) == 1 and get_max_label(data_combined[i+1500]) == 1:
#         print(i, data_combined[i], data_combined[i+500], data_combined[i+1000], data_combined[i+1500])
#         print(i, get_avg_funniness(i), get_avg_funniness(i+500), get_avg_funniness(i+1000), get_avg_funniness(i+1500))

def get_success_rate(data_combined, start, end):
    success = 0
    total_funniness = 0
    total = 0
    for i in range(start, end):
        if get_max_label(data_combined[i]) == 1:
            total_funniness += get_avg_funniness(i)
            success += 1
        total += 1
    return success / total, success, total, total_funniness / success

print(get_success_rate(data_combined, 0, 500))
print(get_success_rate(data_combined, 500, 1000))
print(get_success_rate(data_combined, 1000, 1500))
print(get_success_rate(data_combined, 1500, 2000))

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set seaborn style
# sns.set_theme(style="whitegrid")
# plt.rcParams['font.size'] = 24
# plt.rcParams['font.weight'] = 'bold'

# # Example success rate data for each method (in percentages)
# # Adjust these values to reflect your actual success rate data.
# success_rates = {
#     "Baseline": 20.0,
#     "Contextually Aligned": 38.8,
#     "Question-Answer": 62.6,
#     "Subject-Masked": 43.0
# }

# # Method names and their respective success rates
# methods = list(success_rates.keys())
# rates = list(success_rates.values())

# # Plot each method as an individual pie chart
# fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# for i, method in enumerate(methods):
#     # Define colors for success (green) and failure (dark blue)
#     colors = ["#09b83b", "#e01032"]
#     axs[i].pie(
#         [rates[i], 100 - rates[i]], 
#         labels=[f'Success ({rates[i]}%)', f'Failure ({100 - rates[i]}%)'],
#         autopct='%0.1f%%',
#         startangle=180,
#         colors=colors,
#         wedgeprops={'edgecolor': 'white'},  # To separate slices cleanly
#         textprops={'fontsize': 18, 'fontweight': 'bold'}  # Bold and larger text inside pie chart
#     )
#     axs[i].set_title(method, fontsize=24, fontweight='bold')  # Increased font size for titles

# # Set a main title with larger font size
# plt.suptitle("Success Rates by Method", fontsize=32, fontweight='bold')

# # Adjust layout to make room for the main title
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # Display the pie charts
# plt.show()

# # Save the figure as a high-resolution PNG file
# fig.savefig('success_rates.png', dpi=300)
