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


 
def calculate_fleiss_kappa_unequal(data):
    """
    Calculate Fleiss' Kappa for a dataset with an unequal number of annotations per sample.

    Parameters:
    - data: list of lists where each inner list represents the ratings for a single sample.
            Each rating is an integer category assigned by an annotator.
    
    Returns:
    - Fleiss' Kappa score.
    """
    # Find the total number of categories from the data
    all_categories = {label for sample in data for label in sample}
    num_categories = len(all_categories)
    
    # Initialize counters
    total_agreements = 0
    total_pairwise_comparisons = 0
    total_annotators = 0
    
    # Calculate agreement for each sample
    for sample in data:
        n_annotators = len(sample)
        total_annotators += n_annotators
        
        # Count occurrences of each category in this sample
        label_counts = Counter(sample)
        
        # Calculate observed agreement for this sample
        agreement = sum(count * (count - 1) for count in label_counts.values())
        total_agreements += agreement
        total_pairwise_comparisons += n_annotators * (n_annotators - 1)
    
    # Calculate P_o (observed agreement) and P_e (expected agreement)
    P_o = total_agreements / total_pairwise_comparisons
    
    # Calculate category probabilities across all annotations
    label_counts = Counter([label for sample in data for label in sample])
    total_labels = sum(label_counts.values())
    P_e = sum((count / total_labels) ** 2 for count in label_counts.values())
    
    # Calculate Fleiss' Kappa
    kappa = (P_o - P_e) / (1 - P_e)
    return kappa

def average_pairwise_agreement(data):
    pairwise_agreements = []
    index = 0    
    for sample in data:
        index += 1
        if index <= 250:
            continue
        n = len(sample)
        if n > 1:
            # Count pairwise agreements for this sample
            agreements = sum(1 for (a, b) in combinations(sample, 2) if a == b)
            possible_pairs = n * (n - 1) / 2
            pairwise_agreements.append(agreements / possible_pairs)
        else:
            pairwise_agreements.append(1.0)  # Single rater, perfect agreement by default
    
    return sum(pairwise_agreements) / len(pairwise_agreements)

def pad_data(data):
    """
    Pad the data with np.nan to align the samples to the maximum number of annotations.
    """
    max_len = max(len(sample) for sample in data)
    return [sample + [np.nan] * (max_len - len(sample)) for sample in data]

def calculate_pairwise_kappa(data):
    """
    Calculate pairwise Cohen's Kappa for each pair of annotators.
    
    Parameters:
    - data: list of lists, where each inner list represents the ratings for a single sample by all annotators.
    
    Returns:
    - A list of Cohen's Kappa scores for each pair of annotators.
    """
    # Pad data so each sample has the same number of annotations
    padded_data = pad_data(data)
    
    # Transpose data to have each row represent an annotator's ratings across samples
    transposed_data = np.array(padded_data).T
    
    # Prepare a list to store kappa scores for each pair
    kappa_scores = []

    # Iterate over all pairs of annotators
    for (i, j) in combinations(range(transposed_data.shape[0]), 2):
        # Extract ratings from both annotators, ignoring samples where either has np.nan
        ratings_i = transposed_data[i]
        ratings_j = transposed_data[j]
        
        # Mask NaN values
        valid_indices = ~np.isnan(ratings_i) & ~np.isnan(ratings_j)
        ratings_i = ratings_i[valid_indices]
        ratings_j = ratings_j[valid_indices]
        
        # Calculate Cohen's Kappa if there are valid ratings to compare
        if len(ratings_i) > 0:
            kappa = cohen_kappa_score(ratings_i, ratings_j)
            if isinstance(kappa, (int, float)):  # Ensure kappa is numeric
                kappa_scores.append(kappa)

    return kappa_scores

    
apa_score = average_pairwise_agreement(data_combined.values())
print("Average Pairwise Agreement Score:", apa_score)
max_len = max(len(row) for row in data_combined.values())
print("Max number of raters:", max_len)
data_padded = [row + [np.nan] * (max_len - len(row)) for row in data_combined.values()]
# Calculate Krippendorff's Alpha
# Transpose the data to match krippendorff.alpha expected input shape
data_padded = np.array(data_padded).T

# Calculate Krippendorff's Alpha
alpha_score = krippendorff.alpha(reliability_data=data_padded, level_of_measurement="nominal")
print("Krippendorff's Alpha Score:", alpha_score)

# Calculate pairwise Cohen's Kappa scores
kappa_scores = calculate_pairwise_kappa(data)

# Calculate average pairwise Cohen's Kappa score
kappa_scores = [score for score in kappa_scores if isinstance(score, (int, float))]
# Calculate average pairwise Cohen's Kappa score
if kappa_scores:  # Check if there are valid kappa scores to average
    average_kappa = np.mean(kappa_scores)  # Now we can use np.mean without np.nanmean
    print("Average Pairwise Cohen's Kappa Score:", average_kappa)
else:
    print("No valid pairwise Cohen's Kappa scores were calculated.")


# Set the number of raters (e.g., 2 raters in this case, from file1 and file2)
num_raters = 3

# Get all possible unique categories (labels)
categories = set()
for labels in data_combined.values():
    categories.update(labels)
categories = sorted(categories)  # Sorting to ensure consistent ordering

# Create the frequency matrix
frequency_matrix = []
for entry_id, labels in data_combined.items():
    if len(labels) < num_raters:
        labels += [get_max_label(labels)] * (num_raters - len(labels))

    freqs = [labels.count(category) for category in categories]
    frequency_matrix.append(freqs)
    # print(f"Entry {entry_id}: {labels} -> {freqs}")

# Ensure that each row has the same number of raters
print(len(frequency_matrix), len(frequency_matrix[0]))
n_sub = len(frequency_matrix)
n_rat = num_raters

# Calculate Fleiss' Kappa
kappa = fleiss_kappa(np.array(frequency_matrix))
print(f"Fleiss' Kappa: {kappa}")
