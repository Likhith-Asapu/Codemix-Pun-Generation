import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import json

def average_spearman_correlation(ratings):
    """
    Calculate the average Spearman correlation among all pairs of annotators.
    This function can handle samples with a variable number of annotations (2 or 3).
    
    Parameters:
    ratings (list of lists): A list of lists where each sublist represents an item and contains
                             scores given by annotators for that item. Each sublist can have 2 or 3 ratings.
                             
    Returns:
    float: The average Spearman correlation among all pairs of annotators.
    """
    # Initialize list to store all pairwise correlations
    correlations = []
    
    # Loop through each sample
    for sample_ratings in ratings:
        # Get the number of ratings for the sample
        num_raters = len(sample_ratings)
        
        # Only calculate if there are more than one rating (i.e., 2 or 3 ratings)
        if num_raters > 1:
            # Generate all unique pairs of ratings for the current sample
            rater_pairs = list(combinations(range(num_raters), 2))
            
            # Calculate Spearman correlations for each pair of ratings in this sample
            for r1, r2 in rater_pairs:
                # Extract scores for the two annotators
                score1, score2 = sample_ratings[r1], sample_ratings[r2]
                
                # Spearman correlation between the two ratings (treated as two-element arrays)
                corr, _ = spearmanr([score1], [score2])
                
                # Add correlation to the list
                correlations.append(corr)
    
    # Calculate the average Spearman correlation across all pairs
    print(correlations)
    average_correlation = np.nanmean(correlations)  # Use nanmean to handle potential NaNs
    
    return average_correlation

# Example usage:
# Suppose we have 5 samples, each rated by 2 or 3 annotators
# Each sublist represents ratings for a single item

with open('Data/final_data.json') as f:
    data1 = json.load(f)
    
with open('Data/final_data2.json') as f:
    data2 = json.load(f)
    
with open('Data/final_data3.json') as f:
    data3 = json.load(f)
    
with open('Data/final_data4.json') as f:
    data4 = json.load(f)

data_combined = {}

def add_data(data, data_combined):
    for entry in data:
        entry_id = entry['id']
        if entry_id in data_combined:
            data_combined[entry_id].append(entry['label'])
        else:
            data_combined[entry_id] = [entry['label']]
    return data_combined

data_combined = add_data(data1, data_combined)
data_combined = add_data(data2, data_combined)
data_combined = add_data(data3, data_combined)
data_combined = add_data(data4, data_combined)


ratings = [labels for labels in data_combined.values() if len(labels) > 1]

avg_spearman_corr = average_spearman_correlation(ratings)
print(f"Average Spearman Correlation: {avg_spearman_corr:.4f}")
