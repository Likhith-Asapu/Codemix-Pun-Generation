import json
import numpy as np
import matplotlib.pyplot as plt

def load_data(file):
    with open(file) as f:
        data = json.load(f)
    return data

def calculate_avg_and_distribution(data, start, end):
    funniness = [row['funniness'] for row in data[start:end]]
    acceptability = [row['acceptability'] for row in data[start:end]]
    
    avg_funniness = np.mean(funniness)
    std_funniness = np.std(funniness)
    
    avg_acceptability = np.mean(acceptability)
    std_acceptability = np.std(acceptability)
    
    return funniness, acceptability, avg_funniness, std_funniness, avg_acceptability, std_acceptability

def plot_distribution(funniness1, acceptability1, funniness2, acceptability2, avg_fun1, std_fun1, avg_acc1, std_acc1, avg_fun2, std_fun2, avg_acc2, std_acc2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # First 500 samples funniness distribution
    axes[0, 0].hist(funniness1, bins=6, color='blue', alpha=0.7)
    axes[0, 0].set_title(r'First 500 Samples Funniness: $\mu=%.2f$, $\sigma=%.2f$' % (avg_fun1, std_fun1))
    axes[0, 0].set_xlabel('Funniness')
    axes[0, 0].set_ylabel('Frequency')

    # First 500 samples acceptability distribution
    axes[0, 1].hist(acceptability1, bins=5, color='green', alpha=0.7)
    axes[0, 1].set_title(r'First 500 Samples Acceptability: $\mu=%.2f$, $\sigma=%.2f$' % (avg_acc1, std_acc1))
    axes[0, 1].set_xlabel('Acceptability')
    axes[0, 1].set_ylabel('Frequency')

    # Second 500 samples funniness distribution
    axes[1, 0].hist(funniness2, bins=6, color='blue', alpha=0.7)
    axes[1, 0].set_title(r'Second 500 Samples Funniness: $\mu=%.2f$, $\sigma=%.2f$' % (avg_fun2, std_fun2))
    axes[1, 0].set_xlabel('Funniness')
    axes[1, 0].set_ylabel('Frequency')

    # Second 500 samples acceptability distribution
    axes[1, 1].hist(acceptability2, bins=5, color='green', alpha=0.7)
    axes[1, 1].set_title(r'Second 500 Samples Acceptability: $\mu=%.2f$, $\sigma=%.2f$' % (avg_acc2, std_acc2))
    axes[1, 1].set_xlabel('Acceptability')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('distribution.png')

# Load data
data = load_data('final_data.json')

# First 500 samples
funniness1, acceptability1, avg_fun1, std_fun1, avg_acc1, std_acc1 = calculate_avg_and_distribution(data, 0, 500)

# Second 500 samples
funniness2, acceptability2, avg_fun2, std_fun2, avg_acc2, std_acc2 = calculate_avg_and_distribution(data, 500, 1000)

# Print averages and standard deviations
print(f'First 500 Samples: Avg Funniness = {avg_fun1:.2f}, Std Funniness = {std_fun1:.2f}, Avg Acceptability = {avg_acc1:.2f}, Std Acceptability = {std_acc1:.2f}')
print(f'Second 500 Samples: Avg Funniness = {avg_fun2:.2f}, Std Funniness = {std_fun2:.2f}, Avg Acceptability = {avg_acc2:.2f}, Std Acceptability = {std_acc2:.2f}')

# Plot distributions
plot_distribution(funniness1, acceptability1, funniness2, acceptability2, avg_fun1, std_fun1, avg_acc1, std_acc1, avg_fun2, std_fun2, avg_acc2, std_acc2)
