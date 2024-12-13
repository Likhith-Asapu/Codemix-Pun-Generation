import json
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Load data from JSON
with open('Data/labels.json', 'r') as f:
    data = json.load(f)

# Assuming the JSON file has 'text' and 'label' keys
texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.33, random_state=0)

# Generate random predictions (0 or 1) for the test set
predicted_labels = [random.choice([0]) for _ in test_labels]

# Compute evaluation metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, predicted_labels, average='weighted', zero_division=0
)
accuracy = accuracy_score(test_labels, predicted_labels)

# Print evaluation results
print("\nEvaluation Results on Test Set:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
