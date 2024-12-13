import torch
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from torch.nn import CrossEntropyLoss

# Load data from JSON
with open('final_data.json', 'r') as f:
    data = json.load(f)

# Assuming the JSON file has 'text' and 'label' keys
texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Step 1: Split 70% training and 30% (validation + test)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Step 2: Split the remaining 30% into validation (20%) and test (10%)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.33, random_state=42)

# Create Hugging Face Datasets from the split data
train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

# Initialize the tokenizer and model
model_name = 'facebook/bart-large'  # Replace with BART model of choice
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenizing the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], max_length=512, padding="max_length", truncation=True)
    labels = tokenizer([str(label) for label in examples['label']], max_length=2, padding="max_length", truncation=True)  # Labels as text, for seq2seq
    inputs['labels'] = labels['input_ids']
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format to torch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define custom Trainer for BART with weighted loss
class BartWeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Define class weights (higher weight for minority class)
        class_weights = torch.tensor([1.0, len(train_labels) / sum(train_labels)]).to(logits.device)

        # Compute weighted loss using CrossEntropyLoss
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Define the compute_metrics function
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(torch.tensor(p.label_ids), preds, average='weighted', zero_division=0)
    acc = accuracy_score(torch.tensor(p.label_ids), preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'/scratch/likhithasapu/{model_name}',  # Output directory for the model
    num_train_epochs=5,                               # Number of training epochs
    per_device_train_batch_size=4,                    # Batch size for training
    per_device_eval_batch_size=8,                     # Batch size for evaluation
    warmup_steps=500,                                 # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                                # Strength of weight decay
    logging_steps=100,
    evaluation_strategy="epoch",                      # Evaluate every epoch
    save_strategy="epoch",                            # Save checkpoints every epoch
    load_best_model_at_end=True,                      # Load the best model when finished training
    metric_for_best_model="accuracy",                 # Use accuracy score to determine the best model
)

# Initialize the Trainer
trainer = BartWeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()

# Print validation evaluation results
print("Validation Evaluation results:", eval_results)

# Evaluate the model on the test set
test_results = trainer.evaluate(test_dataset)

# Print test set evaluation results
print("Test Evaluation results:", test_results)
