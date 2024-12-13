import torch
from datasets import Dataset
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import wandb
from huggingface_hub import login
from transformers import EarlyStoppingCallback


# Load data from JSON
with open('compatibility_scores.json', 'r') as f:
    data = json.load(f)

# Prepare input texts and compatibility scores
texts = [f"Pun_word: {item['Pun_word']}, Alternate_word: {item['Alternate_word']}" for item in data]
labels = [float(item['compatibility_score']) for item in data]  # Compatibility scores

model_name = 'bert-base-multilingual-cased'
tokenizer_model_name = 'bert-base-multilingual-cased'

def train_model(random_state):
    wandb.init(project="Regression", name=f"{model_name}")
    print("-"*40)
    print(f"Model: {model_name}")
    print("Random State: " + str(random_state))
    print("-"*40)
    
    # Initialize tokenizer and model for regression
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Split data into 90% training and 10% validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=random_state)

    # Create Hugging Face Datasets from the split data
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    print(train_dataset.shape, val_dataset.shape)

    # Tokenizing function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=64, truncation=True)

    # Tokenize and format datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define the compute_metrics function for regression
    def compute_metrics(p):
        preds = p.predictions.squeeze()  # Regression output
        labels = p.label_ids
        mae = mean_absolute_error(labels, preds)
        mse = mean_squared_error(labels, preds)
        return {"mae": mae, "mse": mse}

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'/scratch/likhithasapu/{model_name}',  # Output directory for the model
        num_train_epochs=19,                              # Number of training epochs
        per_device_train_batch_size=64,                   # Batch size for training
        per_device_eval_batch_size=128,                    # Batch size for evaluation
        warmup_steps=500,                                 # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                                # Strength of weight decay
        logging_steps=10,
        report_to="wandb",
        evaluation_strategy="epoch",                      # Evaluate every epoch
        save_strategy="epoch",                            # Save checkpoints every epoch
        load_best_model_at_end=True,                      # Load the best model when finished training
        metric_for_best_model="mse",                      # Use MAE to determine the best model
        max_grad_norm=1.0  # Clips gradients to prevent exploding gradients
    )

    trainer = Trainer(
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
    print("Validation Evaluation results:", eval_results)
    
random_state = None
train_model(random_state)
