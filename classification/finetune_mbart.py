from transformers import BartForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, GenerationConfig
from datasets import Dataset
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from torch.nn import CrossEntropyLoss

# Load data
with open('Data/labels.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data if item['label'] != 2]
labels = [item['label'] for item in data if item['label'] != 2]

# Model and Tokenizer
model_name = 'facebook/bart-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set num_labels based on binary classification (two labels: 0, 1)
num_labels = 2
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
generation_config = GenerationConfig.from_pretrained(model_name)

# Assign the custom generation configuration to the model
model.generation_config = generation_config

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Training function
def train_model(random_state):
    wandb.init(project="research", name=f"{model_name}")
    
    # Split data into train, validation, and test sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=random_state)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.33, random_state=random_state)

    # Create datasets
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define custom Trainer with weighted loss
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Send labels to the same device as model parameters
            device = next(model.parameters()).device
            labels = inputs.get("labels").to(device)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Class weighting
            weight_for_positive = len(train_labels) / sum(train_labels)
            class_weights = torch.tensor([1.0, weight_for_positive]).to(device)  # Move to correct device

            # Weighted CrossEntropy
            loss_fn = CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels.to(logits.device))
            
            return (loss, outputs) if return_outputs else loss

    # Define metrics
    def compute_metrics(p):
        preds = torch.argmax(torch.tensor(p.predictions[0]), dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
        acc = accuracy_score(p.label_ids, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Define training arguments with DDP for multi-GPU support
    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        report_to="wandb",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        dataloader_num_workers=4,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print("Validation Evaluation results:", eval_results)

    test_results = trainer.evaluate(test_dataset)
    print("Test Evaluation results:", test_results)

# Run training with different random seeds
for random_state in [0, 42, 10, 100, None]:
    train_model(random_state)
