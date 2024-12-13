import torch
from datasets import Dataset
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
import json
import wandb
from huggingface_hub import login

# Load data from JSON
with open('Data/labels.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data if item['label'] != 2]
labels = [item['label'] for item in data if item['label'] != 2]

model_name = 'FacebookAI/xlm-roberta-base'
tokenizer_model_name = 'FacebookAI/xlm-roberta-base'

def train_model(random_state):
    wandb.init(project="research", name=f"{model_name}")
    print("-"*40)
    print(f"Model: {model_name}")
    print("Random State: " + str(random_state))
    print("-"*40)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=random_state)

    # Step 2: Split the remaining 30% into validation (20%) and test (10%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.33, random_state=random_state)

    # Create Hugging Face Datasets from the split data
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    print(train_dataset.shape, val_dataset.shape, test_dataset.shape)
    # Initialize the tokenizer and model

    # Tokenizing the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=256, truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format to torch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define custom Trainer with weighted loss
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Define class weights (e.g., higher weight for the minority class)
            class_weights = torch.tensor([1.0, len(train_labels) / sum(train_labels)]).to(logits.device)

            # Use CrossEntropyLoss with weights
            loss_fn = CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            
            return (loss, outputs) if return_outputs else loss

    # Define the compute_metrics function
    def compute_metrics(p):
        preds = torch.argmax(torch.tensor(p.predictions), dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(torch.tensor(p.label_ids), preds, average='weighted', zero_division=0)
        acc = accuracy_score(torch.tensor(p.label_ids), preds)
        
        per_label_metrics = precision_recall_fscore_support(torch.tensor(p.label_ids), preds, average=None, zero_division=0)
        
        return {
            "accuracy": acc, 
            "f1": f1, 
            "precision": precision, 
            "recall": recall,
            "precision_per_label": per_label_metrics[0].tolist(),  # Precision for each label
            "recall_per_label": per_label_metrics[1].tolist(),     # Recall for each label
            "f1_per_label": per_label_metrics[2].tolist(),         # F1 for each label
            "support_per_label": per_label_metrics[3].tolist()    # Support for each label
        }

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'/scratch/likhithasapu/{model_name}',  # Output directory for the model
        num_train_epochs=30,                              # Number of training epochs
        per_device_train_batch_size=16,                    # Batch size for training
        per_device_eval_batch_size=32,                    # Batch size for evaluation
        warmup_steps=500,                                 # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                                # Strength of weight decay
        logging_steps=100,
        report_to="wandb",
        evaluation_strategy="epoch",                      # Evaluate every epoch
        save_strategy="epoch",                            # Save checkpoints every epoch
        load_best_model_at_end=True,                      # Load the best model when finished training
        metric_for_best_model="accuracy",                 # Use accuracy score to determine the best model
    )

    # Initialize the Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   # Use validation dataset for evaluation
        compute_metrics=compute_metrics,  # Adding the metrics function
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate()

    # Print evaluation results
    print("Validation Evaluation results:", eval_results)

    # Evaluate the model on the test set
    test_results = trainer.evaluate(test_dataset)

    # Print test set evaluation results
    print("Test Evaluation results:", test_results)
    
    tokenizer.push_to_hub("xlmr-pun-detection")
    model.push_to_hub("xlmr-pun-detection")
    
    
    

for random_state in [None]:
    train_model(random_state)

# # Push model to the Hugging Face Hub
# tokenizer.push_to_hub("xlmr-pun-detection")
# model.push_to_hub("xlmr-pun-detection")

