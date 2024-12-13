import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import re
from tqdm import tqdm


# Load data from JSON
with open('Data/labels.json', 'r') as f:
    data = json.load(f)

# Assuming the JSON file has 'text' and 'label' keys
texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Convert labels to a readable format
label_to_text = {0: "Not a Pun", 1: "Pun"}
text_to_label = {v: k for k, v in label_to_text.items()}

# Initialize the tokenizer and model
model_name = 'ai4bharat/Airavata'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
# Few-shot prompt creation

def const_prompt():
    return """
    Classify the following sentences as Pun or Not a Pun.

    Sentence: My son asked me if I'm familiar with computer programming. I said, \"Of course, I know बेटा testing!
    Label: Pun

    Sentence: When you make a donation, it's not just a दान, it's a deed well done!
    Label: Not a Pun

    Sentence: दोस्तों के साथ खेलना हमेशा खुल रहता है, क्योंकि वो हमेशा cool होते हैं!
    Label: Not a Pun

    Sentence: After watching a cooking show, the नयी chef said, "I am such a रूकी in the kitchen, I burnt the water!"
    Label: Not a Pun

    Sentence: Why did the tree go to therapy? Because it had too many वन-sided conversations.
    Label: Pun
    
    Sentece: scissors खोने का खतरा बरकरार रहे, पर उनकी shiny blades ने मेरा attention काफी कैची बना दिया।
    Label: Pun
    """

def create_prompt(input_text, few_shot_examples):
    """
    Creates a prompt for the Llama model by adding few-shot examples and a new input.
    """
    prompt = const_prompt()

    # for example_text, example_label in few_shot_examples:
    #     prompt += f"Sentence: {example_text}\nLabel: {label_to_text[example_label]}\n\n"

    prompt += f"Sentence: {input_text}\nLabel:"
    return prompt

# Function to extract label from the generated text
def extract_label(generated_text):
    """
    Extracts the label ('Pun' or 'Not a Pun') from the generated text.
    """
    generated_text = generated_text.split(const_prompt())[-1]
    # Extract the part after 'Label:'
    match = re.search(r'Label:\s*(Pun|Not a Pun)', generated_text, re.IGNORECASE)
    if match:
        label_text = match.group(1)
        return text_to_label.get(label_text.strip(), None)
    else:
        # Try to find 'Pun' or 'Not a Pun' in the generated text
        if 'Not a Pun' in generated_text:
            return text_to_label['Not a Pun']
        elif 'Pun' in generated_text:
            return text_to_label['Pun']
        else:
            return None  # Unable to extract label

# Tokenize inputs and prepare for few-shot prompting

def inference_model(random_state): 
    # Step 1: Split 70% training and 30% (validation + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=random_state
    )

    # Step 2: Split the remaining 30% into validation (20%) and test (10%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.33, random_state=random_state
    )
    # Select a few-shot subset from the training data
    few_shot_examples = list(zip(train_texts[:5], train_labels[:5]))


    def predict_few_shot(texts):
        prompts = [create_prompt(text, few_shot_examples) for text in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=True,  # Set to True for sampling-based generation
            temperature=1.5,  # Retaining temperature setting for diversity
            top_p=0.9,        # For nucleus sampling
            num_beams=5,      # Enable beam search
            pad_token_id=tokenizer.eos_token_id
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

    # Run predictions on the test set in batches
    def run_test_predictions(test_texts, test_labels):
        batch_size = 8
        predictions = []
        for i in tqdm(range(0, len(test_texts), batch_size)):
            batch_texts = test_texts[i:i + batch_size]
            generated_outputs = predict_few_shot(batch_texts)
            batch_predictions = [extract_label(output) for output in generated_outputs]
            print(batch_predictions)
            for idx, predicted_label in enumerate(batch_predictions):
                if predicted_label is None:
                    # If unable to extract label, assign default value
                    print(f"Warning: Unable to extract label for sample {i + idx}. Assigning 'Not a Pun' by default.")
                    predicted_label = text_to_label['Not a Pun']

                predictions.append(predicted_label)

        # Convert lists to tensors
        true_labels = torch.tensor(test_labels)
        predicted_labels = torch.tensor(predictions)

        # Compute evaluation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Print evaluation results
        print("\nEvaluation Results on Test Set:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy, precision, recall, f1

    # Run predictions on the test set
    test_accuracy, test_precision, test_recall, test_f1 = run_test_predictions(test_texts, test_labels)

    # Run predictions on the validation set
    val_accuracy, val_precision, val_recall, val_f1 = run_test_predictions(val_texts, val_labels)
    
    return test_accuracy, test_precision, test_recall, test_f1, val_accuracy, val_precision, val_recall, val_f1

# Run the inference model
Results = {}

for random_state in [0, 42, 10]:
    test_accuracy, test_precision, test_recall, test_f1, val_accuracy, val_precision, val_recall, val_f1 = inference_model(random_state)
    # convert to percentage
    test_accuracy = test_accuracy * 100
    test_precision = test_precision * 100
    test_recall = test_recall * 100
    test_f1 = test_f1 * 100
    val_accuracy = val_accuracy * 100
    val_precision = val_precision * 100
    val_recall = val_recall * 100
    val_f1 = val_f1 * 100
    Results[random_state] = {
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_accuracy': val_accuracy,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_accuracy': test_accuracy,
    }
    
# Print the average results and standard deviation for these results across different random seeds chooose only top 3 random seeds based on test accuracy
import numpy as np

# Sort results by test_accuracy in descending order and select top 3 seeds
top_3_seeds = sorted(Results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]

# Calculate the average and standard deviation for each metric across the top 3 seeds
metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1',
           'val_accuracy', 'val_precision', 'val_recall', 'val_f1']

averages = {}
std_devs = {}

# Extract values for each metric across the top 3 seeds
for metric in metrics:
    metric_values = [result[1][metric] for result in top_3_seeds]
    averages[metric] = np.mean(metric_values)
    std_devs[metric] = np.std(metric_values) 

# Print the average and standard deviation results
print(f"Average results for top 3 seeds based on test accuracy: {model_name}")
for metric in metrics:
    print(f"{metric}: Average = {averages[metric]:.4f}, Standard Deviation = {std_devs[metric]:.4f}")

    
    
    
