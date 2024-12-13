import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import json
from tqdm import tqdm

from googletrans import Translator

translator = Translator()

# Define POS tags and their one-hot encoding mapping
pos_tags = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV', 'X', 'NUM']
pos_to_onehot = {pos: np.eye(len(pos_tags))[i] for i, pos in enumerate(pos_tags)}
print("POS tag one-hot encoding:", pos_to_onehot)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bertVectorModel = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)

# Function to compute BERT embeddings
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    outputs = bertVectorModel(**inputs)
    return outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).cpu().detach().numpy()

# Define Neural Network model
class CompatibilityModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(CompatibilityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = 1550  # 768 * 2 + 7 * 2
hidden_size1 = 512
hidden_size2 = 256
model_save_path = "compatibility_model_2.pth"
# Load the model from the saved file
loaded_model = CompatibilityModel(input_size, hidden_size1, hidden_size2).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_save_path}")

# Testing the model (example)
def predict_compatibility(pun_word, alt_word, pun_pos, alt_pos):
    # Create feature vector for prediction
    pun_embedding = get_embedding(pun_word).flatten()
    alt_embedding = get_embedding(alt_word).flatten()
    pun_pos_onehot = pos_to_onehot[pun_pos]
    alt_pos_onehot = pos_to_onehot[alt_pos]
    pun_features = np.concatenate((pun_embedding, pun_pos_onehot))
    alt_features = np.concatenate((alt_embedding, alt_pos_onehot))
    combined_features = np.concatenate((pun_features, alt_features))

    # Predict using loaded model
    features = torch.tensor(combined_features, dtype=torch.float32).to(device)  # Move to GPU
    with torch.no_grad():
        score = loaded_model(features)
    return score.item()

# open the file
with open('pun_alternate_word_2.json', 'r') as f:
    data = json.load(f)

final_data = []
for item in tqdm(data, total=len(data)):
    pun_word = item["Pun_word"]
    pun_pos = item["Pun_word_pos"]
    
    pun_word_translation = translator.translate(pun_word).text
    
    # get max compatibility score for all alternate words
    max_score = -1
    max_alt = None
    for alt in item["Alternate_word"]:
        alt_word = alt["word"]
        alt_pos = alt["pos"]
        if pun_word_translation.lower() == alt_word.lower():
            continue
        score = predict_compatibility(pun_word, alt_word, pun_pos, alt_pos)
        if score > max_score:
            max_score = score
            max_alt = alt
    
    print(f"Pun word: {pun_word}, Alternate word: {max_alt['word']}, Compatibility score: {max_score}")
    
    final_data.append({
        "Pun_word": pun_word, 
        "Pun_word_pos": pun_pos,
        "Alternate_word": max_alt["word"],
        "Alternate_word_pos": max_alt["pos"],
        "Compatibility_score": max_score
    })
    
# Save the final data
with open('final_data_2.json', 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)


