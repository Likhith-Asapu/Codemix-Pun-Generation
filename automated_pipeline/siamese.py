import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import json

# Load data from JSON
with open('compatibility_scores.json', 'r') as f:
    data = json.load(f)
    
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

# Dataset and DataLoader
class CompatibilityDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features1 = []
        self.features2 = []
        self.labels = []
        for item in data:
            pun_word = item["Pun_word"]
            alt_word = item["Alternate_word"]
            
            # Calculate embeddings and POS one-hot encodings for each word
            pun_embedding = get_embedding(pun_word).flatten()   # (768,)
            alt_embedding = get_embedding(alt_word).flatten()   # (768,)
            pun_pos_onehot = pos_to_onehot[item["Pun_word_pos"]]  # (7,)
            alt_pos_onehot = pos_to_onehot[item["Alternate_word_pos"]]  # (7,)

            # Concatenate embeddings and one-hot POS vectors
            pun_features = np.concatenate((pun_embedding, pun_pos_onehot))  # (775,)
            alt_features = np.concatenate((alt_embedding, alt_pos_onehot))  # (775,)

            # Append to features and labels
            self.features1.append(pun_features)
            self.features2.append(alt_features)
            self.labels.append(item["compatibility_score"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.features1[idx], dtype=torch.float32),
                torch.tensor(self.features2[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

# Define the Siamese Neural Network model
class SiameseSubnetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(SiameseSubnetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class SiameseCompatibilityModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        super(SiameseCompatibilityModel, self).__init__()
        # Define the Siamese subnetwork
        self.siamese_subnetwork = SiameseSubnetwork(input_size, hidden_size1, hidden_size2)
        # Define the final layer
        self.fc_final = nn.Linear(hidden_size2 * 2, output_size)
        
    def forward(self, x1, x2):
        # Pass both inputs through the Siamese subnetwork
        output1 = self.siamese_subnetwork(x1)
        output2 = self.siamese_subnetwork(x2)
        
        # Concatenate the outputs from both subnetworks
        combined_output = torch.cat((output1, output2), dim=1)
        
        # Pass the concatenated output through the final layer
        score = self.fc_final(combined_output)
        return score

# Initialize dataset
dataset = CompatibilityDataset(data)

# Split the dataset
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1)

# Model, loss, and optimizer
input_size = 775  # 768 + 7 (word embedding + POS one-hot encoding)
hidden_size1 = 512
hidden_size2 = 256
model = SiameseCompatibilityModel(input_size, hidden_size1, hidden_size2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Early stopping parameters
num_epochs = 100
early_stop_threshold = 0.0001
patience = 5  # Number of epochs to wait for improvement
best_eval_loss = float('inf')
no_improvement_epochs = 0

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for features1, features2, labels in train_loader:
        features1, features2, labels = features1.to(device), features2.to(device), labels.to(device)  # Move to GPU
        optimizer.zero_grad()
        outputs = model(features1, features2)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Evaluate model
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for features1, features2, labels in eval_loader:
            features1, features2, labels = features1.to(device), features2.to(device), labels.to(device)  # Move to GPU
            outputs = model(features1, features2)
            loss = criterion(outputs.squeeze(), labels)
            eval_loss += loss.item()
    eval_loss /= len(eval_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Eval Loss: {eval_loss:.4f}')
    
    # Early stopping check
    if eval_loss < best_eval_loss - early_stop_threshold:
        best_eval_loss = eval_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print("Early stopping triggered")
            break

# Save the model after training
model_save_path = "siamese_compatibility_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model from the saved file
loaded_model = SiameseCompatibilityModel(input_size, hidden_size1, hidden_size2).to(device)
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

    # Predict using loaded model
    features1 = torch.tensor(pun_features, dtype=torch.float32).to(device)  # Move to GPU
    features2 = torch.tensor(alt_features, dtype=torch.float32).to(device)  # Move to GPU
    with torch.no_grad():
        score = loaded_model(features1.unsqueeze(0), features2.unsqueeze(0))  # Add batch dimension
    return score.item()

# Example prediction with loaded model
print("Predicted compatibility score:", predict_compatibility("चेक", "chuck", "NOUN", "NOUN"))
print("Predicted compatibility score:", predict_compatibility("चेक", "chick", "NOUN", "NOUN"))
print("Predicted compatibility score:", predict_compatibility("चेक", "choke", "NOUN", "NOUN"))
