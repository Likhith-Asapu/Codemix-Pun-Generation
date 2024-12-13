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
        self.features = []
        self.labels = []
        for item in data:
            pun_word = item["Pun_word"]
            alt_word = item["Alternate_word"]
            
            # Calculate embeddings for each word
            pun_embedding = get_embedding(pun_word).flatten()   # (768,)
            alt_embedding = get_embedding(alt_word).flatten()   # (768,)

            # Concatenate embeddings only (no POS one-hot vectors)
            combined_features = np.concatenate((pun_embedding, alt_embedding))  # (1536,)

            # Append to features and labels
            self.features.append(combined_features)
            self.labels.append(item["compatibility_score"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

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

# Initialize dataset
dataset = CompatibilityDataset(data)

# Split the dataset
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1)

# Model, loss, and optimizer
input_size = 1536  # 768 * 2 (only BERT embeddings)
hidden_size1 = 512
hidden_size2 = 256
model = CompatibilityModel(input_size, hidden_size1, hidden_size2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Early stopping parameters
num_epochs = 100
early_stop_threshold = 0.001
patience = 5  # Number of epochs to wait for improvement
best_eval_loss = float('inf')
no_improvement_epochs = 0

# Training loop with early stopping and RMSE calculation
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_rmse = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)  # Move to GPU
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += ((outputs.squeeze() - labels) ** 2).item()

    train_loss /= len(train_loader)
    train_rmse = (train_rmse / len(train_loader)) ** 0.5  # RMSE calculation

    # Evaluate model
    model.eval()
    eval_loss = 0
    eval_rmse = 0
    with torch.no_grad():
        for features, labels in eval_loader:
            features, labels = features.to(device), labels.to(device)  # Move to GPU
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            eval_loss += loss.item()
            eval_rmse += ((outputs.squeeze() - labels) ** 2).item()

    eval_loss /= len(eval_loader)
    eval_rmse = (eval_rmse / len(eval_loader)) ** 0.5  # RMSE calculation
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Train RMSE: {train_rmse:.4f}, Eval RMSE: {eval_rmse:.4f}')
    
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
model_save_path = "bert_compatibility_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model from the saved file
loaded_model = CompatibilityModel(input_size, hidden_size1, hidden_size2).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_save_path}")

# Testing the model (example)
def predict_compatibility(pun_word, alt_word):
    # Create feature vector for prediction using BERT embeddings only
    pun_embedding = get_embedding(pun_word).flatten()
    alt_embedding = get_embedding(alt_word).flatten()
    combined_features = np.concatenate((pun_embedding, alt_embedding))

    # Predict using loaded model
    features = torch.tensor(combined_features, dtype=torch.float32).to(device)  # Move to GPU
    with torch.no_grad():
        score = loaded_model(features)
    return score.item()

# Example prediction with loaded model
print("Predicted compatibility score:", predict_compatibility("नीत", "neat"))
