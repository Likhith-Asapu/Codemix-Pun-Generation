import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the tokenizer and model with `output_hidden_states=True`
model_name = 'likhithasapu/gcm-xlmr'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Tokenize a sample text for embedding visualization (taking the first example as a demo)
sample_text = "My son asked me if I m familiar with computer programming I said Of course I know बेटा testing"
inputs = tokenizer(sample_text, return_tensors="pt")

# Forward pass to get the embeddings and hidden states
outputs = model(**inputs)

# Extract the second-to-last layer embeddings from hidden states
hidden_states = outputs.hidden_states  # shape: (num_layers, batch_size, seq_len, hidden_dim)
second_last_layer_embeddings = hidden_states[-7].squeeze(0)  # shape: (seq_len, hidden_dim)

# Convert to numpy
embeddings = second_last_layer_embeddings.detach().numpy()

# Compute cosine similarity between embeddings
similarity_matrix = cosine_similarity(embeddings)

# Tokenize to retrieve the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())

# Plotting the similarity matrix
plt.figure(figsize=(20, 16))  # Increased figure size for better readability

# Use a font that supports Hindi characters
hindi_font = font_manager.FontProperties(fname='NotoSansDevanagari.ttf')

# Plot the heatmap
sns.heatmap(similarity_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis', annot=True, fmt=".2f", cbar=True)

# Set title and labels, using Hindi-supporting font
plt.title(f"Cosine Similarity between Second-to-Last Layer Embeddings for: {sample_text}", fontproperties=hindi_font, fontsize=16)
plt.xlabel("Tokens", fontproperties=hindi_font, fontsize=14)
plt.ylabel("Tokens", fontproperties=hindi_font, fontsize=14)

# Show the plot with appropriate font settings for Hindi
plt.xticks(rotation=90, fontproperties=hindi_font, fontsize=12)
plt.yticks(rotation=0, fontproperties=hindi_font, fontsize=12)
plt.savefig("similarity.png")
