import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

with open('../Data/final_data.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data if item['label'] != 2]
labels = [item['label'] for item in data if item['label'] != 2]

# Initialize the tokenizer and model
model_name = 'likhithasapu/gcm-xlmr-pun-detection'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Enable attention output

# Tokenize a sample text for attention visualization (taking the first example as a demo)
sample_text = "My son asked me if I m familiar with computer programming I said Of course I know बेटा testing"
inputs = tokenizer(sample_text, return_tensors="pt")

# Forward pass to get the attention weights
outputs = model(**inputs)

# Extract the attention weights from the last layer
attention = outputs.attentions[-1]  # shape: (batch_size, num_heads, seq_len, seq_len)
print(len(outputs.attentions))

# Since attention is multi-headed, you can take the mean of attention across heads
attention_mean = attention.mean(dim=1).squeeze(0).detach().numpy()  # shape: (seq_len, seq_len)
print(attention_mean.shape)

# Tokenize to retrieve the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())

# Plotting the attention weights
plt.figure(figsize=(20, 16))  # Increased figure size for better readability

# Use a font that supports Hindi characters
hindi_font = font_manager.FontProperties(fname='NotoSansDevanagari.ttf')

# Plot the heatmap
sns.heatmap(attention_mean, xticklabels=tokens, yticklabels=tokens, cmap='viridis', annot=True, fmt=".2f", cbar=True)

# Set title and labels, using Hindi-supporting font
plt.title(f"Attention weights for: {sample_text}", fontproperties=hindi_font, fontsize=16)
plt.xlabel("Tokens", fontproperties=hindi_font, fontsize=14)
plt.ylabel("Tokens", fontproperties=hindi_font, fontsize=14)

# Show the plot with appropriate font settings for Hindi
plt.xticks(rotation=90, fontproperties=hindi_font, fontsize=12)
plt.yticks(rotation=0, fontproperties=hindi_font, fontsize=12)
plt.savefig("attention_weights_1.png")
