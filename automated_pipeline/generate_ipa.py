import json
with open('compatibility_scores.json', 'r') as f:
    data = json.load(f)
    
# read data from csv
import pandas as pd
english_words = pd.read_csv('english_words.csv')

english_ipa = {}
for index, row in english_words.iterrows():
    english_ipa[row['word']] = row['ipa_edited']
    
hindi_words = pd.read_csv('hindi_words.csv')
hindi_ipa = {}
for index, row in hindi_words.iterrows():
    hindi_ipa[row['word']] = row['ipa_edited']
    
for row in data:
    row['Pun_word_ipa'] = hindi_ipa.get(row['Pun_word'])
    row['Alternate_word_ipa'] = english_ipa.get(row['Alternate_word'])
    
with open('compatibility_scores.json', 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
