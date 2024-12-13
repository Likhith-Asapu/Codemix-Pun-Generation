# read csv file and sample 50 words from first 1000 words randomly from the file

import pandas as pd
import Levenshtein
from Levenshtein import distance
from tqdm import tqdm
import json
pos_tags = ['NOUN','VERB','ADJ','PROPN','ADV','X','NUM']

hindi_data = pd.read_csv('hindi_words.csv')
hindi_data = hindi_data[hindi_data['pos'].isin(pos_tags)]
hindi_data = hindi_data[hindi_data['syns'] == True]
print("Number of words in Hindi data:", hindi_data.shape[0])
hindi_data = hindi_data[:5000]

# Sample 50 words randomly from the data
sampled_data = hindi_data.sample(800)
sampled_data = sampled_data.reset_index(drop=True)

english_data = pd.read_csv('english_words.csv')
english_data = english_data[english_data['pos'].isin(pos_tags)]
english_data = english_data[english_data['syns'] == True]

data = []
    
for index, row in tqdm(sampled_data.iterrows(), total=sampled_data.shape[0]):
    hindi_word = row['word']
    hindi_ipa = row['ipa_edited']
    word_data = {
        "Pun_word": hindi_word,
        "Pun_word_ipa": hindi_ipa,
        "Pun_word_pos": row['pos'],
        "Alternate_word": [],
    }
    for english_index, english_row in english_data.iterrows():
        english_word = english_row['word']
        english_ipa = english_row['ipa_edited']
        distance = Levenshtein.distance(hindi_ipa, english_ipa, weights=(2, 2, 1))
        if distance <= 1:
            if distance == 1 and hindi_ipa[0] != english_ipa[0]:
                continue
            change = False
            if distance == 1 and len(hindi_ipa) == len(english_ipa):
                for i in range(len(hindi_ipa)):
                    if hindi_ipa[i] != english_ipa[i]:
                        if hindi_ipa[i] in ['a', 'e', 'i', 'o', 'u'] and english_ipa[i] not in ['a', 'e', 'i', 'o', 'u']:
                            change = True
                        elif hindi_ipa[i] not in ['a', 'e', 'i', 'o', 'u'] and english_ipa[i] in ['a', 'e', 'i', 'o', 'u']:
                            change = True
                        elif hindi_ipa[i] not in ['a', 'e', 'i', 'o', 'u'] and english_ipa[i] not in ['a', 'e', 'i', 'o', 'u']:
                            change = True
            if change:
                continue
                        
            word_data["Alternate_word"].append({
                "word": english_word,
                "ipa": english_ipa,
                "pos": english_row['pos']
            })
            
    if len(word_data["Alternate_word"]) >= 3:
        data.append(word_data)
        
print("Number of words with at least 3 compatible words:", len(data))

with open('pun_alternate_word_2.json', 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
        
    
    