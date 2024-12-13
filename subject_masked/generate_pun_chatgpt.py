import re
from tqdm import tqdm
import json
import random
# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI
import os
import json
import time
# Example OpenAI Python library request
MODEL = "gpt-4o"

file_path = "../samples.json"
with open(file_path, "r") as file:
    data = json.load(file)

pun_data = []

for row in tqdm(data, total = len(data)):
    pun_word = row['Pun_word']
    alternate_word = row['Alternate_word']
    pun_word_meaning = row['Pun_word_meaning']

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a helpful assistant. 
              Given a pair of pun words and alternate words use the following steps to generate a Hindi-English codemixed pun. First generate a short sentence ENDING with the alternate word. Then replace the alternate word with the pun word. Then replace the subject of the sentence to add relevance to the pun. 
                Example:
                The pair is Pun Word = 'लाख'(meaning lakh) and Alternate Word = luck

                Short Sentence: The man attributed all his success to luck
                Replace Alternate Word: The man attributed all his success to लाख
                Mask the Subject: [MASK] attributed all his success to लाख
                Pun Sentence: The भाग्यशाली अमीर व्यापारी attributed all his success to लाख
              """},
            {"role": "user", "content": "The pair is Pun Word = 'सच'(meaning truth) and Alternate Word = such"},
            {"role": "assistant", "content": """Short Sentence: She was never faced with such challenges before.\nReplace Alternate Word: She was never faced with सच challenges before.\nMask the Subject: [MASK] was never faced with सच challenges before.\nPun Sentence: The बेईमान वकील was never faced with सच challenges before."""},
            {"role": "user", "content": "The pair is Pun Word = 'पीपल'(meaning tree) and Alternate Word = people"},
            {"role": "assistant", "content": """Short Sentence: The park was filled with people enjoying the evening.\nReplace Alternate Word: The park was filled with पीपल enjoying the evening.\nMask the Subject: [MASK] was filled with पीपल enjoying the evening.\nPun Sentence: The वन was filled with पीपल enjoying the evening."""},
            {"role": "user", "content": "The pair is Pun Word = 'लाख'(meaning lakh) and Alternate Word = luck"},
            {"role": "assistant", "content": """Short Sentence: She felt like all her success was due to luck.\nReplace Alternate Word: She felt like all her success was due to लाख.\nMask the Subject: [MASK] felt like all her success was due to लाख.\nPun Sentence: The करोड़पति उद्यमी felt like all her success was due to लाख."""},
            {"role": "user", "content": "The pair is Pun Word = 'धन'(meaning money) and Alternate Word = done"},
            {"role": "assistant", "content": """Short Sentence: The deal was finally done after long negotiations.\nReplace Alternate Word: The deal was finally धन after long negotiations.\nMask the Subject: [MASK] was finally धन after long negotiations.\nPun Sentence: The चालाक व्यापारी was finally धन after long negotiations."""},
            {"role": "user", "content": f"The pair is Pun Word = '{pun_word}'(meaning {pun_word_meaning}) and Alternate Word = {alternate_word}"},
            {"role": "user", "content": "The pair is Pun Word = 'दान'(meaning donation) and Alternate Word = done"},
            {"role": "assistant", "content": """Short Sentence: The project was finally done after months of hard work.\nReplace Alternate Word: The project was finally दान after months of hard work.\nMask the Subject: [MASK] was finally दान after months of hard work.\nPun Sentence: The सामाजिक कार्यकर्ता was finally दान after months of hard work."""},
            {"role": "user", "content": f"The pair is Pun Word = '{pun_word}'(meaning {pun_word_meaning}) and Alternate Word = {alternate_word}"},
            
            ],
        temperature=1,
    )
    
    row['Response'] = response.choices[0].message.content
    phrase_pattern = r"Pun Sentence:\s*(.*)"

    # Extract using regex
    generated_phrase = re.search(phrase_pattern, row['Response']).group(1)
    
    row['Sentence_chosen'] = generated_phrase
    pun_data.append(row)
    
# Write the modified data back to a new file
with open("Chatgpt_pun_codemixed.json", "w") as file:
    json.dump(pun_data, file, indent=4, ensure_ascii=False)