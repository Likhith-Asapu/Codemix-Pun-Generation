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

file_path = "/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/puns_2.json"
with open(file_path, "r") as file:
    data = json.load(file)

pun_data = []

for row in tqdm(data, total = len(data)):
    pun_word = row['Pun_word']
    alternate_word = row['Alternate_word']
    pun_word_meaning = row['Pun_word_translation']

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a helpful assistant. 
              Given a pair of pun words and alternate words use the following steps to generate a Hindi-English codemixed pun. First generate a short sentence ending with the alternate word. Then replace the alternate word with the pun word. Then replace the subject of the sentence to add relevance to the pun. Then translate the subject to codemixed Hindi-English version.
                Example:
                The pair is Pun Word = 'लाख'(meaning lakh) and Alternate Word = luck

                Short Sentence: The man attributed all his success to luck
                Replace Alternate Word: The man attributed all his success to लाख
                Mask the Subject: [MASK] attributed all his success to लाख
                Pun Sentence: The lucky rich businessman attributed all his success to लाख
                Translated Sentence: The lucky अमीर businessman attributed all his सफलता to लाख.
              """},
            
            {"role": "user", "content": "The pair is Pun Word = 'बिक'(meaning sold) and Alternate Word = big"},
            {"role": "assistant", "content": """Short Sentence: The event was a big success.\nReplace Alternate Word: The event was a बिक success.\nMask the Subject: [MASK] was a बिक success.\nPun Sentence: The sold-out concert was a बिक success.\nTranslated Sentence: The sold-out संगीत सभा was a बिक success."""},
            {"role": "user", "content": "The pair is Pun Word = 'सच'(meaning truth) and Alternate Word = such"},
            {"role": "assistant", "content": """Short Sentence: She was never faced with such challenges before.\nReplace Alternate Word: She was never faced with सच challenges before.\nMask the Subject: [MASK] was never faced with सच challenges before.\nPun Sentence: The dishonest lawyer was never faced with सच challenges before.\nTranslated Sentence: The dishonest वकील was never faced with सच challenges before."""},
            {"role": "user", "content": "The pair is Pun Word = 'पीपल'(meaning tree) and Alternate Word = people"},
            {"role": "assistant", "content": """Short Sentence: The park was filled with people enjoying the evening.\nReplace Alternate Word: The park was filled with पीपल enjoying the evening.\nMask the Subject: [MASK] was filled with पीपल enjoying the evening.\nPun Sentence: The forest was filled with पीपल enjoying the evening.\nTranslated Sentence: The जंगल was filled with पीपल enjoying the evening."""},
            
            {"role": "user", "content": f"The pair is Pun Word = '{pun_word}'(meaning {pun_word_meaning}) and Alternate Word = {alternate_word}"},
            ],
        temperature=1,
    )
    
    text_response = response.choices[0].message.content
    phrase_pattern = r"Translated Sentence:\s*(.*)"

    # Extract using regex
    generated_phrase = re.search(phrase_pattern, text_response).group(1)
    
    row['task3'] = generated_phrase
    pun_data.append(row)
    
# Write the modified data back to a new file
with open("/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/puns_3.json", "w") as file:
    json.dump(pun_data, file, indent=4, ensure_ascii=False)