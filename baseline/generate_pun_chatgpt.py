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

file_path = "/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/final_data_2.json"
with open(file_path, "r") as file:
    data = json.load(file)

pun_data = []

for row in tqdm(data, total = len(data)):
    pun_word = row['Pun_word']
    alternate_word = row['Alternate_word']

    # response = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant. Try to generate creative puns. Make sure to write the English words in roman script and Hindi words in Devanagari."},
    #         {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'ढाई' and the alternate word is english word 'die'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
    #         {"role": "assistant", "content": "मेरा watch 2 and 2.30 के बीच stuck है. It's a do or ढाई situation."},
    #         {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'दूध' and the alternate word is english word 'dude'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
    #         {"role": "assistant", "content": "American milk ने Indian milk से क्या कहा? “What's up दूध?"},
    #         {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'बेटा' and the alternate word is english word 'beta'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
    #         {"role": "assistant", "content": "A daughter is the perfect child. A son is just a बेटा version."},
    #         {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'स्नान' and the alternate word is english word 'none'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
    #         {"role": "assistant", "content": "I really don't care कि कौन प्रतिदिन bath करता है।...Its स्नान of my business..."},
    #         {"role": "user", "content": f"Generate a hinglish pun where the pun word is the hindi word '{pun_word}' and the alternate word is english word '{alternate_word}'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
    #         ],
    #     temperature=1,
    # )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Try to generate creative puns. Make sure to write the English words in roman script and Hindi words in Devanagari."},
            {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'ढाई'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
            {"role": "assistant", "content": "मेरा watch 2 and 2.30 के बीच stuck है. It's a do or ढाई situation."},
            {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'दूध'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
            {"role": "assistant", "content": "American milk ने Indian milk से क्या कहा? “What's up दूध?"},
            {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'बेटा'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
            {"role": "assistant", "content": "A daughter is the perfect child. A son is just a बेटा version."},
            {"role": "user", "content": "Generate a hinglish pun where the pun word is the hindi word 'स्नान'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
            {"role": "assistant", "content": "I really don't care कि कौन प्रतिदिन bath करता है।...Its स्नान of my business..."},
            {"role": "user", "content": f"Generate a hinglish pun where the pun word is the hindi word '{pun_word}'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."},
            ],
        temperature=1,
    )
    
    row['baseline'] = response.choices[0].message.content
    pun_data.append(row)
    
# Write the modified data back to a new file
with open("/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/puns.json", "w") as file:
    json.dump(pun_data, file, indent=4, ensure_ascii=False)