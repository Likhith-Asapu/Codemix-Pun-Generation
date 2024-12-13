import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

import re
from tqdm import tqdm
import json
import random

file_path = "Gemini_pun_1.json"
with open(file_path, "r") as file:
    data = json.load(file)

pun_data = []

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

generation_config = {
  "temperature": 0.75,
#   "top_p": 1,
#   "top_k": 1,
  "max_output_tokens": 2048,
}
model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings, generation_config=generation_config)

total_zero = 1
attempts = 0
while total_zero != 0 and attempts < 10:
  attempts += 1
  total_zero = 0
  pun_data = []
  for row in tqdm(data, total = len(data)):
    pun_word = row['Pun_word']
    alternate_word = row['Alternate_word']
    pun_word_meaning = row['Pun_word_meaning']
    
    pun_entry = {
        'Pun_word': pun_word,
        'Alternate_word': alternate_word,
        'Pun_word_meaning': pun_word_meaning,
        'Pun_word_pos': row['Pun_word_pos'],
        'Alternate_word_pos': row['Alternate_word_pos'],
    }
    
    if 'Sentence_chosen' not in row:
        pun_entry['Sentence_chosen'] = ''

    if len(pun_entry['Sentence_chosen']) != 0:
        continue

    response = model.generate_content(["""Generate a hindi-english codemixed pun where the pun word is the hindi word 'ढाई' and the alternate word is english word 'die'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari.
    मेरा watch 2 and 2.30 के बीच stuck है. It's a do or ढाई situation.

    Generate a hinglish pun where the pun word is the hindi word 'दूध' and the alternate word is english word 'dude'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari.
    "American milk ने Indian milk से क्या कहा? “What’s up दूध?"

    Generate a hinglish  pun where the pun word is the hindi word 'बेटा' and the alternate word is english word 'beta'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari.
    A daughter is the perfect child. A son is just a बेटा version.

    Generate a hinglish  pun where the pun word is the hindi word 'स्नान' and the alternate word is english word 'none'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari.
    I really don't care कि कौन प्रतिदिन bath करता है।...Its स्नान of my business...

    Generate a hinglish  pun where the pun word is the hindi word 'अंडा' and the alternate word is english word 'under'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari.
    Eggs चुराने वाले लोगों के साथ desi police क्या करती है? They tell them they are अंडा arrest."
    
    Similar to the above examples generate a hinglish pun where the pun word is the hindi word '{pun_word}' and the alternate word is english word '{alternate_word}'. Maintain a balanced ratio between english and hindi words. Make sure to write the English words in roman script and Hindi words in Devanagari."""
    ],safety_settings = safety_settings)

    try:
        sentence = response.text.lower().strip()
        pun_entry['Sentence_chosen'] = sentence
    except:
        pun_entry['Sentence_chosen'] = ""
    
    pun_data.append(pun_entry)

  print(f"No. of zero pun sentences = {total_zero}")

  # Write the modified data back to a new file
  with open(file_path, "w") as file:
      json.dump(pun_data, file, indent=4, ensure_ascii=False)
  with open(file_path, "r") as file:
      data = json.load(file)