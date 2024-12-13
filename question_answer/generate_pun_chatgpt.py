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
MODEL = "gpt-3.5-turbo"

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
            {"role": "system", "content": """
              Follow these steps to generate a pun:
              First generate a very short phrase(which can be responses to questions) with the alternate word. For example if the alternate word is 'guy' the generated sentence is "A cool guy". The replace the alternate word with the pun word. For example if the pun word is 'गाय' the new sentence would become "A cool गाय". Then generate a pun question for this "What do you call a cow wearing sunglasses?". Write it in the below format ill give pairs of pun words and alteranate word

              Generated Small Phrase: A cool guy
              Replace Pun Word: A cool गाय
              Generated Question: What do you call a cow wearing sunglasses?"""},
            
            {"role": "user", "content": "The pair is Pun Word = 'लाख'(meaning lakh) and Alternate Word = luck"},
            {"role": "assistant", "content": """Generated Small Phrase: A stroke of luck\nReplace Pun Word: A stroke of लाख\nGenerated Question: What did the lottery winner say about their fortune?"""},
            {"role": "user", "content": "The pair is Pun Word = 'पीपल'(meaning tree) and Alternate Word = people"},
            {"role": "assistant", "content": """Generated Small Phrase: A group of people\nReplace Pun Word: A group of पीपल\nGenerated Question: What do you call a forest gathering for a meeting?\n"""},
            {"role": "user", "content": "The pair is Pun Word = 'बिक'(meaning sold) and Alternate Word = big"},
            {"role": "assistant", "content": """Generated Small Phrase: A big deal\nReplace Pun Word: A बिक deal\nGenerated Question: What do you call something that was successfully sold for a high price?"""},
            {"role": "user", "content": "The pair is Pun Word = 'वन'(meaning Forest) and Alternate Word = won"},
            {"role": "assistant", "content": """Generated Small Phrase: A victory won\nReplace Pun Word: A victory वन\nGenerated Question: What do you call a forest that achieved success?"""},
            
            {"role": "user", "content": f"The pair is Pun Word = '{pun_word}'(meaning {pun_word_meaning}) and Alternate Word = {alternate_word}"},
            ],
        temperature=1,
    )
    
    row['Response'] = response.choices[0].message.content
    phrase_pattern = r"Generated Small Phrase:\s*(.*)"
    pun_word_pattern = r"Replace Pun Word:\s*(.*)"
    question_pattern = r"Generated Question:\s*(.*)"

    # Extract using regex
    generated_phrase = re.search(phrase_pattern, row['Response']).group(1)
    replace_pun_word = re.search(pun_word_pattern, row['Response']).group(1)
    generated_question = re.search(question_pattern, row['Response']).group(1)
    
    row['Sentence_chosen'] = generated_question + "\n" + replace_pun_word
    pun_data.append(row)
    
# Write the modified data back to a new file
with open("Chatgpt_pun.json", "w") as file:
    json.dump(pun_data, file, indent=4, ensure_ascii=False)