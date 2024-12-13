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

file_path = "/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/puns.json"
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
            {"role": "system", "content": """
              Follow these steps to generate a pun:
              First generate a very short phrase(which can be a response to questions such as Noun Phrase) with the given alternate word. For example if the alternate word is 'guy' the generated sentence is "A cool guy". The replace the alternate word with the pun word. For example if the pun word is 'गाय' the new sentence would become "A cool गाय". Then generate a pun question for this "What do you call a cow wearing sunglasses?". Write it in the below format ill give pairs of pun words and alteranate word

              Generated Small Phrase: A cool guy
              Replace Pun Word: A cool गाय
              Generated Question: What do you call a cow wearing sunglasses?
              Generated Translated Question: What do you call a cow जो sunglasses पहनती है?"""},
            
            {"role": "user", "content": "The pair is Pun Word = 'लाख'(meaning lakh) and Alternate Word = luck"},
            {"role": "assistant", "content": """Generated Small Phrase: A stroke of luck\nReplace Pun Word: A stroke of लाख\nGenerated Question: What do you call a huge fortune that strikes suddenly?\nGenerated Translated Question: What do you call a huge fortune जो suddenly हासिल होता हे?"""},
            {"role": "user", "content": "The pair is Pun Word = 'पीपल'(meaning tree) and Alternate Word = people"},
            {"role": "assistant", "content": """Generated Small Phrase: A group of people\nReplace Pun Word: A group of पीपल\nGenerated Question: What do you call a gathering under a sacred tree?\nGenerated Translated Question: What do you call a समूह under a sacred पेड़?"""},
            {"role": "user", "content": "The pair is Pun Word = 'बिक'(meaning sold) and Alternate Word = big"},
            {"role": "assistant", "content": """Generated Small Phrase: A big sale\nReplace Pun Word: A बिक sale\nGenerated Question: What do you call a huge sale where everything is sold out?\nGenerated Translated Question: What do you call a huge sale जहां सब कुछ बिक जाता है?"""},
            {"role": "user", "content": "The pair is Pun Word = 'वन'(meaning Forest) and Alternate Word = won"},
            {"role": "assistant", "content": """Generated Small Phrase: He won the game\nReplace Pun Word: He वन the game\nGenerated Question: What do you call it when someone conquers a forest in a game?\nGenerated Translated Question: What do you call it जब कोई खेल में जंगल conquer करता है?"""},
            {"role": "user", "content": "The pair is Pun Word = 'धन'(meaning money) and Alternate Word = done"},
            {"role": "assistant", "content": """Generated Small Phrase: All done\nReplace Pun Word: All धन\nGenerated Question: What do you call it when you've earning making a lot of money?\nGenerated Translated Question: What do you call it जब तुमने बोहोत सारा पैसे कमा लिया हो?"""},
            
            {"role": "user", "content": f"The pair is Pun Word = '{pun_word}'(meaning {pun_word_meaning}) and Alternate Word = {alternate_word}"},
            ],
        temperature=1,
    )
    
    text_response = response.choices[0].message.content
    phrase_pattern = r"Generated Small Phrase:\s*(.*)"
    pun_word_pattern = r"Replace Pun Word:\s*(.*)"
    question_pattern = r"Generated Question:\s*(.*)"
    translated_question_pattern = r"Generated Translated Question:\s*(.*)"

    # Extract using regex
    generated_phrase = re.search(phrase_pattern, text_response).group(1)
    replace_pun_word = re.search(pun_word_pattern, text_response).group(1)
    generated_question = re.search(question_pattern, text_response).group(1)
    transalated_question = re.search(translated_question_pattern, text_response).group(1)
    
    row['task2'] = transalated_question + "\n" + replace_pun_word
    pun_data.append(row)
    
# Write the modified data back to a new file
with open("/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/puns_2.json", "w") as file:
    json.dump(pun_data, file, indent=4, ensure_ascii=False)