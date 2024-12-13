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

total_zero = 1
attempts = 0

while total_zero != 0 and attempts < 10:
  attempts += 1
  total_zero = 0
  pun_data = []
  index = 0
  for row in tqdm(data, total = len(data)):
    pun_word = row['Pun_word']
    alternate_word = row['Alternate_word']
    pun_word_meaning = row['Pun_word_translation']
    pun_word_pos = row['Pun_word_pos']
    alternate_word_pos = row['Alternate_word_pos']
    if 'No_of_sentences' not in row:
      row['No_of_sentences'] = 0
      row['Candidates'] = []
    no_of_sentences = row['No_of_sentences']

    if no_of_sentences > 0:
      pun_data.append(row)
      continue

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a helpful assistant. Try to generate pun-style sentences containing the English words, don't use the roman transliterated hindi words. Make sure to write the English words in Roman script and Hindi words in Devanagari. And here are some pun-style sentences:
                                              मैं हमेशा करता हूँ अपने काम का Best, क्योंकि मैं हूँ एक Perfectionist.
                                              जब मैंने उसे देखा, मेरे दिल ने कहा, "Oh my God, she's Gorgeous!"
                                              क्या तुमने सुना? वह नया Rest-O-Bar बहुत ही शानदार है।
                                              मेरे दोस्त ने कहा, "Hey Bro, तुम बिल्कुल Rockstar लग रहे हो!"
                                              उस Restaurant का खाना इतना Delicious था, कि मैंने extra Order कर दिया।"""},
            {"role": "user", "content": "Generate 5 creative Hindi-English code-mixed pun-style sentences ending with the english word 'share'. Have the english word 'lion' as a context in each of these sentences. Ensure the word 'share' occurs after the word 'lion'. Avoid providing English translations. Make sure to write the English words in roman script and Hindi words in Devanagari. Maintain a balanced 1:1 ratio between hindi and english words."},
            {"role": "assistant", "content": """
              1. Jungle में lion का status high है, पर उसकी fame को भी एक अच्छा सा share चाहिए!
              2. Lion अपने territory को protect करता है, लेकिन कभी-कभी उसको अपनी success का share भी देना पड़ता है.
              3. अगर lion के साथ photo खिंचनी है, तो उसके साथ share भी करना पड़ेगा!
              4. Lion की leadership को challenge करना है, तो उसका power share भी जरूरी है!
              5. Lion की roar की आवाज सुनी है, तो उसका impact share करना भी तो पड़ेगा!
              """
            },
            {"role": "user", "content": "Generate 5 creative Hindi-English code-mixed pun-style sentences ending with the english word 'luck'. Have the english word 'money' as a context in each of these sentences. Ensure the word 'luck' occurs after the word 'money'. Avoid providing English translations. Make sure to write the English words in roman script and Hindi words in Devanagari. Maintain a balanced 1:1 ratio between hindi and english words."},
            {"role": "assistant", "content": """
              1. उसने कहा, "When I have enough money in my pocket, it feels like my luck is always on my side."
              2. मेरे friend ने मजाक में कहा, "Money can buy you almost everything, but you still need a bit of luck."
              3. आजकल money की कमी तो सबको है, पर luck से सब कुछ मिलता है।
              4. जितना भी invest करो, कभी न कभी तो मिलेगी तुम्हें money with a bit of luck!
              5. अगर तुम्हारे पास money नहीं है, तो बस तुम्हें चाहिए थोड़ा सा luck!
              """
            },
            {"role": "user", "content": "Generate 5 creative Hindi-English code-mixed pun-style sentences ending with the english word 'bat'. Have the english word 'chair' as a context in each of these sentences. Ensure the word 'bat' occurs after the word 'chair'. Avoid providing English translations. Make sure to write the English words in roman script and Hindi words in Devanagari. Maintain a balanced 1:1 ratio between hindi and english words."},
            {"role": "assistant", "content": """
              1. जब वह अपनी comfy chair पर बैठा, उसने सोचा, "I really need a better bat to improve my game."
              2. उसने अपनी favorite chair पर बैठते हुए कहा, "This chair is perfect, but I could really use a new bat."
              3. Chair पर आराम करते हुए उसने कहा, "Yaar, I wish I had a bat like that for my cricket matches."
              4. उसने अपनी chair पर बैठते हुए कहा, "This chair is great for relaxing, but I'm missing a good bat."
              5. जैसे ही वह chair से उठा, उसने अपने भाई से पूछा, "Have you seen my cricket bat around here?"
              """
            },
            {"role": "user", "content": "Generate 5 creative Hindi-English code-mixed pun-style sentences ending with the english word 'book'. Have the english word 'hunger' as a context in each of these sentences. Ensure the word 'book' occurs after the word 'hunger'. Avoid providing English translations. Make sure to write the English words in roman script and Hindi words in Devanagari. Maintain a balanced 1:1 ratio between hindi and english words."},
            {"role": "assistant", "content": """
              1. जब भी late-night की hunger शुरू होती है, मैं तुरंत अपने favorite snack ideas book को open कर लेता हूँ।
              2. Office के काम के बाद, जब hunger मुझसे मिलती है, तो मैं always अपनी comfort food recipe book देखता हूँ।
              3. दिनभर की मेहनत के बाद, hunger से deal करने के लिए मैंने एक खास meal prep book खरीदी।
              4. अगर weekend पर hunger strike करे, तो एक quick fixes की book मेरे पास हमेशा ready रहती है।
              5. शाम की hunger को curb करने के लिए, मैंने recently एक new healthy eating book buy की है।
              """
            },
            {"role": "user", "content": "Generate 5 creative Hindi-English code-mixed pun-style sentences ending with the english word 'deal'. Have the english word 'heart' as a context in each of these sentences. Ensure the word 'deal' occurs after the word 'heart'. Avoid providing English translations. Make sure to write the English words in roman script and Hindi words in Devanagari. Maintain a balanced 1:1 ratio between hindi and english words."},
            {"role": "assistant", "content": """
              1. उसके दिल में बहुत सारी hopes हैं, but finding true love is not an easy deal.
              2. मेरे heart कह रहा है कि उससे मिलने जाऊं, पर money की deal ठीक नहीं है।
              3. उसने अपने दिल की बात सबसे पहले share की, but betrayal was the final deal.
              4. उसका दिल तोड़ा गया, and now trust is a hard deal to make.
              5. उसके दिल में बहुत सारी dreams हैं, but reality is a tough deal to face.
              """
            },

            {"role": "user", "content": f"Generate 5 creative Hindi-English code-mixed pun-style sentences ending with the english word '{alternate_word}'. Have the english word '{pun_word_meaning}' as a context in each of these sentences. Ensure the word '{alternate_word}' occurs after the word '{pun_word_meaning}'. Avoid providing English translations. Make sure to write the English words in roman script and Hindi words in Devanagari. Maintain a balanced 1:1 ratio between hindi and english words."},
            ],
        
        temperature=1,
    )
    try:
      text = re.sub(alternate_word,pun_word,response.choices[0].message.content.lower())
      lines = text.split('\n')
      if len(lines[0].strip()) > 0:
          pun_entry = {
              'Pun_word': pun_word,
              'Alternate_word': alternate_word,
              'Pun_word_translation': pun_word_meaning,
              'Pun_word_pos': pun_word_pos,
              'Alternate_word_pos': alternate_word_pos,
          }
          sentences = [lines[0].strip(),lines[1].strip(),lines[2].strip(),lines[3].strip(),lines[4].strip()]
      else:
          pun_entry = {
              'Pun_word': pun_word,
              'Alternate_word': alternate_word,
              'Pun_word_translation': pun_word_meaning,
              'Pun_word_pos': pun_word_pos,
              'Alternate_word_pos': alternate_word_pos,
          }
          sentences = [lines[1].strip(),lines[2].strip(),lines[3].strip(),lines[4].strip(),lines[5].strip()]
      filtered_sentences = []
      for sentence in sentences:
        if re.search(pun_word, sentence, re.IGNORECASE):
          filtered_sentences.append(sentence[3:])
      pun_entry["No_of_sentences"] =  len(filtered_sentences)
      pun_entry["Candidates"] =  filtered_sentences
      if len(filtered_sentences) > 0:
          pun_entry['Sentence_chosen'] = random.choice(filtered_sentences)
      else:
          pun_entry['Sentence_chosen'] = ""
          
      pun_entry['index'] = index
      index += 1
    except:
      pun_entry = {
            'Pun_word': pun_word,
            'Alternate_word': alternate_word,
            'Pun_word_translation': pun_word_meaning,
            'Pun_word_pos': pun_word_pos,
            'Alternate_word_pos': alternate_word_pos,
            'No_of_sentences': 0,
            'Candidates': [],
            'index': index,
        }

    if pun_entry["No_of_sentences"] == 0:
      total_zero += 1
    pun_data.append(pun_entry)

  print(f"No. of zero pun sentences = {total_zero}")

  # Write the modified data back to a new file
  write_file_path = "/home2/likhithasapu/Codemixed-Pun-Generation/pun/automated_gen/pun_4.json"
  with open(write_file_path, "w") as file:
      json.dump(pun_data, file, indent=4, ensure_ascii=False)

  with open(write_file_path, "r") as file:
      data = json.load(file)