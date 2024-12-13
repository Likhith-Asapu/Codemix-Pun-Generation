from googletrans import Translator
import json
from tqdm import tqdm

translator = Translator()

with open('final_data_2.json', 'r') as f:
    data = json.load(f)
    
for item in tqdm(data, total=len(data)):
    pun_word = item["Pun_word"]
    pun_word_translation = translator.translate(pun_word).text
    item["Pun_word_translation"] = pun_word_translation
    
with open('final_data_2.json', 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

