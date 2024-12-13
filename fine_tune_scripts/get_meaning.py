import json
from tqdm import tqdm
from PyDictionary import PyDictionary
dictionary = PyDictionary()
# Function to get the definition from PyDictionary
def get_definition(word, pos=None):
    definitions = dictionary.meaning(word)
    
    if definitions:
        if pos and pos in definitions:
            return definitions[pos]
        else:
            # Return the first available definition if the specific POS is not found
            for pos_type in definitions:
                return definitions[pos_type]
    return None

# Function to update each JSON entry with the alternate word meaning
def update_json_with_meaning(json_data):
    for entry in tqdm(json_data[:10], total=len(json_data[:10])):
        pun_pos = entry.get("Pun_word_pos")
        alternate_word = entry.get("Alternate_word")
        
        if pun_pos == "X":
            pos = "Noun"
        else:
            pos = entry.get("Alternate_word_pos").upper()
            if pos == 'ADJ':
                pos = 'Adjective'
            elif pos == 'ADV':
                pos = 'Adverb'
            elif pos == 'NUM':
                pos = 'Noun'
            

        alternate_word_meaning = get_definition(alternate_word, pos)
        entry["Alternate_word_meaning"] = alternate_word_meaning[0] if alternate_word_meaning else "Meaning not found"
    
    return json_data

# Reading from the samples.json file
with open('samples.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Updating the JSON data
updated_data = update_json_with_meaning(data)

# Writing the updated data back to a new file or overwriting the original file
with open('samples_updated.json', 'w', encoding='utf-8') as file:
    json.dump(updated_data, file, ensure_ascii=False, indent=4)

print("Updated JSON data written to 'samples_updated.json'.")

# from PyDictionary import PyDictionary
# dictionary=PyDictionary()

# print (dictionary.meaning("one"))
# print (dictionary.meaning("qatar"))
