# Write code to read from sample.json file generate indexes

import json
import re
from tqdm import tqdm

BASE_INDEX = 1000
file_path = "./task2/Chatgpt_pun_translated_2.json"
with open(file_path, "r") as file:
    data = json.load(file)
    
for ind, row in enumerate(data):
    row["index"] = ind + BASE_INDEX
    
with open(file_path, "w") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)