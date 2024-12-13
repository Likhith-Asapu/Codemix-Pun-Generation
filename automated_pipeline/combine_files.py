import json
with open('puns_3.json', 'r') as f:
    data1 = json.load(f)
    
with open('pun_4_pos.json', 'r') as f:
    data2 = json.load(f)
    
final_data = []
for item1, item2 in zip(data1, data2):
    item1["task1"] = item2["Sentence_chosen"]
    
    final_data.append(item1)
    
with open('puns_5.json', 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)