import json
def choose_samples(data, ids):
    chosen_samples = []
    for entry in data:
        if entry['index'] in ids:
            chosen_samples.append(entry)
    return chosen_samples

with open('disagreement.json') as f:
    data = json.load(f)
ids = [entry['id'] for entry in data]

annotation_data = []
with open('data.json') as f:
    data = json.load(f)
    for entry in data:
            annotation_data.append(entry['data'])
            
# sort the data by index
data = sorted(annotation_data, key=lambda x: x['index']) 
chosen_samples = choose_samples(data, ids)

with open('annotation_data.json', 'w') as f:
    json.dump(chosen_samples, f, indent=4, ensure_ascii=False)    
    print(len(chosen_samples))
    
    
    