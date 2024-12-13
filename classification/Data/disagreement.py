import json

with open('final_data.json') as f:
    data1 = json.load(f)
    
with open('final_data2.json') as f:
    data2 = json.load(f)
    
with open('final_data3.json') as f:
    data3 = json.load(f)

with open('final_data4.json') as f:
    data4 = json.load(f)

data_combined = {}

def add_data(data, data_combined):
    for entry in data:
        entry_id = entry['id']
        if entry_id in data_combined:
            data_combined[entry_id].append(entry['label'])
        else:
            data_combined[entry_id] = [entry['label']]
    return data_combined

data_combined = add_data(data1, data_combined)
data_combined = add_data(data2, data_combined)
data_combined = add_data(data3, data_combined)
data_combined = add_data(data4, data_combined)

def calculate_disagreement(data_combined):
    count = 0
    data = []
    print(len(data_combined))
    for entry_id, labels in data_combined.items():    
        if len(set(labels)) > 1 and len(labels) <= 2 and len(labels) > 0:
            
            count += 1
            data.append({
                'id': entry_id,
            })
            # print(f"Disagreement found for entry {entry_id}: {labels}")
    return count, data

count, data = calculate_disagreement(data_combined)
print(f"Total number of disagreements: {count}")
with open('disagreement.json', 'w') as f:
    json.dump(data, f)
    
