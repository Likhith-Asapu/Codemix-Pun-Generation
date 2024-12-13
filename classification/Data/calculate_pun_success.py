import json

with open('final_data4.json') as f:
    data = json.load(f)

def calulate_pun_success(data, startIndex, endIndex):
    pun_count = 0
    total_count = 0
    for row in data:
        if row['label'] == 1 and row['id'] >= startIndex and row['id'] < endIndex:
            pun_count += 1
        if row['id'] >= startIndex and row['id'] < endIndex:
            total_count += 1
    return pun_count, total_count
        
print(calulate_pun_success(data, 0, 500))
print(calulate_pun_success(data, 500, 1000))
print(calulate_pun_success(data, 1000, 1500))
print(calulate_pun_success(data, 1500, 2000))

