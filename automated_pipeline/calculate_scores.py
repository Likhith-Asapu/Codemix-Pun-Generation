import json

with open('refactored_data.json', 'r') as f:
    data = json.load(f)
    
funniness_senetence1 = 0
funniness_senetence2 = 0
funniness_senetence1_count = 0
funniness_senetence2_count = 0

funniness_senetence1_wins = 0
funniness_senetence2_wins = 0

funniness_senetence1_wins_votes = 0
funniness_senetence2_wins_votes = 0

funniness_senetence1_when_pun = 0
funniness_senetence2_when_pun = 0
total_pun = 0

for row in data:
    if row['funniness_senetence1'] != 0:
        funniness_senetence1 += row['funniness_senetence1']
        funniness_senetence1_count += 1
    if row['funniness_senetence2'] != 0:
        funniness_senetence2 += row['funniness_senetence2']
        funniness_senetence2_count += 1
    if row['funniness_senetence1'] > row['funniness_senetence2']:
        funniness_senetence1_wins += 1
    elif row['funniness_senetence1'] < row['funniness_senetence2']:
        funniness_senetence2_wins += 1
        
    if row['better_pun'] == 1:
        funniness_senetence1_wins_votes += 1
    elif row['better_pun'] == 2:
        funniness_senetence2_wins_votes += 1
        
    if row['better_pun'] == 1 or row['better_pun'] == 2:
        funniness_senetence1_when_pun += row['funniness_senetence1']
        funniness_senetence2_when_pun += row['funniness_senetence2']
        total_pun += 1
            
funniness_senetence1 /= funniness_senetence1_count
funniness_senetence2 /= funniness_senetence2_count

funniness_senetence1_when_pun /= total_pun
funniness_senetence2_when_pun /= total_pun

print("Average funniness_senetence1: ", funniness_senetence1)
print("Average funniness_senetence2: ", funniness_senetence2)

print("funniness_senetence1 count: ", funniness_senetence1_count)
print("funniness_senetence2 count: ", funniness_senetence2_count)

print("funniness_senetence1 wins: ", funniness_senetence1_wins)
print("funniness_senetence2 wins: ", funniness_senetence2_wins)

print("funniness_senetence1 wins votes: ", funniness_senetence1_wins_votes)
print("funniness_senetence2 wins votes: ", funniness_senetence2_wins_votes)

print("funniness_senetence1 when pun: ", funniness_senetence1_when_pun)
print("funniness_senetence2 when pun: ", funniness_senetence2_when_pun)
