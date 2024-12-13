import json

# Open the JSON file
with open('dad_jokes.json') as file:
    data = json.load(file)

# Filter out sentences with a question mark
filtered_sentences = [row['joke'] for row in data if '?' in row['joke']]

# Create json object with question and answer split based on question mark and write to a new file 

data = []
for sentence in filtered_sentences:
    if len(sentence.split('?')) > 2:
        continue
    question = sentence.split('?')[0]
    answer = " ".join(sentence.split('?')[1:]).strip()
    if answer != '':
        data.append({'question': question + '?', 'answer': answer})

with open('questions.json', 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)
