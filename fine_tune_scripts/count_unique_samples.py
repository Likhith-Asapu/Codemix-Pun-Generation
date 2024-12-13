from collections import Counter
import json
with open("samples.json") as f:
    samples = json.load(f)
    counter = Counter()
    for sample in samples:
        counter[sample["Alternate_word"]] += 1

print(len(counter))
print(counter.most_common(10))
    
    