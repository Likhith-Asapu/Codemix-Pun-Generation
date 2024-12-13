from tqdm import tqdm
import json

def get_funniness_score(annotation):
    if annotation == "Incomprehensible/ Not a pun":
        return 0
    elif annotation == '1 - Not Funny':
        return 1
    elif annotation == '2 - Mildly Funny':
        return 2
    elif annotation == '3 - Moderately Funny':
        return 3
    elif annotation == '4 - Quite Funny':
        return 4
    elif annotation == '5 - Hilarious':
        return 5
    else:
        print(annotation)
        raise ValueError("Invalid funniness annotation")
    
def get_better_pun(annotation):
    if annotation == "None":
        return 0
    elif annotation == 'Sentence 1':
        return 1
    elif annotation == 'Sentence 2':
        return 2
    else:
        raise ValueError("Invalid better pun annotation")

def annotation_refactor(file):
    with open(file, 'r') as f:
        data = json.load(f)
    
    final_data = []
    
    for row in data:
        annotation = row['annotations'][0]
        results = annotation['result']
        annotation_data = {}
        for result in results:
            if result['from_name'] == 'funniness_senetence1':
                funniness = result['value']['choices'][0]
                funniness_score = get_funniness_score(funniness)
                annotation_data['funniness_senetence1'] = funniness_score
            elif result['from_name'] == 'funniness_senetence2':
                funniness = result['value']['choices'][0]
                funniness_score = get_funniness_score(funniness)
                annotation_data['funniness_senetence2'] = funniness_score
            elif result['from_name'] == 'better_pun':
                better_pun = result['value']['choices'][0]
                better_pun_score = get_better_pun(better_pun)
                annotation_data['better_pun'] = better_pun_score
            else:
                continue
        for key, value in row['data'].items():
            annotation_data[key] = value
        final_data.append(annotation_data)

    return final_data

final_data = annotation_refactor('annotated_data.json')
with open('refactored_data.json', 'w') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)
    