from tqdm import tqdm
import json

def calculate_funniness(funniness):
    if funniness == 'Incomprehensible':
        return 0
    elif funniness == '1 - Not Funny':
        return 1
    elif funniness == '2 - Mildly Funny':
        return 2
    elif funniness == '3 - Moderately Funny':
        return 3
    elif funniness == '4 - Quite Funny':
        return 4
    elif funniness == '5 - Hilarious':
        return 5
    else:
        return 1

def calculate_acceptability(acceptability):
    if acceptability == 'Definitely Unacceptable':
        return 0
    elif acceptability == 'Leaning towards unacceptable':
        return 1
    elif acceptability == 'Uncertain whether it is acceptable or unacceptable':
        return 2
    elif acceptability == 'Acceptable sentence but not very fluent':
        return 3
    elif acceptability == 'Definitely acceptable and very fluent':
        return 4
    else:
        return 4

def calculate_pun_label(total_pun, total_pun_not_pair, total_not_pun):
    if total_pun_not_pair > total_pun and total_pun_not_pair > total_not_pun:
        return 2  # Pun but not formed with Pun word and Alternate word pair
    else:
        return 1 if total_pun > total_not_pun else 0  # 1 for Pun, 0 for Not Pun
    
def get_annotation_indices(annotation):
    funniness_index = -1
    acceptability_index = -1
    pun_index = -1
    for i, result in enumerate(annotation['result']):
        if result['from_name'] == 'punsuccess':
            pun_index = i
        elif result['from_name'] == 'funniness':
            funniness_index = i
        elif result['from_name'] == 'acceptability':
            acceptability_index = i

    return pun_index, funniness_index, acceptability_index

def refactor_file(file):

    with open(file) as f:
        data = json.load(f)

    final_data = []

    for row in tqdm(data, total=len(data)):
        total_pun = 0
        total_pun_not_pair = 0
        total_not_pun = 0
        funniness = 0
        acceptability = 0
        
        pun_index = 0
        funniness_index = 1
        acceptability_index = 2

        for annotation in row['annotations']:
            pun_index, funniness_index, acceptability_index = get_annotation_indices(annotation)
            if pun_index != -1:
                total_pun += (annotation['result'][pun_index]['value']['choices'][0] == 'Yes')
                total_pun_not_pair += (annotation['result'][pun_index]['value']['choices'][0] == 'Pun but not formed with Pun word and Alternate word pair')
                total_not_pun += (annotation['result'][pun_index]['value']['choices'][0] == 'No')
                
            if funniness_index != -1:
                funniness += int(calculate_funniness(annotation['result'][funniness_index]['value']['choices'][0]))
            else:
                funniness += 1
            
            if acceptability_index != -1:
                acceptability += int(calculate_acceptability(annotation['result'][acceptability_index]['value']['choices'][0]))
            else:
                acceptability += 4

        funniness /= len(row['annotations'])
        acceptability /= len(row['annotations'])

        pun_label = calculate_pun_label(total_pun, total_pun_not_pair, total_not_pun)

        # if pun_label == 2:
        #     continue
        
        final_data.append({
            'text': row['data']['Sentence_chosen'],
            'label': pun_label,
            # 'funniness': funniness,
            # 'acceptability': acceptability,
            'id': row['data']['index'],
            'Pun_word': row['data']['Pun_word'],
            'Alternate_word': row['data']['Alternate_word'],
            'Pun_word_pos': row['data']['Pun_word_pos'],
            'Alternate_word_pos': row['data']['Alternate_word_pos'],
            'Pun_word_meaning': row['data']['Pun_word_meaning'],
        })

    # Sort rows by id
    final_data = sorted(final_data, key=lambda x: x['id'])

    # Count number of rows with label 1 for pun
    count = sum(1 for row in final_data if row['label'] == 1)

    print(count)
    print(len(final_data))

    with open('Sorted_' + file, 'w') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

# Example usage
refactor_file('data.json')