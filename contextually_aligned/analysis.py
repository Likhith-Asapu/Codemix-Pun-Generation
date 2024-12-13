import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

file_path = "Chatgpt_pun.json"
with open(file_path, "r") as file:
    data = json.load(file)

def last_word_occurrence(sentence, word):
    words = sentence.split()
    last_index = -1
    for index, w in enumerate(words):
        if w == word or re.search(word, w):
            last_index = index
    return last_index

def relative_postion_graphs():
    relative_pun_positions = []
    # Track the position of last occurance of pun word in the sentence
    for row in tqdm(data, total=len(data)):
        pun_word = row["Pun_word"]
        for sentence in row["Candidates"]:
            if len(sentence) == 0:
                continue
            last_index = last_word_occurrence(sentence, pun_word)
            last_index += 1
            relative_pun_positions.append(last_index /(len(sentence.split()) + 1))
        # sentence = row["Sentence_chosen"]
        # if len(sentence) == 0:
        #     continue
        # last_index = last_word_occurrence(sentence, pun_word)
        # relative_pun_positions.append(last_index/len(sentence.split()))
            
    # plot histogram of relative pun positions

    plt.hist(relative_pun_positions, bins=10, range=[0, 1])
    plt.xlabel("Relative Pun Position")
    plt.ylabel("Frequency")
    plt.title("Histogram of Relative Pun Position")
    plt.savefig("pun_position_histogram_chatgpt.png")
    
def no_of_candidates():
    no_of_candidates = []
    for row in data:
        no_of_candidates.append(len(row["Candidates"]))
    plt.hist(no_of_candidates, bins=6, range=[0, 5])
    plt.xlabel("Number of Candidates")
    plt.ylabel("Frequency")
    plt.title("Histogram of Number of Candidates")
    plt.savefig("no_of_candidates_histogram_chatgpt.png")
    
# no_of_candidates()
relative_postion_graphs()