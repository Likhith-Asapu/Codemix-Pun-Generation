# Codemix-Pun-Generation

This repository contains the code for the paper "Bridging Laughter Across Languages: Generation of Hindi-English Code-mixed Puns" accepted at the 1st Workshop on Computational Humor (CHum 2025) at COLING 2025. The paper can be found [here](https://aclanthology.org/2025.chum-1.5/).

## Requirements
- Python 3.11
- PyTorch 2.2.1
- Transformers 4.45.1
- NLTK 3.8.1
- Pandas 2.0.3
- Numpy 1.25.2
- Scikit-learn 1.4.1
- Huggingface Datasets 2.14.3

## Dataset
The dataset used in the paper can be found in folder `classification/Data/labels.json`. The dataset is in the format of a json file with the following columns:

- `id`: Unique identifier for each data point
- `text`: Code-mixed text
- `label`: 1 if the text is a pun, 0 otherwise

## Methodology
The code for the four methods for generating puns proposed in the paper can be found in the following folders:

- `baseline`: Contains the code for the baseline pun generation model
- `contextually_aligned`: Contains the code for the contextually aligned pun generation model
- `question_answer` : Contains the code for the question-answer based pun generation model
- `subject_masked`: Contains the code for the subject masked pun generation model

## Automated Pipeline
The code for the automated pipeline for generating puns can be found in the folder `automated_pipeline`. The pipeline generates word pairs and then creates puns using the three methods proposed in the paper on these word pairs. It then evaluates the generated puns using the classification model.

## Citation
If you use this code for your research, please cite the paper as follows:
```
@inproceedings{asapu-etal-2025-bridging,
    title = "Bridging Laughter Across Languages: Generation of {H}indi-{E}nglish Code-mixed Puns",
    author = "Asapu, Likhith  and
      Kodali, Prashant  and
      Dua, Ashna  and
      Rajesh Kavitha, Kapil  and
      Shrivastava, Manish",
    editor = "Hempelmann, Christian F.  and
      Rayz, Julia  and
      Dong, Tiansi  and
      Miller, Tristan",
    booktitle = "Proceedings of the 1st Workshop on Computational Humor (CHum)",
    month = jan,
    year = "2025",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.chum-1.5/",
    pages = "32--57",
    abstract = "Puns, as a linguistic phenomenon, hold significant importance in both humor and language comprehension. While extensive research has been conducted in the realm of pun generation in English, there exists a notable gap in the exploration of pun generation within code-mixed text, particularly in Hindi-English code-mixed text. This study addresses this gap by offering a computational method specifically designed to create puns in Hindi-English code-mixed text. In our investigation, we delve into three distinct methodologies aimed at pun generation utilizing pun-alternate word pairs. Furthermore, this novel dataset, HECoP, comprising of 2000 human-annotated sentences serves as a foundational resource for training diverse pun detection models. Additionally, we developed a structured pun generation pipeline capable of generating puns from a single input word without relying on predefined word pairs. Through rigorous human evaluations, our study demonstrates the efficacy of our proposed models in generating code-mixed puns. The findings presented herein lay a solid groundwork for future endeavours in pun generation and computational humor within diverse linguistic contexts."
}
```


