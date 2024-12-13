# Codemix-Pun-Generation

This repository contains the code for the paper "Bridging Laughter Across Languages: Generation of Hindi-English Codemixed Puns" accepted at the 1st Workshop on Computational Humor (CHum 2025) at COLING 2025. The paper can be found [here]().

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
@inproceedings{asapu2025generatingpuns,
  title={Generating Code-Mixed Puns},
  author={Likhith Asapu, Prashant Kodali, Ashna Dua, Kapil Rajesh Kavitha and Manish Shrivastava},
  booktitle={Proceedings of the 1st Workshop on Computational Humor (CHum 2025) at COLING 2025},
  year={2025}
}
```


