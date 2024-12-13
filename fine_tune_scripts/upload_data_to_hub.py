import os
from datasets import Dataset
import pandas as pd
import re
from datasets import load_dataset
from huggingface_hub import login

# Load the data
dataset = load_dataset("json", data_files={'train': "annotated-dataset/train.json", 'validation': "annotated-dataset/validation.json", 'test': "annotated-dataset/test.json"})
print(dataset)

# keep only certain columns in the dataset
dataset = dataset.select_columns(["data.idx", "data.L1", "data.L2", "data.alignments", "data.CM_candidates", "data.CM_candidates_transliterated_indictrans", "average_rating", "int_annotations", "LID", "PoSTags"])
print(dataset)
# push the dataset to the hub
dataset.push_to_hub("likhithasapu/codemix-annotated-dataset", private=True)
