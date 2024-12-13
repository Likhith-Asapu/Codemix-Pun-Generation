
from datasets import load_metric
metric = load_metric("sacrebleu")
decoded_preds = [
    "hello there general kenobi",                             # tokenized prediction of the first sample
    "foo bar foobar"                                            # tokenized prediction of the second sample
]
decoded_labels = [
    ["hello there general kenobi"],  # tokenized references for the first sample (2 references)
    ["foo bar foobar"]                                           # tokenized references for the second sample (1 reference)
]
print(metric.compute(predictions=decoded_preds, references=decoded_labels))