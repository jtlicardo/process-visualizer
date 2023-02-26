import sys

sys.path.append("src/token_classification")

from preprocess_data import *

print("\nLoading dataset...")
dataset = load_jsonl("src/token_classification/annotated_data.jsonl")
print("\nDataset loaded:\n", dataset)

print("\nExtracting labels...")
labels = extract_labels(dataset)
print("Labels:", labels)

print("\nMapping labels...")
id2label, label2id = map_labels(labels)
print("Labels:\n", labels)
print("\nid2label:", id2label)
print("label2id:", label2id)

print("\nMapping dataset...")
dataset = map_dataset(dataset)
print("\nNew dataset:\n", dataset)
print("\nExample tokens:", dataset["train"][0]["tokens"][:13])
print("Example labels:", dataset["train"][0]["labels"][:13])
