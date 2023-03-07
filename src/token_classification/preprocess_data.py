from typing import Any, Callable, Dict, List, Set, Tuple

import datasets
from datasets import load_dataset


def load_jsonl(filepath: str) -> datasets.DatasetDict:
    """
    Creates a DatasetDict from a Prodigy .jsonl file

    Args:
        filepath: path to the source .jsonl file

    Returns:
        A DatasetDict object with a 10% test split
    """
    dataset = load_dataset("json", data_files=filepath, split="train")

    dataset = dataset.filter(lambda example: example["answer"] == "accept")

    dataset = dataset.remove_columns(
        [
            "text",
            "_input_hash",
            "_task_hash",
            "meta",
            "_view_id",
            "answer",
            "_timestamp",
        ]
    )

    dataset_dict = dataset.train_test_split(test_size=0.1)

    return dataset_dict


def extract_labels(dataset: datasets.DatasetDict) -> Set[str]:
    """
    Extracts the unique labels from a NER dataset

    Args:
        dataset: a DatasetDict object created using the load_jsonl function

    Returns:
        A set of labels in the dataset
    """

    labels = set()

    for sample in dataset["train"]:
        for span in sample["spans"]:
            labels.add(span["label"])

    for sample in dataset["test"]:
        for span in sample["spans"]:
            labels.add(span["label"])

    return labels


def map_labels(labels: Set[str]) -> Tuple[Dict[int, str], Dict[str, int]]:

    """
    Creates a dictionary mapping labels to indices

    Args:
        labels: a set of labels in string format

    Returns:
        A tuple of 2 dictionaries - the first dictionary maps integers to labels,
        and the second dictionary maps labels to integers
    """

    # Intially add only the default "O" label
    id2label = {0: "O"}

    for i, label in enumerate(labels, start=1):
        id2label[2 * i - 1] = "B-" + label
        id2label[2 * i] = "I-" + label

    label2id = {value: key for key, value in id2label.items()}

    return (id2label, label2id)


def add_labels_and_tokens_to_dataset(
    example: Dict[str, Any], label_set: Set[str]
) -> Dict[str, Any]:

    """
    Creates a dataset suitable for token classification

    Used in the .map function provided by the datasets library. The function
    transforms a DatasetDict object that contains "spans" and "tokens" attributes
    by adding a new "labels" attribute to each sample. The goal is to create a
    dataset that contains lists of tokens and corresponding lists of labels in
    integer format. The labels conform to the IOB format: https://w.wiki/qsm

    Args:
        example: a sample in the dataset
        label_set: a set of labels in string format

    Returns:
        An updated sample of the dataset
    """

    # Get the label indices
    id2label, label2id = map_labels(label_set)

    labels = []

    for token in example["tokens"]:
        # Set a flag to indicate whether the token is within a span
        within_span = False
        for span in example["spans"]:
            # Check if the current token is within the span
            if span["token_start"] <= token["id"] <= span["token_end"]:
                # Set the flag to True because the token is within a span
                within_span = True
                if span["token_start"] == token["id"]:
                    # The token is the beginning of the span, so it should have the "B-" prefix
                    # Get the index of the desired label with the "B-" prefix
                    for key, value in id2label.items():
                        if value.endswith(span["label"]) and value.startswith("B-"):
                            index = key
                    # Append the index
                    labels.append(index)
                    break
                else:
                    # The token is within the span, but not at the beginning, so it should have the "I-" prefix
                    # Get the index of the desired label with the "I-" prefix
                    for key, value in id2label.items():
                        if value.endswith(span["label"]) and value.startswith("I-"):
                            index = key
                    # Append the index
                    labels.append(index)
                    break
        # If the token is not within any of the spans, append the default 0 value
        if not within_span:
            labels.append(0)

    example["labels"] = labels
    example["tokens"] = [token["text"] for token in example["tokens"]]

    return example


def map_dataset(
    dataset: datasets.DatasetDict,
    func: Callable = add_labels_and_tokens_to_dataset,
    cols: List["str"] = ["spans"],
) -> datasets.DatasetDict:

    """
    Process the dataset using a specified function while removing the specified columns

    Args:
        dataset: a DatasetDict object
        func: a function used to process the dataset
        cols: a list of columns to be removed from the dataset

    Returns:
        A processed dataset
    """

    loaded_dataset = load_jsonl("annotated_data.jsonl")
    labels = extract_labels(loaded_dataset)

    updated_dataset = dataset.map(
        func,
        fn_kwargs={"label_set": labels},
        remove_columns=cols,
    )
    return updated_dataset
