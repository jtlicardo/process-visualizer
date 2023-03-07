import evaluate
import numpy as np
from preprocess_data import extract_labels, load_jsonl, map_labels

dataset = load_jsonl("annotated_data.jsonl")
labels = extract_labels(dataset)
id2label, label2id = map_labels(labels)

metric = evaluate.load("seqeval")


def compute_metrics(eval_preds):

    logits, labels = eval_preds

    # Extract index of logit with maximum value (i.e. the predicted labels)
    predictions = np.argmax(logits, axis=-1)

    # Convert labels and predictions from integers to strings and remove special tokens (-100)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
