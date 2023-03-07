import argparse
import os

import evaluate
from evaluation import compute_metrics
from preprocess_data import extract_labels, load_jsonl, map_dataset, map_labels
from tokenize_data import tokenize_and_align_labels
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--name", help="Name of the repo the model will be pushed to", required=True
)
parser.add_argument(
    "-e", "--epochs", help="Numbers of epochs to train for", required=True, type=int
)
parser.add_argument(
    "-m",
    "--model",
    help="Model checkpoint from Hugging Face",
    required=False,
    default="bert-base-cased",
)
args = parser.parse_args()

metric = evaluate.load("seqeval")

model_checkpoint = args.model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Functions from preprocess_data.py
dataset = load_jsonl("annotated_data.jsonl")
labels = extract_labels(dataset)
id2label, label2id = map_labels(labels)
dataset = map_dataset(dataset)

# Function from tokenize_data.py
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir=args.name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,
    learning_rate=2e-5,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    logging_steps=10,
    logging_strategy="epoch",
    push_to_hub=True,
    hub_private_repo=True,
    report_to=["tensorboard", "wandb"],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.push_to_hub()
