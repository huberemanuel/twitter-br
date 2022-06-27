import html
import re

import numpy as np
import torch
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def rename_user(example):
    if isinstance(example["text"], str):
        example["text"] = re.sub(
            r"@\w{1,15}", "@user", example["text"], flags=re.I | re.M
        )
    else:
        example["text"] = [
            re.sub(r"@\w{1,15}", "@user", e, flags=re.I | re.M) for e in example["text"]
        ]
    return example


def rename_url(example):
    example["text"] = [
        re.sub(
            r"(\{?http\}?s\}??:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|\{?http\}?s\}??:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            "http",
            e,
            flags=re.I | re.M,
        )
        for e in example["text"]
    ]
    return example


def escape_html(example):
    example["text"] = [html.unescape(e) for e in example["text"]]
    return example


def load_and_prepare_dataset(dataset_name: str, model_name: str):

    dataset = load_dataset(f"Emanuel/{dataset_name}", use_auth_token=True)
    # Cleaning datasets
    dataset.map(rename_user, batched=True)
    dataset.map(rename_url, batched=True)
    dataset.map(escape_html, batched=True)

    labels = [str(x) for x in set(dataset["train"]["label"])]
    num_labels = len(labels)

    label_encoder = ClassLabel(num_classes=num_labels, names=labels)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=False, use_auth_token=True, from_flax=True
    )

    def tokenize_function(examples):
        # global label_encoder
        # TODO: get max_length from model
        tokenized_batch = tokenizer(
            examples["text"], max_length=140, padding="max_length", truncation=True
        )
        tokenized_batch["label"] = [
            label_encoder.str2int(str(x)) for x in examples["label"]
        ]
        return tokenized_batch

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
    # TODO: crossvalidation
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    return full_train_dataset, full_eval_dataset, num_labels


def load_model(model_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, use_auth_token=True, from_flax=True
    )


def compute_metrics(eval_pred):
    metric = load_metric("f1", average="macro")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    # logs
    logging_strategy = "steps"
    logging_first_step = True
    logging_steps = 40  # if logging_strategy = "steps"
    eval_steps = logging_steps
    # checkpoints
    evaluation_strategy = logging_strategy
    save_strategy = logging_strategy
    save_steps = logging_steps
    save_total_limit = 3

    ttbr_base = "Emanuel/ttbr-roberta-base"
    ttbr_base_v2 = "ttbr-roberta-base-flax-v2/ttbr-roberta-base"
    ttbr_base_v3 = "Emanuel/ttbr-roberta-base-v3"
    bert_base = "neuralmind/bert-base-portuguese-cased"
    selected_model = ttbr_base_v3

    # eleicoes2018
    # covid19-desinformation
    # covid19-fakenews
    # tt-depression
    dataset_name = "covid19-fakenews"
    full_train_dataset, full_eval_dataset, num_labels = load_and_prepare_dataset(
        dataset_name, selected_model
    )
    model = load_model(selected_model, num_labels)

    training_args = TrainingArguments(
        "test_trainer",
        per_device_train_batch_size=32,
        evaluation_strategy=evaluation_strategy,
        do_train=True,
        do_eval=True,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        num_train_epochs=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_dataset,  # small_train_dataset
        eval_dataset=full_eval_dataset,  # small_eval_dataset
        compute_metrics=compute_metrics,
    )

    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()
