import html
import json
import random
import re

import numpy as np
import torch
import transformers
from datasets import ClassLabel, load_dataset, load_metric
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
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
        example["text"] = re.sub(r"@\w{1,15}", "@user", example["text"], flags=re.I | re.M)
    else:
        example["text"] = [re.sub(r"@\w{1,15}", "@user", e, flags=re.I | re.M) for e in example["text"]]
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


def deEmojify(example):
    regrex_pattern = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    example["text"] = [regrex_pattern.sub(r"", x) for x in example["text"]]
    return example


def removeruidos(example):
    # sentenca = sentenca.replace("username","")
    example["text"] = [sentenca.replace("hashtag", "") for sentenca in example["text"]]
    example["text"] = [sentenca.replace("retweeet", "") for sentenca in example["text"]]
    return example


def load_and_prepare_dataset(dataset_name: str, model_name: str, do_preprocess: bool = True):

    dataset = load_dataset(f"Emanuel/{dataset_name}", use_auth_token=True)
    # Cleaning datasets
    if do_preprocess:
        dataset = dataset.map(rename_user, batched=True)
        dataset = dataset.map(rename_url, batched=True)
        dataset = dataset.map(escape_html, batched=True)
        # dataset = dataset.map(removeruidos, batched=True)
        # dataset = dataset.map(deEmojify, batched=True)

    labels = [str(x) for x in set(dataset["train"]["label"])]
    num_labels = len(labels)

    label_encoder = ClassLabel(num_classes=num_labels, names=labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_auth_token=True, from_flax=True)

    def tokenize_function(examples):
        # TODO: get max_length from model
        tokenized_batch = tokenizer(examples["text"], max_length=140, padding="max_length", truncation=True)
        tokenized_batch["label"] = [label_encoder.str2int(str(x)) for x in examples["label"]]
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
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    cf = json.dumps(confusion_matrix(labels, predictions).tolist())
    report = classification_report(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "cf": cf, "report": report}


def main():
    # Setup experiment
    seed = 43  # 2020, 42, 43
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # logs
    logging_strategy = "steps"
    logging_first_step = True
    logging_steps = 100  # if logging_strategy = "steps"
    eval_steps = logging_steps

    # checkpoints
    evaluation_strategy = logging_strategy
    save_strategy = logging_strategy
    save_steps = logging_steps
    save_total_limit = 3

    # other params
    epochs = 15
    learning_rate = 2e-5
    batch_size = 24

    ttbr_base = "Emanuel/ttbr-roberta-base"
    ttbr_base_v2 = "ttbr-roberta-base-flax-v2/ttbr-roberta-base"
    ttbr_base_v3 = "Emanuel/ttbr-roberta-base-v3"
    bert_base = "neuralmind/bert-base-portuguese-cased"
    bert_large = "neuralmind/bert-large-portuguese-cased"
    selected_model = bert_base

    # eleicoes2018
    # covid19-desinformation
    # covid19-fakenews
    # tt-depression
    dataset_name = "covid19-desinformation"

    full_train_dataset, full_eval_dataset, num_labels = load_and_prepare_dataset(dataset_name, selected_model)
    # Teste my own training routine
    full_train_dataset = full_train_dataset.remove_columns(["text"])
    full_eval_dataset = full_eval_dataset.remove_columns(["text"])
    full_train_dataset = full_train_dataset.rename_column("label", "labels")
    full_eval_dataset = full_eval_dataset.rename_column("label", "labels")
    full_train_dataset.set_format("torch")
    full_eval_dataset.set_format("torch")
    # small_train_dataset = full_train_dataset.shuffle(seed=42).select(range(200))
    # small_eval_dataset = full_eval_dataset.shuffle(seed=42).select(range(1000))

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(full_train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(full_eval_dataset, batch_size=batch_size)

    model = load_model(selected_model, num_labels)
    model.config.hidden_dropout = 0.99

    # from torch.optim import AdamW
    from transformers import get_scheduler

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0},
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=True)
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(epochs):
        model.train()
        metric = load_metric("f1")
        acc = load_metric("accuracy")
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # loss_func = nn.BCEWithLogitsLoss()
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            acc.add_batch(predictions=predictions, references=batch["labels"])
        f_score = metric.compute(average="macro")
        acc_score = acc.compute()
        print("Training: ", f_score, acc_score)

        model.eval()
        metric = load_metric("f1")
        acc = load_metric("accuracy")
        preds, labels = [], []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            acc.add_batch(predictions=predictions, references=batch["labels"])
            preds += predictions.detach().cpu().tolist()
            labels += batch["labels"].detach().cpu().tolist()

        f_score = metric.compute(average="macro")
        acc_score = acc.compute()
        print("Validation: ", f_score, acc_score)
        print(confusion_matrix(labels, preds))
    exit(0)
    # model.config.hidden_dropout_prob = 0.85
    run_name = f"{selected_model}-{dataset_name}"
    training_args = TrainingArguments(
        run_name,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy=evaluation_strategy,
        do_train=True,
        do_eval=True,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        seed=seed,
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
    print(trainer.evaluate())

    trainer.save_model(run_name)


if __name__ == "__main__":
    main()
