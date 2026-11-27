"""DistilBERT training with dynamic padding and reproducible seeds."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from tickets_data import LABELS

BASE = 'distilbert/distilbert-base-uncased'
MAX_LENGTH = 96


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tokenize(frame, tokenizer):
    rows = []
    for row in frame.itertuples():
        item = tokenizer(row.text, truncation=True, max_length=MAX_LENGTH)
        item['labels'] = LABELS.index(row.label)
        rows.append(item)
    return rows


def configure(model, variant):
    if variant not in ('head', 'full'):
        raise ValueError('Unknown training variant')
    for param in model.base_model.parameters():
        param.requires_grad = variant == 'full'
    return model


def load_model(seed, source=BASE):
    seed_all(seed)
    return AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=4, id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )


def predict_scores(model, tokenizer, texts, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                list(texts[start:start + batch_size]), padding=True,
                truncation=True, max_length=MAX_LENGTH, return_tensors='pt',
            ).to(device)
            scores.extend(model(**batch).logits.softmax(-1).cpu().numpy())
    return np.asarray(scores)


def predict(model, tokenizer, texts):
    return [LABELS[i] for i in predict_scores(model, tokenizer, list(texts)).argmax(1)]


def train_model(train, val, seed, variant, lr, epochs, tokenizer=None, model=None):
    seed_all(seed)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(BASE)
    model = configure(model if model is not None else load_model(seed), variant)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loader = DataLoader(
        tokenize(train, tokenizer), batch_size=16, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        if variant == 'head':
            model.base_model.eval()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch['labels'])
        accuracy = float(np.mean(np.asarray(predict(model, tokenizer, val.text)) == val.label.to_numpy()))
        record = {'epoch': epoch, 'loss': total / len(train), 'val_accuracy': accuracy}
        history.append(record)
        print(variant, seed, len(train), lr, record, flush=True)
    return model, tokenizer, history
