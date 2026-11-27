"""Small local DistilBERT configuration; no Hub downloads during unit tests."""
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from finetune import MAX_LENGTH, configure, predict, seed_all, tokenize, train_model
from share import model_card, save_model
from tickets_data import LABELS


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / 'vocab.txt'
    path.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfee\nwheel\ndock\nlogin\n', encoding='utf-8')
    return DistilBertTokenizerFast(vocab_file=str(path))


@pytest.fixture
def rows():
    return pd.DataFrame({'text': ['fee', 'wheel', 'dock', 'login'] * 2,
                         'label': LABELS * 2})


def tiny(seed=42):
    seed_all(seed)
    return DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=9, n_layers=1, n_heads=2, dim=16, hidden_dim=32,
        num_labels=4, dropout=0.0, attention_dropout=0.0, seq_classif_dropout=0.0,
        id2label=dict(enumerate(LABELS)), label2id={v: i for i, v in enumerate(LABELS)},
    ))


def weights(model):
    return [p.detach().cpu().clone() for p in model.base_model.parameters()]


def test_tokenization(tokenizer):
    row = pd.DataFrame({'text': ['fee ' * 500], 'label': ['billing']})
    token = tokenize(row, tokenizer)[0]
    assert len(token['input_ids']) == MAX_LENGTH
    assert token['labels'] == 0


def test_freeze(tokenizer, rows):
    model = configure(tiny(), 'head')
    before = weights(model)
    head = model.classifier.weight.detach().clone()
    assert not any(p.requires_grad for p in model.base_model.parameters())
    assert model.classifier.weight.requires_grad
    model, _, _ = train_model(rows, rows, 42, 'head', 0.01, 1, tokenizer, model)
    assert all(torch.equal(a, b) for a, b in zip(before, weights(model), strict=True))
    assert not torch.equal(head.cpu(), model.classifier.weight.detach().cpu())


def test_full_changes_and_repeats(tokenizer, rows):
    before = weights(tiny())
    models = [train_model(rows, rows, 42, 'full', 0.01, 1, tokenizer, tiny())[0] for _ in range(2)]
    assert any(not torch.equal(a, b) for a, b in zip(before, weights(models[0]), strict=True))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(weights(models[0]), weights(models[1]), strict=True))


def test_memorize(tokenizer):
    # Single class sanity check: one tiny batch repeated, not evidence of generalization.
    rows = pd.DataFrame({'text': ['fee'] * 4, 'label': ['billing'] * 4})
    model, _, _ = train_model(rows, rows, 42, 'full', 0.05, 8, tokenizer, tiny())
    assert predict(model, tokenizer, rows.text) == ['billing'] * 4


def test_save_reload(tokenizer, rows, tmp_path):
    model, _, _ = train_model(rows, rows, 42, 'head', 0.01, 1, tokenizer, tiny())
    folder = tmp_path / 'model'
    save_model(model, tokenizer, {'n': 8}, folder)
    loaded = DistilBertForSequenceClassification.from_pretrained(folder)
    assert predict(model, tokenizer, rows.text) == predict(loaded, tokenizer, rows.text)
    card = model_card({'n': 8})
    for heading in ('Intended use', 'Training data', 'Results', 'Limitations'):
        assert f'## {heading}' in card


def test_app(tokenizer, monkeypatch):
    import gradio as gr

    import app

    monkeypatch.setattr(app, 'load', lambda: (tiny(), tokenizer))
    scores = app.classify('fee')
    assert set(scores) == set(LABELS)
    assert np.isclose(sum(scores.values()), 1)
    with pytest.raises(gr.Error, match='Please enter'):
        app.classify('  ')
