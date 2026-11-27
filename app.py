"""Local Gradio inference app."""
import os
from functools import lru_cache

import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import predict_scores
from tickets_data import LABELS


@lru_cache(maxsize=1)
def load():
    source = os.environ.get('MODEL_ID', 'model')
    return (AutoModelForSequenceClassification.from_pretrained(source),
            AutoTokenizer.from_pretrained(source))


def classify(text):
    if not text or not text.strip():
        raise gr.Error('Please enter a support message.')
    model, tok = load()
    values = predict_scores(model, tok, [text])[0]
    return {label: float(value) for label, value in zip(LABELS, values, strict=True)}


def build():
    return gr.Interface(fn=classify, inputs=gr.Textbox(lines=4, label='Support ticket'),
                        outputs=gr.Label(num_top_classes=4), title='Lakeshore Ticket Sorter')


if __name__ == '__main__':
    build().launch()
