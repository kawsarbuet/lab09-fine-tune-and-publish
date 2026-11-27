"""Standalone Hub inference, without local project imports."""

from functools import lru_cache

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "kawsarbuet/lakeshore-ticket-sorter"


@lru_cache(maxsize=1)
def load():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return model, AutoTokenizer.from_pretrained(MODEL_ID)


def classify(text):
    if not text or not text.strip():
        raise gr.Error("Please enter a support message.")
    model, tok = load()
    batch = tok(text, truncation=True, max_length=96, return_tensors="pt")
    with torch.inference_mode():
        values = model(**batch).logits.softmax(-1)[0].tolist()
    return {model.config.id2label[i]: value for i, value in enumerate(values)}


if __name__ == "__main__":
    gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=4),
        outputs=gr.Label(num_top_classes=4),
        title="Lakeshore Ticket Sorter",
    ).launch()
