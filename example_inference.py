"""
Minimal example: run the published MARBERT ISIS detector on two sample
tweets — one clearly pro-ISIS and one neutral — and print the model's
label + ISIS-class probability for each.

The model weights and tokenizer are downloaded on first use from:
    https://huggingface.co/alfifi/marbert-isis-detector

Usage:
    python3 example_inference.py
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "alfifi/marbert-isis-detector"

SAMPLES = [
    # Clearly pro-ISIS rhetoric (endorses the "caliphate" and violence).
    "والله لا نرضى إلا بالدولة الإسلامية وقتال الكفار في كل مكان",
    # Neutral — weather in Riyadh, entirely benign.
    "اليوم طقس جميل جدا في الرياض ارتفاع درجة الحرارة يدعو للذهاب للبحر",
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    enc = tokenizer(
        SAMPLES,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    id2label = model.config.id2label

    for text, pred, p in zip(SAMPLES, preds.tolist(), probs.tolist()):
        label = id2label[pred]
        print(f"[{label}]  P(ISIS)={p[0]:.4f}  P(NOT-ISIS)={p[1]:.4f}")
        print(f"    {text}\n")


if __name__ == "__main__":
    main()
