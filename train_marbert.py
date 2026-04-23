"""
Fine-tune MARBERT (UBC-NLP/MARBERT) for Arabic ISIS binary classification.

The script trains one classifier at the sample size you pass via --size
(balanced 50/50 across the two input files), with a 90/10 train/test split,
and prints test-set metrics via sklearn's `classification_report`. Run it
once per size to sweep the scaling curve; we typically sweep 1K, 10K, 100K,
250K, and 500K.

Expected metrics at size 500,000 on this corpus:
    Accuracy ~= 0.90    ISIS Prec ~= 0.88    Recall ~= 0.94    F1 ~= 0.91

Data (not shipped in the repo; Twitter ToS forbids redistribution):
    data/isis-250k.txt  - one pro-ISIS Arabic tweet per line (250,000 lines)
    data/neg-250k.txt   - one non-ISIS Arabic tweet per line (250,000 lines)
See the README for how to obtain a sample.

Usage:
    # Train once per size:
    python3 train_marbert.py --size 1000
    python3 train_marbert.py --size 10000
    python3 train_marbert.py --size 100000
    python3 train_marbert.py --size 250000
    python3 train_marbert.py --size 500000     # default if --size omitted

    # Evaluate a saved checkpoint on the same 10% test split:
    python3 train_marbert.py --eval-only --checkpoint checkpoints/n500000/model_5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
CKPT_DIR = REPO / "checkpoints"

MODEL_NAME = "UBC-NLP/MARBERT"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-6
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
RANDOM_STATE = 42

# Binary labels: ISIS is the positive class (index 0).
LABEL2ID = {"ISIS": 0, "NOT-ISIS": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
TARGET_NAMES = [ID2LABEL[0], ID2LABEL[1]]

log = logging.getLogger("train_marbert")


def load_data(sample_size: int | None = None) -> pd.DataFrame:
    """Load the two tweet files into a shuffled DataFrame of (text, label, label_id).

    If `sample_size` is given, keeps `sample_size // 2` tweets from each class
    so runs at different sizes stay balanced.
    """
    isis_path = DATA / "isis-250k.txt"
    neg_path = DATA / "neg-250k.txt"
    for p in (isis_path, neg_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. These tweet files are not shipped in the repo "
                "(Twitter ToS). See the README section 'Obtaining the training "
                "data' for how to reconstruct them from seed account IDs."
            )

    isis = [l.strip() for l in isis_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    neg = [l.strip() for l in neg_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if sample_size is not None:
        per_class = sample_size // 2
        if per_class > min(len(isis), len(neg)):
            raise ValueError(
                f"sample_size={sample_size} (per_class={per_class}) exceeds available "
                f"data (isis={len(isis)}, neg={len(neg)})."
            )
        isis, neg = isis[:per_class], neg[:per_class]

    df = pd.concat(
        [pd.DataFrame({"text": isis, "label": "ISIS"}),
         pd.DataFrame({"text": neg, "label": "NOT-ISIS"})],
        ignore_index=True,
    ).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)
    log.info("Loaded %d tweets (%d ISIS, %d NOT-ISIS)",
             len(df), (df["label"] == "ISIS").sum(), (df["label"] == "NOT-ISIS").sum())
    return df


def build_dataloaders(df: pd.DataFrame, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """Split 90/10 (stratified, seed=42), tokenize, and build train/test DataLoaders."""
    train_df, test_df = train_test_split(
        df, test_size=0.1, random_state=RANDOM_STATE, shuffle=True,
        stratify=df["label_id"],
    )
    log.info("Train size %d, test size %d", len(train_df), len(test_df))

    def _encode(frame: pd.DataFrame) -> TensorDataset:
        enc = tokenizer(frame["text"].tolist(), max_length=MAX_SEQ_LENGTH,
                        padding="max_length", truncation=True, return_tensors="pt")
        y = torch.tensor(frame["label_id"].values, dtype=torch.long)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train_dl = DataLoader(_encode(train_df), batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(_encode(test_df), batch_size=BATCH_SIZE, shuffle=False)
    return train_dl, test_dl


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    """Run one training epoch, returning mean batch loss."""
    model.train()
    total = 0.0
    for step, (input_ids, mask, labels) in enumerate(loader):
        input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        loss = criterion(logits, labels)
        if loss.dim() > 0:  # DataParallel returns one loss per GPU
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if step % 100 == 0:
            log.info("  step %d/%d  loss=%.4f", step, len(loader), loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as numpy arrays."""
    model.eval()
    preds, trues = [], []
    for input_ids, mask, labels in loader:
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(labels.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def log_epoch(epoch: int, train_loss: float, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rep = classification_report(y_true, y_pred, target_names=TARGET_NAMES,
                                output_dict=True, digits=4)
    log.info(
        "Epoch %d  train_loss=%.4f  val_acc=%.4f  ISIS P=%.4f R=%.4f F1=%.4f",
        epoch, train_loss, rep["accuracy"],
        rep["ISIS"]["precision"], rep["ISIS"]["recall"], rep["ISIS"]["f1-score"],
    )


def run_training(sample_size: int | None, save_tag: str | None = None) -> None:
    """Train from scratch on the given sample size and print the final report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s (cuda devices: %d)", device, torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = load_data(sample_size=sample_size)
    train_dl, test_dl = build_dataloaders(df, tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID)
    ).to(device)
    model = base_model
    if torch.cuda.device_count() > 1:
        log.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(base_model)

    num_training_steps = len(train_dl) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * WARMUP_PROPORTION),
        num_training_steps=num_training_steps,
    )
    criterion = nn.CrossEntropyLoss()

    tag = save_tag or (f"n{sample_size}" if sample_size else "full")
    for epoch in range(1, EPOCHS + 1):
        log.info("=== Epoch %d/%d ===", epoch, EPOCHS)
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, criterion, device)
        y_true, y_pred = evaluate(model, test_dl, device)
        log_epoch(epoch, train_loss, y_true, y_pred)

        save_path = CKPT_DIR / tag / f"model_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        base_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.info("Saved checkpoint to %s", save_path)

    print(f"\n=== Final test-set report ({len(df)} tweets, {EPOCHS} epochs) ===")
    y_true, y_pred = evaluate(model, test_dl, device)
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=4))


def run_eval_only(checkpoint: Path, sample_size: int | None) -> None:
    """Load `checkpoint` and evaluate on the same 10% test split used in training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

    df = load_data(sample_size=sample_size)
    _, test_dl = build_dataloaders(df, tokenizer)

    print(f"\n=== Eval-only report for {checkpoint} ===")
    y_true, y_pred = evaluate(model, test_dl, device)
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--size", type=int, default=500_000,
        help="Total training+test tweets (balanced 50/50). "
             "The scaling sweep uses 1000, 10000, 100000, 250000, or 500000. "
             "Default: 500000.",
    )
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; evaluate the checkpoint at --checkpoint.")
    p.add_argument("--checkpoint", type=Path,
                   help="Path to a saved MARBERT checkpoint (required with --eval-only).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.eval_only:
        if not args.checkpoint:
            log.error("--eval-only requires --checkpoint")
            sys.exit(2)
        run_eval_only(args.checkpoint, sample_size=args.size)
        return

    run_training(sample_size=args.size)


if __name__ == "__main__":
    main()
