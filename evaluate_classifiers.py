"""
Evaluate GPT-4o and Claude 3.5 Sonnet on ISIS classification under four
prompting strategies (Zero-Shot, One-Shot, Few-Shot, Taxonomy).

For each strategy the script reads the per-tweet `*_classification.json`
predictions under `./data/` and reports accuracy, precision, recall, F1
(with ISIS as the positive class) plus GPT-vs-Claude inter-model agreement
(Cohen's kappa and percent match) on the joint 2,000-tweet evaluation set.

Usage:
    python3 evaluate_classifiers.py
"""

import json
import os
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


REPO = Path(__file__).resolve().parent
DATA = REPO / "data/classification"

# (setting name, GPT-pos dir, GPT-neg dir, Claude-pos dir, Claude-neg dir)
SETTINGS = [
    ("Zero-Shot",
        "zero_shot_positive_gpt4o",
        "zero_shot_negative_gpt4o",
        "zero_shot_positive_claude35sonnet",
        "zero_shot_negative_claude35sonnet"),
    ("One-Shot",
        "one_shot_positive_gpt4o",
        "one_shot_negative_gpt4o",
        "one_shot_positive_claude35sonnet",
        "one_shot_negative_claude35sonnet"),
    ("Few-Shot",
        "few_shot_positive_gpt4o",
        "few_shot_negative_gpt4o",
        "few_shot_positive_claude35sonnet",
        "few_shot_negative_claude35sonnet"),
    ("Taxonomy",
        "taxonomy_positive_gpt4o",
        "taxonomy_negative_gpt4o",
        "taxonomy_positive_claude35sonnet",
        "taxonomy_negative_claude35sonnet"),
]


def load_preds(relpath):
    """Load {sample_id -> normalized_label} from a directory of *_classification.json files."""
    d = DATA / relpath
    out = {}
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        sid = fn.split("_", 1)[0]
        with open(d / fn) as f:
            data = json.load(f)
        label = str(data.get("classification", "")).strip().upper()
        if label == "ISIS":
            out[sid] = "ISIS"
        elif label in ("NOT-ISIS", "NOT_ISIS", "NOTISIS", "NON-ISIS"):
            out[sid] = "NOT-ISIS"
    return out


def metrics(pos_preds, neg_preds):
    """Compute Acc/Prec/Rec/F1 with ISIS as the positive class."""
    tp = sum(1 for v in pos_preds.values() if v == "ISIS")
    fn = sum(1 for v in pos_preds.values() if v == "NOT-ISIS")
    fp = sum(1 for v in neg_preds.values() if v == "ISIS")
    tn = sum(1 for v in neg_preds.values() if v == "NOT-ISIS")
    n = tp + fn + fp + tn
    return {
        "acc": (tp + tn) / n if n else 0.0,
        "prec": tp / (tp + fp) if (tp + fp) else 0.0,
        "rec": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }


def agreement(gpt_pos, gpt_neg, cla_pos, cla_neg):
    """Cohen's kappa and % agreement between GPT and Claude on the joint 2k set."""
    # Namespace sample IDs by class so positive sid=0 and negative sid=0 don't collide.
    g = {("P", k): v for k, v in gpt_pos.items()} | {("N", k): v for k, v in gpt_neg.items()}
    c = {("P", k): v for k, v in cla_pos.items()} | {("N", k): v for k, v in cla_neg.items()}
    common = sorted(set(g) & set(c))
    yg = [g[k] for k in common]
    yc = [c[k] for k in common]
    kappa = cohen_kappa_score(yg, yc)
    pct = 100.0 * sum(1 for a, b in zip(yg, yc) if a == b) / len(common)
    return kappa, pct


def compute_row(name, gpt_pos_dir, gpt_neg_dir, cla_pos_dir, cla_neg_dir):
    gpt_pos = load_preds(gpt_pos_dir)
    gpt_neg = load_preds(gpt_neg_dir)
    cla_pos = load_preds(cla_pos_dir)
    cla_neg = load_preds(cla_neg_dir)
    gpt = metrics(gpt_pos, gpt_neg)
    cla = metrics(cla_pos, cla_neg)
    kappa, pct = agreement(gpt_pos, gpt_neg, cla_pos, cla_neg)
    return name, gpt, cla, kappa, pct


def print_table(rows):
    cw = 8  # width per metric cell
    header = (
        f"{'Classifier':<12}│ "
        f"{'GPT-4o':^{cw*4}}│ "
        f"{'Claude 3.5 Sonnet':^{cw*4}}│ "
        f"{'Agreement':^{cw*2}}"
    )
    sub = (
        f"{'':<12}│ "
        f"{'Acc':>{cw}}{'Prec':>{cw}}{'Rec':>{cw}}{'F1':>{cw}}│ "
        f"{'Acc':>{cw}}{'Prec':>{cw}}{'Rec':>{cw}}{'F1':>{cw}}│ "
        f"{'κ':>{cw}}{'%':>{cw}}"
    )
    print(header)
    print(sub)
    print("─" * len(sub))
    for name, g, c, k, p in rows:
        print(
            f"{name:<12}│ "
            f"{g['acc']:>{cw}.3f}{g['prec']:>{cw}.3f}{g['rec']:>{cw}.3f}{g['f1']:>{cw}.3f}│ "
            f"{c['acc']:>{cw}.3f}{c['prec']:>{cw}.3f}{c['rec']:>{cw}.3f}{c['f1']:>{cw}.3f}│ "
            f"{k:>{cw}.3f}{p:>{cw}.1f}"
        )


def main():
    rows = [compute_row(*s) for s in SETTINGS]
    print_table(rows)


if __name__ == "__main__":
    main()
