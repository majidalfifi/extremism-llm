"""
Print the taxonomy-label distribution for the extremist (positive) and
non-extremist (negative) eval tweets.

Reads per-tweet *_classification.json files from:
    data/taxonomy-distribution/positive/  (1,000 ISIS tweets)
    data/taxonomy-distribution/negative/  (1,000 NOT-ISIS tweets)

Rule: each tweet is attributed to its first-listed taxonomy label (primary).
On the non-extremist side, "Religion and Spirituality → Islamic Teachings
and Quotes" and "Religion and Spirituality → Prayers and Supplications"
are merged into one row ("Islamic Teachings & Supplications"). Tweets
with no taxonomy label are counted under "Others".

Usage:
    python3 taxonomy_distribution.py
"""

import json
import os
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parent
DATA = REPO / "data/taxonomy-distribution"

# Merges applied on both the extremist and non-extremist sides.
COMMON_MERGE = {
    # Religion: Prayers/Supplications + Islamic Teachings → one row
    "Religion and Spirituality -> Prayers and Supplications":
        "Religion and Spirituality -> Islamic Teachings & Supplications",
    "Religion and Spirituality -> Islamic Teachings and Quotes":
        "Religion and Spirituality -> Islamic Teachings & Supplications",
    # Bare parent-level labels (no subcategory) → Others
    "Sports and Entertainment": "Others",
}

# Merges applied only to the extremist (positive) side.
POS_MERGE = {
    **COMMON_MERGE,
    # Fold the bare "Extremism and Radical Ideologies" label into the
    # Politics/Extremism row so it doesn't appear as a separate tail.
    "Extremism and Radical Ideologies":
        "Politics and Current Events -> Extremism and Radical Ideologies",
    # Singleton tail categories on the extremist side → Others
    "Inappropriate Content -> Explicit Content": "Others",
    "Politics and Current Events -> International Relations": "Others",
    "Social Issues and Activism -> Legal and Justice": "Others",
}

# Merges applied only to the non-extremist (negative) side.
NEG_MERGE = {
    **COMMON_MERGE,
    # Non-extremist tweets that still hit ME/Extremism → single starred row
    "Politics and Current Events -> Middle East Conflicts":
        "Politics and Current Events -> Extremism, Radical Ideol.",
    "Politics and Current Events -> Extremism and Radical Ideologies":
        "Politics and Current Events -> Extremism, Radical Ideol.",
    # Social Issues: Legal and Justice + Women's Rights + base → one row
    "Social Issues and Activism -> Legal and Justice":
        "Social Issues and Activism -> Legal Justice & Women's Rights",
    "Social Issues and Activism -> Women's Rights":
        "Social Issues and Activism -> Legal Justice & Women's Rights",
    "Social Issues and Activism":
        "Social Issues and Activism -> Legal Justice & Women's Rights",
}


def primary_label_counts(dir_path, merge=None):
    """Return Counter of first-listed taxonomy_labels across files in dir_path.
    Labels that MERGE remaps to "Others" are counted as unlabeled."""
    merge = merge or {}
    c = Counter()
    unlabeled = 0
    total = 0
    for fn in os.listdir(dir_path):
        if not fn.endswith(".json"):
            continue
        total += 1
        with open(dir_path / fn) as f:
            data = json.load(f)
        labels = data.get("taxonomy_labels") or []
        if not labels:
            unlabeled += 1
            continue
        primary = merge.get(labels[0], labels[0])
        if primary == "Others":
            unlabeled += 1
        else:
            c[primary] += 1
    return c, unlabeled, total


def summarize(counts, unlabeled, total):
    """Return list of (label, count, pct_of_total). Unlabeled tweets → Others."""
    rows = [(lbl, n, 100 * n / total) for lbl, n in counts.most_common()]
    if unlabeled:
        rows.append(("Others", unlabeled, 100 * unlabeled / total))
    return rows


def print_side(title, rows, total):
    print(f"{title}  (N={total})")
    print("-" * 72)
    for lbl, n, pct in rows:
        # Split the "Category -> Subcategory" for nicer two-line display
        if " -> " in lbl:
            cat, sub = lbl.split(" -> ", 1)
        else:
            cat, sub = lbl, ""
        print(f"  {cat:<42} {n:>4}  ({pct:4.1f}%)")
        if sub:
            print(f"    {sub}")
    print()


def main():
    pos_counts, pos_unl, pos_total = primary_label_counts(
        DATA / "positive", merge=POS_MERGE)
    neg_counts, neg_unl, neg_total = primary_label_counts(
        DATA / "negative", merge=NEG_MERGE)

    print_side("EXTREMIST CONTENT",
               summarize(pos_counts, pos_unl, pos_total),
               pos_total)
    print_side("NON-EXTREMIST CONTENT",
               summarize(neg_counts, neg_unl, neg_total),
               neg_total)


if __name__ == "__main__":
    main()
