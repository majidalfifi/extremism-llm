"""
Aggregate the LLM-as-judge preferences over scholar-vs-LLM counter-message
pairs and print the five-row distribution shown in the paper's counter-
messaging evaluation figure (humans + four LLM evaluators).

Reads per-pair judgment files from:

    data/llm-judge-preferences/{model}/seed{42,50}/{post_id}_evaluation.json

Each file records one LLM's verdict on a single pair:

    {
      "winner":       "A" | "B" | "EQUAL",
      "reason":       str,      # the judge's free-text justification
      "a_was_human":  bool      # whether A was the scholar-authored message
    }

Positions A/B are randomised per pair, so "A was human" has to be combined
with the winner to derive whether the scholar or the LLM message was
preferred. Each model is run twice (seed 42 and seed 50) and the two runs
are pooled, giving N=2000 judgments per model.

The human row is loaded from `data/counter_message_evaluations.json` and
resolved by majority vote (see evaluate_counter_messages.py).

Usage:
    python3 evaluate_llm_judges.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parent
JUDGE_ROOT = REPO / "data" / "llm-judge-preferences"
HUMAN_VOTES = REPO / "data" / "counter_message_evaluations.json"

# Display order matches the paper's figure (humans first, then LLMs in
# roughly ascending capability).
LLM_JUDGES = [
    "gpt-3.5-turbo",
    "gpt-4o",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
]


def preference(record: dict) -> str:
    """Map (winner, a_was_human) to Scholar / LLM / Equal."""
    winner = record["winner"]
    a_was_human = record["a_was_human"]
    if winner == "EQUAL":
        return "Equal"
    if (winner == "A" and a_was_human) or (winner == "B" and not a_was_human):
        return "Scholar"
    return "LLM"


def aggregate_judge(model_dir: Path) -> Counter:
    """Pool all judgment files under a model's seed directories."""
    counts: Counter = Counter()
    for seed_dir in sorted(model_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        for f in sorted(seed_dir.glob("*.json")):
            with f.open() as fh:
                counts[preference(json.load(fh))] += 1
    return counts


def aggregate_humans() -> Counter:
    """Resolve each pair in counter_message_evaluations.json by majority vote."""
    with HUMAN_VOTES.open() as f:
        records = json.load(f)
    by_pair: dict[int, list[str]] = {}
    for r in records:
        by_pair.setdefault(r["post_id"], []).extend(r.get("labels", []))
    label_to_pref = {
        "human_good": "Scholar",
        "llm_good":   "LLM",
        "both_good":  "Equal",
    }
    counts: Counter = Counter()
    for votes in by_pair.values():
        c = Counter(votes)
        top, top_n = c.most_common(1)[0]
        if top_n > len(votes) / 2:
            counts[label_to_pref[top]] += 1
    return counts


def print_row(name: str, counts: Counter) -> None:
    n = sum(counts.values())
    s, l, e = counts["Scholar"], counts["LLM"], counts["Equal"]
    print(f"  {name:<20s} N={n:<5d}  Scholar={100*s/n:5.1f}%  "
          f"LLM={100*l/n:5.1f}%  Equal={100*e/n:5.1f}%")


def main() -> None:
    print("=" * 68)
    print("Counter-message preferences by evaluator")
    print("=" * 68)
    print_row("Humans (majority)", aggregate_humans())
    for model in LLM_JUDGES:
        print_row(model, aggregate_judge(JUDGE_ROOT / model))


if __name__ == "__main__":
    main()
