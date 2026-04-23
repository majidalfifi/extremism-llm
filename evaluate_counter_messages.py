"""
Aggregate the raw counter-message evaluator votes and report the
preference distribution plus inter-rater agreement.

Reads `data/counter_message_evaluations.json`, an array of vote records:

    {
      "post_id":      int,       # pair identifier
      "post_text":    str,       # original Arabic pro-ISIS tweet
      "human":        str,       # scholar-authored counter-message
      "llm":          str,       # LLM-authored counter-message
      "labeler_id":   str,       # unique evaluator ID
      "labels":       [str, ...] # one of "llm_good", "human_good", "both_good"
    }

Each pair is resolved by simple majority vote across its evaluator labels
(pairs with no majority are flagged separately for a tiebreak reviewer).
The script prints:

  * Structural counts (unique evaluators, pairs, total votes, no-majority
    pairs) so readers can confirm the study's size.
  * Per-pair preference distribution after majority vote, split into
    "LLM better / Equal / Scholar better".
  * Fleiss' kappa on the first three votes per pair (3 raters, 3 categories).

Usage:
    python3 evaluate_counter_messages.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa


REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "counter_message_evaluations.json"

# Raw vote label -> preference category used for reporting.
LABEL_TO_CATEGORY = {
    "llm_good":    "LLM better",
    "both_good":   "Equal",
    "human_good":  "Scholar better",
}
CATEGORIES = ["LLM better", "Equal", "Scholar better"]


def load_votes(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of vote records.")
    return data


def group_by_pair(records: list[dict]) -> dict[int, list[str]]:
    """Return {post_id: [label1, label2, ...]} in the order they were cast."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for r in records:
        pid = r["post_id"]
        for lab in r.get("labels", []):
            grouped[pid].append(lab)
    return grouped


def majority_vote_distribution(pair_votes: dict[int, list[str]]) -> tuple[Counter, int]:
    """
    Resolve each pair by simple majority over its votes. If no label strictly
    exceeds half of the votes for a pair, that pair is flagged as having no
    majority (deferred to a tiebreak reviewer). Returns a Counter of
    preference categories for resolved pairs and the no-majority count.
    """
    resolved: Counter = Counter()
    ties = 0
    for _pid, votes in pair_votes.items():
        c = Counter(votes)
        top_label, top_count = c.most_common(1)[0]
        if top_count > len(votes) / 2:
            resolved[LABEL_TO_CATEGORY[top_label]] += 1
        else:
            ties += 1
    return resolved, ties


def fleiss_on_first_three(pair_votes: dict[int, list[str]]) -> tuple[float, int]:
    """
    Build an (N_pairs x 3_categories) matrix of vote counts using only the
    first 3 votes per pair (Fleiss' kappa requires equal ratings per item),
    return the kappa and the number of pairs used.
    """
    cat_idx = {"llm_good": 0, "both_good": 1, "human_good": 2}
    rows = []
    for pid in sorted(pair_votes):
        votes = pair_votes[pid][:3]
        if len(votes) < 3:
            continue
        row = [0, 0, 0]
        for lab in votes:
            row[cat_idx[lab]] += 1
        rows.append(row)
    matrix = np.array(rows)
    return float(fleiss_kappa(matrix)), len(matrix)


def _fmt_pct(value: int, total: int) -> str:
    return f"{100 * value / total:.1f}%"


def main() -> None:
    records = load_votes(DATA)
    pair_votes = group_by_pair(records)

    n_evaluators = len({r["labeler_id"] for r in records})
    n_pairs = len(pair_votes)
    n_votes = sum(len(v) for v in pair_votes.values())

    resolved, ties = majority_vote_distribution(pair_votes)
    total_resolved = sum(resolved.values())

    kappa, n_kappa = fleiss_on_first_three(pair_votes)

    print("=" * 52)
    print("Counter-message evaluator preferences")
    print("=" * 52)
    print(f"{'Evaluators':<30s} {n_evaluators:>6d}")
    print(f"{'Pairs':<30s} {n_pairs:>6d}")
    print(f"{'Total votes':<30s} {n_votes:>6d}")
    print(f"{'Pairs w/ no majority':<30s} {ties:>6d}")
    print()
    print("Per-pair majority distribution")
    print("-" * 52)
    for cat in CATEGORIES:
        n = resolved[cat]
        print(f"  {cat:<28s} {_fmt_pct(n, total_resolved):>6s}")
    print()
    print(f"Fleiss' kappa (first 3 votes, N={n_kappa}, 3 categories):  {kappa:.3f}")


if __name__ == "__main__":
    main()
