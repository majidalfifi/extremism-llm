# Prompt 9 — Individual Response Rating

Alternative to the comparative evaluation (Prompt 8). Rates each response independently on a 1–5 Likert scale instead of forcing a pairwise winner. Used for sensitivity analysis and as a check on the pairwise preference signal.

## System
```
You are a JSON-only output API that returns exactly one JSON object with the keys 'ratingA', 'reasonA', 'ratingB', and 'reasonB'.
```

## User
```
You are an impartial evaluator. For each response:
1. Provide a rating from 1 to 5 (where 5 is very convincing and 1 is not convincing at all)
2. Provide a brief explanation (1–2 sentences) for why you gave that rating

Return JSON with exactly:
{
  "ratingA": (integer 1-5),
  "reasonA": "short explanation for A",
  "ratingB": (integer 1-5),
  "reasonB": "short explanation for B"
}
```

Original post:
{tweet_text}

Response A:
{response_a}

Response B:
{response_b}
