# Prompt 8 — Comparative (A vs B) Evaluation

Used in Section 5.2 for LLM-based evaluation of counter-messages. Fed back into four LLM evaluators (GPT-4o, Claude 3.5 Sonnet, GPT-3.5-Turbo, and Claude 3 Sonnet) to compare a human-authored response against an LLM-generated response for the same extremist post. Positions of A/B were randomly alternated.

## System
```
You are a JSON-only output API that returns exactly one JSON object with keys 'winner' and 'reason'. No extra text.
```

## User
```
You are an impartial evaluator. Below is an original post (tweet), plus two responses:
- Response A
- Response B

1. Read the original post and both responses carefully.
2. Decide which response is more convincing or effective at countering the extremist message,
   or whether they are equal in quality.
3. Provide a concise explanation (1–3 sentences) explaining your choice.

Return JSON with exactly:
{
  "winner": "A or B or EQUAL",
  "reason": "a short explanation"
}
```

Original post:
{tweet_text}

Response A:
{response_a}

Response B:
{response_b}
