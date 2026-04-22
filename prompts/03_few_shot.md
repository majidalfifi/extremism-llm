# Prompt 3 — Few-Shot Classification

Used in Section 4.1 (Baseline Classification with LLMs). The model is given five manually selected pro-ISIS tweets as reference examples.

## System
```
You are a JSON-only output API. Return one JSON object with keys "classification", "reason", and "original_post". No extra text.
```

## User
```
You are an impartial classifier. You have multiple labeled examples:

{example_post_1}
{example_post_2}
{example_post_3}
{example_post_4}
{example_post_5}

Now, classify this new tweet as either "ISIS" or "NOT-ISIS".
Provide a short reason, and include the original tweet text for reference.

Tweet text:
{tweet_text}

Return a JSON object with these exact fields:
{
  "classification": "ISIS or NOT-ISIS",
  "reason": "a short explanation",
  "original_post": "the original tweet text"
}
```
