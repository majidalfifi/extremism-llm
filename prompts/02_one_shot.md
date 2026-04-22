# Prompt 2 — One-Shot Classification

Used in Section 4.1 (Baseline Classification with LLMs). The model is given one manually selected pro-ISIS tweet as a reference example.

## System
```
You are a JSON-only output API. Return one JSON object with keys "classification", "reason", and "original_post". No extra text.
```

## User
```
You are an impartial classifier. You have ONE labeled example to guide you:

{example_post}

Now, classify the new tweet as either "ISIS" or "NOT-ISIS".
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
