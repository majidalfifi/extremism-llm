# Prompt 1 — Zero-Shot Classification

Used in Section 4.1 (Baseline Classification with LLMs). The model is asked to classify a tweet as ISIS or NOT-ISIS with no examples.

## System
```
You are a JSON-only output API that returns exactly one JSON object with keys "classification" and "reason". No extra text.
```

## User
```
You are an impartial classifier. You will receive a single tweet text.
Determine if this tweet is from or in support of ISIS (i.e., "ISIS"), or if it is not ("NOT-ISIS").
Provide a concise reason for your decision.

Tweet text:
{tweet_text}

Return a JSON object with these exact fields:
{
  "classification": "ISIS or NOT-ISIS",
  "reason": "a short explanation"
}
```
