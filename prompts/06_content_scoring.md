# Prompt 6 — Finding Counter-Messaging Candidate Posts

Used in Section 5.1 to score extremist posts for counter-messaging potential on a 1–10 scale. The top-scoring 1,000 posts (from a 20,000-tweet pool) were sent to human annotators to write counter-messages.

## System
```
You are a JSON-only output API that evaluates social media content. Only output a valid JSON object with the exact requested fields.
```

## User
```
Return a JSON object rating this tweet's potential for a meaningful counter-message (1-10).
The JSON must have exactly these fields:
{
  "tweet": "full tweet text",
  "score": integer between 1 and 10,
  "reason": "brief explanation"
}

Specifically, focus on whether the tweet:
- Contains extremist (especially ISIS-related) rhetoric or calls to violence
- Attempts to justify violence via religious or ideological arguments
- Encourages harming or killing others
- Could benefit from a contextual or counter-message response

If a tweet is purely news or informational with no extremist call-to-action, it should receive a lower score.
If the tweet is explicitly urging violence, promoting extremism, or misusing religious concepts to justify harm, it should receive a higher score (i.e., more likely to warrant a response).
```
