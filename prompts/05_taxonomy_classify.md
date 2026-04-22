# Prompt 5 — Taxonomy-Based Classification

Used in Section 4.2 (Iterative Taxonomy-Based LLM Classification) and Section 4.4 (building the large-scale labeled dataset). The LLM is given the full taxonomy and classifies a post as extremist or not, also assigning one or more matching taxonomy categories.

## System
```
You are a JSON-only output API that classifies social media content using a predefined taxonomy. Output a single JSON object containing the classification and reasoning.
```

## User
```
Using the provided taxonomy of extremist content, analyze the following tweet and determine if it contains extremist content. Consider all categories and examples in the taxonomy for your decision.

Taxonomy:
{taxonomy_json}

Tweet to analyze:
{tweet_text}

Return a JSON object with these exact fields:
{
  "is_extremist": boolean,
  "confidence": float between 0 and 1,
  "matching_categories": ["list of relevant taxonomy categories"],
  "reasoning": "brief explanation of classification"
}
```
