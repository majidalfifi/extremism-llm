# Prompt 4 — Iterative Taxonomy Generation

Used in Section 4.2 (Iterative Taxonomy-Based LLM Classification). The model builds or refines a hierarchical taxonomy from a batch of tweets, optionally merging with an existing taxonomy. Called iteratively over batches of ~100 tweets until the full sample (20K extremist + 20K negative in our experiments) is processed.

## System
```
You are an assistant that organizes tweets into a hierarchical taxonomy.
```

## User
```
Create a hierarchical taxonomy for the following tweets. Group related tweets into categories and subcategories based on recurring themes or topics. Each category should contain a list of example tweets (no more than 5) that fall under it. Return the taxonomy in JSON format.

If an existing taxonomy is provided, incorporate the new tweets into the existing taxonomy, updating or merging categories as needed. If new themes emerge, create additional categories. Resolve any conflicts or overlapping categories in a logical manner.

Tweets:
{tweets_batch}

Existing taxonomy (optional):
{existing_taxonomy}

Return the resulting taxonomy in JSON format, reflecting any changes due to the new tweets.
```
