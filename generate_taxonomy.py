"""
Iteratively induce a hierarchical tweet taxonomy from a balanced pool of
pro-ISIS and non-ISIS tweets.

The script walks the input pool in fixed-size batches, calling GPT-4o
(`gpt-4o-2024-08-06`) once per batch. Each call is sent the taxonomy
produced by the previous batch and asked to merge the new tweets into it,
so the taxonomy grows and stabilises as the run progresses. The prompt
template is version-controlled at `prompts/04_taxonomy_generate.md`; this
script is the canonical driver for that prompt. Every intermediate batch
output is saved as a JSON file for inspection, and the final taxonomy is
written to `final_taxonomy.json` under the output directory.

Expected input format: a directory of per-tweet JSON files (one tweet per
file), each file shaped like:

    {"tweet": "<arabic text>", "label": "pro-ISIS"}   # or "NOT-ISIS"

Data is not shipped in the repo (Twitter ToS); see the README for how to
reconstruct it from seed account IDs.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 generate_taxonomy.py \\
        --input-dir data/per_tweet_jsons \\
        --output-dir outputs/taxonomy_run1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)


REPO = Path(__file__).resolve().parent
PROMPT_TEMPLATE_PATH = REPO / "prompts" / "04_taxonomy_generate.md"

SYSTEM_PROMPT = "You are an assistant that organizes tweets into a hierarchical taxonomy."

USER_PROMPT_TEMPLATE = (
    "Create a hierarchical taxonomy for the following tweets. Group related tweets into "
    "categories and subcategories based on recurring themes or topics. Each category should "
    "contain a list of example tweets (no more than 5) that fall under it. Return the "
    "taxonomy in JSON format.\n"
    "\n"
    "If an existing taxonomy is provided, incorporate the new tweets into the existing "
    "taxonomy, updating or merging categories as needed. If new themes emerge, create "
    "additional categories. Resolve any conflicts or overlapping categories in a logical "
    "manner.\n"
    "\n"
    "Tweets:\n"
    "{tweets_batch}\n"
    "\n"
    "Existing taxonomy (optional):\n"
    "{existing_taxonomy}\n"
    "\n"
    "Return the resulting taxonomy in JSON format, reflecting any changes due to the new "
    "tweets."
)

# Label values expected in the per-tweet JSON files. Matches the original
# research script (`extremism-ai-tools/isis/generate-taxonomy-combined.py`)
# and the label strings used elsewhere in the pipeline.
LABEL_EXTREMIST = "pro-ISIS"
LABEL_NEGATIVE = "NOT-ISIS"

# Transient errors worth retrying.
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

log = logging.getLogger("generate_taxonomy")


def load_tweets(input_dir: Path) -> List[Dict[str, Any]]:
    """Read every `*.json` file in `input_dir` into a list of dicts.

    Each file is expected to be a single-tweet object with at least `tweet`
    and `label` keys. Malformed files are skipped with a warning.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    tweets: List[Dict[str, Any]] = []
    skipped = 0
    for path in sorted(input_dir.iterdir()):
        if path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("Skipping %s (invalid JSON: %s)", path.name, e)
            skipped += 1
            continue
        if "tweet" not in data or "label" not in data:
            log.warning("Skipping %s (missing 'tweet' or 'label')", path.name)
            skipped += 1
            continue
        tweets.append(data)

    log.info("Loaded %d tweet files from %s (%d skipped)",
             len(tweets), input_dir, skipped)
    return tweets


def sample_balanced(
    tweets: Sequence[Dict[str, Any]],
    extremist_count: int,
    negative_count: int,
    seed: int,
) -> List[str]:
    """Filter by label, keep `N` of each, then shuffle the union deterministically.

    Returns a flat list of tweet text strings in the order to iterate.
    """
    extremist = [t["tweet"] for t in tweets if t.get("label") == LABEL_EXTREMIST]
    negative = [t["tweet"] for t in tweets if t.get("label") == LABEL_NEGATIVE]

    log.info("Found %d '%s' and %d '%s' tweets before sampling",
             len(extremist), LABEL_EXTREMIST, len(negative), LABEL_NEGATIVE)

    if len(extremist) < extremist_count:
        raise ValueError(
            f"Need {extremist_count} '{LABEL_EXTREMIST}' tweets, "
            f"found only {len(extremist)}."
        )
    if len(negative) < negative_count:
        raise ValueError(
            f"Need {negative_count} '{LABEL_NEGATIVE}' tweets, "
            f"found only {len(negative)}."
        )

    rng = random.Random(seed)
    rng.shuffle(extremist)
    rng.shuffle(negative)
    extremist = extremist[:extremist_count]
    negative = negative[:negative_count]

    combined = extremist + negative
    rng.shuffle(combined)
    log.info("Sampled %d tweets (%d extremist + %d negative), shuffled with seed=%d",
             len(combined), extremist_count, negative_count, seed)
    return combined


def build_prompt(tweets_batch: Sequence[str], existing_taxonomy: Dict[str, Any] | None) -> str:
    """Render the user-message prompt for one batch.

    - `tweets_batch` is rendered as a 1-indexed numbered list for readability.
    - `existing_taxonomy` is serialized as JSON, or the literal "(none)" on
      the first iteration.
    """
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(tweets_batch, start=1))
    if existing_taxonomy:
        existing_str = json.dumps(existing_taxonomy, ensure_ascii=False, indent=2)
    else:
        existing_str = "(none)"
    return USER_PROMPT_TEMPLATE.format(
        tweets_batch=numbered,
        existing_taxonomy=existing_str,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
def call_llm_with_retry(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """One chat-completions call with JSON mode. Retries transient errors up to 3 times."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return json.loads(content)


def generate_taxonomy(
    tweets: Sequence[str],
    client: OpenAI,
    model: str,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the iterative induction loop and return the final taxonomy."""
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    total_batches = math.ceil(len(tweets) / batch_size)
    taxonomy: Dict[str, Any] | None = None

    t0 = time.time()
    for i in range(total_batches):
        batch = tweets[i * batch_size : (i + 1) * batch_size]
        user_prompt = build_prompt(batch, taxonomy)

        t_batch = time.time()
        taxonomy = call_llm_with_retry(
            client=client,
            model=model,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        batch_elapsed = time.time() - t_batch
        total_elapsed = time.time() - t0

        out_path = intermediate_dir / f"batch_{i + 1:03d}.json"
        out_path.write_text(
            json.dumps(taxonomy, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            "Batch %d/%d  size=%d  batch=%.1fs  total=%.1fs  saved=%s",
            i + 1, total_batches, len(batch), batch_elapsed, total_elapsed, out_path.name,
        )

    if taxonomy is None:
        raise RuntimeError("No batches were processed; taxonomy is empty.")
    return taxonomy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory of per-tweet JSON files (each with 'tweet' and 'label').")
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "taxonomy",
                   help="Where to write intermediate/ and final_taxonomy.json.")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Tweets per LLM call. Paper §3.2 uses 100. Default: 100.")
    p.add_argument("--extremist-count", type=int, default=20_000,
                   help=f"Number of '{LABEL_EXTREMIST}' tweets to sample. Default: 20000.")
    p.add_argument("--negative-count", type=int, default=20_000,
                   help=f"Number of '{LABEL_NEGATIVE}' tweets to sample. Default: 20000.")
    p.add_argument("--model", type=str, default="gpt-4o-2024-08-06",
                   help="OpenAI chat-completions model. Default: gpt-4o-2024-08-06.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature. Default: 0.7.")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Max completion tokens per call. Default: 8192.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for deterministic sampling and shuffle. Default: 42.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY is not set. Export it before running this script.")
        sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tweets_raw = load_tweets(args.input_dir)
    tweets = sample_balanced(
        tweets_raw,
        extremist_count=args.extremist_count,
        negative_count=args.negative_count,
        seed=args.seed,
    )

    client = OpenAI(api_key=api_key)
    log.info(
        "Starting taxonomy induction: model=%s batch_size=%d total_tweets=%d",
        args.model, args.batch_size, len(tweets),
    )

    final_taxonomy = generate_taxonomy(
        tweets=tweets,
        client=client,
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )

    final_path = args.output_dir / "final_taxonomy.json"
    final_path.write_text(
        json.dumps(final_taxonomy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Final taxonomy saved to %s", final_path)


if __name__ == "__main__":
    main()
