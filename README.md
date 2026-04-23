# extremism-llm

Code, prompts, and (where shareable) data accompanying the paper *"Extremism Detection and Counter-Messaging with Large Language Models"* (Alfifi, Kaghazgaran, Caverlee).

This repository reproduces the LLM prompting experiments and the distilled classifier evaluation reported in the paper. Notably, we release the **full text of all 2,000 evaluation tweets** (1,000 pro-ISIS + 1,000 negatives) used in the paper's Table 2, alongside the per-tweet LLM classifications and reasoning. The iteratively induced taxonomy, the LLM- and human-authored counter-messages, and all LLM prompts used in the paper are also provided.

The larger 500K LLM-labeled corpus used for MARBERT training is **not** redistributed here (Twitter/X Terms of Service).

The paper's appendix — prompts, human-instruction text, and evaluation-interface screenshots — is reproduced below on this page.

## Environment

The three scripts were developed and tested in the following environment:

| Component | Version |
|---|---|
| Python | **3.8.18** (any 3.8+ works; `train_marbert.py` uses `from __future__ import annotations` so newer-style type hints remain compatible) |
| PyTorch | 2.1.2 (+cu121 on the machine we ran on) |
| transformers | 4.37.2 |
| scikit-learn | 1.3.2 |
| pandas | 2.0.3 |
| numpy | 1.24.4 |
| CUDA | 12.1 runtime with NVIDIA drivers matching it (only required for `train_marbert.py`; the other scripts run on CPU) |
| Hardware used | 4× NVIDIA RTX A6000 (48 GB each) with `torch.nn.DataParallel`; the script falls back to a single GPU or CPU automatically |

A `requirements.txt` is included. To set up a fresh environment:

```
conda create -n extremism-llm python=3.8 -y
conda activate extremism-llm
pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch wheel, follow the selector at [pytorch.org](https://pytorch.org/get-started/locally/) and install the matching torch/torchvision/torchaudio combination before running `pip install -r requirements.txt`.

**Runtime notes for `train_marbert.py`**: MARBERT (~650 MB) is downloaded from the HuggingFace Hub on first use and cached. At `--size 1000` training finishes in ~2 minutes on 4 NVIDIA RTX A6000 GPUs; at `--size 500000` budget a few hours on similar hardware. Per-epoch checkpoints land in `checkpoints/n<SIZE>/model_<epoch>/`.

`evaluate_classifiers.py` and `taxonomy_distribution.py` only need Python + scikit-learn (for `cohen_kappa_score`) and the standard library — they run in a few seconds on CPU.

## Repository structure

```
extremism-llm/
├── prompts/                        # All LLM and human-annotator instruction prompts
│   ├── 01_zero_shot.md             # Baseline classification prompts (Sec 4.1)
│   ├── 02_one_shot.md
│   ├── 03_few_shot.md
│   ├── 04_taxonomy_generate.md     # Iterative taxonomy induction (Sec 4.2)
│   ├── 05_taxonomy_classify.md     # Taxonomy-guided classification (Sec 4.2, 4.4)
│   ├── 06_content_scoring.md       # Counter-messaging candidate scoring (Sec 5.1)
│   ├── 07_counter_message_generate.md  # LLM counter-message generation (Sec 5.1)
│   ├── 08_comparative_eval.md      # LLM-judge pairwise A/B evaluation (Sec 5.2)
│   ├── 09_individual_rating.md     # LLM-judge Likert rating alternative
│   ├── 10_human_scholar_instructions.md     # Islamic-studies student task
│   └── 11_human_evaluator_instructions.md   # Lay Arabic-speaker evaluator task
├── screenshots/                    # Evaluation interface and labeling tool snapshots
│   ├── human-eval-arabic.pdf|.png  # Scholar-response view of the eval UI
│   ├── human-eval-english.pdf|.png # LLM-response view of the eval UI
│   └── taxonomy_heatmap.png        # Evolution of the taxonomy across iterations
├── data/
│   ├── classification/             # The 2,000 evaluation tweets (full text) +
│   │                               # per-tweet LLM predictions and reasoning for
│   │                               # each prompting strategy and each model (Table 2)
│   ├── taxonomy-distribution/      # Per-tweet taxonomy labels assigned by the
│   │                               # LLM on the 2,000-tweet eval set (Table 1)
│   ├── taxonomy.json               # The final induced taxonomy (output of
│   │                               # generate_taxonomy.py on the paper's run)
│   ├── counter_message_candidates.json  # The 1,000 extremist tweets selected
│   │                               # for counter-messaging, with the LLM's
│   │                               # counter-messaging-potential score (1-10)
│   │                               # and rationale per tweet
│   ├── counter_message_evaluations.json  # Raw 3,043 counter-message votes from
│   │                               # 93 Arabic-speaking evaluators (Fig 1 human bar)
│   └── llm-judge-preferences/      # Per-pair scholar-vs-LLM verdicts from four
│                                   # LLM judges (GPT-3.5-Turbo, GPT-4o,
│                                   # Claude-3-Sonnet, Claude-3.5-Sonnet),
│                                   # each run twice (seed 42 + seed 50);
│                                   # drives the four LLM bars of Fig 1
├── evaluate_classifiers.py         # Reproduces Table 2 (k-shot and taxonomy LLM
│                                   # accuracy, precision, recall, F1, and
│                                   # inter-model kappa)
├── train_marbert.py                # Reproduces any row of Table 3 (MARBERT binary
│                                   # classifier at 1K / 10K / 100K / 250K / 500K
│                                   # training-set sizes)
├── example_inference.py            # Minimal demo: load the published HF model
│                                   # and classify two sample tweets
├── generate_taxonomy.py            # Iteratively induces the taxonomy via GPT-4o
│                                   # (Section 4.2), using Prompt 4 verbatim and
│                                   # a balanced 20K + 20K tweet sample in
│                                   # batches of 100
├── evaluate_counter_messages.py    # Reproduces the human bar of Fig 1 and
│                                   # Fleiss' kappa from the raw 3,043 evaluator
│                                   # votes
├── evaluate_llm_judges.py          # Reproduces the four LLM-judge bars of Fig 1
│                                   # from the raw per-pair verdict files
└── taxonomy_distribution.py        # Reproduces Table 1 (taxonomy label
                                    # distribution across extremist and
                                    # non-extremist eval tweets)
```

## Reproducing Table 2 (LLM classification performance)

```
python3 evaluate_classifiers.py
```

Reads per-tweet JSON outputs under `data/classification/` — each file contains the **original tweet text**, the LLM's classification, and the LLM's reasoning — for each prompting strategy (zero-/one-/few-shot and taxonomy-guided) and each model (GPT-4o, Claude 3.5 Sonnet), then reports accuracy, precision, recall, F1, and GPT-vs-Claude agreement (Cohen's kappa plus percent match) on the joint 2,000-tweet evaluation set. Because the full tweet text is included, researchers can inspect, filter, re-label, or re-run classifiers against the exact same set we used in the paper.

## Reproducing Table 3 (MARBERT classifier at different training sizes)

`train_marbert.py` fine-tunes [UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT) on the LLM-labeled corpus and prints a sklearn `classification_report` that reproduces the corresponding row of Table 3. Each run operates on a balanced 50/50 sub-sample of the 500K pool; the sub-sample is deterministic (`random_state=42`), and training uses the same hyperparameters as the paper (5 epochs, batch 64, lr 2e-6, max_seq 128, AdamW + linear warmup + linear decay, grad clip 1.0).

```
# Reproduce one row of Table 3 (run once per size):
python3 train_marbert.py --size 1000
python3 train_marbert.py --size 10000
python3 train_marbert.py --size 100000
python3 train_marbert.py --size 250000
python3 train_marbert.py --size 500000      # default if --size is omitted

# Skip training and evaluate a saved checkpoint on the same 10% test split.
# You can point --checkpoint at a local directory OR the published HF model:
python3 train_marbert.py --eval-only --checkpoint alfifi/marbert-isis-detector
python3 train_marbert.py --eval-only --checkpoint checkpoints/n500000/model_5
```

The script expects `data/isis-250k.txt` and `data/neg-250k.txt` (one tweet per line, 250K lines each). These are not redistributed here (Twitter/X Terms of Service). The 500K row reported in the paper achieves Accuracy ≈ 0.90, ISIS Precision ≈ 0.88, Recall ≈ 0.94, F1 ≈ 0.91. Per-epoch checkpoints are saved under `checkpoints/n<SIZE>/model_<epoch>/` for later evaluation or downstream use.

### Pre-trained model on Hugging Face

The 500K-trained model is published at [**alfifi/marbert-isis-detector**](https://huggingface.co/alfifi/marbert-isis-detector). Load it directly from your own code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("alfifi/marbert-isis-detector")
model = AutoModelForSequenceClassification.from_pretrained("alfifi/marbert-isis-detector")

enc = tokenizer("your Arabic tweet here", return_tensors="pt",
                max_length=128, truncation=True, padding="max_length")
with torch.no_grad():
    logits = model(**enc).logits
print(model.config.id2label[int(logits.argmax(-1))])  # -> "ISIS" or "NOT-ISIS"
```

Label mapping: `{0: "ISIS", 1: "NOT-ISIS"}`. See the Hugging Face model card for training details, intended-use notes, and limitations.

For a copy-pasteable runnable example, see [`example_inference.py`](example_inference.py) — it loads the Hugging Face model and classifies two sample tweets (one pro-ISIS, one neutral), printing the model's label and per-class probabilities:

```
python3 example_inference.py
```

Expected output:

```
[ISIS]     P(ISIS)=0.9994  P(NOT-ISIS)=0.0006
    والله لا نرضى إلا بالدولة الإسلامية وقتال الكفار في كل مكان

[NOT-ISIS] P(ISIS)=0.0007  P(NOT-ISIS)=0.9993
    اليوم طقس جميل جدا في الرياض ارتفاع درجة الحرارة يدعو للذهاب للبحر
```

## Reproducing Fig 1 (counter-message evaluation)

Figure 1 compares how humans and four LLM evaluators rank scholar-authored versus LLM-authored counter-messages. The human bar and the four LLM bars come from two different data sources and are reproduced by two scripts.

**Human bar.** `evaluate_counter_messages.py` loads the raw vote table and reproduces the human bar plus the inter-rater agreement statistic referenced in Section 5.2.

```
python3 evaluate_counter_messages.py
```

Reads `data/counter_message_evaluations.json` — 3,043 vote records from 93 Arabic-speaking evaluators rating 1,000 scholar-vs-LLM counter-message pairs. The script resolves each pair by majority vote (43 pairs with no majority among the three raters are counted separately; the paper resolved these with a fourth team-member review), prints the distribution, and computes Fleiss' κ over the first three votes per pair. Expected output:

```
LLM better      36.1%   (paper: 36%)
Equal           49.0%   (paper: 49%)
Scholar better  14.9%   (paper: 15%)
Fleiss' kappa   0.411   (moderate agreement)
```

**LLM-judge bars.** `evaluate_llm_judges.py` aggregates the per-pair scholar-vs-LLM verdicts cast by GPT-3.5-Turbo, GPT-4o, Claude-3-Sonnet, and Claude-3.5-Sonnet (see `prompts/08_comparative_eval.md`). Each model was run twice with randomised A/B positions (seed 42 and seed 50) for 1,000 pairs, giving N=2000 judgments per model.

```
python3 evaluate_llm_judges.py
```

Reads `data/llm-judge-preferences/{model}/seed{42,50}/{post_id}_evaluation.json` and prints a single table with the human bar and all four LLM bars side-by-side. Expected output:

```
Humans (majority)    N=957    Scholar= 14.9%  LLM= 36.1%  Equal= 49.0%
gpt-3.5-turbo        N=2000   Scholar= 84.0%  LLM= 16.0%  Equal=  0.0%
gpt-4o               N=2000   Scholar= 98.0%  LLM=  0.5%  Equal=  1.6%
claude-3-sonnet      N=2000   Scholar= 65.3%  LLM=  5.2%  Equal= 29.5%
claude-3.5-sonnet    N=2000   Scholar= 99.0%  LLM=  0.2%  Equal=  0.8%
```

### How the 1,000 counter-message candidates were selected

`data/counter_message_candidates.json` documents the selection step that precedes the evaluation: 20,000 extremist tweets were each scored 1–10 by GPT-4o for counter-messaging potential (see `prompts/06_content_scoring.md`), and the top-scoring 1,000 were kept. The file records the tweet text, the score, and the one-sentence LLM rationale for each of the 1,000 selected tweets. Its index aligns with the `post_id` field in `counter_message_evaluations.json`.

## Reproducing Table 1 (taxonomy distribution)

```
python3 taxonomy_distribution.py
```

Reads the LLM-assigned taxonomy labels under `data/taxonomy-distribution/` (1,000 extremist + 1,000 non-extremist eval tweets, one classification JSON per tweet) and prints the two-panel label distribution shown in Table 1.

## What is NOT in this repository

- **Raw 2015 Arabic Twitter Firehose** — redistribution is not permitted by Twitter/X. The paper was written against the author's full-access archive at the time.
- **The 500K LLM-labeled tweet corpus used for MARBERT training** — full tweet text for this much larger corpus is withheld to stay within Twitter/X's ToS. Note, however, that the smaller but much more carefully curated **2,000-tweet evaluation set** that drives the paper's headline LLM-classification results *is* included with full text (see `data/classification/`).
- **Ground-truth seed account list** — derived from a now-defunct Anonymous-hacking-group crowdsourcing effort; see the archived copy linked in the paper (Section 3).

## Citation

If you use these prompts, predictions, or the taxonomy in your own work, please cite the paper. BibTeX will be added upon acceptance.

## Contact

For questions about reproducing the LLM experiments, please contact the corresponding author.

---

# Appendix: Prompts and System Screenshots

The following reproduces the paper's appendix, organized as it was intended to appear but removed from the submitted PDF to fit the ASONAM page limit. Machine-readable copies of each prompt (for copy-paste into scripts) are in the `prompts/` directory.

## B.1  Prompts

### Prompt 1 — Zero-Shot Classification

Used in Section 4.1 (Baseline Classification with LLMs). The model is asked to classify a tweet as ISIS or NOT-ISIS with no examples.

**System:**
```
You are a JSON-only output API that returns exactly one JSON object with keys "classification" and "reason". No extra text.
```

**User:**
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

---

### Prompt 2 — One-Shot Classification

Used in Section 4.1. The model is given one manually selected pro-ISIS tweet as a reference example.

**System:**
```
You are a JSON-only output API. Return one JSON object with keys "classification", "reason", and "original_post". No extra text.
```

**User:**
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

---

### Prompt 3 — Few-Shot Classification

Used in Section 4.1. The model is given five manually selected pro-ISIS tweets as reference examples.

**System:**
```
You are a JSON-only output API. Return one JSON object with keys "classification", "reason", and "original_post". No extra text.
```

**User:**
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

---

### Prompt 4 — Iterative Taxonomy Generation

Used in Section 4.2. The model builds or refines a hierarchical taxonomy from a batch of tweets, optionally merging with an existing taxonomy. Called iteratively over batches of ~100 tweets until the full sample (20K extremist + 20K negative in our experiments) is processed.

**System:**
```
You are an assistant that organizes tweets into a hierarchical taxonomy.
```

**User:**
```
Create a hierarchical taxonomy for the following tweets. Group related tweets into categories and subcategories based on recurring themes or topics. Each category should contain a list of example tweets (no more than 5) that fall under it. Return the taxonomy in JSON format.

If an existing taxonomy is provided, incorporate the new tweets into the existing taxonomy, updating or merging categories as needed. If new themes emerge, create additional categories. Resolve any conflicts or overlapping categories in a logical manner.

Tweets:
{tweets_batch}

Existing taxonomy (optional):
{existing_taxonomy}

Return the resulting taxonomy in JSON format, reflecting any changes due to the new tweets.
```

---

### Prompt 5 — Taxonomy-Based Classification

Used in Section 4.2 and Section 4.4 (large-scale labeling). The LLM is given the full taxonomy and classifies a post as extremist or not, also assigning one or more matching taxonomy categories.

**System:**
```
You are a JSON-only output API that classifies social media content using a predefined taxonomy. Output a single JSON object containing the classification and reasoning.
```

**User:**
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

---

### Prompt 6 — Finding Counter-Messaging Candidate Posts

Used in Section 5.1 to score extremist posts for counter-messaging potential on a 1–10 scale. The top-scoring 1,000 posts (from a 20,000-tweet pool) were sent to human annotators to write counter-messages.

**System:**
```
You are a JSON-only output API that evaluates social media content. Only output a valid JSON object with the exact requested fields.
```

**User:**
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

---

### Prompt 7 — Counter-Message Generation

Used in Section 5.1 to automatically generate Arabic-language counter-messages for flagged extremist posts. GPT-4o was the backbone for generation in our experiments.

**System:**
```
You are a JSON-only output API that returns exactly one JSON object with keys 'original_tweet' and 'counter_message'. No extra text.
```

**User:**
```
You are a writing assistant specializing in counter-messaging extremist content in the Arabic language.
Please craft a concise counter-message that aims to dissuade readers from accepting or endorsing extremist ideas, especially any calls to violence or misuse of religious concepts, making sure it's in the Arabic language.

Return valid JSON with the following fields:
{
  "original_tweet": "...",
  "counter_message": "..."
}
```

---

### Prompt 8 — Comparative (A vs B) Evaluation

Used in Section 5.2 for LLM-based evaluation of counter-messages. Fed into four LLM evaluators (GPT-4o, Claude 3.5 Sonnet, GPT-3.5-Turbo, and Claude 3 Sonnet) to compare a human-authored response against an LLM-generated response for the same extremist post. Positions of A/B were randomly alternated.

**System:**
```
You are a JSON-only output API that returns exactly one JSON object with keys 'winner' and 'reason'. No extra text.
```

**User:**
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

---

### Prompt 9 — Individual Response Rating

Alternative to the comparative evaluation (Prompt 8). Rates each response independently on a 1–5 Likert scale instead of forcing a pairwise winner. Used for sensitivity analysis and as a check on the pairwise preference signal.

**System:**
```
You are a JSON-only output API that returns exactly one JSON object with the keys 'ratingA', 'reasonA', 'ratingB', and 'reasonB'.
```

**User:**
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

---

### Prompt 10 — Counter-Message Human Instructions (Islamic-Studies Students)

Instructions shown to the Islamic-studies undergraduate volunteers recruited to write the human counter-messages used as ground truth against which LLM-generated counter-messages were compared in Section 5.

**Arabic (original):**

> كطالب علم في الدرسات الإسلامية نقدر لك مساعدتنا في هذا البحث وصياغة رد اسلامي على حوالي 15 تغريدات متطرفة من حسابات داعمة لداعش على تويتر. الغرض من الرد هو دحض اي شبهه او استخدام خاطئ للآيات بشكل مقنع و غير متعالي على القارء. النتيجة المأمولة هي أن القارء لن يتبني الرأي المتطرف في التغريبة بعد قراء الرد.

**English (translation):**

> As a student of Islamic studies, we would appreciate your help in this research and formulating an Islamic response to about 15 extremist tweets from pro-ISIS accounts on Twitter. The purpose of the response is to refute any doubts or misuse of the verses in a convincing and non-condescending manner towards the reader. The desired outcome is that the reader will not adopt the extremist view of the tweet after reading the response.

---

### Prompt 11 — Counter-Message Evaluator Instructions (Lay Arabic Readers)

Instructions shown to the 93 Arabic-speaking lay evaluators (recruited via WhatsApp, mostly in Saudi Arabia) who rated pairs of human-authored and LLM-generated counter-messages in the Section 5.2 evaluation.

> Below is a tweet from one of the extremist group accounts, such as ISIS or others, followed by two replies to this tweet.
>
> - If you believe the reply to the tweet is appropriate, click thumbs up.
> - If you believe the reply is inappropriate, click thumbs down.
> - Finally, choose which reply you personally think is more convincing than the other.

Each pair received three independent evaluations. Final labels were assigned by majority vote; 43 pairs with no majority were broken by a fourth team-member review.

---

## B.2  System Screenshots

### Figure B.1 — Counter-Message Evaluation Interface

Snapshot of the system used to collect evaluation of human-vs-LLM counter-messages. In this example, Response 1 is from an Islamic-studies student and Response 2 is from an LLM. Positions are randomly alternated between pairs.

**(a) A scholar response**

![Counter-message evaluation interface showing a scholar-authored response](screenshots/human-eval-arabic.png)

**(b) An AI response**

![Counter-message evaluation interface showing an LLM-authored response](screenshots/human-eval-english.png)

---

### Figure B.2 — Taxonomy Evolution (illustrative)

Illustrative heatmap from a pre-stabilization run showing how the iterative taxonomy-induction procedure behaves across batches: labels are merged (e.g., Sports Commentary), removed (e.g., Uncategorized), or introduced mid-run (e.g., "Inappropriate Content"). This figure is included as a visualization of the *process*, not as the exact evolution trace of the taxonomy shipped in this repo — the final stabilized taxonomy used for the paper's Section 3.2 and Figure 2 analyses is committed at [`data/taxonomy.json`](data/taxonomy.json).

![Heatmap showing which taxonomy categories are present at each iteration](screenshots/taxonomy_heatmap.png)
