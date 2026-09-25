"""
benchmarks/financial_phrasebank.py
────────────────────────────────────
BYOB benchmark for Financial PhraseBank sentiment classification.

Scorer:
  - Lower-case the model response.
  - Find the FIRST occurrence of "negative", "neutral", or "positive" (word boundary).
  - Exact-match that word against the gold `label_text`.

⚠️  Requires preprocessing first (adds `label_text` field to each row):
    python benchmarks/preprocess_financial_phrasebank.py

Dataset : data/processed/financial_phrasebank/train.jsonl
target_field : "label_text"   (values: "negative" | "neutral" | "positive")

Install & run:
    venv\\Scripts\\nemo-evaluator-byob benchmarks\\financial_phrasebank.py
    venv\\Scripts\\nemo-evaluator run_eval \\
        --eval_type byob_financial_phrasebank.financial_phrasebank \\
        --model_url http://localhost:1234/v1 \\
        --model_id <model-id> \\
        --model_type chat \\
        --output_dir results\\financial_phrasebank
"""

import re
from pathlib import Path

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

_HERE    = Path(__file__).resolve().parent
_DATASET = str(_HERE.parent / "data" / "processed" / "financial_phrasebank" / "train.jsonl")

# Valid sentiment words (order matters for first-occurrence search)
_SENTIMENT_WORDS = ("negative", "neutral", "positive")

# ── Prompt template ───────────────────────────────────────────────────────────
# Jinja2 mode (block-comment at top) to safely handle sentences with braces.
_PROMPT = """\
{# financial_phrasebank prompt — Jinja2 mode #}
You are a financial sentiment analysis expert. \
Analyze the sentiment of the following sentence from a financial disclosure or news report.

Sentence: "{{ sentence }}"

Question: Is the sentiment of this sentence negative, neutral, or positive?
State your answer clearly as one of: negative, neutral, or positive.\
"""


# ── BYOB benchmark registration ───────────────────────────────────────────────
@benchmark(
    name="financial-phrasebank",
    dataset=_DATASET,
    prompt=_PROMPT,
    target_field="label_text",
    endpoint_type="chat",
)
@scorer
def financial_phrasebank_scorer(sample: ScorerInput) -> dict:
    """
    Score a Financial PhraseBank sentiment prediction.

    Logic:
      1. Lower-case the full model response.
      2. Find the earliest occurrence of "negative", "neutral", or "positive"
         as a whole word.
      3. Return {"correct": True} only if the found word matches the gold label_text.

    Returns:
        {"correct": bool}
    """
    response_lower = sample.response.lower()
    target         = str(sample.target).strip().lower()

    # Find the position of each sentiment word; take the earliest one.
    first_word  = None
    first_pos   = len(response_lower) + 1  # sentinel

    for word in _SENTIMENT_WORDS:
        m = re.search(r"\b" + word + r"\b", response_lower)
        if m and m.start() < first_pos:
            first_pos  = m.start()
            first_word = word

    return {"correct": first_word is not None and first_word == target}
