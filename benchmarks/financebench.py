"""
benchmarks/financebench.py
───────────────────────────
BYOB benchmark for FinanceBench SEC-filing question answering.

Scorer (placeholder — LLM-as-judge scorer can replace this later):
  - If both gold answer and model response look numeric:
      Extract first number from each, compare within 1 % relative tolerance.
  - Otherwise:
      Case-insensitive substring-containment of gold answer in model response.

⚠️  Requires preprocessing first (flattens the `evidence` array):
    python benchmarks/preprocess_financebench.py

Dataset : data/processed/financebench/train.jsonl
target_field : "answer"

Install & run:
    venv\\Scripts\\nemo-evaluator-byob benchmarks\\financebench.py
    venv\\Scripts\\nemo-evaluator run_eval \\
        --eval_type byob_financebench.financebench \\
        --model_url http://localhost:1234/v1 \\
        --model_id <model-id> \\
        --model_type chat \\
        --output_dir results\\financebench
"""

import re
from pathlib import Path

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

_HERE    = Path(__file__).resolve().parent
_DATASET = str(_HERE.parent / "data" / "processed" / "financebench" / "train.jsonl")

# ── Prompt template ───────────────────────────────────────────────────────────
# Jinja2 mode for safe {{ }} substitution (evidence_text can contain braces).
_PROMPT = """\
{# financebench prompt — Jinja2 mode #}
You are a financial analyst reviewing SEC filing '{{ doc_name }}'. \
Answer the following question relying directly on the provided context evidence.

Context Evidence:
{{ evidence_text }}

Question: {{ question }}

Provide a direct, factual answer with the exact numeric value or conclusion based on the filing.\
"""


# ── Numeric helpers ───────────────────────────────────────────────────────────
_STRIP_CURRENCY = re.compile(r"[$,% ]")
_NUMBER_RE      = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_number(text: str):
    """
    Extract the FIRST numeric value from `text` after stripping $, commas, %.
    Returns float or None.
    """
    cleaned = _STRIP_CURRENCY.sub("", text)
    m = _NUMBER_RE.search(cleaned)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


# ── BYOB benchmark registration ───────────────────────────────────────────────
@benchmark(
    name="financebench",
    dataset=_DATASET,
    prompt=_PROMPT,
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def financebench_scorer(sample: ScorerInput) -> dict:
    """
    Score a FinanceBench answer.

    Two-stage scoring:
      1. Numeric path  — if both gold and response parse to a number, compare
                         within 1 % relative tolerance.
      2. String path   — case-insensitive substring check (gold ⊆ response).

    Returns:
        {"correct": bool, "numeric_match": bool}
          numeric_match=True only when the numeric path was taken AND passed.
    """
    gold     = str(sample.target).strip()
    response = sample.response.strip()

    gold_num = _parse_number(gold)
    resp_num = _parse_number(response)

    if gold_num is not None and resp_num is not None:
        if gold_num == 0.0:
            correct = abs(resp_num - gold_num) < 1e-6
        else:
            correct = abs(resp_num - gold_num) / abs(gold_num) <= 0.01
        return {"correct": correct, "numeric_match": correct}

    # Fallback: is the gold answer (cleaned) contained in the response?
    correct = gold.lower() in response.lower()
    return {"correct": correct, "numeric_match": False}
