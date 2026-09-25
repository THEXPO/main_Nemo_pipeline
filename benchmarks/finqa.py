"""
benchmarks/finqa.py
─────────────────────
BYOB benchmark for FinQA numerical reasoning over corporate financial reports.

Scorer:
  Extract the first number from the model response and compare it against
  the gold exe_ans value within 1 % relative tolerance.

⚠️  Requires preprocessing first (flattens JSON array to JSONL):
    python benchmarks/preprocess_finqa.py

Dataset : data/processed/finqa/dev.jsonl
target_field : "answer"  (str representation of qa.exe_ans)

Install & run:
    venv\\Scripts\\nemo-evaluator-byob benchmarks\\finqa.py
    venv\\Scripts\\nemo-evaluator run_eval \\
        --eval_type byob_finqa.finqa \\
        --model_url http://localhost:1234/v1 \\
        --model_id <model-id> \\
        --model_type chat \\
        --output_dir results\\finqa
"""

import re
from pathlib import Path

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

_HERE    = Path(__file__).resolve().parent
_DATASET = str(_HERE.parent / "data" / "processed" / "finqa" / "dev.jsonl")

# ── Prompt template ───────────────────────────────────────────────────────────
# Jinja2 mode (block-comment) so that financial text containing {{ or }}
# does not break template rendering.
_PROMPT = """\
{# finqa prompt — Jinja2 mode #}
You are an expert financial auditor performing numerical reasoning over corporate disclosures.

{{ context }}

Question: {{ question }}

Please provide:
1. Step-by-step arithmetic calculations showing where each number came from.
2. The final numerical answer.\
"""


# ── Numeric helpers ───────────────────────────────────────────────────────────
_STRIP_CURRENCY = re.compile(r"[$%]")
# Match numbers with optional commas as thousands-separators, then optional decimals
_NUMBER_RE      = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _extract_number(text: str):
    """
    Extract the FIRST numeric value found in `text` after stripping $ and %.
    Returns float or None.
    """
    cleaned = _STRIP_CURRENCY.sub("", text)
    m = _NUMBER_RE.search(cleaned)
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            pass
    return None


# ── BYOB benchmark registration ───────────────────────────────────────────────
@benchmark(
    name="finqa",
    dataset=_DATASET,
    prompt=_PROMPT,
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def finqa_scorer(sample: ScorerInput) -> dict:
    """
    Score a FinQA numerical answer.

    Extracts the first number from both the gold answer and the model response,
    then checks whether they agree within 1 % relative tolerance.

    If either value cannot be parsed as a number the sample is marked incorrect.

    Returns:
        {"correct": bool, "numeric_match": bool}
    """
    gold_num = _extract_number(str(sample.target))
    resp_num = _extract_number(sample.response)

    if gold_num is None or resp_num is None:
        return {"correct": False, "numeric_match": False}

    if gold_num == 0.0:
        correct = abs(resp_num - gold_num) < 1e-6
    else:
        correct = abs(resp_num - gold_num) / abs(gold_num) <= 0.01

    return {"correct": correct, "numeric_match": correct}
