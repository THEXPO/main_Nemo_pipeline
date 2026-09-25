"""
benchmarks/convfinqa.py
─────────────────────────
BYOB benchmark for ConvFinQA conversational financial question answering.

Scorer: same tolerance-based numeric match as FinQA —
  extract the first number from the model response and compare to the gold
  final-turn exe_ans within 1 % relative tolerance.

⚠️  Requires preprocessing first (flattens JSON array to JSONL with dialogue):
    python benchmarks/preprocess_convfinqa.py

Dataset : data/processed/convfinqa/dev.jsonl
target_field : "answer"  (str of the last turn's exe_ans)

Install & run:
    venv\\Scripts\\nemo-evaluator-byob benchmarks\\convfinqa.py
    venv\\Scripts\\nemo-evaluator run_eval \\
        --eval_type byob_convfinqa.convfinqa \\
        --model_url http://localhost:1234/v1 \\
        --model_id <model-id> \\
        --model_type chat \\
        --output_dir results\\convfinqa
"""

import re
from pathlib import Path

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

_HERE    = Path(__file__).resolve().parent
_DATASET = str(_HERE.parent / "data" / "processed" / "convfinqa" / "dev.jsonl")

# ── Prompt template ───────────────────────────────────────────────────────────
# Jinja2 mode (block-comment) so that financial text containing {{ or }}
# does not break template rendering.
_PROMPT = """\
{# convfinqa prompt — Jinja2 mode #}
You are an interactive financial analyst participating in a multi-turn dialogue \
based on the following financial report.

{{ context }}

Conversation History:
{{ dialogue_history }}

Current Turn Question: {{ final_question }}

Answer the current turn question directly based on the table, report context, \
and preceding dialogue.\
"""


# ── Numeric helpers ───────────────────────────────────────────────────────────
_STRIP_CURRENCY = re.compile(r"[$%]")
_NUMBER_RE      = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _extract_number(text: str):
    """Extract the first numeric value from `text` after stripping $ and %."""
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
    name="convfinqa",
    dataset=_DATASET,
    prompt=_PROMPT,
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def convfinqa_scorer(sample: ScorerInput) -> dict:
    """
    Score a ConvFinQA final-turn numerical answer.

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
