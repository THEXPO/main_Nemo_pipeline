"""
benchmarks/banking77.py
────────────────────────
BYOB benchmark for Banking77 customer-intent classification.

The full list of 77 valid category labels is loaded from categories.json at
module import time and embedded directly into the prompt template string.
That means the prompt BYOB sends to the model already contains the complete
label list; the only per-row substitution is `{{ text }}` (the customer message).

Scorer mirrors the existing parse_banking77_output() logic from parsers.py:
  1. Strip common prefixes (Category: / Intent: / etc.)
  2. Exact-match against valid labels
  3. Substring scan across the full response
  4. Fuzzy fallback via difflib.get_close_matches (cutoff 0.6)

Dataset : data/raw/banking77/train.jsonl   (raw file is already flat — no preprocessing needed)
target_field : "label"

Install & run:
    venv\\Scripts\\nemo-evaluator-byob benchmarks\\banking77.py
    venv\\Scripts\\nemo-evaluator run_eval \\
        --eval_type byob_banking77.banking77 \\
        --model_url http://localhost:1234/v1 \\
        --model_id <model-id> \\
        --model_type chat \\
        --output_dir results\\banking77
"""

import difflib
import json
import re
from pathlib import Path

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

# ── 1. Load label list once at module level ──────────────────────────────────
_HERE = Path(__file__).resolve().parent
_CATEGORIES_PATH = _HERE.parent / "data" / "raw" / "banking77" / "categories.json"

if not _CATEGORIES_PATH.exists():
    raise FileNotFoundError(
        f"Banking77 categories file not found at: {_CATEGORIES_PATH}\n"
        "Make sure you have downloaded the dataset with download_bfsi_datasets.py."
    )

with _CATEGORIES_PATH.open("r", encoding="utf-8") as _f:
    _BANKING77_LABELS: list = json.load(_f)

_CATEGORIES_BLOCK = "\n".join(f"- {cat}" for cat in _BANKING77_LABELS)

# ── 2. Build prompt template (categories embedded, {text} filled per-row) ────
# A Jinja2 block comment triggers Jinja2 mode — this lets Jinja2 handle
# the {{ text }} substitution safely, even if the customer message contains
# literal curly braces.
_PROMPT = (
    "{# banking77 prompt — Jinja2 mode for safe {{ }} substitution #}\n"
    "You are an AI assistant for a retail bank. "
    "Classify the customer's inquiry into exactly ONE valid intent category from the list below.\n\n"
    "Valid Intent Categories:\n"
    + _CATEGORIES_BLOCK
    + "\n\n"
    "Customer Message: \"{{ text }}\"\n\n"
    "Instructions: Respond with ONLY the exact category label from the list above. "
    "Do not include any explanation, extra text, quotation marks, or punctuation.\n"
    "Do not call any function or tool. Do not output JSON. Respond with plain text containing only the category label.\n\n"
    "Category:"
)

# ── 3. Dataset path (absolute, resolved from this file's location) ────────────
_DATASET = str(_HERE.parent / "data" / "raw" / "banking77" / "train.jsonl")


# ── 4. Scorer helper (mirrors parsers.parse_banking77_output) ─────────────────
def _match_label(response: str, valid_labels: list, cutoff: float = 0.6):
    """
    Normalize model response and fuzzy-match against valid_labels.
    Returns the matched label string, or None if no match found.

    Steps (same as existing parsers.py logic):
      1. Extract first non-empty line and strip common prefixes.
      2. Exact match (lowercase, underscored).
      3. Whole-word scan of the full response.
      4. Fuzzy match with difflib (cutoff=0.6).
      5. Fuzzy match on first 4 word-pieces.
    """
    if not response:
        return None

    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if not lines:
        return None

    # Step 1 — strip prefixes like "Category:", "Intent:", "Label:", "-"
    first = lines[0]
    clean = re.sub(
        r"^(category|intent|label|answer)\s*:\s*", "", first, flags=re.IGNORECASE
    ).strip()
    clean = clean.lstrip("-*`'\"").rstrip(".*`'\"").strip()

    candidate = clean.lower().replace(" ", "_").replace("-", "_")

    # Step 2 — exact match
    if candidate in valid_labels:
        return candidate

    sanitized = re.sub(r"[^\w]", "", candidate)
    for lbl in valid_labels:
        if sanitized == lbl:
            return lbl

    # Step 3 — whole-word scan across full response
    full_lower = response.lower()
    for lbl in valid_labels:
        if re.search(r"\b" + re.escape(lbl) + r"\b", full_lower):
            return lbl

    # Step 4 — fuzzy on candidate
    hits = difflib.get_close_matches(candidate, valid_labels, n=1, cutoff=cutoff)
    if hits:
        return hits[0]

    # Step 5 — fuzzy on first 4 underscore-separated pieces
    words = candidate.split("_")
    if len(words) > 1:
        phrase = "_".join(words[:4])
        hits = difflib.get_close_matches(phrase, valid_labels, n=1, cutoff=cutoff)
        if hits:
            return hits[0]

    return None


# ── 5. BYOB benchmark registration ───────────────────────────────────────────
@benchmark(
    name="banking77",
    dataset=_DATASET,
    prompt=_PROMPT,
    target_field="label",
    endpoint_type="chat",
)
@scorer
def banking77_scorer(sample: ScorerInput) -> dict:
    """
    Score a Banking77 prediction.

    Returns:
        {"correct": bool}   — True if the parsed label matches the gold label.

    The scorer normalises the model output with the same fuzzy-matching
    pipeline used in parsers.parse_banking77_output so results are comparable
    with the original smoke_test.py numbers.
    """
    parsed = _match_label(sample.response, _BANKING77_LABELS)
    gold   = str(sample.target).strip().lower()
    return {"correct": parsed is not None and parsed == gold}
