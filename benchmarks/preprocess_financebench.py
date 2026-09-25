"""
benchmarks/preprocess_financebench.py
──────────────────────────────────────
Flattens the nested `evidence` array in FinanceBench into a single joined
text field so the BYOB prompt template can use a simple `{evidence_text}`
placeholder instead of needing a Jinja2 loop.

Input  : data/raw/financebench/train.jsonl
           → rows with {"question": "...", "answer": "...", "evidence": [...]}
Output : data/processed/financebench/train.jsonl
           → flat rows with {"doc_name", "question", "answer", "evidence_text"}

Run from project root:
    python benchmarks/preprocess_financebench.py
"""

import argparse
import json
from pathlib import Path

_DEFAULT_INPUT  = "data/raw/financebench/train.jsonl"
_DEFAULT_OUTPUT = "data/processed/financebench/train.jsonl"


def _join_evidence(evidence_list) -> str:
    """
    Join evidence items into a single readable block.
    Prefers `evidence_text_full_page`; falls back to `evidence_text`.
    Prepends [Page N] header when available.
    """
    if not isinstance(evidence_list, list):
        return str(evidence_list).strip() if evidence_list else "[No context provided]"

    parts = []
    for ev in evidence_list:
        if isinstance(ev, dict):
            text = (ev.get("evidence_text_full_page") or ev.get("evidence_text", "")).strip()
            page = ev.get("evidence_page_num")
            header = f"[Page {page}]\n" if page is not None else ""
            if text:
                parts.append(f"{header}{text}")
        elif isinstance(ev, str) and ev.strip():
            parts.append(ev.strip())

    return "\n\n---\n\n".join(parts) if parts else "[No context provided]"


def preprocess(input_path: str = _DEFAULT_INPUT, output_path: str = _DEFAULT_OUTPUT) -> int:
    """
    Read every raw JSONL row, flatten evidence, write flat rows.
    Returns number of rows written.
    """
    in_path  = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            flat = {
                "financebench_id": row.get("financebench_id", ""),
                "doc_name":        row.get("doc_name", "SEC Filing"),
                "question":        row.get("question", ""),
                "answer":          row.get("answer", ""),
                "evidence_text":   _join_evidence(row.get("evidence", [])),
            }
            fout.write(json.dumps(flat, ensure_ascii=False) + "\n")
            count += 1

    print(f"[FinanceBench] Wrote {count:,} rows → {out_path.resolve()}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten FinanceBench evidence array into a single text field."
    )
    parser.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help=f"Path to raw JSONL (default: {_DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help=f"Path for processed JSONL (default: {_DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)
