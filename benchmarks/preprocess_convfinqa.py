"""
benchmarks/preprocess_convfinqa.py
────────────────────────────────────
Converts ConvFinQA dev.json (a JSON array of multi-turn dialogue records) into
a flat JSONL file that BYOB can stream row by row.

Each raw record has N dialogue turns stored in annotation.dialogue_break /
annotation.exe_ans_list.  We flatten to one BYOB row per record where:

    - `context`          = pre_text + table (as readable text)
    - `dialogue_history` = all turns EXCEPT the last (Q + A pairs rendered
                           as "User: …\nAssistant: …")
    - `final_question`   = the last question in dialogue_break
    - `answer`           = str(exe_ans_list[-1])  — final turn's numeric answer

Input  : data/raw/convfinqa/dev.json
Output : data/processed/convfinqa/dev.jsonl

Run from project root:
    python benchmarks/preprocess_convfinqa.py
"""

import argparse
import json
from pathlib import Path

_DEFAULT_INPUT  = "data/raw/convfinqa/dev.json"
_DEFAULT_OUTPUT = "data/processed/convfinqa/dev.jsonl"


def _format_table(table) -> str:
    """Convert a 2-D list into pipe-separated text rows."""
    if not table:
        return "[No table provided]"
    if isinstance(table, str):
        return table
    if not isinstance(table, list):
        return str(table)
    rows = []
    for row in table:
        if isinstance(row, list):
            rows.append(" | ".join(str(cell).strip() for cell in row))
        else:
            rows.append(str(row).strip())
    return "\n".join(rows)


def _flatten(item: dict) -> dict:
    """
    Convert one ConvFinQA JSON record into a flat dict suitable for BYOB.
    Fields produced:
        id               – original record id
        context          – readable string (pre_text + table)
        dialogue_history – all completed Q/A turns rendered as prose
        final_question   – the current (last) dialogue turn question
        answer           – the last executed numeric answer (as string)
    """
    # --- Context ---------------------------------------------------------
    pre_text = item.get("pre_text", [])
    pre_str  = " ".join(str(s) for s in pre_text).strip() \
               if isinstance(pre_text, list) else str(pre_text).strip()
    table_str = _format_table(item.get("table", []))

    context_parts = []
    if pre_str:
        context_parts.append(f"Report Excerpt:\n{pre_str}")
    if table_str and table_str != "[No table provided]":
        context_parts.append(f"Financial Table:\n{table_str}")

    # --- Dialogue --------------------------------------------------------
    annotation = item.get("annotation") or {}
    if not isinstance(annotation, dict):
        annotation = {}

    questions = annotation.get("dialogue_break", []) or []
    answers   = annotation.get("exe_ans_list",   []) or []
    total     = len(questions)

    # History = all turns before the final one
    history_parts = []
    for i in range(max(0, total - 1)):
        ans_val = answers[i] if i < len(answers) else "[Answered]"
        history_parts.append(f"User: {questions[i]}\nAssistant: {ans_val}")

    final_q = questions[-1] if questions else "[No question]"
    final_a = str(answers[-1]) if answers else ""

    return {
        "id":               item.get("id", ""),
        "context":          "\n\n".join(context_parts),
        "dialogue_history": "\n".join(history_parts) if history_parts else "(Beginning of conversation)",
        "final_question":   final_q,
        "answer":           final_a,
    }


def preprocess(input_path: str = _DEFAULT_INPUT, output_path: str = _DEFAULT_OUTPUT) -> int:
    """
    Load JSON array, flatten each item, write JSONL.
    Returns number of rows written.
    """
    in_path  = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a top-level JSON array in {in_path}, got {type(data).__name__}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for item in data:
            flat = _flatten(item)
            fout.write(json.dumps(flat, ensure_ascii=False) + "\n")
            count += 1

    print(f"[ConvFinQA] Wrote {count:,} rows → {out_path.resolve()}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten ConvFinQA dev.json (JSON array) to a flat JSONL file."
    )
    parser.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help=f"Path to ConvFinQA JSON array (default: {_DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help=f"Path for output JSONL (default: {_DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)
