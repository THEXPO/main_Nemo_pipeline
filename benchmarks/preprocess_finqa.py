"""
benchmarks/preprocess_finqa.py
────────────────────────────────
Converts FinQA dev.json (a JSON array of nested records) into a flat JSONL
file that BYOB can stream row by row.

Input  : data/raw/finqa/dev.json
           → top-level list of {"pre_text", "table", "post_text", "qa": {"question", "exe_ans", ...}}
Output : data/processed/finqa/dev.jsonl
           → flat rows: {"id", "context", "question", "answer"}
             where `context` = pre_text + pipe-delimited table + post_text
             and   `answer`  = str(qa.exe_ans)

Run from project root:
    python benchmarks/preprocess_finqa.py
"""

import argparse
import json
from pathlib import Path

_DEFAULT_INPUT  = "data/raw/finqa/dev.json"
_DEFAULT_OUTPUT = "data/processed/finqa/dev.jsonl"


def _format_table(table) -> str:
    """
    Convert a 2-D list (list of rows, each row a list of cells) to pipe-separated
    text rows.  Handles missing / malformed tables gracefully.
    """
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
    Convert one FinQA JSON record into a flat dict suitable for BYOB.
    Fields produced:
        id        – original record id (or "")
        context   – readable string combining pre_text, table, post_text
        question  – the QA question
        answer    – string representation of exe_ans (the executable answer)
    """
    # --- Context: pre_text -----------------------------------------------
    pre_text = item.get("pre_text", [])
    pre_str  = " ".join(str(s) for s in pre_text).strip() \
               if isinstance(pre_text, list) else str(pre_text).strip()

    # --- Context: table --------------------------------------------------
    table_str = _format_table(item.get("table", []))

    # --- Context: post_text ----------------------------------------------
    post_text = item.get("post_text", [])
    post_str  = " ".join(str(s) for s in post_text).strip() \
                if isinstance(post_text, list) else str(post_text).strip()

    # --- QA block --------------------------------------------------------
    qa       = item.get("qa", {}) or {}
    question = qa.get("question", "") if isinstance(qa, dict) else ""
    # `exe_ans` is the executed numerical answer (float/int); fall back to
    # the string `answer` field if exe_ans is absent.
    exe_ans  = qa.get("exe_ans")
    if exe_ans is None:
        exe_ans = qa.get("answer", "")
    answer = str(exe_ans)

    # --- Assemble context ------------------------------------------------
    context_parts = []
    if pre_str:
        context_parts.append(f"Context Introduction:\n{pre_str}")
    if table_str and table_str != "[No table provided]":
        context_parts.append(f"Financial Table:\n{table_str}")
    if post_str:
        context_parts.append(f"Footnotes / Notes:\n{post_str}")

    return {
        "id":       item.get("id", ""),
        "context":  "\n\n".join(context_parts),
        "question": question,
        "answer":   answer,
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
        raise ValueError(f"Expected a top-level JSON array in {in_path}, got {type(data).__name__}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for item in data:
            flat = _flatten(item)
            fout.write(json.dumps(flat, ensure_ascii=False) + "\n")
            count += 1

    print(f"[FinQA] Wrote {count:,} rows → {out_path.resolve()}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten FinQA dev.json (JSON array) to a flat JSONL file."
    )
    parser.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help=f"Path to FinQA JSON array (default: {_DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help=f"Path for output JSONL (default: {_DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)
