"""
benchmarks/preprocess_financial_phrasebank.py
─────────────────────────────────────────────
Adds a `label_text` field (negative / neutral / positive) to every row of the
Financial PhraseBank JSONL so that the BYOB benchmark can use it directly as
`target_field`.

Input  : data/raw/financial_phrasebank/sentences_allagree/train.jsonl
           → rows like  {"sentence": "...", "label": 0|1|2}
Output : data/processed/financial_phrasebank/train.jsonl
           → rows like  {"sentence": "...", "label": 1, "label_text": "neutral"}

Run from project root:
    python benchmarks/preprocess_financial_phrasebank.py
"""

import argparse
import json
from pathlib import Path

# Mapping from integer label → string sentiment used as the BYOB target
_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

_DEFAULT_INPUT = "data/raw/financial_phrasebank/sentences_allagree/train.jsonl"
_DEFAULT_OUTPUT = "data/processed/financial_phrasebank/train.jsonl"


def preprocess(input_path: str = _DEFAULT_INPUT, output_path: str = _DEFAULT_OUTPUT) -> int:
    """
    Read every JSONL row, inject `label_text`, write to output_path.
    Returns the number of rows written.
    """
    in_path = Path(input_path)
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
            # Inject label_text (default "neutral" if label not in map)
            row["label_text"] = _LABEL_MAP.get(row.get("label"), "neutral")
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"[Financial PhraseBank] Wrote {count:,} rows → {out_path.resolve()}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add label_text field to Financial PhraseBank JSONL."
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
