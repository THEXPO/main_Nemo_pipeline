"""
load_samples.py
Loads a small sample (5 items each) from all 5 BFSI datasets in data/raw/
for pipeline smoke testing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

FPB_LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def load_banking77(data_dir: str = "data/raw", n: int = 5) -> List[Dict[str, Any]]:
    """
    Loads samples from Banking77 (data/raw/banking77/train.jsonl).
    Expected schema: {"text": str, "label": str}
    """
    path = Path(data_dir) / "banking77" / "train.jsonl"
    if not path.exists():
        print(f"[ERROR] Banking77 file not found at: {path}")
        return []

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Schema validation
            if "text" not in item or "label" not in item:
                print(f"[ERROR] Banking77 schema mismatch in {path}. Expected keys 'text', 'label', got: {list(item.keys())}")
                return []
            items.append(item)
            if len(items) >= n:
                break

    print(f"[Banking77] Loaded {len(items)} samples. Sample keys: {list(items[0].keys()) if items else 'None'}")
    return items


def load_financial_phrasebank(data_dir: str = "data/raw", n: int = 5) -> List[Dict[str, Any]]:
    """
    Loads samples from Financial PhraseBank (data/raw/financial_phrasebank/sentences_allagree/train.jsonl).
    Expected schema: {"sentence": str, "label": int, "label_text": str}
    """
    path = Path(data_dir) / "financial_phrasebank" / "sentences_allagree" / "train.jsonl"
    if not path.exists():
        print(f"[ERROR] Financial PhraseBank file not found at: {path}")
        return []

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "sentence" not in item or "label" not in item:
                print(f"[ERROR] PhraseBank schema mismatch in {path}. Expected 'sentence', 'label', got: {list(item.keys())}")
                return []
            # Ensure label_text is present
            if "label_text" not in item:
                item["label_text"] = FPB_LABEL_NAMES.get(item["label"], str(item["label"]))
            items.append(item)
            if len(items) >= n:
                break

    print(f"[Financial PhraseBank] Loaded {len(items)} samples. Sample keys: {list(items[0].keys()) if items else 'None'}")
    return items


def load_financebench(data_dir: str = "data/raw", n: int = 5) -> List[Dict[str, Any]]:
    """
    Loads samples from FinanceBench (data/raw/financebench/train.jsonl).
    Expected schema: {"financebench_id", "doc_name", "question", "answer", "justification", "evidence": [...]}
    """
    path = Path(data_dir) / "financebench" / "train.jsonl"
    if not path.exists():
        print(f"[ERROR] FinanceBench file not found at: {path}")
        return []

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            required_keys = {"question", "answer"}
            if not required_keys.issubset(item.keys()):
                print(f"[ERROR] FinanceBench schema mismatch in {path}. Missing required keys. Got: {list(item.keys())}")
                return []
            items.append(item)
            if len(items) >= n:
                break

    print(f"[FinanceBench] Loaded {len(items)} samples. Sample keys: {list(items[0].keys()) if items else 'None'}")
    return items


def load_finqa(data_dir: str = "data/raw", n: int = 5) -> List[Dict[str, Any]]:
    """
    Loads samples from FinQA (data/raw/finqa/dev.json).
    Expected schema: list of {"id", "pre_text": [...], "table": [[...]], "post_text": [...], "qa": {...}}
    """
    path = Path(data_dir) / "finqa" / "dev.json"
    if not path.exists():
        print(f"[ERROR] FinQA file not found at: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[ERROR] FinQA expected top-level list, got {type(data)}")
        return []

    items = data[:n]
    if items:
        first = items[0]
        if "qa" not in first or "table" not in first:
            print(f"[ERROR] FinQA schema mismatch in {path}. Expected 'qa', 'table' in records, got: {list(first.keys())}")
            return []

    print(f"[FinQA] Loaded {len(items)} samples. Sample keys: {list(items[0].keys()) if items else 'None'}")
    return items


def load_convfinqa(data_dir: str = "data/raw", n: int = 5) -> List[Dict[str, Any]]:
    """
    Loads samples from ConvFinQA (data/raw/convfinqa/dev.json).
    Expected schema: list of {"id", "pre_text", "table", "annotation": {"dialogue_break": [...], "exe_ans_list": [...], "turn_program": [...]}}
    """
    path = Path(data_dir) / "convfinqa" / "dev.json"
    if not path.exists():
        print(f"[ERROR] ConvFinQA file not found at: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[ERROR] ConvFinQA expected top-level list, got {type(data)}")
        return []

    items = data[:n]
    if items:
        first = items[0]
        if "annotation" not in first or "dialogue_break" not in first.get("annotation", {}):
            print(f"[ERROR] ConvFinQA schema mismatch in {path}. Expected 'annotation.dialogue_break', got: {list(first.keys())}")
            return []

    print(f"[ConvFinQA] Loaded {len(items)} samples. Sample keys: {list(items[0].keys()) if items else 'None'}")
    return items


def load_all_samples(data_dir: str = "data/raw", n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Loads n samples for each of the 5 datasets."""
    print("=" * 60)
    print(f"LOADING SAMPLES (n={n} per dataset) FROM: {data_dir}")
    print("=" * 60)
    datasets = {
        "banking77": load_banking77(data_dir, n),
        "financial_phrasebank": load_financial_phrasebank(data_dir, n),
        "financebench": load_financebench(data_dir, n),
        "finqa": load_finqa(data_dir, n),
        "convfinqa": load_convfinqa(data_dir, n),
    }
    print("=" * 60 + "\n")
    return datasets


if __name__ == "__main__":
    load_all_samples()
