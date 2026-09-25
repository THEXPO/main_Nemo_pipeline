"""
parsers.py
Output parsing and normalization utilities for benchmark evaluations.
"""

import difflib
import json
import re
from pathlib import Path
from typing import List, Optional

_DEFAULT_CATEGORIES_PATH = Path("data/raw/banking77/categories.json")
_CACHED_BANKING77_LABELS: Optional[List[str]] = None


def get_banking77_labels(categories_path: Path = _DEFAULT_CATEGORIES_PATH) -> List[str]:
    """Loads and caches the 77 valid Banking77 category labels."""
    global _CACHED_BANKING77_LABELS
    if _CACHED_BANKING77_LABELS is None:
        if categories_path.exists():
            with open(categories_path, "r", encoding="utf-8") as f:
                _CACHED_BANKING77_LABELS = json.load(f)
        else:
            # Fallback path if run from a subfolder
            alt_path = Path(__file__).resolve().parent / "data" / "raw" / "banking77" / "categories.json"
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    _CACHED_BANKING77_LABELS = json.load(f)
            else:
                _CACHED_BANKING77_LABELS = []
    return _CACHED_BANKING77_LABELS


def parse_banking77_output(
    output_text: Optional[str],
    valid_labels: Optional[List[str]] = None,
    cutoff: float = 0.6,
) -> Optional[str]:
    """
    Parses and normalizes model output for Banking77 intent classification.

    1. Cleans the output string (strips whitespace, common prefixes, markdown, lowercase).
    2. Tests exact match against valid_labels.
    3. Falls back to fuzzy matching (difflib.get_close_matches with cutoff).
    4. Returns None if no valid label can be matched.
    """
    if not output_text:
        return None

    if valid_labels is None:
        valid_labels = get_banking77_labels()

    if not valid_labels:
        return None

    # Step 1: Extract the first meaningful line or phrase
    raw = output_text.strip()
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if not lines:
        return None

    first_line = lines[0]

    # Remove common prefixes models often output (e.g. "Category:", "Intent:", "Label:", "-")
    clean_line = re.sub(r"^(category|intent|label|answer)\s*:\s*", "", first_line, flags=re.IGNORECASE).strip()
    clean_line = clean_line.lstrip("-*`'\"").rstrip(".*`'\"").strip()

    # Normalize candidate: lowercase and replace spaces/hyphens with underscores
    candidate = clean_line.lower().replace(" ", "_").replace("-", "_")

    # Step 2: Exact match check on cleaned candidate
    if candidate in valid_labels:
        return candidate

    # Also check if candidate with only alphanumeric + underscore matches
    candidate_sanitized = re.sub(r"[^\w]", "", candidate)
    for label in valid_labels:
        if candidate_sanitized == label:
            return label

    # Check if any valid label appears as a standalone word/token in the first line
    for label in valid_labels:
        # e.g., if model answered: "The category is card_arrival."
        pattern = r"\b" + re.escape(label) + r"\b"
        if re.search(pattern, raw.lower()):
            return label

    # Step 3: Fuzzy match using difflib
    matches = difflib.get_close_matches(candidate, valid_labels, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    # Also try fuzzy matching against the sanitized version or individual words
    words = candidate.split("_")
    if len(words) > 1:
        phrase_guess = "_".join(words[:4])
        matches = difflib.get_close_matches(phrase_guess, valid_labels, n=1, cutoff=cutoff)
        if matches:
            return matches[0]

    # Step 4: No match found
    return None


if __name__ == "__main__":
    labels = get_banking77_labels()
    print(f"Loaded {len(labels)} labels. Testing parser:")
    test_cases = [
        ("card_arrival", "card_arrival"),
        ("Category: card_arrival", "card_arrival"),
        ("card_arrival.", "card_arrival"),
        ("Card Arrival", "card_arrival"),
        ("card_arival", "card_arrival"),  # slight typo
        ("lost_or_stolen_card", "lost_or_stolen_card"),
        ("I think this is a general inquiry.", None),
    ]
    for raw_input, expected in test_cases:
        parsed = parse_banking77_output(raw_input, labels)
        print(f"  Input: {raw_input!r:<35} -> Parsed: {parsed!r:<20} (Expected: {expected!r})")
