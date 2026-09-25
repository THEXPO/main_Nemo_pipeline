"""
scripts/normalize_datasets.py
==============================
BFSI LLM Benchmarking -- NVIDIA NeMo Evaluator FYP
Preprocessing pipeline: normalizes all 5 raw datasets into a unified JSONL schema.

Output layout
-------------
data/processed/
  banking77_train.jsonl
  banking77_test.jsonl
  phrasebank_<tier>_train.jsonl
  financebench_train.jsonl
  finqa_train.jsonl  finqa_dev.jsonl  finqa_test.jsonl
  convfinqa_train.jsonl  convfinqa_dev.jsonl  convfinqa_test_private.jsonl
  manifest.json
  _rejects.log

Design decisions
----------------
* Every normalize_* function is a *pure function* -- no global state, fully
  unit-testable in isolation.
* Bad records are written to _rejects.log; the pipeline never crashes on a
  single malformed record.
* ConvFinQA: we use the pre-built _turn.json files (train_turn.json /
  dev_turn.json / test_turn_private.json) as the single source of truth for
  per-turn records. Re-flattening the dialogue-level files ourselves would
  require reproducing the exact qa_split grouping logic that the ConvFinQA
  authors already encoded in *_turn.json, increasing the risk of subtle
  turn-boundary errors. The turn files are the authoritative flattened view.
* PhraseBank deduplication: when multiple tiers are processed in one run, a
  sentence that appears in a higher-agreement tier is kept only once (the
  occurrence from the tightest-agreement tier wins -- allagree > 75agree >
  66agree > 50agree), matching the nested-subset structure described in the
  project brief.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup -- stderr only so stdout stays clean for piping
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified record schema
# ---------------------------------------------------------------------------
@dataclass
class UnifiedRecord:
    """One example in the normalised benchmark corpus."""
    id: str                               # globally unique identifier
    dataset: str                          # e.g. "banking77"
    split: str                            # "train" | "dev" | "test" | "test_private"
    task_type: str                        # see constants below
    input_text: str                       # primary question / utterance
    context: Any                          # table string, passage, or prior-turn list
    target: Optional[str]                 # gold label / answer (None for blind sets)
    target_type: str                      # "label" | "numeric" | "program" | "unlabeled"
    normalized_answer: Optional[float]    # parsed numeric form when applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Task-type constants
TASK_INTENT_CLASSIFICATION = "intent_classification"
TASK_SENTIMENT_ANALYSIS = "sentiment_analysis"
TASK_QA_NUMERIC = "qa_numeric"
TASK_QA_NUMERIC_PROGRAM = "qa_numeric_program"
TASK_QA_CONVERSATIONAL = "qa_conversational"


# ---------------------------------------------------------------------------
# Shared utility: reject logger
# ---------------------------------------------------------------------------
class RejectLogger:
    """Accumulates reject entries and flushes them to _rejects.log."""

    def __init__(self) -> None:
        self._entries: List[dict] = []

    def log(
        self,
        dataset: str,
        record_id: Optional[str],
        reason: str,
        raw: Optional[Any] = None,
    ) -> None:
        entry: dict = {"dataset": dataset, "id": record_id, "reason": reason}
        if raw is not None:
            try:
                entry["raw_preview"] = str(raw)[:300]
            except Exception:
                pass
        self._entries.append(entry)
        logger.warning("REJECT [%s] id=%s -- %s", dataset, record_id, reason)

    def flush(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for e in self._entries:
                fh.write(json.dumps(e) + "\n")
        return len(self._entries)


# ---------------------------------------------------------------------------
# Shared utility: Markdown table serializer
#
# Reused by both normalize_finqa and normalize_convfinqa.
# Input:  list[list[str]]  -- rows (first row is header)
# Output: a pipe-delimited Markdown table string
#
# Rules:
#   * Every cell is stripped of leading/trailing whitespace.
#   * The separator row uses --- for every column.
#   * Empty tables return an empty string.
#   * If there is only one row it is treated as a header with no data rows.
# ---------------------------------------------------------------------------
def table_to_markdown(table: List[List[str]]) -> str:
    """
    Serialise a list-of-lists table to a Markdown pipe table.

    This function is intentionally strict about formatting so that two
    identical tables always produce identical strings (important for
    reproducibility and caching by downstream components).
    """
    if not table:
        return ""

    # Normalise: stringify every cell, strip whitespace
    rows = [[str(cell).strip() for cell in row] for row in table]

    # Determine column count from the widest row
    n_cols = max(len(row) for row in rows)

    # Pad short rows with empty strings so every row has the same width
    padded = [row + [""] * (n_cols - len(row)) for row in rows]

    header = padded[0]
    separator = ["---"] * n_cols
    data_rows = padded[1:]

    def _fmt(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [_fmt(header), _fmt(separator)]
    lines.extend(_fmt(row) for row in data_rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared utility: numeric answer normaliser
#
# Used by normalize_financebench, normalize_finqa, normalize_convfinqa.
# ---------------------------------------------------------------------------
_NUMERIC_CLEAN_RE = re.compile(r"[$,%\s]")


def parse_numeric_answer(raw: Optional[str], decimal_places: int = 4) -> Optional[float]:
    """
    Attempt to parse a raw answer string into a float.

    Steps:
      1. Strip currency symbols ($), commas, percent signs, and whitespace.
      2. Handle parenthesised negatives like "(1,577)" -> -1577.
      3. Cast to float and round to `decimal_places`.
      4. Return None if the string is not numeric-ish.

    The function never raises -- callers always receive None on failure.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Parenthesised negative: (1,577) -> -1577
    paren_match = re.fullmatch(r"\(([^)]+)\)", s)
    if paren_match:
        s = "-" + paren_match.group(1)

    # Remove currency / percentage / comma / whitespace
    s = _NUMERIC_CLEAN_RE.sub("", s)

    # Remove trailing period (e.g. "127.")
    s = s.rstrip(".")

    try:
        value = float(s)
        return round(value, decimal_places)
    except (ValueError, OverflowError):
        return None


def _stable_id(*parts: str) -> str:
    """Generate a stable deterministic ID by hashing parts together."""
    key = "|".join(str(p) for p in parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Banking77 normaliser
# ---------------------------------------------------------------------------
def normalize_banking77(
    path: Path,
    categories_path: Path,
    split: str,
    rejects: Optional[RejectLogger] = None,
) -> List[UnifiedRecord]:
    """
    Normalise Banking77 train.jsonl or test.jsonl.

    Schema expected per line:
        {"text": str, "label": str}

    Rules:
    * Every label is validated against categories.json.
    * Unknown labels are logged to rejects and the record is dropped.
      Labels are NEVER silently accepted if they are absent from the
      canonical set -- this is a hard validation, not a soft warning.
    * IDs: "banking77_<split>_<zero-padded-line-index>"
    """
    if rejects is None:
        rejects = RejectLogger()

    # Load canonical label set from categories.json
    with categories_path.open("r", encoding="utf-8") as fh:
        raw_cats = json.load(fh)

    # categories.json may be a list[str] or list[dict]; handle both
    if isinstance(raw_cats, list) and raw_cats and isinstance(raw_cats[0], str):
        canonical_labels: set = set(raw_cats)
    elif isinstance(raw_cats, list) and raw_cats and isinstance(raw_cats[0], dict):
        canonical_labels = {item.get("name", "") for item in raw_cats}
    else:
        canonical_labels = set(raw_cats)

    records: List[UnifiedRecord] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record_id = f"banking77_{split}_{line_no:06d}"
            try:
                obj = json.loads(raw_line)
                text: str = obj["text"]
                label: str = obj["label"]

                # Hard label validation -- never silently accept unknown labels
                if label not in canonical_labels:
                    rejects.log(
                        "banking77",
                        record_id,
                        f"Unknown label '{label}' not in canonical categories.json",
                        raw=obj,
                    )
                    continue

                records.append(
                    UnifiedRecord(
                        id=record_id,
                        dataset="banking77",
                        split=split,
                        task_type=TASK_INTENT_CLASSIFICATION,
                        input_text=text,
                        context="",
                        target=label,
                        target_type="label",
                        normalized_answer=None,
                        metadata={"line_no": line_no},
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                rejects.log("banking77", record_id,
                             f"Parse error: {exc}", raw=raw_line)

    logger.info(
        "banking77 [%s]: %d records loaded",
        split,
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# 2. Financial PhraseBank normaliser
# ---------------------------------------------------------------------------

# Tiers ordered from tightest to loosest agreement
PHRASEBANK_TIERS = ("allagree", "75agree", "66agree", "50agree")

# Lower priority index = higher dedup priority (allagree wins)
_TIER_PRIORITY: Dict[str, int] = {t: i for i, t in enumerate(PHRASEBANK_TIERS)}

_LABEL_TEXT_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}


def normalize_phrasebank(
    path: Path,
    tier: str,
    split: str,
    rejects: Optional[RejectLogger] = None,
    _seen_sentences: Optional[dict] = None,
) -> List[UnifiedRecord]:
    """
    Normalise one Financial PhraseBank tier file.

    Schema expected per line:
        {"sentence": str, "label": int}   (label: 0=negative, 1=neutral, 2=positive)

    Parameters
    ----------
    tier : str
        One of PHRASEBANK_TIERS. Stored in metadata.tier for downstream use.
    split : str
        Split name (always "train" for PhraseBank, which has no test file).
    rejects : RejectLogger
        Shared reject logger.
    _seen_sentences : dict (optional, internal)
        Shared deduplication dict across tier calls.
        Maps sentence_text -> tier_priority_int.
        Pass the SAME dict object when normalising multiple tiers in a single
        pipeline run. A sentence from a tighter-agreement tier (lower priority
        index) takes precedence over the same sentence in a looser tier.
        The function simply skips any sentence whose priority was already
        claimed by a higher-priority tier.
    """
    if rejects is None:
        rejects = RejectLogger()
    if _seen_sentences is None:
        _seen_sentences = {}

    if tier not in _TIER_PRIORITY:
        raise ValueError(
            f"Unknown phrasebank tier '{tier}'. Valid options: {PHRASEBANK_TIERS}"
        )
    current_priority = _TIER_PRIORITY[tier]

    records: List[UnifiedRecord] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record_id = f"phrasebank_{tier}_{split}_{line_no:06d}"
            try:
                obj = json.loads(raw_line)
                sentence: str = obj["sentence"]
                label_int: int = int(obj["label"])

                if label_int not in _LABEL_TEXT_MAP:
                    rejects.log(
                        "financial_phrasebank",
                        record_id,
                        f"Invalid label int {label_int}; expected 0, 1, or 2",
                        raw=obj,
                    )
                    continue

                label_text = _LABEL_TEXT_MAP[label_int]

                # Cross-tier deduplication:
                # Skip if a higher-priority (tighter-agreement) tier already
                # owns this sentence. Tiers must be processed tightest-first
                # for dedup to be effective.
                existing_priority = _seen_sentences.get(sentence)
                if existing_priority is not None and existing_priority <= current_priority:
                    continue  # already emitted by a tighter tier

                _seen_sentences[sentence] = current_priority

                records.append(
                    UnifiedRecord(
                        id=record_id,
                        dataset="financial_phrasebank",
                        split=split,
                        task_type=TASK_SENTIMENT_ANALYSIS,
                        input_text=sentence,
                        context="",
                        target=label_text,
                        target_type="label",
                        normalized_answer=None,
                        metadata={
                            "label_int": label_int,
                            "label_text": label_text,
                            "tier": tier,
                            "line_no": line_no,
                        },
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                rejects.log("financial_phrasebank", record_id,
                             f"Parse error: {exc}", raw=raw_line)

    logger.info(
        "financial_phrasebank [tier=%s, split=%s]: %d records",
        tier,
        split,
        len(records),
    )
    return records


# ---------------------------------------------------------------------------
# 3. FinanceBench normaliser
# ---------------------------------------------------------------------------
def normalize_financebench(
    path: Path,
    split: str = "train",
    rejects: Optional[RejectLogger] = None,
) -> List[UnifiedRecord]:
    """
    Normalise FinanceBench train.jsonl.

    NOTE: "train" is the ONLY split that exists in the FinanceBench release
    (150 expert-curated examples). Do NOT invent a test split. The evaluation
    harness is responsible for any hold-out sampling.

    Schema:
        financebench_id, company, doc_name, question, answer, justification,
        evidence: [{evidence_text, doc_name, evidence_page_num,
                    evidence_text_full_page}], ...

    Rules:
    * evidence is flattened to metadata.evidence: [{text, page, doc_name}].
      evidence_text_full_page is NOT carried forward (too large; use evidence_text).
    * answer -> normalized_answer via parse_numeric_answer.
    * Raw answer string is preserved verbatim in metadata.answer_raw.
    """
    if rejects is None:
        rejects = RejectLogger()

    records: List[UnifiedRecord] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record_id = f"financebench_{split}_{line_no:06d}"
            try:
                obj = json.loads(raw_line)

                fb_id: str = obj.get("financebench_id", record_id)
                question: str = obj["question"]
                answer_raw: str = obj.get("answer", "")
                justification: str = obj.get("justification", "")
                doc_name: str = obj.get("doc_name", "")

                # Flatten evidence list -- drop full-page blobs to keep output compact
                evidence_raw = obj.get("evidence", [])
                evidence_flat = []
                for ev in evidence_raw:
                    evidence_flat.append({
                        "text": ev.get("evidence_text", ""),
                        "page": ev.get("evidence_page_num"),
                        "doc_name": ev.get("doc_name", doc_name),
                    })

                norm_ans = parse_numeric_answer(answer_raw)

                records.append(
                    UnifiedRecord(
                        id=fb_id,
                        dataset="financebench",
                        split=split,
                        task_type=TASK_QA_NUMERIC,
                        input_text=question,
                        context="",   # context = the SEC filing passage(s) in evidence
                        target=answer_raw,
                        target_type="numeric",
                        normalized_answer=norm_ans,
                        metadata={
                            "answer_raw": answer_raw,
                            "justification": justification,
                            "evidence": evidence_flat,
                            "company": obj.get("company", ""),
                            "doc_name": doc_name,
                            "doc_type": obj.get("doc_type", ""),
                            "doc_period": obj.get("doc_period"),
                            "question_type": obj.get("question_type", ""),
                            "question_reasoning": obj.get("question_reasoning", ""),
                            "dataset_subset_label": obj.get("dataset_subset_label", ""),
                            "line_no": line_no,
                        },
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                rejects.log("financebench", record_id,
                             f"Parse error: {exc}", raw=raw_line)

    logger.info("financebench [%s]: %d records", split, len(records))
    return records


# ---------------------------------------------------------------------------
# 4. FinQA normaliser
# ---------------------------------------------------------------------------
def normalize_finqa(
    path: Path,
    split: str,
    rejects: Optional[RejectLogger] = None,
) -> List[UnifiedRecord]:
    """
    Normalise FinQA train.json / dev.json / test.json.

    Schema:
        filename (or id), pre_text: [str], table: [[str]], table_ori: [[str]],
        post_text: [str],
        qa: {question, answer, program, steps: [{op, arg1, arg2, res}], ...}

    Rules:
    * context = table serialised via table_to_markdown() -- the same shared
      function used by ConvFinQA. Using `table` (lowercased, cleaned) not
      `table_ori` (raw HTML-annotated) to keep context consistent.
    * input_text = pre_text sentences joined, then question, then post_text
      sentences joined (newline-separated sections).
    * program and steps are kept verbatim in metadata for program-equivalence
      scoring by the evaluation harness.
    * normalized_answer = parse_numeric_answer(qa.answer).
    """
    if rejects is None:
        rejects = RejectLogger()

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    records: List[UnifiedRecord] = []

    for idx, obj in enumerate(data):
        # FinQA uses both "id" and "filename" depending on the split
        raw_id = obj.get("id", obj.get("filename", f"finqa_{split}_{idx}"))
        record_id = f"finqa_{split}_{idx:06d}"
        try:
            pre_text: List[str] = obj.get("pre_text", [])
            post_text: List[str] = obj.get("post_text", [])
            # Use the cleaned `table` field (not table_ori which has HTML tags)
            table: List[List[str]] = obj.get("table", [])
            qa: dict = obj["qa"]

            question: str = qa["question"]
            answer_raw: str = qa.get("answer", "")
            program: str = qa.get("program", "")
            steps: list = qa.get("steps", [])

            # Compose input_text: pre-context prose + question + post-context prose
            pre_joined = " ".join(pre_text)
            post_joined = " ".join(post_text)
            input_text = "\n\n".join(part for part in [pre_joined, question, post_joined] if part)

            # Serialise table to Markdown -- shared, deterministic function
            context_md = table_to_markdown(table)

            norm_ans = parse_numeric_answer(answer_raw)

            records.append(
                UnifiedRecord(
                    id=record_id,
                    dataset="finqa",
                    split=split,
                    task_type=TASK_QA_NUMERIC_PROGRAM,
                    input_text=input_text,
                    context=context_md,
                    target=answer_raw,
                    target_type="program",
                    normalized_answer=norm_ans,
                    metadata={
                        "source_id": raw_id,
                        "program": program,
                        "steps": steps,
                        "answer_raw": answer_raw,
                        "pre_text": pre_text,
                        "post_text": post_text,
                        "ann_table_rows": qa.get("ann_table_rows", []),
                        "ann_text_rows": qa.get("ann_text_rows", []),
                        "gold_inds": qa.get("gold_inds", {}),
                        "idx": idx,
                    },
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            rejects.log("finqa", record_id,
                         f"Parse error: {exc}", raw=str(obj)[:200])

    logger.info("finqa [%s]: %d records", split, len(records))
    return records


# ---------------------------------------------------------------------------
# 5. ConvFinQA normaliser
# ---------------------------------------------------------------------------
def _convfinqa_current_question(annotation: dict, qa: Optional[dict] = None) -> str:
    """
    Question for one flattened turn record.

    In *_turn.json, qa.question is often the final-turn paraphrase shared across
    every row in the same dialogue. The per-turn utterance lives in
    annotation.dialogue_break[turn_ind] or, for blind splits, cur_dial[-1].
    """
    turn_ind: int = annotation.get("turn_ind", 0)
    dialogue_break: List[str] = annotation.get("dialogue_break", [])
    cur_dial: List[str] = annotation.get("cur_dial", [])

    if dialogue_break and turn_ind < len(dialogue_break):
        return str(dialogue_break[turn_ind])
    if cur_dial:
        return str(cur_dial[-1])
    if qa:
        fallback = qa.get("question")
        if fallback:
            return str(fallback)
    return ""


def _debug_print_json_keys(path: Path, max_depth: int = 4, sample_n: int = 2) -> None:
    """Print nested key structure for the first records in a JSON array file."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        print(f"{path}: top-level type is {type(data).__name__}, expected list")
        return

    def _keys_tree(obj: Any, depth: int) -> Any:
        if depth <= 0:
            return type(obj).__name__
        if isinstance(obj, dict):
            return {k: _keys_tree(v, depth - 1) for k, v in obj.items()}
        if isinstance(obj, list):
            if not obj:
                return "list(empty)"
            return [_keys_tree(obj[0], depth - 1), f"... len={len(obj)}"]
        return type(obj).__name__

    print(f"{path}: {len(data)} records")
    for i in range(min(sample_n, len(data))):
        print(f"--- record[{i}] key tree (depth={max_depth}) ---")
        print(json.dumps(_keys_tree(data[i], max_depth), indent=2))


def normalize_convfinqa(
    dialogue_path: Optional[Path],
    turn_path: Path,
    split: str,
    use_prebuilt_turns: bool = True,
    rejects: Optional[RejectLogger] = None,
) -> List[UnifiedRecord]:
    """
    Normalise ConvFinQA per-turn records.

    Source-of-truth decision
    ------------------------
    We ALWAYS use the pre-built *_turn.json files as primary source
    (use_prebuilt_turns=True, the default). The ConvFinQA authors encode the
    correct qa_split turn-boundary logic in those files. Re-deriving turns
    from the dialogue-level JSON would risk subtle mis-alignment.

    If use_prebuilt_turns=False the function falls back to re-flattening the
    dialogue-level file. This mode is provided for completeness only and is
    NOT recommended for production runs.

    Schema (turn file, labelled splits)
    ------------------------------------
    id, filename, pre_text, post_text, table, table_ori,
    qa: {question, answer, exe_ans, program, steps, ...},
    annotation: {
        turn_ind,          # 0-based index of current turn in its dialogue
        cur_dial,          # [current question text]
        dialogue_break,    # [Q0, Q1, ..., Qn] full dialogue questions
        exe_ans,           # gold numeric answer for current turn
        exe_ans_list,      # [ans0, ans1, ..., ansn] all gold answers
        cur_program,       # program for current turn
        turn_program,      # [prog0, prog1, ...] programs for all turns
        qa_split,          # grouping metadata
        ...
    }

    Blind test (test_turn_private.json)
    ------------------------------------
    annotation only contains cur_dial (the question). No exe_ans, no programs.
    We emit:
        target=None, target_type="unlabeled", split="test_private"
    The pipeline never crashes or drops these records.

    Context field
    -------------
    A list of {"question": str, "answer": str} dicts for all turns that
    precede the current turn (turn_ind=0 has an empty context list).
    Reconstructed from annotation.dialogue_break + annotation.exe_ans_list.
    """
    if rejects is None:
        rejects = RejectLogger()

    is_blind = (split == "test_private")

    # -----------------------------------------------------------------------
    # Load source file
    # -----------------------------------------------------------------------
    if use_prebuilt_turns or is_blind:
        source_path = turn_path
    else:
        if dialogue_path is None:
            raise ValueError(
                "dialogue_path must be provided when use_prebuilt_turns=False"
            )
        source_path = dialogue_path

    with source_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    records: List[UnifiedRecord] = []

    # -----------------------------------------------------------------------
    # Helper: build prior-turn context list
    # -----------------------------------------------------------------------
    def _build_prior_context(
        dialogue_break: List[str],
        exe_ans_list: List[Any],
        turn_ind: int,
    ) -> List[dict]:
        """
        Return an ordered list of {"question": ..., "answer": ...} pairs
        for all turns strictly before `turn_ind` in this dialogue.
        """
        ctx: List[dict] = []
        for t in range(turn_ind):
            if t >= len(dialogue_break):
                break
            q = str(dialogue_break[t])
            a = str(exe_ans_list[t]) if t < len(exe_ans_list) else ""
            ctx.append({"question": q, "answer": a})
        return ctx

    # -----------------------------------------------------------------------
    # Branch: pre-built turn files (recommended path)
    # -----------------------------------------------------------------------
    if use_prebuilt_turns or is_blind:
        for idx, obj in enumerate(data):
            raw_id: str = obj.get("id", f"convfinqa_{split}_{idx}")
            record_id = f"convfinqa_{split}_{idx:07d}"
            try:
                annotation: dict = obj.get("annotation", {})
                table: List[List[str]] = obj.get("table", [])
                pre_text: List[str] = obj.get("pre_text", [])
                post_text: List[str] = obj.get("post_text", [])

                # Shared table serializer (same function as FinQA)
                context_md = table_to_markdown(table)

                if is_blind:
                    # Blind: annotation only has cur_dial, no gold labels
                    cur_dial: List[str] = annotation.get("cur_dial", [])
                    question = _convfinqa_current_question(annotation)
                    prior_context_blind = [
                        {"question": str(q), "answer": ""} for q in cur_dial[:-1]
                    ]

                    records.append(
                        UnifiedRecord(
                            id=raw_id,
                            dataset="convfinqa",
                            split="test_private",
                            task_type=TASK_QA_CONVERSATIONAL,
                            input_text=question,
                            context=context_md,
                            target=None,
                            target_type="unlabeled",
                            normalized_answer=None,
                            metadata={
                                "source_id": raw_id,
                                "filename": obj.get("filename", ""),
                                "pre_text": pre_text,
                                "post_text": post_text,
                                "cur_dial": cur_dial,
                                "prior_context": prior_context_blind,
                                "idx": idx,
                            },
                        )
                    )
                    continue

                # Labelled train / dev
                qa: dict = obj.get("qa", {})
                question = _convfinqa_current_question(annotation, qa)

                turn_ind: int = annotation.get("turn_ind", 0)
                dialogue_break: List[str] = annotation.get("dialogue_break", [])
                exe_ans_list: List[Any] = annotation.get("exe_ans_list", [])
                cur_program: str = annotation.get("cur_program", qa.get("program", ""))
                turn_program: List[str] = annotation.get("turn_program", [])

                # Gold answer for the current turn
                exe_ans_raw = annotation.get("exe_ans", qa.get("exe_ans"))
                target_str = str(exe_ans_raw) if exe_ans_raw is not None else ""
                norm_ans = parse_numeric_answer(target_str)

                prior_context = _build_prior_context(
                    dialogue_break, exe_ans_list, turn_ind
                )

                steps: list = qa.get("steps", [])

                records.append(
                    UnifiedRecord(
                        id=raw_id,
                        dataset="convfinqa",
                        split=split,
                        task_type=TASK_QA_CONVERSATIONAL,
                        input_text=question,
                        context=context_md,
                        target=target_str,
                        target_type="numeric",
                        normalized_answer=norm_ans,
                        metadata={
                            "source_id": raw_id,
                            "filename": obj.get("filename", ""),
                            "pre_text": pre_text,
                            "post_text": post_text,
                            "turn_ind": turn_ind,
                            "prior_context": prior_context,
                            "dialogue_break": dialogue_break,
                            "exe_ans_list": exe_ans_list,
                            "cur_program": cur_program,
                            "turn_program": turn_program,
                            "steps": steps,
                            "program": qa.get("program", ""),
                            "program_re": qa.get("program_re", ""),
                            "qa_split": annotation.get("qa_split", []),
                            "idx": idx,
                        },
                    )
                )
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                rejects.log("convfinqa", record_id,
                             f"Parse error: {exc}", raw=str(obj)[:200])

    # -----------------------------------------------------------------------
    # Branch: dialogue-level re-flattening (fallback, not recommended)
    # NOTE: This path is provided for completeness. The pre-built turn files
    # are the canonical source; use them unless you have a specific reason not to.
    # -----------------------------------------------------------------------
    else:
        if dialogue_path is None:
            raise ValueError(
                "dialogue_path must be provided when use_prebuilt_turns=False"
            )
        with dialogue_path.open("r", encoding="utf-8") as fh:
            dialogues = json.load(fh)

        for d_idx, dialogue in enumerate(dialogues):
            raw_id = dialogue.get("id", f"convfinqa_{split}_d{d_idx}")
            try:
                annotation: dict = dialogue.get("annotation", {})
                table: List[List[str]] = dialogue.get("table", [])
                pre_text: List[str] = dialogue.get("pre_text", [])
                post_text: List[str] = dialogue.get("post_text", [])
                context_md = table_to_markdown(table)

                dialogue_break: List[str] = annotation.get("dialogue_break", [])
                exe_ans_list: List[Any] = annotation.get("exe_ans_list", [])
                turn_program: List[str] = annotation.get("turn_program", [])

                for turn_ind, question in enumerate(dialogue_break):
                    turn_id = f"{raw_id}_t{turn_ind}"
                    exe_ans_raw = (
                        exe_ans_list[turn_ind]
                        if turn_ind < len(exe_ans_list) else None
                    )
                    target_str = str(exe_ans_raw) if exe_ans_raw is not None else ""
                    norm_ans = parse_numeric_answer(target_str)
                    cur_program = (
                        turn_program[turn_ind] if turn_ind < len(turn_program) else ""
                    )
                    prior_context = _build_prior_context(
                        dialogue_break, exe_ans_list, turn_ind
                    )

                    records.append(
                        UnifiedRecord(
                            id=turn_id,
                            dataset="convfinqa",
                            split=split,
                            task_type=TASK_QA_CONVERSATIONAL,
                            input_text=question,
                            context=context_md,
                            target=target_str,
                            target_type="numeric",
                            normalized_answer=norm_ans,
                            metadata={
                                "source_id": raw_id,
                                "filename": dialogue.get("filename", ""),
                                "pre_text": pre_text,
                                "post_text": post_text,
                                "turn_ind": turn_ind,
                                "prior_context": prior_context,
                                "dialogue_break": dialogue_break,
                                "exe_ans_list": exe_ans_list,
                                "cur_program": cur_program,
                                "turn_program": turn_program,
                                "d_idx": d_idx,
                            },
                        )
                    )
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                rejects.log(
                    "convfinqa",
                    f"convfinqa_{split}_d{d_idx}",
                    f"Parse error: {exc}",
                    raw=str(dialogue)[:200],
                )

    logger.info(
        "convfinqa [%s]: %d records (use_prebuilt=%s)",
        split,
        len(records),
        use_prebuilt_turns,
    )
    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_jsonl(records: List[UnifiedRecord], path: Path) -> None:
    """Write records to JSONL, overwriting cleanly (idempotent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


def update_manifest(
    manifest: dict,
    dataset: str,
    split: str,
    task_type: str,
    count: int,
    output_file: str,
) -> None:
    """Upsert a manifest entry for (dataset, split)."""
    key = f"{dataset}_{split}"
    manifest[key] = {
        "dataset": dataset,
        "split": split,
        "task_type": task_type,
        "record_count": count,
        "output_file": output_file,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Normalise BFSI benchmark datasets into a unified JSONL schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw dataset subdirectories.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where normalised JSONL files and manifest are written.",
    )
    p.add_argument(
        "--phrasebank-tier",
        choices=list(PHRASEBANK_TIERS),
        default="allagree",
        help=(
            "Which Financial PhraseBank agreement tier to process. "
            "allagree is the smallest (2,264 sentences) and most reliable. "
            "50agree is the largest (4,846) but includes noisier annotations. "
            "Tiers are nested (allagree is a subset of all others), so mixing "
            "tiers in one run without --all-phrasebank-tiers will not deduplicate."
        ),
    )
    p.add_argument(
        "--all-phrasebank-tiers",
        action="store_true",
        default=False,
        help=(
            "Process ALL four PhraseBank tiers, deduplicating across them. "
            "Tiers are processed tightest-first so allagree sentences are never "
            "duplicated in looser tiers. Each tier gets its own output file. "
            "When set, --phrasebank-tier is ignored."
        ),
    )
    p.add_argument(
        "--use-prebuilt-convfinqa-turns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the pre-built *_turn.json files as source of truth for "
            "ConvFinQA (recommended and default). "
            "Pass --no-use-prebuilt-convfinqa-turns to re-flatten from "
            "dialogue-level files instead (not recommended -- see module docstring)."
        ),
    )
    # Per-dataset skip flags for partial runs / debugging
    p.add_argument("--skip-banking77",    action="store_true", help="Skip Banking77.")
    p.add_argument("--skip-phrasebank",   action="store_true", help="Skip Financial PhraseBank.")
    p.add_argument("--skip-financebench", action="store_true", help="Skip FinanceBench.")
    p.add_argument("--skip-finqa",        action="store_true", help="Skip FinQA.")
    p.add_argument("--skip-convfinqa",    action="store_true", help="Skip ConvFinQA.")
    p.add_argument(
        "--debug-keys",
        type=Path,
        metavar="PATH",
        help="Print nested key structure for a raw JSON array file and exit.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.debug_keys is not None:
        _debug_print_json_keys(args.debug_keys)
        return 0

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rejects = RejectLogger()
    manifest: dict = {}

    # -----------------------------------------------------------------------
    # 1. Banking77
    # -----------------------------------------------------------------------
    if not args.skip_banking77:
        logger.info("=== Banking77 ===")
        b77_dir = input_dir / "banking77"
        cats_path = b77_dir / "categories.json"
        for split_name, fname in [("train", "train.jsonl"), ("test", "test.jsonl")]:
            split_path = b77_dir / fname
            if not split_path.exists():
                logger.warning("banking77/%s not found, skipping.", fname)
                continue
            recs = normalize_banking77(split_path, cats_path, split_name, rejects)
            out = output_dir / f"banking77_{split_name}.jsonl"
            write_jsonl(recs, out)
            update_manifest(manifest, "banking77", split_name,
                            TASK_INTENT_CLASSIFICATION, len(recs), out.name)

    # -----------------------------------------------------------------------
    # 2. Financial PhraseBank
    # -----------------------------------------------------------------------
    if not args.skip_phrasebank:
        logger.info("=== Financial PhraseBank ===")
        pb_base = input_dir / "financial_phrasebank"

        tiers_to_process = list(PHRASEBANK_TIERS) if args.all_phrasebank_tiers else [args.phrasebank_tier]

        # Shared cross-tier dedup dict -- pass the same object to all tier calls
        _seen: dict = {}

        for tier in tiers_to_process:
            tier_dir = pb_base / f"sentences_{tier}"
            train_path = tier_dir / "train.jsonl"
            if not train_path.exists():
                logger.warning("phrasebank sentences_%s/train.jsonl not found, skipping.", tier)
                continue
            recs = normalize_phrasebank(train_path, tier, "train", rejects, _seen)
            out = output_dir / f"phrasebank_{tier}_train.jsonl"
            write_jsonl(recs, out)
            update_manifest(
                manifest,
                f"financial_phrasebank_{tier}",
                "train",
                TASK_SENTIMENT_ANALYSIS,
                len(recs),
                out.name,
            )

    # -----------------------------------------------------------------------
    # 3. FinanceBench
    # -----------------------------------------------------------------------
    if not args.skip_financebench:
        logger.info("=== FinanceBench ===")
        fb_path = input_dir / "financebench" / "train.jsonl"
        if not fb_path.exists():
            logger.warning("financebench/train.jsonl not found, skipping.")
        else:
            recs = normalize_financebench(fb_path, "train", rejects)
            out = output_dir / "financebench_train.jsonl"
            write_jsonl(recs, out)
            update_manifest(manifest, "financebench", "train",
                            TASK_QA_NUMERIC, len(recs), out.name)

    # -----------------------------------------------------------------------
    # 4. FinQA
    # -----------------------------------------------------------------------
    if not args.skip_finqa:
        logger.info("=== FinQA ===")
        fq_dir = input_dir / "finqa"
        for split_name, fname in [
            ("train", "train.json"),
            ("dev",   "dev.json"),
            ("test",  "test.json"),
        ]:
            split_path = fq_dir / fname
            if not split_path.exists():
                logger.warning("finqa/%s not found, skipping.", fname)
                continue
            recs = normalize_finqa(split_path, split_name, rejects)
            out = output_dir / f"finqa_{split_name}.jsonl"
            write_jsonl(recs, out)
            update_manifest(manifest, "finqa", split_name,
                            TASK_QA_NUMERIC_PROGRAM, len(recs), out.name)

    # -----------------------------------------------------------------------
    # 5. ConvFinQA
    # -----------------------------------------------------------------------
    if not args.skip_convfinqa:
        logger.info("=== ConvFinQA ===")
        cq_dir = input_dir / "convfinqa"
        use_prebuilt = args.use_prebuilt_convfinqa_turns

        # (dialogue_filename, turn_filename, split_name)
        # dialogue_filename is None for the blind test split (no gold dialogue file)
        conv_configs = [
            ("train.json",       "train_turn.json",          "train"),
            ("dev.json",         "dev_turn.json",            "dev"),
            (None,               "test_turn_private.json",   "test_private"),
        ]

        for dial_fname, turn_fname, split_name in conv_configs:
            turn_path = cq_dir / turn_fname
            dial_path = cq_dir / dial_fname if dial_fname else None

            if not turn_path.exists():
                logger.warning("convfinqa/%s not found, skipping.", turn_fname)
                continue

            # Blind test always uses the turn file regardless of CLI flag
            effective_prebuilt = use_prebuilt or (split_name == "test_private")

            recs = normalize_convfinqa(
                dialogue_path=dial_path,
                turn_path=turn_path,
                split=split_name,
                use_prebuilt_turns=effective_prebuilt,
                rejects=rejects,
            )
            out = output_dir / f"convfinqa_{split_name}.jsonl"
            write_jsonl(recs, out)
            update_manifest(manifest, "convfinqa", split_name,
                            TASK_QA_CONVERSATIONAL, len(recs), out.name)

    # -----------------------------------------------------------------------
    # Write manifest.json (idempotent overwrite)
    # -----------------------------------------------------------------------
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest -> %s  (%d entries)", manifest_path, len(manifest))

    # -----------------------------------------------------------------------
    # Flush reject log
    # -----------------------------------------------------------------------
    reject_path = output_dir / "_rejects.log"
    n_rejects = rejects.flush(reject_path)
    if n_rejects:
        logger.warning("%d records rejected -- see %s", n_rejects, reject_path)
    else:
        logger.info("No rejects.")

    # Print summary to stderr
    logger.info("=== Run Summary ===")
    for key, info in manifest.items():
        logger.info(
            "  %-50s  %6d records  ->  %s",
            key,
            info["record_count"],
            info["output_file"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())