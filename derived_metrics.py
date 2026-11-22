"""
derived_metrics.py
==================
Pure computation module and CLI adapters for the NeMo Evaluator "derived metrics" layer.
Computes 27 model-level features across Format, Task, Judge, Reliability, Efficiency,
and Robustness on top of existing evaluation predictions and aggregates.

Hard constraints:
1. ADDITIVE ONLY: Does not modify original prediction or aggregate schemas.
2. FAIL-SAFE: Pure feature computations run within try/except, logging warnings and
   returning None (JSON null) if inputs are missing or computations fail.
3. NO EXTRA CALLS: Deterministic computation using saved rows + aggregates (seed 0).
4. KILL SWITCH: DERIVED_METRICS=0 env var or config flag disables processing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("DerivedMetrics")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# CONFIGURATION & FEATURE CATALOG
# ============================================================================

FEATURE_CATALOG: Dict[str, Dict[str, str]] = {
    # FORMAT
    "fn_call_leak_rate": {
        "group": "FORMAT",
        "description": "Share of model outputs containing raw function-call JSON syntax or headers.",
    },
    "extraction_success_rate": {
        "group": "FORMAT",
        "description": "Share of predictions with non-empty extracted answer after whitespace stripping.",
    },
    "final_answer_line_rate": {
        "group": "FORMAT",
        "description": "Share of outputs containing an explicit 'Final Answer' line (case-insensitive).",
    },
    "answer_recoverable_rate": {
        "group": "FORMAT",
        "description": "Share of outputs where reference answer is literally stated in the pre-JSON text.",
    },
    # TASK
    "label_exact_match": {
        "group": "TASK",
        "description": "Mean exact match accuracy for label / classification questions (qa_label).",
    },
    "numeric_exact_match": {
        "group": "TASK",
        "description": "Mean exact match accuracy for numerical / arithmetic questions (qa_numeric_program).",
    },
    # JUDGE
    "judge_em_gap": {
        "group": "JUDGE",
        "description": "Normalized gap between LLM judge score and exact match: mean(judge_score)/5 - mean(EM).",
    },
    "judge_composite": {
        "group": "JUDGE",
        "description": "Composite quality score: arithmetic mean of mean judge score, completeness, and helpfulness.",
    },
    "judge_leniency_rate": {
        "group": "JUDGE",
        "description": "Share of predictions awarded high judge score (>= 4) despite zero exact match.",
    },
    "judge_fail_rate": {
        "group": "JUDGE",
        "description": "Share of predictions receiving a failing judge score (<= 2).",
    },
    # RELIABILITY
    "reliable_accuracy": {
        "group": "RELIABILITY",
        "description": "Share of predictions that achieved exact match (1.0) without being flagged hallucinated.",
    },
    "hallucinated_but_correct_rate": {
        "group": "RELIABILITY",
        "description": "Share of predictions achieving exact match (1.0) while being flagged hallucinated.",
    },
    "ungrounded_number_ratio": {
        "group": "RELIABILITY",
        "description": "Average ratio of distinct reasoning numbers not present in prompt, context, or target.",
    },
    "ungrounded_flag_delta": {
        "group": "RELIABILITY",
        "description": "Mean ungrounded ratio of hallucinated samples minus that of non-hallucinated samples.",
    },
    "hallucination_separation": {
        "group": "RELIABILITY",
        "description": "Separation power of hallucination flag: P(Hallucinated | EM == 0) - P(Hallucinated | EM == 1).",
    },
    "label_hedge_rate": {
        "group": "RELIABILITY",
        "description": "Share of label questions matching uncertainty/hedging phrases in output text.",
    },
    # EFFICIENCY
    "latency_per_correct": {
        "group": "EFFICIENCY",
        "description": "Average generation latency in seconds required per unit of exact match accuracy.",
    },
    "tokens_per_correct": {
        "group": "EFFICIENCY",
        "description": "Average generated tokens consumed per unit of exact match accuracy.",
    },
    "latency_p95": {
        "group": "EFFICIENCY",
        "description": "95th percentile generation latency across samples (seconds).",
    },
    "latency_cv": {
        "group": "EFFICIENCY",
        "description": "Coefficient of variation of generation latency: sample std / mean latency.",
    },
    "quality_per_second": {
        "group": "EFFICIENCY",
        "description": "Mean judge score delivered per second of generation latency.",
    },
    "output_input_token_ratio": {
        "group": "EFFICIENCY",
        "description": "Average ratio of generated output tokens to prompt input tokens.",
    },
    "verbosity_accuracy_corr": {
        "group": "EFFICIENCY",
        "description": "Pearson correlation coefficient between output token count and exact match.",
    },
    # ROBUSTNESS
    "macro_exact_match": {
        "group": "ROBUSTNESS",
        "description": "Macro-averaged exact match across all category subgroups.",
    },
    "worst_category_em": {
        "group": "ROBUSTNESS",
        "description": "Minimum exact match accuracy among categories with at least 5 samples.",
    },
    "category_em_std": {
        "group": "ROBUSTNESS",
        "description": "Sample standard deviation of exact match across categories with at least 5 samples.",
    },
    "em_ci95_halfwidth": {
        "group": "ROBUSTNESS",
        "description": "Half-width of 95% bootstrap confidence interval of mean exact match (2,000 resamples, seed 0).",
    },
}

ORIGINAL_15_METRIC_KEYS = [
    "exact_match",
    "f1",
    "bleu",
    "rouge",
    "judge_score",
    "completeness",
    "helpfulness",
    "reasoning_faithfulness",
    "bert_score",
    "hallucination_rate",
    "failure_rate",
    "latency",
    "throughput",
    "tokens_per_sec",
    "category_accuracy",
]


@dataclass
class DerivedConfig:
    """Configurable settings, field names, and heuristic regexes for derived metrics."""

    enabled: bool = True
    fn_call_regex: str = r'\{\s*"name"\s*:\s*"'
    line_prefix_regex: str = r"^(?:JSON for|Function call)"
    hedge_regex: str = (
        r"does not specify|not specif|cannot definitively|cannot provide a definitive|"
        r"no specific provision|not explicit|definitive answer|not clear|unsure"
    )
    list_numeral_regex: str = r"^\s*\d+[.)]\s"
    number_regex: str = r"\b\d+(?:\.\d+)?\b"

    # Field mappings
    task_type_field: str = "task_type"
    category_field: str = "company"
    output_text_field: str = "output_text"
    extracted_answer_field: str = "extracted_answer"
    target_field: str = "target"
    normalized_answer_field: str = "normalized_answer"
    input_text_field: str = "input_text"
    context_field: str = "context"
    prompt_field: str = "prompt"
    exact_match_field: str = "exact_match"
    is_hallucinated_field: str = "is_hallucinated"
    judge_score_field: str = "judge_score"
    completeness_field: str = "completeness"
    helpfulness_field: str = "helpfulness"
    latency_seconds_field: str = "latency_seconds"
    tokens_in_field: str = "tokens_in"
    tokens_out_field: str = "tokens_out"

    # Task type identifiers
    label_task_name: str = "qa_label"
    numeric_task_name: str = "qa_numeric_program"

    # Robustness thresholds
    min_category_samples: int = 5
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    round_decimals: int = 4

    @classmethod
    def from_env_or_dict(cls, data: Optional[Dict[str, Any]] = None) -> "DerivedConfig":
        env_val = os.environ.get("DERIVED_METRICS", "1").strip().lower()
        is_enabled = env_val not in ("0", "false", "no", "off", "disable", "disabled")
        cfg = cls(enabled=is_enabled)
        if data:
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg


def is_derived_metrics_enabled() -> bool:
    """Check whether the kill switch is active via environment variable."""
    val = os.environ.get("DERIVED_METRICS", "1").strip().lower()
    return val not in ("0", "false", "no", "off", "disable", "disabled")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _clean_round(val: Any, decimals: int = 4) -> Optional[float]:
    """Rounds float values to specified decimals; converts NaN/Inf to None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (ValueError, TypeError):
        return None


def get_pre_json_text(
    text: Optional[str],
    fn_call_pattern: re.Pattern,
    line_prefix_pattern: re.Pattern,
) -> str:
    """Extracts text appearing before any function-call blob or header."""
    if not text:
        return ""
    cut_idx = len(text)
    m1 = fn_call_pattern.search(text)
    if m1 and m1.start() < cut_idx:
        cut_idx = m1.start()
    m2 = line_prefix_pattern.search(text)
    if m2 and m2.start() < cut_idx:
        cut_idx = m2.start()
    return text[:cut_idx]


def check_answer_recoverable(
    row: Dict[str, Any],
    pre_json_text: str,
    config: DerivedConfig,
) -> int:
    """
    Checks if reference answer is literally stated in pre-JSON text:
    - Numeric tasks: normalized_answer (commas removed) as a standalone number.
    - Yes/No targets: last Yes/No right after 'answer' (within 25 chars) or line start equals target.
    - Other targets: normalized containment (lowercase, strip rupee sign, commas, punctuation).
    """
    try:
        target = str(row.get(config.target_field) or "").strip()
        norm_ans = str(row.get(config.normalized_answer_field) or "").strip()
        task_type = str(row.get(config.task_type_field) or "")

        if task_type == config.numeric_task_name:
            clean_num = norm_ans.replace(",", "").strip()
            if not clean_num:
                return 0
            pattern = r"(?<![\d.,])" + re.escape(clean_num) + r"(?![\d.,])"
            return 1 if re.search(pattern, pre_json_text.replace(",", "")) else 0

        elif target.lower() in ("yes", "no"):
            matches = []
            for line in pre_json_text.splitlines():
                line_str = line.strip()
                # Line start match (handles optional bullet/numbering)
                m_start = re.match(r"^(?:(?:\d+[.)]|-|\*)\s*)?(yes|no)\b", line_str, re.IGNORECASE)
                if m_start:
                    matches.append(m_start.group(1).capitalize())
                # Within 25 chars after 'answer'
                for m in re.finditer(r"\banswer\b(.{0,25}?)\b(yes|no)\b", line_str, re.IGNORECASE):
                    matches.append(m.group(2).capitalize())
            if matches:
                last_ans = matches[-1]
                return 1 if last_ans.lower() == target.lower() else 0
            return 0

        else:
            def clean(s: str) -> str:
                s = s.lower().replace("₹", "").replace(",", "")
                return re.sub(r"[^\w\s]", "", s).strip()

            c_target = clean(target)
            c_text = clean(pre_json_text)
            return 1 if c_target and c_target in c_text else 0
    except Exception as e:
        logger.warning("Error checking answer recoverability for row: %s", e)
        return 0


# ============================================================================
# PURE COMPUTATION ENGINE
# ============================================================================

def compute_derived(
    rows: List[Dict[str, Any]],
    aggregates: Optional[Dict[str, Any]] = None,
    config: Optional[DerivedConfig] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Pure computation of all 27 derived features from sample rows and aggregates.
    No file I/O is performed in this function.

    Returns:
        (derived_features_dict, per_sample_df)
    """
    cfg = config or DerivedConfig.from_env_or_dict()
    if not cfg.enabled:
        logger.info("Derived metrics computation is disabled by config/kill-switch.")
        return {}, pd.DataFrame()

    n_samples = len(rows)
    aggs = aggregates or {}

    # Compile regexes once
    fn_call_pat = re.compile(cfg.fn_call_regex)
    line_prefix_pat = re.compile(cfg.line_prefix_regex, re.IGNORECASE | re.MULTILINE)
    hedge_pat = re.compile(cfg.hedge_regex, re.IGNORECASE)
    list_num_pat = re.compile(cfg.list_numeral_regex, re.MULTILINE)
    num_pat = re.compile(cfg.number_regex)

    derived: Dict[str, Any] = {}
    per_sample_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------------
    # Per-sample analysis loop
    # ------------------------------------------------------------------------
    for r in rows:
        rec: Dict[str, Any] = {
            "sample_id": r.get("id") or r.get("sample_id"),
            "task_type": r.get(cfg.task_type_field),
            "category": r.get(cfg.category_field),
            "target": r.get(cfg.target_field),
            "normalized_answer": r.get(cfg.normalized_answer_field),
            "extracted_answer": r.get(cfg.extracted_answer_field),
            "exact_match": r.get(cfg.exact_match_field),
            "judge_score": r.get(cfg.judge_score_field),
            "completeness": r.get(cfg.completeness_field),
            "helpfulness": r.get(cfg.helpfulness_field),
            "is_hallucinated": r.get(cfg.is_hallucinated_field),
            "is_failure": r.get("is_failure"),
            "latency_seconds": r.get(cfg.latency_seconds_field),
            "tokens_in": r.get(cfg.tokens_in_field),
            "tokens_out": r.get(cfg.tokens_out_field),
        }

        output_text = str(r.get(cfg.output_text_field) or "")
        pre_json = get_pre_json_text(output_text, fn_call_pat, line_prefix_pat)

        # Helpers
        has_fn_leak = bool(fn_call_pat.search(output_text))
        has_ext_success = bool(str(r.get(cfg.extracted_answer_field) or "").strip())
        has_fa_line = "final answer" in output_text.lower()
        is_ans_rec = check_answer_recoverable(r, pre_json, cfg)

        rec["has_fn_call_leak"] = has_fn_leak
        rec["has_extraction_success"] = has_ext_success
        rec["has_final_answer_line"] = has_fa_line
        rec["is_answer_recoverable"] = is_ans_rec

        # Number extraction for ungrounded ratio
        pj_no_lists = list_num_pat.sub("", pre_json).replace(",", "")
        out_nums = set(num_pat.findall(pj_no_lists))

        if out_nums:
            src_text = (
                f"{r.get(cfg.input_text_field) or ''} "
                f"{r.get(cfg.context_field) or ''} "
                f"{r.get(cfg.target_field) or ''}"
            ).replace(",", "")
            src_nums = set(num_pat.findall(src_text))
            ungrounded_count = sum(1 for n in out_nums if n not in src_nums)
            rec["ungrounded_number_ratio"] = ungrounded_count / len(out_nums)
        else:
            rec["ungrounded_number_ratio"] = None

        tin = r.get(cfg.tokens_in_field)
        tout = r.get(cfg.tokens_out_field)
        if tin is not None and tout is not None and tin > 0:
            rec["token_ratio"] = tout / tin
        else:
            rec["token_ratio"] = None

        per_sample_records.append(rec)

    per_sample_df = pd.DataFrame(per_sample_records)

    # ------------------------------------------------------------------------
    # Feature Calculations (Each in try/except)
    # ------------------------------------------------------------------------

    # 1. fn_call_leak_rate
    try:
        leaks = [1 if rec["has_fn_call_leak"] else 0 for rec in per_sample_records]
        derived["fn_call_leak_rate"] = _clean_round(np.mean(leaks) if leaks else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing fn_call_leak_rate: %s", e)
        derived["fn_call_leak_rate"] = None

    # 2. extraction_success_rate
    try:
        exts = [1 if rec["has_extraction_success"] else 0 for rec in per_sample_records]
        derived["extraction_success_rate"] = _clean_round(np.mean(exts) if exts else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing extraction_success_rate: %s", e)
        derived["extraction_success_rate"] = None

    # 3. final_answer_line_rate
    try:
        fas = [1 if rec["has_final_answer_line"] else 0 for rec in per_sample_records]
        derived["final_answer_line_rate"] = _clean_round(np.mean(fas) if fas else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing final_answer_line_rate: %s", e)
        derived["final_answer_line_rate"] = None

    # 4. answer_recoverable_rate
    try:
        recs = [rec["is_answer_recoverable"] for rec in per_sample_records]
        derived["answer_recoverable_rate"] = _clean_round(np.mean(recs) if recs else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing answer_recoverable_rate: %s", e)
        derived["answer_recoverable_rate"] = None

    # 5. label_exact_match
    try:
        label_ems = [
            rec["exact_match"]
            for rec in per_sample_records
            if rec["task_type"] == cfg.label_task_name and rec["exact_match"] is not None
        ]
        derived["label_exact_match"] = _clean_round(np.mean(label_ems) if label_ems else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing label_exact_match: %s", e)
        derived["label_exact_match"] = None

    # 6. numeric_exact_match
    try:
        num_ems = [
            rec["exact_match"]
            for rec in per_sample_records
            if rec["task_type"] == cfg.numeric_task_name and rec["exact_match"] is not None
        ]
        derived["numeric_exact_match"] = _clean_round(np.mean(num_ems) if num_ems else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing numeric_exact_match: %s", e)
        derived["numeric_exact_match"] = None

    # Extra task-level breakdowns if other task types exist
    try:
        all_task_types = {str(rec["task_type"]) for rec in per_sample_records if rec["task_type"]}
        extra_tasks = all_task_types - {cfg.label_task_name, cfg.numeric_task_name}
        if extra_tasks:
            task_breakdown = {}
            for tt in sorted(all_task_types):
                tt_ems = [
                    rec["exact_match"]
                    for rec in per_sample_records
                    if rec["task_type"] == tt and rec["exact_match"] is not None
                ]
                task_breakdown[tt] = _clean_round(np.mean(tt_ems) if tt_ems else None, cfg.round_decimals)
            derived["exact_match_by_task"] = task_breakdown
    except Exception as e:
        logger.warning("Error computing exact_match_by_task: %s", e)

    # 7. judge_em_gap: mean(judge_score)/5 - mean(EM)
    try:
        js_list = [rec["judge_score"] for rec in per_sample_records if rec["judge_score"] is not None]
        em_list = [rec["exact_match"] for rec in per_sample_records if rec["exact_match"] is not None]
        mean_js = float(np.mean(js_list)) if js_list else (aggs.get("judge_score") or 0.0)
        mean_em = float(np.mean(em_list)) if em_list else (aggs.get("exact_match") or 0.0)
        derived["judge_em_gap"] = _clean_round((mean_js / 5.0) - mean_em, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing judge_em_gap: %s", e)
        derived["judge_em_gap"] = None

    # 8. judge_composite: mean of mean(judge_score), mean(completeness), mean(helpfulness)
    try:
        comp_list = [rec["completeness"] for rec in per_sample_records if rec["completeness"] is not None]
        help_list = [rec["helpfulness"] for rec in per_sample_records if rec["helpfulness"] is not None]
        mean_comp = float(np.mean(comp_list)) if comp_list else (aggs.get("completeness") or 0.0)
        mean_help = float(np.mean(help_list)) if help_list else (aggs.get("helpfulness") or 0.0)
        derived["judge_composite"] = _clean_round(
            float(np.mean([mean_js, mean_comp, mean_help])), cfg.round_decimals
        )
    except Exception as e:
        logger.warning("Error computing judge_composite: %s", e)
        derived["judge_composite"] = None

    # 9. judge_leniency_rate: share with judge_score >= 4 AND EM == 0
    try:
        lenient = [
            1 if (rec["judge_score"] or 0) >= 4 and (rec["exact_match"] or 0) == 0 else 0
            for rec in per_sample_records
        ]
        derived["judge_leniency_rate"] = _clean_round(
            np.mean(lenient) if lenient else None, cfg.round_decimals
        )
    except Exception as e:
        logger.warning("Error computing judge_leniency_rate: %s", e)
        derived["judge_leniency_rate"] = None

    # 10. judge_fail_rate: share with judge_score <= 2
    try:
        fails = [1 if (rec["judge_score"] or 0) <= 2 else 0 for rec in per_sample_records]
        derived["judge_fail_rate"] = _clean_round(np.mean(fails) if fails else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing judge_fail_rate: %s", e)
        derived["judge_fail_rate"] = None

    # 11. reliable_accuracy: share with EM == 1 AND not H
    try:
        rel = [
            1 if (rec["exact_match"] or 0) == 1 and not rec["is_hallucinated"] else 0
            for rec in per_sample_records
        ]
        derived["reliable_accuracy"] = _clean_round(np.mean(rel) if rel else None, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing reliable_accuracy: %s", e)
        derived["reliable_accuracy"] = None

    # 12. hallucinated_but_correct_rate: share with EM == 1 AND H
    try:
        h_corr = [
            1 if (rec["exact_match"] or 0) == 1 and bool(rec["is_hallucinated"]) else 0
            for rec in per_sample_records
        ]
        derived["hallucinated_but_correct_rate"] = _clean_round(
            np.mean(h_corr) if h_corr else None, cfg.round_decimals
        )
    except Exception as e:
        logger.warning("Error computing hallucinated_but_correct_rate: %s", e)
        derived["hallucinated_but_correct_rate"] = None

    # 13. ungrounded_number_ratio & 14. ungrounded_flag_delta
    try:
        valid_ratios = [rec["ungrounded_number_ratio"] for rec in per_sample_records if rec["ungrounded_number_ratio"] is not None]
        derived["ungrounded_number_ratio"] = _clean_round(
            np.mean(valid_ratios) if valid_ratios else None, cfg.round_decimals
        )

        h_ratios = [
            rec["ungrounded_number_ratio"]
            for rec in per_sample_records
            if rec["ungrounded_number_ratio"] is not None and rec["is_hallucinated"]
        ]
        non_h_ratios = [
            rec["ungrounded_number_ratio"]
            for rec in per_sample_records
            if rec["ungrounded_number_ratio"] is not None and not rec["is_hallucinated"]
        ]
        if h_ratios and non_h_ratios:
            delta = float(np.mean(h_ratios) - np.mean(non_h_ratios))
            derived["ungrounded_flag_delta"] = _clean_round(delta, cfg.round_decimals)
        else:
            derived["ungrounded_flag_delta"] = None
    except Exception as e:
        logger.warning("Error computing ungrounded number metrics: %s", e)
        derived["ungrounded_number_ratio"] = None
        derived["ungrounded_flag_delta"] = None

    # 15. hallucination_separation: P(H | EM==0) - P(H | EM==1)
    try:
        em0_h = [bool(rec["is_hallucinated"]) for rec in per_sample_records if (rec["exact_match"] or 0) == 0]
        em1_h = [bool(rec["is_hallucinated"]) for rec in per_sample_records if (rec["exact_match"] or 0) == 1]
        p_h0 = float(np.mean(em0_h)) if em0_h else 0.0
        p_h1 = float(np.mean(em1_h)) if em1_h else 0.0
        derived["hallucination_separation"] = _clean_round(p_h0 - p_h1, cfg.round_decimals)
    except Exception as e:
        logger.warning("Error computing hallucination_separation: %s", e)
        derived["hallucination_separation"] = None

    # 16. label_hedge_rate
    try:
        qa_label_rows = [r for r in rows if r.get(cfg.task_type_field) == cfg.label_task_name]
        if qa_label_rows:
            hedged = [
                1 if hedge_pat.search(str(r.get(cfg.output_text_field) or "")) else 0
                for r in qa_label_rows
            ]
            derived["label_hedge_rate"] = _clean_round(np.mean(hedged), cfg.round_decimals)
        else:
            derived["label_hedge_rate"] = None
    except Exception as e:
        logger.warning("Error computing label_hedge_rate: %s", e)
        derived["label_hedge_rate"] = None

    # 17. latency_per_correct & 18. tokens_per_correct
    try:
        lat_vals = [rec["latency_seconds"] for rec in per_sample_records if rec["latency_seconds"] is not None]
        tok_out_vals = [rec["tokens_out"] for rec in per_sample_records if rec["tokens_out"] is not None]
        mean_lat = float(np.mean(lat_vals)) if lat_vals else None
        mean_tok_out = float(np.mean(tok_out_vals)) if tok_out_vals else None

        if mean_em > 0:
            derived["latency_per_correct"] = _clean_round(
                mean_lat / mean_em if mean_lat is not None else None, cfg.round_decimals
            )
            derived["tokens_per_correct"] = _clean_round(
                mean_tok_out / mean_em if mean_tok_out is not None else None, cfg.round_decimals
            )
        else:
            derived["latency_per_correct"] = None
            derived["tokens_per_correct"] = None
    except Exception as e:
        logger.warning("Error computing latency/tokens per correct: %s", e)
        derived["latency_per_correct"] = None
        derived["tokens_per_correct"] = None

    # 19. latency_p95
    try:
        derived["latency_p95"] = _clean_round(
            float(np.percentile(lat_vals, 95)) if lat_vals else None, cfg.round_decimals
        )
    except Exception as e:
        logger.warning("Error computing latency_p95: %s", e)
        derived["latency_p95"] = None

    # 20. latency_cv: std(latency_seconds, ddof=1) / mean
    try:
        if lat_vals and len(lat_vals) > 1 and mean_lat and mean_lat > 0:
            lat_std = float(np.std(lat_vals, ddof=1))
            derived["latency_cv"] = _clean_round(lat_std / mean_lat, cfg.round_decimals)
        else:
            derived["latency_cv"] = None
    except Exception as e:
        logger.warning("Error computing latency_cv: %s", e)
        derived["latency_cv"] = None

    # 21. quality_per_second: mean judge_score / mean latency_seconds
    try:
        if mean_lat and mean_lat > 0:
            derived["quality_per_second"] = _clean_round(mean_js / mean_lat, cfg.round_decimals)
        else:
            derived["quality_per_second"] = None
    except Exception as e:
        logger.warning("Error computing quality_per_second: %s", e)
        derived["quality_per_second"] = None

    # 22. output_input_token_ratio: mean of per-sample tokens_out/tokens_in
    try:
        valid_ti_to = [rec["token_ratio"] for rec in per_sample_records if rec["token_ratio"] is not None]
        derived["output_input_token_ratio"] = _clean_round(
            float(np.mean(valid_ti_to)) if valid_ti_to else None, cfg.round_decimals
        )
    except Exception as e:
        logger.warning("Error computing output_input_token_ratio: %s", e)
        derived["output_input_token_ratio"] = None

    # 23. verbosity_accuracy_corr: Pearson correlation between tokens_out and EM
    try:
        touts = [rec["tokens_out"] for rec in per_sample_records if rec["tokens_out"] is not None]
        ems = [rec["exact_match"] for rec in per_sample_records if rec["exact_match"] is not None]
        if len(touts) == len(ems) and len(touts) > 1 and np.std(touts) > 0 and np.std(ems) > 0:
            corr, _ = stats.pearsonr(touts, ems)
            derived["verbosity_accuracy_corr"] = _clean_round(corr, cfg.round_decimals)
        else:
            derived["verbosity_accuracy_corr"] = None
    except Exception as e:
        logger.warning("Error computing verbosity_accuracy_corr: %s", e)
        derived["verbosity_accuracy_corr"] = None

    # 24. macro_exact_match, 25. worst_category_em, 26. category_em_std
    try:
        cat_ems: Dict[str, List[float]] = {}
        for rec in per_sample_records:
            c = rec["category"]
            em = rec["exact_match"]
            if c is not None and em is not None:
                cat_ems.setdefault(str(c), []).append(float(em))

        if cat_ems:
            cat_means = {c: float(np.mean(vals)) for c, vals in cat_ems.items()}
            derived["macro_exact_match"] = _clean_round(
                float(np.mean(list(cat_means.values()))), cfg.round_decimals
            )

            cats_ge_thresh = {c: m for c, m in cat_means.items() if len(cat_ems[c]) >= cfg.min_category_samples}
            if cats_ge_thresh:
                derived["worst_category_em"] = _clean_round(min(cats_ge_thresh.values()), cfg.round_decimals)
                if len(cats_ge_thresh) > 1:
                    c_std = float(np.std(list(cats_ge_thresh.values()), ddof=1))
                    derived["category_em_std"] = _clean_round(c_std, cfg.round_decimals)
                else:
                    derived["category_em_std"] = None
            else:
                derived["worst_category_em"] = None
                derived["category_em_std"] = None
        else:
            derived["macro_exact_match"] = None
            derived["worst_category_em"] = None
            derived["category_em_std"] = None
    except Exception as e:
        logger.warning("Error computing category robustness metrics: %s", e)
        derived["macro_exact_match"] = None
        derived["worst_category_em"] = None
        derived["category_em_std"] = None

    # 27. em_ci95_halfwidth: bootstrap of mean EM, 2000 resamples, numpy default_rng(0)
    try:
        em_array = np.array([rec["exact_match"] for rec in per_sample_records if rec["exact_match"] is not None], dtype=float)
        if len(em_array) > 0:
            rng = np.random.default_rng(cfg.bootstrap_seed)
            n_obs = len(em_array)
            resampled_means = [rng.choice(em_array, size=n_obs, replace=True).mean() for _ in range(cfg.bootstrap_resamples)]
            p2_5 = np.percentile(resampled_means, 2.5)
            p97_5 = np.percentile(resampled_means, 97.5)
            derived["em_ci95_halfwidth"] = _clean_round((p97_5 - p2_5) / 2.0, cfg.round_decimals)
        else:
            derived["em_ci95_halfwidth"] = None
    except Exception as e:
        logger.warning("Error computing em_ci95_halfwidth: %s", e)
        derived["em_ci95_halfwidth"] = None

    return derived, per_sample_df


# ============================================================================
# FILE GENERATION & I/O ADAPTERS
# ============================================================================

def normalize_model_slug(name: str) -> str:
    """
    Normalizes model identifier to filesystem-safe slug, consistent with project standards.
    """
    slug = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    # Remove any trailing suffixes like _predictions or _aggregates
    if slug.endswith("_predictions.jsonl"):
        slug = slug[:-len("_predictions.jsonl")]
    elif slug.endswith("_aggregates.json"):
        slug = slug[:-len("_aggregates.json")]
    return slug


def process_model_files(
    predictions_path: Union[str, Path],
    aggregates_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[DerivedConfig] = None,
    merge_into_aggregates: bool = False,
) -> Dict[str, Any]:
    """
    Reads predictions and aggregates, computes derived features, and writes:
    - <model>_derived_features.json
    - <model>_all_metrics.json
    - <model>_per_sample_features.csv
    - (Optionally) adds 'derived' key into aggregates JSON if merge_into_aggregates is True.
    Existing files are never overwritten or altered unless merge_into_aggregates is True.
    """
    cfg = config or DerivedConfig.from_env_or_dict()
    if not cfg.enabled:
        logger.info("Derived metrics disabled via kill switch (DERIVED_METRICS). Skipping %s", predictions_path)
        return {}

    preds_file = Path(predictions_path)
    if not preds_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_file}")

    target_dir = Path(output_dir) if output_dir else preds_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    rows: List[Dict[str, Any]] = []
    with open(preds_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rows.append(json.loads(line_str))

    # 2. Load aggregates if available
    aggs: Dict[str, Any] = {}
    aggs_file = Path(aggregates_path) if aggregates_path else None
    if aggs_file and aggs_file.exists():
        with open(aggs_file, "r", encoding="utf-8") as f:
            aggs = json.load(f)
    else:
        # Check automatic match in target_dir or parent
        stem = preds_file.name.replace("_predictions.jsonl", "")
        auto_agg = preds_file.parent / f"{stem}_aggregates.json"
        if auto_agg.exists():
            aggs_file = auto_agg
            with open(auto_agg, "r", encoding="utf-8") as f:
                aggs = json.load(f)

    # Determine model identifier and slug
    model_name = aggs.get("model") or (rows[0].get("model") if rows else None)
    if not model_name:
        model_name = preds_file.stem.replace("_predictions", "")
    model_slug = preds_file.name[:-len("_predictions.jsonl")] if preds_file.name.endswith("_predictions.jsonl") else normalize_model_slug(model_name)

    # 3. Compute derived features
    derived, per_sample_df = compute_derived(rows=rows, aggregates=aggs, config=cfg)

    # 4. Write <model>_derived_features.json
    derived_out_path = target_dir / f"{model_slug}_derived_features.json"
    derived_payload = {
        "model": model_name,
        "n_samples": len(rows),
        "derived_features": derived,
        "catalog": FEATURE_CATALOG,
    }
    with open(derived_out_path, "w", encoding="utf-8") as f:
        json.dump(derived_payload, f, indent=2)
    logger.info("Saved derived features to: %s", derived_out_path)

    # 5. Write <model>_all_metrics.json (15 original aggregates + 27 derived features, flat)
    all_metrics_path = target_dir / f"{model_slug}_all_metrics.json"
    all_metrics_payload: Dict[str, Any] = {
        "model": model_name,
        "n_samples": len(rows),
    }
    for k in ORIGINAL_15_METRIC_KEYS:
        all_metrics_payload[k] = aggs.get(k)
    for k, v in derived.items():
        all_metrics_payload[k] = v

    with open(all_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics_payload, f, indent=2)
    logger.info("Saved consolidated metrics to: %s", all_metrics_path)

    # 6. Write <model>_per_sample_features.csv
    csv_out_path = target_dir / f"{model_slug}_per_sample_features.csv"
    per_sample_df.to_csv(csv_out_path, index=False, encoding="utf-8")
    logger.info("Saved per-sample features CSV to: %s", csv_out_path)

    # 7. Optional merge into aggregates
    if merge_into_aggregates and aggs_file and aggs_file.exists():
        try:
            with open(aggs_file, "r", encoding="utf-8") as f:
                existing_aggs = json.load(f)
            existing_aggs["derived"] = derived
            with open(aggs_file, "w", encoding="utf-8") as f:
                json.dump(existing_aggs, f, indent=2)
            logger.info("Merged 'derived' features into aggregates file: %s", aggs_file)
        except Exception as e:
            logger.warning("Failed to merge derived metrics into aggregates: %s", e)

    return derived_payload


def process_directory(
    directory: Union[str, Path],
    config: Optional[DerivedConfig] = None,
    merge_into_aggregates: bool = False,
    recompute: bool = False,
) -> List[Dict[str, Any]]:
    """
    Finds all predictions.jsonl files in directory (flat or nested) and runs derived metrics.
    Modes A and C support.
    """
    target_dir = Path(directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    pred_files = sorted(list(target_dir.glob("*_predictions.jsonl")))
    # Also include nested <model>/predictions.jsonl if present
    for d in target_dir.iterdir():
        if d.is_dir() and (d / "predictions.jsonl").exists():
            pred_files.append(d / "predictions.jsonl")

    if not pred_files and (target_dir / "predictions.jsonl").exists():
        pred_files = [target_dir / "predictions.jsonl"]

    logger.info("Discovered %d prediction files in %s", len(pred_files), target_dir)
    results = []
    for pf in pred_files:
        try:
            res = process_model_files(
                predictions_path=pf,
                output_dir=target_dir if pf.name.endswith("_predictions.jsonl") else pf.parent,
                config=config,
                merge_into_aggregates=merge_into_aggregates,
            )
            if res:
                results.append(res)
        except Exception as e:
            logger.warning("Failed processing %s: %s", pf, e)

    return results


# ============================================================================
# CLI ENTRY POINT (MODES A & C)
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeMo Evaluator Derived Metrics Layer (27 Model-Level Features)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=Path,
        default=None,
        help="Path to a single model's *_predictions.jsonl file",
    )
    parser.add_argument(
        "--aggregates",
        "-a",
        type=Path,
        default=None,
        help="Path to matching *_aggregates.json file",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=None,
        help="Directory containing predictions and aggregates to process",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Destination directory for derived outputs (defaults to input file's parent)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Mode C: Recompute derived metrics from saved predictions without touching originals",
    )
    parser.add_argument(
        "--merge-into-aggregates",
        action="store_true",
        help="Optional: add derived features under a single 'derived' key in the aggregates JSON",
    )
    parser.add_argument(
        "--no-derived-metrics",
        action="store_true",
        help="Kill switch flag: completely disables derived metric computation",
    )

    args = parser.parse_args()

    cfg = DerivedConfig.from_env_or_dict()
    if args.no_derived_metrics:
        cfg.enabled = False

    if not cfg.enabled:
        logger.info("Derived metrics is disabled. Exiting.")
        return

    if args.predictions:
        process_model_files(
            predictions_path=args.predictions,
            aggregates_path=args.aggregates,
            output_dir=args.output_dir,
            config=cfg,
            merge_into_aggregates=args.merge_into_aggregates,
        )
    elif args.dir:
        process_directory(
            directory=args.dir,
            config=cfg,
            merge_into_aggregates=args.merge_into_aggregates,
            recompute=args.recompute,
        )
    else:
        # Default fallback: check ./results
        default_results = Path("results")
        if default_results.exists():
            process_directory(
                directory=default_results,
                config=cfg,
                merge_into_aggregates=args.merge_into_aggregates,
                recompute=args.recompute,
            )
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
