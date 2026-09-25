"""
bfsi_llm_benchmark.py
=====================
Comprehensive 15-Metric LLM Benchmarking Pipeline for BFSI Synthetic Dataset
Using NVIDIA NeMo Evaluator & LM Studio Local Inference.

THE 15 METRICS EVALUATED:
--------------------------
1.  Exact Match (EM)            -> NeMo Evaluator / task-aware normalization (numeric_match / label match)
2.  F1                          -> Token F1 for labels; aligned to exact_match for numeric tasks
3.  BLEU                        -> Text similarity over extracted final-answer substring
4.  ROUGE                       -> Text ROUGE for labels; aligned to exact_match for numeric tasks
5.  Judge Score (1-5)           -> NeMo Evaluator LLM-as-a-Judge (anchored rubric)
6.  Completeness (1-5)          -> NeMo Evaluator LLM-as-a-Judge (capped <= 2 on incorrect answers)
7.  Helpfulness (1-5)           -> NeMo Evaluator LLM-as-a-Judge (capped <= 2 on incorrect answers)
8.  Reasoning Faithfulness      -> Rule-based: fraction of ground-truth step results appearing in reasoning
9.  BERTScore                   -> Scored on extracted final-answer substring vs reference
10. Hallucination Rate          -> Unverified numbers outside expanded whitelist (raw_values, steps, constants)
11. Failure Rate                -> Failed API call, empty output, or refusal phrases
12. Latency                     -> Generation latency per request (seconds)
13. Throughput                  -> Total completed requests per second
14. Tokens/sec                  -> Total output tokens divided by total generation time
15. Category Accuracy           -> Tidy breakdown by task_type and company

MODULAR & STAGE-INDEPENDENT EXECUTION:
--------------------------------------
Stage 1: Generation           -> Runs models against dataset prompts; writes results/predictions.jsonl
Stage 2: NeMo Similarity     -> Evaluates EM, F1, BLEU, ROUGE
Stage 3: LLM Judge            -> Evaluates Judge Score, Completeness, Helpfulness (anchored 1-5)
Stage 4: Custom & Efficiency  -> Reasoning Faithfulness, BERTScore, Hallucination, Failure, Latency, Throughput, Tokens/sec
Stage 5: Master Table & Plots -> Consolidates master_table.csv, tidy category_accuracy.csv, heatmaps, radar, bar & consistency charts

Run each stage independently via CLI flags:
  --skip-generation --predictions results/predictions.jsonl   (re-evaluate existing predictions)
  --skip-judge                                                (run without LLM-as-a-Judge)
  --sample-size 5                                             (test run on 5 rows; 0 or -1 for full dataset)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Set non-interactive Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NeMo Evaluator imports
try:
    from nemo_evaluator.contrib.byob import ScorerInput
    from nemo_evaluator.contrib.byob.scorers import exact_match, f1_token, bleu, rouge
    from nemo_evaluator.contrib.byob.judge import JudgeConfig, judge_call
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

# Scipy statistical test
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BFSI_Benchmark")

# Defaults
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_DATASET = Path("data/synthetic_bfsi_train.jsonl")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_PLOTS_DIR = Path("plots")
DEFAULT_JUDGE_MAX_TOKENS = 1500

# Per-model default generation max_tokens (reasoning models need >= 2048 headroom)
MODEL_DEFAULT_MAX_TOKENS: Dict[str, int] = {
    "deepseek-r1": 2048,
    "deepseek": 2048,
    "r1": 2048,
    "ministral": 2048,
    "phi4": 512,
    "phi-4": 512,
    "llama": 512,
    "qwen": 512,
}


def get_model_max_tokens(model_name: str, override_tokens: Optional[int] = None) -> int:
    """Returns generation max_tokens budget for a given model. Explicit override takes precedence."""
    if override_tokens is not None and override_tokens > 0:
        return override_tokens
    m_lower = model_name.lower()
    for key, val in MODEL_DEFAULT_MAX_TOKENS.items():
        if key in m_lower:
            return val
    return 512


def _safe_model_dir(model_name: str) -> str:
    """
    Converts a model identifier into a filesystem-safe directory name.
    e.g. "lmstudio/Meta-Llama-3-8B-Instruct-GGUF" -> "lmstudio_Meta-Llama-3-8B-Instruct-GGUF"
    """
    safe = re.sub(r"[\\/:*?\"<>|\s]+", "_", model_name)
    return safe.strip("_") or "model"



# ============================================================================
# 1. JUDGE TEMPLATE & PROMPTS (FIX A & FIX B)
# ============================================================================

JUDGE_TEMPLATE = """You are an impartial, highly rigorous financial AI judge.
Evaluate the model response against the gold target reference and context for a financial task.

[Task Question]:
{question}

[Reference Context]:
{context}

[Gold Target Reference]:
{gold_target}

[Model Response]:
{model_response}

[Automated Verification]:
Note: automated scoring found this answer to be {correct_status} against the reference.

Score each dimension on a scale of 1 to 5 (integer or float 1.0 to 5.0):
- judge_score (1 to 5): overall quality, correctness, and soundness of the response.
- completeness (1 to 5): did the response use the relevant table values and walk through all necessary steps?
  * IMPORTANT: if the final numeric answer does not match the reference answer (within reasonable rounding), cap this score at 2, regardless of how thorough the explanation looks.
- helpfulness (1 to 5): would this response actually help a financial analyst get the right number?
  * IMPORTANT: if the final answer is wrong, cap this at 2 as well — a confident wrong answer is not helpful.

Since this answer is {correct_status}, completeness and helpfulness must be {score_guidance}.

Return ONLY compact JSON:
{{
  "judge_score": <1-5>,
  "completeness": <1-5>,
  "helpfulness": <1-5>,
  "reasoning": "<brief explanation>"
}}
"""


def build_benchmark_prompt(row: Dict[str, Any]) -> str:
    """Builds a structured prompt from a row of the synthetic BFSI dataset."""
    task_type = row.get("task_type", "qa_numeric_program")
    question = row.get("input_text", "").strip()
    context = row.get("context", "").strip()

    if task_type == "qa_numeric_program":
        prompt = (
            "You are an expert financial auditor performing numerical reasoning over corporate disclosures.\n\n"
            f"Financial Table:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. Carefully examine the table and perform the required calculation step-by-step.\n"
            "2. State your final numeric answer clearly at the end (e.g., Final Answer: <number>)."
        )
    else:  # qa_label
        prompt = (
            "You are an expert financial analyst answering questions based on corporate disclosures.\n\n"
            f"Financial Table:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. Identify the requested label, period, or entity from the table.\n"
            "2. State your final answer clearly at the end (e.g., Final Answer: <label>)."
        )
    return prompt


# ============================================================================
# 2. MODEL INFERENCE & API CLIENT
# ============================================================================

def call_model(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = 512,
    temperature: float = 0.0,
    connect_timeout: float = 5.0,
    read_timeout: float = 120.0,
    max_retries: int = 2,
    assistant_prefill: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls the configured OpenAI-compatible endpoint (LM Studio).
    Handles retries and timeouts gracefully; on failure, records failed_call=True
    rather than crashing the run.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio",
    }

    # For Llama models in LM Studio, LM Studio's chat template injects function-calling
    # instructions into user messages, causing the model to produce tool-call JSON.
    # Prefilling an assistant turn forces the model to respond in plain text or valid JSON.
    effective_prefill = assistant_prefill
    if effective_prefill is None:
        if "judge_score" in prompt:
            effective_prefill = '```json\n{\n  "judge_score": '
        elif "llama" in model_id.lower():
            effective_prefill = "Analysis:\n1. "

    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    if effective_prefill:
        messages.append({"role": "assistant", "content": effective_prefill})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,       # override any LM Studio UI default → greedy decode
        "seed": 42,             # deterministic output (honoured when backend supports it)
    }

    start_time = time.perf_counter()
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout, read_timeout),
            )
            elapsed = time.perf_counter() - start_time

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    choice = choices[0]
                    msg = choice.get("message", {})
                    out_text = msg.get("content", "") or ""
                    reasoning_text = msg.get("reasoning_content", "") or ""
                    finish_reason = choice.get("finish_reason", "unknown")
                    # If content is empty (e.g. model cut off mid-reasoning), preserve reasoning_content
                    if not out_text and reasoning_text:
                        out_text = reasoning_text
                    if effective_prefill and out_text and not out_text.startswith(effective_prefill):
                        out_text = effective_prefill + out_text
                    usage = data.get("usage", {})
                    tokens_out = usage.get("completion_tokens", len(out_text.split()))
                    tokens_in = usage.get("prompt_tokens", len(prompt.split()))
                    return {
                        "output_text": out_text,
                        "reasoning_content": reasoning_text,
                        "latency_seconds": round(elapsed, 4),
                        "tokens_out": tokens_out,
                        "tokens_in": tokens_in,
                        "failed_call": False,
                        "error": None,
                        "finish_reason": finish_reason,
                    }
                else:
                    last_error = f"No choices in API response: {data}"
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"

        except requests.exceptions.Timeout:
            last_error = f"Request timed out after {read_timeout}s (attempt {attempt + 1}/{max_retries + 1})"
        except requests.exceptions.ConnectionError:
            last_error = f"Connection error to {url}. Ensure LM Studio server is running on port 1234."
        except Exception as e:
            last_error = f"Unexpected error during call: {str(e)}"

        if attempt < max_retries:
            time.sleep(1.0 * (attempt + 1))

    # All retries exhausted
    elapsed = time.perf_counter() - start_time
    logger.warning("Model call failed for model '%s': %s", model_id, last_error)
    return {
        "output_text": "",
        "reasoning_content": "",
        "latency_seconds": round(elapsed, 4),
        "tokens_out": 0,
        "tokens_in": 0,
        "failed_call": True,
        "error": last_error,
        "finish_reason": "error",
    }


# ============================================================================
# 3. NORMALIZATION & FINAL-ANSWER EXTRACTION
# ============================================================================

def get_extracted_final_answer(output_text: str) -> str:
    """
    Extracts the final-answer substring from output_text.
    Takes everything after the LAST case-insensitive occurrence of 'final answer'.
    Fallback: returns output_text if 'final answer' marker is not found.
    """
    if not output_text:
        return ""
    matches = list(re.finditer(r"final answer", output_text, re.IGNORECASE))
    if matches:
        sub = output_text[matches[-1].end():]
        sub = re.sub(r"^[\s:\-*`'\"]+", "", sub)
        return sub.strip()
    return output_text.strip()


def extract_candidate_numbers(output_text: str) -> List[float]:
    """
    Finds the LAST occurrence of 'final answer' and extracts all numbers
    from that substring as a list of floats.
    """
    if not output_text:
        return []

    matches = list(re.finditer(r"final answer", output_text, re.IGNORECASE))
    if matches:
        substr = output_text[matches[-1].end():]
    else:
        substr = output_text

    clean_substr = substr.replace(",", "")
    raw_nums = re.findall(r"-?\d+(?:\.\d+)?", clean_substr)
    cands: List[float] = []
    for s in raw_nums:
        try:
            cands.append(float(s))
        except ValueError:
            pass
    return cands


def numeric_match(reference: float, candidates: List[float]) -> bool:
    """
    Checks if any candidate number matches reference under scale/sign variants:
    [c, c/100, c*100, -c, -c/100, -c*100] within max(0.02, abs(reference) * 0.01).
    """
    tol = max(0.02, abs(reference) * 0.01)
    for c in candidates:
        variants = [c, c / 100.0, c * 100.0, -c, -c / 100.0, -c * 100.0]
        for v in variants:
            if abs(v - reference) <= tol:
                return True
    return False


def normalize_label(text: str) -> str:
    """Normalizes a label string for qa_label comparisons."""
    if not text:
        return ""
    m = re.search(r"(?:final answer|answer is|the answer is)[:\s]*([^\n\.]+)", text, re.IGNORECASE)
    cand = m.group(1) if m else text
    line = cand.strip().split("\n")[0]
    line = re.sub(r"^(category|intent|label|answer|period)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
    line = line.strip(".*`'\" ")
    return line.lower()


def evaluate_exact_match(
    response: str,
    target: Any,
    task_type: str,
    normalized_answer: Optional[float] = None,
) -> bool:
    """Evaluates Exact Match for qa_numeric_program and qa_label."""
    if not response or not response.strip():
        return False

    if task_type == "qa_numeric_program":
        ref_val = normalized_answer if normalized_answer is not None else target
        try:
            ref_float = float(str(ref_val).replace("$", "").replace(",", "").replace("%", ""))
        except (ValueError, TypeError):
            return False

        cands = extract_candidate_numbers(response)
        if not cands:
            return False
        return numeric_match(ref_float, cands)

    else:  # qa_label
        gold_norm = normalize_label(str(target))
        pred_norm = normalize_label(response)
        if not gold_norm or not pred_norm:
            return False
        return (gold_norm == pred_norm) or (gold_norm in pred_norm.split())


# ============================================================================
# 4. NEMO EVALUATOR SIMILARITY METRICS (EM, F1, BLEU, ROUGE)
# ============================================================================

def run_similarity_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs similarity metrics offline over the predictions list:
      - Exact Match: evaluated via numeric_match (numeric) or label string match (label)
      - F1 & ROUGE: for numeric rows, equal to exact_match (1.0 or 0.0); for labels, token-based
      - BLEU: text similarity evaluated on the extracted final-answer substring
    """
    scored_rows = []
    em_list, f1_list, bleu_list, rouge_list = [], [], [], []

    for row in predictions:
        pred_text = row.get("output_text", "")
        target = row.get("target", "")
        task_type = row.get("task_type", "qa_numeric_program")
        norm_ans = row.get("normalized_answer")
        row_id = row.get("id", "unknown")

        try:
            is_em = evaluate_exact_match(pred_text, target, task_type, norm_ans)
            em_val = 1.0 if is_em else 0.0

            # BLEU: evaluated on extracted final-answer substring
            final_sub = get_extracted_final_answer(pred_text)
            target_str = str(target).strip()

            if NEMO_AVAILABLE and final_sub.strip():
                si_bleu = ScorerInput(
                    response=final_sub,
                    target=target_str,
                    metadata=row.get("metadata", {}),
                )
                b_res = bleu(si_bleu)
                bleu_val = float(b_res.get("bleu_1", 0.0))
            else:
                pred_tokens = set(final_sub.lower().split())
                target_tokens = set(target_str.lower().split())
                if pred_tokens and target_tokens:
                    intersect = len(pred_tokens & target_tokens)
                    bleu_val = intersect / len(pred_tokens)
                else:
                    bleu_val = 1.0 if is_em else 0.0

            if task_type == "qa_numeric_program":
                f1_val = em_val
                rouge_val = em_val
            else:
                if NEMO_AVAILABLE and final_sub.strip():
                    si_label = ScorerInput(
                        response=normalize_label(final_sub),
                        target=normalize_label(target_str),
                        metadata=row.get("metadata", {}),
                    )
                    f1_val = float(f1_token(si_label).get("f1", 0.0))
                    r_res = rouge(si_label)
                    rouge_val = float(r_res.get("rouge_l", r_res.get("rouge_1", 0.0)))
                else:
                    f1_val = em_val
                    rouge_val = em_val

        except Exception as e:
            logger.error("Error computing similarity metrics for row '%s': %s", row_id, e)
            em_val, f1_val, bleu_val, rouge_val = 0.0, 0.0, 0.0, 0.0

        em_list.append(em_val)
        f1_list.append(f1_val)
        bleu_list.append(bleu_val)
        rouge_list.append(rouge_val)

        row_copy = dict(row)
        row_copy["exact_match"] = em_val
        row_copy["f1"] = round(f1_val, 4)
        row_copy["bleu"] = round(bleu_val, 4)
        row_copy["rouge"] = round(rouge_val, 4)
        scored_rows.append(row_copy)

    aggregates = {
        "exact_match": round(float(np.mean(em_list)) if em_list else 0.0, 4),
        "f1": round(float(np.mean(f1_list)) if f1_list else 0.0, 4),
        "bleu": round(float(np.mean(bleu_list)) if bleu_list else 0.0, 4),
        "rouge": round(float(np.mean(rouge_list)) if rouge_list else 0.0, 4),
    }
    return {"scored_rows": scored_rows, "aggregates": aggregates}


# ============================================================================
# 5. LLM-AS-A-JUDGE (FIX A: CORRECTNESS-ANCHORED RUBRIC)
# ============================================================================

def parse_judge_response(raw_text: str, is_correct: bool = True) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Parses judge JSON response on a 1-5 scale for judge_score, completeness, helpfulness.
    Applies the mandatory capping rule: if not is_correct, completeness and helpfulness
    are strictly capped at 2.0.
    """
    scores = {"judge_score": 0.0, "completeness": 0.0, "helpfulness": 0.0}
    if not raw_text or not raw_text.strip():
        return scores, "Empty response from judge"

    text = raw_text.strip()
    reasoning: Optional[str] = None

    # 1. Clean markdown code fence extraction: ```json ... ``` or ```{ ... }```
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        try:
            data = json.loads(json_block.group(1))
            for k in scores.keys():
                if k in data:
                    val = float(data[k])
                    scores[k] = round(max(1.0, min(5.0, val)), 2)
            reasoning = data.get("reasoning")
        except Exception:
            pass

    # 2. Extract JSON block specifically containing "judge_score" (bypasses LaTeX braces)
    if scores["judge_score"] == 0.0:
        blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        for b in reversed(blocks):
            if "judge_score" in b:
                try:
                    data = json.loads(b)
                    for k in scores.keys():
                        if k in data:
                            val = float(data[k])
                            scores[k] = round(max(1.0, min(5.0, val)), 2)
                    reasoning = data.get("reasoning")
                    if scores["judge_score"] > 0.0:
                        break
                except Exception:
                    pass

    # 3. Greedy JSON extraction as fallback
    if scores["judge_score"] == 0.0:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                for k in scores.keys():
                    if k in data:
                        val = float(data[k])
                        scores[k] = round(max(1.0, min(5.0, val)), 2)
                reasoning = data.get("reasoning")
            except Exception:
                pass

    # 4. Regex fallback for individual keys if json.loads failed
    if scores["judge_score"] == 0.0:
        for k in scores.keys():
            m = re.search(rf'"{k}"\s*:\s*([0-9.]+)', text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    scores[k] = round(max(1.0, min(5.0, val)), 2)
                except ValueError:
                    pass

    # FIX A: Enforce capping rule as a deterministic safeguard
    if not is_correct:
        scores["completeness"] = min(2.0, scores["completeness"] if scores["completeness"] > 0 else 2.0)
        scores["helpfulness"] = min(2.0, scores["helpfulness"] if scores["helpfulness"] > 0 else 2.0)
        scores["judge_score"] = min(2.0, scores["judge_score"] if scores["judge_score"] > 0 else 2.0)

    return scores, reasoning or "Parsed successfully"


def run_llm_judge(
    predictions: List[Dict[str, Any]],
    judge_model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    skip_judge: bool = False,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Runs LLM-as-a-Judge using JUDGE_TEMPLATE with correctness anchoring.
    One call per row returning:
      - judge_score (1-5)
      - completeness (1-5, capped <= 2 on incorrect answers)
      - helpfulness (1-5, capped <= 2 on incorrect answers)
    Captures finish_reason from raw API response for every row to diagnose
    token budget ("length") vs structural issues ("stop").
    """
    if skip_judge:
        logger.info("Skipping LLM-as-a-Judge (--skip-judge enabled). Assigning default 0.0.")
        for r in predictions:
            r.update({"judge_score": 0.0, "completeness": 0.0, "helpfulness": 0.0, "judge_finish_reason": "skipped"})
        return {
            "scored_rows": predictions,
            "aggregates": {"judge_score": 0.0, "completeness": 0.0, "helpfulness": 0.0},
        }

    scored_rows = []
    judge_scores, completeness_scores, helpfulness_scores = [], [], []

    judge_cfg = None
    if NEMO_AVAILABLE:
        try:
            judge_cfg = JudgeConfig(
                url=base_url,
                model_id=judge_model,
                api_key=None,
                temperature=0.0,
                max_new_tokens=judge_max_tokens,
                request_timeout=120.0,
            )
        except Exception as e:
            logger.warning("Could not initialize NeMo JudgeConfig: %s. Using HTTP client.", e)

    total = len(predictions)
    logger.info("Starting LLM-as-a-Judge on %d rows with max_tokens=%d...", total, judge_max_tokens)

    for i, row in enumerate(predictions):
        row_id = row.get("id", f"row_{i}")
        q = row.get("input_text", "")
        ctx = row.get("context", "")
        gold = row.get("target", "")
        model_out = row.get("output_text", "")

        # FIX A: Anchor correctness status in prompt
        is_em = bool(row.get("exact_match", 0.0) == 1.0)
        correct_status = "CORRECT" if is_em else "INCORRECT"
        score_guidance = "between 4 and 5" if is_em else "1 or 2 (MUST be capped at 2)"

        judge_prompt = JUDGE_TEMPLATE.format(
            question=q,
            context=ctx,
            gold_target=gold,
            model_response=model_out if model_out.strip() else "[NO RESPONSE / CALL FAILED]",
            correct_status=correct_status,
            score_guidance=score_guidance,
        )

        parsed = {"judge_score": 0.0, "completeness": 0.0, "helpfulness": 0.0}
        call_error = None
        finish_reason = "unknown"
        judge_tokens = 0
        raw_judge_text = ""
        max_judge_attempts = 3
        judge_call_failed = False
        retries_used = 0
        resp = {}

        for attempt in range(1, max_judge_attempts + 1):
            resp = call_model(
                judge_prompt,
                model_id=judge_model,
                base_url=base_url,
                max_tokens=judge_max_tokens,
                connect_timeout=5.0,
                read_timeout=120.0,
            )
            raw_judge_text = resp.get("output_text", "")
            call_error = resp.get("error")
            finish_reason = resp.get("finish_reason", "unknown")
            judge_tokens = resp.get("tokens_out", 0)

            parsed, parse_msg = parse_judge_response(raw_judge_text, is_correct=is_em)

            # Succeeded if judge_score > 0.0 and raw_judge_text is not empty
            if parsed["judge_score"] > 0.0 and raw_judge_text.strip():
                if attempt > 1:
                    logger.info(
                        "Judge row '%s' succeeded on retry attempt %d/%d (score=%.1f).",
                        row_id, attempt, max_judge_attempts, parsed["judge_score"],
                    )
                break
            else:
                retries_used = attempt
                err_detail = call_error or parse_msg or "unparseable response"
                if attempt < max_judge_attempts:
                    logger.warning(
                        "Judge attempt %d/%d for row '%s' returned empty/invalid response (finish_reason='%s', error='%s'). Retrying...",
                        attempt, max_judge_attempts, row_id, finish_reason, err_detail,
                    )
                    time.sleep(1.0 * attempt)
                else:
                    judge_call_failed = True
                    logger.warning(
                        "Judge row '%s' failed all %d attempts. Defaulting to 0.0 (judge_call_failed=True). Last finish_reason: '%s', error: '%s', raw: %s",
                        row_id, max_judge_attempts, finish_reason, err_detail, raw_judge_text[:120],
                    )

        logger.info(
            "Judge [%d/%d] row '%s': finish_reason='%s', tokens_out=%d, score=%.1f, latency=%.2fs%s",
            i + 1, total, row_id, finish_reason, judge_tokens, parsed["judge_score"],
            resp.get("latency_seconds", 0.0),
            f" (retries={retries_used})" if retries_used > 0 else "",
        )

        judge_scores.append(parsed["judge_score"])
        completeness_scores.append(parsed["completeness"])
        helpfulness_scores.append(parsed["helpfulness"])

        row_copy = dict(row)
        # Drop safety if present from previous runs
        row_copy.pop("safety", None)
        row_copy.update(parsed)
        row_copy["judge_finish_reason"] = finish_reason
        row_copy["judge_tokens_out"] = judge_tokens
        row_copy["judge_call_failed"] = judge_call_failed
        row_copy["judge_retries"] = retries_used
        scored_rows.append(row_copy)

    aggregates = {
        "judge_score": round(float(np.mean(judge_scores)) if judge_scores else 0.0, 4),
        "completeness": round(float(np.mean(completeness_scores)) if completeness_scores else 0.0, 4),
        "helpfulness": round(float(np.mean(helpfulness_scores)) if helpfulness_scores else 0.0, 4),
    }
    return {"scored_rows": scored_rows, "aggregates": aggregates}


# ============================================================================
# 6. CUSTOM METRICS: REASONING FAITHFULNESS (FIX B), BERTSCORE, HALLUCINATION
# ============================================================================

def reasoning_faithfulness(output_text: str, steps: list[dict]) -> float:
    """
    FIX B: Fraction of the row's ground-truth intermediate/final step results
    that appear (within tolerance max(0.02, abs(v) * 0.01)) somewhere in output_text.
    1.0 = every step's result is reflected in reasoning; 0.0 = none are.
    """
    if not steps:
        return 1.0

    clean_text = output_text.replace(",", "")
    extracted_nums = [abs(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", clean_text)]

    matched_count = 0
    total_res = 0

    for step in steps:
        if not isinstance(step, dict) or "res" not in step or step["res"] is None:
            continue
        try:
            res_val = abs(float(step["res"]))
        except (ValueError, TypeError):
            continue

        total_res += 1
        variants = [res_val, res_val * 100.0, res_val / 100.0]
        step_matched = False

        for ex in extracted_nums:
            for v in variants:
                tol = max(0.02, abs(v) * 0.01)
                if abs(ex - v) <= tol:
                    step_matched = True
                    break
            if step_matched:
                break

        if step_matched:
            matched_count += 1

    return round(matched_count / total_res, 4) if total_res > 0 else 1.0


def compute_bertscore(
    candidates: List[str],
    references: List[str],
    base_url: str = DEFAULT_BASE_URL,
) -> List[float]:
    """
    Computes BERTScore F1 between candidate extracted final-answer substrings
    and reference targets.
    """
    try:
        from bert_score import score
        logger.info("Computing BERTScore with bert_score library...")
        P_raw, R_raw, F1_raw = score(candidates, references, lang="en", rescale_with_baseline=False, verbose=False)
        return [round(float(x), 4) for x in F1_raw]

    except Exception as bert_err:
        logger.info("bert_score library unavailable (%s). Checking embedding similarity...", bert_err)

        # Try LM Studio /v1/embeddings endpoint
        try:
            embed_url = f"{base_url.rstrip('/')}/embeddings"
            embed_scores = []
            for cand, ref in zip(candidates, references):
                c_text = cand.strip()
                r_text = str(ref).strip()
                if not c_text or not r_text:
                    embed_scores.append(0.0)
                    continue
                resp = requests.post(embed_url, json={
                    "model": "text-embedding-nomic-embed-text-v1.5",
                    "input": [c_text, r_text]
                }, timeout=5)
                if resp.status_code == 200:
                    d = resp.json()["data"]
                    e1 = np.array(d[0]["embedding"])
                    e2 = np.array(d[1]["embedding"])
                    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
                    embed_scores.append(round(max(0.0, min(1.0, sim)), 4))
                else:
                    break
            if len(embed_scores) == len(candidates):
                logger.info("Computed embedding similarity via LM Studio embeddings endpoint.")
                return embed_scores
        except Exception:
            pass

        # Fallback: TF-IDF semantic cosine similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        scores = []
        for cand, ref in zip(candidates, references):
            c_clean = cand.strip()
            r_clean = str(ref).strip()
            if not c_clean or not r_clean:
                scores.append(0.0)
                continue
            try:
                vec = TfidfVectorizer().fit([c_clean, r_clean])
                m = vec.transform([c_clean, r_clean])
                sim = float(cosine_similarity(m[0:1], m[1:2])[0][0])
                scores.append(round(max(0.0, min(1.0, sim)), 4))
            except Exception:
                scores.append(0.0)
        return scores


def is_hallucinated(response: str, row: Dict[str, Any]) -> bool:
    """
    Checks if any number in output_text is NOT in the expanded whitelist:
      - raw_values, target, normalized_answer
      - res, arg1, arg2, args from metadata.steps
      - integers 1..12, len(periods), periods
      - 100, and v*100 / v/100 scale variants
    """
    if not response or not response.strip():
        return False

    clean_text = response.replace(",", "")
    nums_str = re.findall(r"-?\d+(?:\.\d+)?", clean_text)
    if not nums_str:
        return False

    metadata = row.get("metadata", {}) or {}
    vals: set[float] = set()

    # 1. metadata.raw_values
    raw_vals = metadata.get("raw_values", {})
    if isinstance(raw_vals, dict):
        for k, v_list in raw_vals.items():
            if isinstance(v_list, list):
                for item in v_list:
                    try:
                        vals.add(abs(float(str(item).replace("$", "").replace("%", ""))))
                    except (ValueError, TypeError):
                        pass

    # 2. target & normalized_answer
    for t in [row.get("target"), row.get("normalized_answer")]:
        if t is not None:
            try:
                vals.add(abs(float(str(t).replace("$", "").replace("%", ""))))
            except (ValueError, TypeError):
                pass

    # 3. metadata.steps: res, arg1, arg2, args
    steps = metadata.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                for k in ["arg1", "arg2", "res"]:
                    if step.get(k) is not None:
                        try:
                            vals.add(abs(float(str(step[k]).replace("$", "").replace("%", ""))))
                        except (ValueError, TypeError):
                            pass
                args = step.get("args", [])
                if isinstance(args, list):
                    for a in args:
                        try:
                            vals.add(abs(float(str(a).replace("$", "").replace("%", ""))))
                        except (ValueError, TypeError):
                            pass

    # 4. small integers 1 through 12, plus len(metadata.periods)
    for i in range(1, 13):
        vals.add(float(i))

    periods = metadata.get("periods", [])
    if isinstance(periods, list):
        vals.add(float(len(periods)))
        for p in periods:
            try:
                vals.add(abs(float(p)))
            except (ValueError, TypeError):
                pass

    # 5. constant 100
    vals.add(100.0)

    # 6. scale variants: v*100 and v/100
    expanded = set(vals)
    for v in vals:
        expanded.add(v * 100.0)
        expanded.add(v / 100.0)

    for s in nums_str:
        try:
            val = abs(float(s))
        except ValueError:
            continue

        matched = False
        for w in expanded:
            tol = max(0.02, abs(w) * 0.01)
            if abs(val - w) <= tol:
                matched = True
                break

        if not matched:
            return True

    return False


def is_failure(response: str, row: Dict[str, Any], failed_call: bool, error: Optional[str]) -> bool:
    """Checks if output represents a failure."""
    if failed_call or error:
        return True
    if not response or not response.strip():
        return True

    refusal_patterns = [
        r"\b(i cannot|i can't|i am unable|as an ai|as a language model|i do not have access|i refuse|not authorized|sorry, but i cannot|apologize, but)\b"
    ]
    resp_lower = response.lower()
    for pat in refusal_patterns:
        if re.search(pat, resp_lower):
            return True
    return False


# ============================================================================
# 7. EFFICIENCY METRICS
# ============================================================================

def compute_efficiency_metrics(
    predictions: List[Dict[str, Any]],
    total_wall_time: float,
) -> Dict[str, float]:
    """Computes mean latency, throughput (req/s), and tokens/sec."""
    latencies = [r.get("latency_seconds", 0.0) for r in predictions if not r.get("failed_call")]
    tokens_out = [r.get("tokens_out", 0) for r in predictions if not r.get("failed_call")]

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    throughput = (len(latencies) / total_wall_time) if total_wall_time > 0 else 0.0
    sum_latency = sum(latencies)
    tokens_per_sec = (sum(tokens_out) / sum_latency) if sum_latency > 0 else 0.0

    return {
        "latency": round(mean_latency, 4),
        "throughput": round(throughput, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


# ============================================================================
# 8. MASTER TABLE & TIDY CATEGORY ACCURACY (FIX B: REASONING FAITHFULNESS)
# ============================================================================

def build_master_table(
    all_model_results: Dict[str, Dict[str, Any]],
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolidates all 15 metrics into one master DataFrame:
      1. Exact Match
      2. F1
      3. BLEU
      4. ROUGE
      5. Judge Score (1-5)
      6. Completeness (1-5)
      7. Helpfulness (1-5)
      8. Reasoning Faithfulness (FIX B: replaces Safety)
      9. BERTScore
      10. Hallucination Rate
      11. Failure Rate
      12. Latency (s)
      13. Throughput (req/s)
      14. Tokens/sec
      15. Category Accuracy (Overall)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tidy_category_rows = []

    for model_name, res in all_model_results.items():
        aggs = res.get("aggregates", {})
        scored_rows = res.get("scored_rows", [])

        # Master Table row with Reasoning Faithfulness replacing Safety
        row_dict = {
            "Model": model_name,
            "Exact Match": aggs.get("exact_match", 0.0),
            "F1": aggs.get("f1", 0.0),
            "BLEU": aggs.get("bleu", 0.0),
            "ROUGE": aggs.get("rouge", 0.0),
            "Judge Score": aggs.get("judge_score", 0.0),
            "Completeness": aggs.get("completeness", 0.0),
            "Helpfulness": aggs.get("helpfulness", 0.0),
            "Reasoning Faithfulness": aggs.get("reasoning_faithfulness", 0.0),
            "BERTScore": aggs.get("bert_score", 0.0),
            "Hallucination Rate": aggs.get("hallucination_rate", 0.0),
            "Failure Rate": aggs.get("failure_rate", 0.0),
            "Latency (s)": aggs.get("latency", 0.0),
            "Throughput (req/s)": aggs.get("throughput", 0.0),
            "Tokens/sec": aggs.get("tokens_per_sec", 0.0),
            "Category Accuracy": aggs.get("category_accuracy", 0.0),
        }
        rows.append(row_dict)

        # Tidy Category Breakdown
        df_rows = pd.DataFrame(scored_rows)
        if not df_rows.empty and "exact_match" in df_rows.columns:
            if "task_type" in df_rows.columns:
                t_grp = df_rows.groupby("task_type")["exact_match"].mean()
                for task_val, acc in t_grp.items():
                    tidy_category_rows.append({
                        "Model": model_name,
                        "group_type": "task_type",
                        "group": task_val,
                        "accuracy": round(float(acc), 4),
                    })

            df_rows["company"] = df_rows["metadata"].apply(
                lambda m: m.get("company", "Unknown") if isinstance(m, dict) else "Unknown"
            )
            c_grp = df_rows.groupby("company")["exact_match"].mean()
            for comp_val, acc in c_grp.items():
                tidy_category_rows.append({
                    "Model": model_name,
                    "group_type": "company",
                    "group": comp_val,
                    "accuracy": round(float(acc), 4),
                })

    master_df = pd.DataFrame(rows)
    cat_df = pd.DataFrame(tidy_category_rows, columns=["Model", "group_type", "group", "accuracy"])

    master_path = output_dir / "master_table.csv"
    master_df.to_csv(master_path, index=False)
    logger.info("Saved master table to: %s", master_path)

    category_path = output_dir / "category_accuracy.csv"
    cat_df.to_csv(category_path, index=False)
    logger.info("Saved tidy category accuracy to: %s", category_path)

    return master_df, cat_df


# ============================================================================
# 9. VISUALIZATIONS & STATISTICAL SIGNIFICANCE
# ============================================================================

def plot_bar_per_metric(df: pd.DataFrame, output_dir: Path = DEFAULT_PLOTS_DIR):
    """Generates grouped bar charts for quality metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    quality_cols = ["Exact Match", "F1", "BLEU", "ROUGE", "Reasoning Faithfulness", "BERTScore"]
    avail_cols = [c for c in quality_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(avail_cols))
    width = 0.8 / max(1, len(df))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[c] for c in avail_cols]
        ax.bar(x + idx * width, values, width, label=row["Model"])

    ax.set_title("BFSI Benchmark: Quality Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(avail_cols, fontsize=10, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = output_dir / "metrics_bar_chart.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved quality bar chart to: %s", out_file)


def plot_radar(df: pd.DataFrame, output_dir: Path = DEFAULT_PLOTS_DIR):
    """Generates radar chart across quality dimensions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    dimensions = ["Exact Match", "F1", "BLEU", "ROUGE", "Reasoning Faithfulness", "BERTScore"]
    avail_dims = [d for d in dimensions if d in df.columns]
    if len(avail_dims) < 3:
        return

    num_vars = len(avail_dims)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in df.iterrows():
        values = [row[d] for d in avail_dims]
        values += values[:1]
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(avail_dims, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Dimensional Model Profile (Radar)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    out_file = output_dir / "radar_chart.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved radar chart to: %s", out_file)


def plot_quality_vs_latency(df: pd.DataFrame, output_dir: Path = DEFAULT_PLOTS_DIR):
    """Generates Quality (Exact Match) vs Latency trade-off scatter plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or "Latency (s)" not in df.columns or "Exact Match" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        ax.scatter(row["Latency (s)"], row["Exact Match"], s=150, label=row["Model"], alpha=0.8)
        ax.annotate(
            row["Model"],
            (row["Latency (s)"], row["Exact Match"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Quality vs. Latency Trade-Off", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Latency (seconds) -> Lower is Better", fontsize=11)
    ax.set_ylabel("Exact Match Score -> Higher is Better", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = output_dir / "quality_vs_latency.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved quality vs latency scatter plot to: %s", out_file)


def plot_category_heatmaps(
    cat_df: pd.DataFrame,
    output_dir: Path = DEFAULT_PLOTS_DIR,
):
    """Generates heatmaps of category-wise accuracy from tidy DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if cat_df.empty:
        return

    task_sub = cat_df[cat_df["group_type"] == "task_type"]
    if not task_sub.empty:
        task_pivot = task_sub.pivot(index="Model", columns="group", values="accuracy")
        fig, ax = plt.subplots(figsize=(8, max(3, len(task_pivot) * 1.2)))
        cax = ax.matshow(task_pivot.values, cmap="YlGnBu", vmin=0, vmax=1.0)
        fig.colorbar(cax)

        ax.set_xticks(range(len(task_pivot.columns)))
        ax.set_xticklabels(task_pivot.columns, rotation=25, ha="left", fontsize=10)
        ax.set_yticks(range(len(task_pivot.index)))
        ax.set_yticklabels(task_pivot.index, fontsize=10)
        ax.set_title("Model Accuracy by Task Type", fontsize=12, fontweight="bold", pad=25)

        for i in range(len(task_pivot.index)):
            for j in range(len(task_pivot.columns)):
                val = task_pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.7 else "white")

        plt.tight_layout()
        out_task = output_dir / "heatmap_task_type.png"
        plt.savefig(out_task, dpi=300)
        plt.close()
        logger.info("Saved task type heatmap to: %s", out_task)

    comp_sub = cat_df[cat_df["group_type"] == "company"]
    if not comp_sub.empty:
        comp_pivot = comp_sub.pivot(index="Model", columns="group", values="accuracy")
        fig, ax = plt.subplots(figsize=(max(8, len(comp_pivot.columns) * 1.0), max(4, len(comp_pivot) * 1.5)))
        cax = ax.matshow(comp_pivot.values, cmap="Blues", vmin=0, vmax=1.0)
        fig.colorbar(cax)

        ax.set_xticks(range(len(comp_pivot.columns)))
        ax.set_xticklabels(comp_pivot.columns, rotation=45, ha="left", fontsize=9)
        ax.set_yticks(range(len(comp_pivot.index)))
        ax.set_yticklabels(comp_pivot.index, fontsize=10)
        ax.set_title("Model Accuracy across BFSI Companies", fontsize=12, fontweight="bold", pad=30)

        for i in range(len(comp_pivot.index)):
            for j in range(len(comp_pivot.columns)):
                val = comp_pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white", fontsize=8)

        plt.tight_layout()
        out_comp = output_dir / "heatmap_company.png"
        plt.savefig(out_comp, dpi=300)
        plt.close()
        logger.info("Saved company heatmap to: %s", out_comp)


def plot_consistency_spread(
    consistency_data: Dict[str, List[float]],
    output_dir: Path = DEFAULT_PLOTS_DIR,
):
    """Plots score spread across repeated runs to visualize model consistency."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not consistency_data:
        return

    models = list(consistency_data.keys())
    data = [consistency_data[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=models, patch_artist=True)
    ax.set_title("Model Consistency Spread (Repeated Runs)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Exact Match Score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = output_dir / "consistency_spread.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info("Saved consistency spread plot to: %s", out_file)


def run_statistical_significance(
    all_model_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Computes paired t-test between any two models' per-row exact match scores."""
    models = list(all_model_results.keys())
    if len(models) < 2 or not SCIPY_AVAILABLE:
        return pd.DataFrame()

    records = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            rows1 = all_model_results[m1].get("scored_rows", [])
            rows2 = all_model_results[m2].get("scored_rows", [])

            s1 = [r.get("exact_match", 0.0) for r in rows1]
            s2 = [r.get("exact_match", 0.0) for r in rows2]

            min_len = min(len(s1), len(s2))
            if min_len > 1:
                t_stat, p_val = stats.ttest_rel(s1[:min_len], s2[:min_len])
                sig = "Yes (p < 0.05)" if p_val < 0.05 else "No"
                records.append({
                    "Model A": m1,
                    "Model B": m2,
                    "Mean A": round(float(np.mean(s1[:min_len])), 4),
                    "Mean B": round(float(np.mean(s2[:min_len])), 4),
                    "t-statistic": round(float(t_stat), 4),
                    "p-value": round(float(p_val), 4),
                    "Significant?": sig,
                })

    sig_df = pd.DataFrame(records)
    if not sig_df.empty:
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE SUMMARY (Paired t-test):")
        print("=" * 70)
        print(sig_df.to_string(index=False))
        print("=" * 70 + "\n")
    return sig_df


# ============================================================================
# 10. END-TO-END BENCHMARKING ENGINE
# ============================================================================

def benchmark_model(
    model_name: str,
    dataset_rows: List[Dict[str, Any]],
    base_url: str = DEFAULT_BASE_URL,
    judge_model: str = DEFAULT_MODEL,
    skip_judge: bool = False,
    existing_predictions: Optional[List[Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Executes the full 15-metric evaluation pipeline for a single model:
      Step 1: Response Generation (with latency & token tracking)
      Step 2: NeMo Similarity Metrics (EM, F1, BLEU, ROUGE)
      Step 3: NeMo LLM-as-a-Judge (Judge Score, Completeness, Helpfulness 1-5 with anchoring)
      Step 4: Custom Metrics:
              - Reasoning Faithfulness (FIX B)
              - BERTScore (on extracted final answer)
              - Hallucination Rate (expanded whitelist)
              - Failure Rate
      Step 5: Efficiency Metrics (Latency, Throughput, Tokens/sec)
    """
    gen_max_tokens = get_model_max_tokens(model_name, override_tokens=max_tokens)
    logger.info("==================================================")
    logger.info("BENCHMARKING MODEL: %s (%d rows | gen max_tokens=%d | judge max_tokens=%d)",
                model_name, len(dataset_rows), gen_max_tokens, judge_max_tokens)
    logger.info("==================================================")

    # Step 1: Inference / Response Generation
    predictions: List[Dict[str, Any]] = []
    start_wall = time.perf_counter()

    if existing_predictions:
        logger.info("Using %d existing predictions for '%s'", len(existing_predictions), model_name)
        predictions = existing_predictions
        total_wall_time = sum(p.get("latency_seconds", 1.0) for p in predictions)
    else:
        for idx, row in enumerate(dataset_rows):
            prompt = build_benchmark_prompt(row)
            resp = call_model(prompt, model_id=model_name, base_url=base_url, max_tokens=gen_max_tokens)

            pred_record = dict(row)
            pred_record["model"] = model_name
            pred_record["prompt"] = prompt
            pred_record["output_text"] = resp["output_text"]
            pred_record["latency_seconds"] = resp["latency_seconds"]
            pred_record["tokens_out"] = resp["tokens_out"]
            pred_record["tokens_in"] = resp["tokens_in"]
            pred_record["finish_reason"] = resp.get("finish_reason", "unknown")
            pred_record["failed_call"] = resp["failed_call"]
            pred_record["error"] = resp["error"]

            predictions.append(pred_record)
            if (idx + 1) % 5 == 0 or (idx + 1) == len(dataset_rows):
                logger.info("Progress [%s]: %d/%d completed (last row finish_reason='%s', tokens_out=%d)",
                            model_name, idx + 1, len(dataset_rows), pred_record["finish_reason"], pred_record["tokens_out"])

        total_wall_time = time.perf_counter() - start_wall

    # Step 2: NeMo Evaluator Similarity Metrics (EM, F1, BLEU, ROUGE)
    logger.info("Evaluating NeMo similarity metrics (EM, F1, BLEU, ROUGE)...")
    sim_res = run_similarity_metrics(predictions)
    scored_rows = sim_res["scored_rows"]
    aggs = sim_res["aggregates"]

    # Step 3: LLM-as-a-Judge (FIX A: Anchored 1-5 Scoring)
    logger.info("Evaluating LLM-as-a-Judge (Judge Score, Completeness, Helpfulness)...")
    judge_res = run_llm_judge(
        scored_rows,
        judge_model=judge_model,
        base_url=base_url,
        skip_judge=skip_judge,
        judge_max_tokens=judge_max_tokens,
    )
    scored_rows = judge_res["scored_rows"]
    aggs.update(judge_res["aggregates"])

    # Step 4: Custom Metrics: Reasoning Faithfulness, BERTScore, Hallucination, Failure
    logger.info("Evaluating Custom Metrics (Reasoning Faithfulness, BERTScore, Hallucination, Failure)...")
    cands = [get_extracted_final_answer(r.get("output_text", "")) for r in scored_rows]
    refs = [str(r.get("target", "")) for r in scored_rows]
    bert_scores = compute_bertscore(cands, refs, base_url=base_url)

    faithfulness_scores = []
    hallucination_flags = []
    failure_flags = []

    for r, b_score in zip(scored_rows, bert_scores):
        r["bert_score"] = b_score

        # FIX B: Reasoning Faithfulness
        steps = r.get("metadata", {}).get("steps", [])
        rf = reasoning_faithfulness(r.get("output_text", ""), steps)
        r["reasoning_faithfulness"] = rf
        faithfulness_scores.append(rf)

        h_flag = is_hallucinated(r.get("output_text", ""), r)
        f_flag = is_failure(
            r.get("output_text", ""),
            r,
            failed_call=r.get("failed_call", False),
            error=r.get("error"),
        )
        r["is_hallucinated"] = h_flag
        r["is_failure"] = f_flag
        hallucination_flags.append(1.0 if h_flag else 0.0)
        failure_flags.append(1.0 if f_flag else 0.0)

    aggs["reasoning_faithfulness"] = round(float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0, 4)
    aggs["bert_score"] = round(float(np.mean(bert_scores)) if bert_scores else 0.0, 4)
    aggs["hallucination_rate"] = round(float(np.mean(hallucination_flags)) if hallucination_flags else 0.0, 4)
    aggs["failure_rate"] = round(float(np.mean(failure_flags)) if failure_flags else 0.0, 4)

    # Step 5: Efficiency Metrics
    eff_metrics = compute_efficiency_metrics(scored_rows, total_wall_time)
    aggs.update(eff_metrics)

    # Category Accuracy Summary
    em_scores = [r.get("exact_match", 0.0) for r in scored_rows]
    aggs["category_accuracy"] = round(float(np.mean(em_scores)) if em_scores else 0.0, 4)

    logger.info("Model '%s' Benchmark Complete. EM: %.4f | Judge: %.4f | Faithfulness: %.4f",
                model_name, aggs["exact_match"], aggs["judge_score"], aggs["reasoning_faithfulness"])

    return {
        "model": model_name,
        "scored_rows": scored_rows,
        "aggregates": aggs,
    }


# ============================================================================
# 11. MAIN ENTRYPOINT & CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BFSI LLM 15-Metric Benchmark Pipeline using NVIDIA NeMo Evaluator & LM Studio"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to synthetic BFSI dataset (default: data/synthetic_bfsi_train.jsonl)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="List of model identifiers to benchmark (default: ['local-model'])",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="LM Studio OpenAI-compatible endpoint URL (default: http://127.0.0.1:1234/v1)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model identifier to use as LLM Judge (default: 'local-model')",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of rows to benchmark for initial verification (default: 5; use 0 or -1 for all 240 rows)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max generation tokens for model inference (default: per-model default, e.g. 2048 for reasoning models, 512 for others)",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help=f"Max tokens for LLM-as-a-Judge call (default: {DEFAULT_JUDGE_MAX_TOKENS})",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-a-Judge calls to speed up initial sanity testing",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip inference generation and score existing predictions from --predictions file",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to existing predictions.jsonl for re-scoring without re-running models",
    )
    parser.add_argument(
        "--consistency-sample",
        type=int,
        default=0,
        help="Number of sample rows to re-run 3x per model for consistency spread analysis (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save master_table.csv and predictions (default: results)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to save visualization charts (default: plots)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("   BFSI LLM 15-METRIC BENCHMARKING PIPELINE")
    print("   NVIDIA NeMo Evaluator & LM Studio Integration")
    print("=" * 80)
    print(f"Dataset path:      {args.dataset}")
    print(f"Models to test:    {args.models}")
    print(f"Sample size:       {'ALL rows' if args.sample_size <= 0 else f'{args.sample_size} rows'}")
    print(f"LM Studio endpoint:{args.base_url}")
    print(f"Judge model:       {args.judge_model} (Skip judge: {args.skip_judge} | max_tokens: {args.judge_max_tokens})")
    print(f"Gen max_tokens:    {'Per-model auto' if args.max_tokens is None else args.max_tokens}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Plots directory:   {args.plots_dir}")
    print("=" * 80)

    # 1. Load Dataset
    if not args.dataset.exists():
        logger.error("Dataset file not found at: %s", args.dataset)
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    total_available = len(all_rows)
    if args.sample_size > 0:
        rows_to_run = all_rows[: args.sample_size]
        logger.info("Loaded %d rows for verification (out of %d total).", len(rows_to_run), total_available)
    else:
        rows_to_run = all_rows
        logger.info("Loaded full dataset: %d rows.", len(rows_to_run))

    # 2. Benchmark Each Model
    all_results: Dict[str, Dict[str, Any]] = {}
    all_predictions_to_save: List[Dict[str, Any]] = []

    for model_name in args.models:
        existing_preds = None
        if args.skip_generation:
            # Priority 1: per-model folder written by a previous run
            per_model_preds = args.output_dir / _safe_model_dir(model_name) / "predictions.jsonl"
            # Priority 2: explicit --predictions path (combined file, filtered by model)
            combined_preds = args.predictions

            if per_model_preds.exists():
                with open(per_model_preds, "r", encoding="utf-8") as pf:
                    existing_preds = [json.loads(line) for line in pf if line.strip()]
                logger.info("Resuming '%s' from per-model folder (%d rows): %s",
                            model_name, len(existing_preds), per_model_preds)
            elif combined_preds and combined_preds.exists():
                with open(combined_preds, "r", encoding="utf-8") as pf:
                    existing_preds = [
                        rec for rec in (json.loads(line) for line in pf if line.strip())
                        if rec.get("model") == model_name
                    ]
                logger.info("Resuming '%s' from combined predictions (%d rows): %s",
                            model_name, len(existing_preds), combined_preds)
            else:
                logger.warning(
                    "--skip-generation requested for '%s' but no predictions file found. "
                    "Running generation instead.", model_name
                )

            if existing_preds and args.sample_size > 0 and len(existing_preds) > args.sample_size:
                existing_preds = existing_preds[: args.sample_size]
                logger.info("Sample-size %d applied to existing predictions: using first %d rows.",
                            args.sample_size, len(existing_preds))

        res = benchmark_model(
            model_name=model_name,
            dataset_rows=rows_to_run,
            base_url=args.base_url,
            judge_model=args.judge_model,
            skip_judge=args.skip_judge,
            existing_predictions=existing_preds,
            max_tokens=args.max_tokens,
            judge_max_tokens=args.judge_max_tokens,
        )
        all_results[model_name] = res
        all_predictions_to_save.extend(res["scored_rows"])

        # ── Per-model folder ───────────────────────────────────────────────
        model_dir = args.output_dir / _safe_model_dir(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        # predictions.jsonl – every scored row for this model
        model_preds_path = model_dir / "predictions.jsonl"
        with open(model_preds_path, "w", encoding="utf-8") as mpf:
            for row in res["scored_rows"]:
                mpf.write(json.dumps(row) + "\n")
        logger.info("[%s] Saved %d predictions to: %s",
                    model_name, len(res["scored_rows"]), model_preds_path)

        # aggregates.json – the 15 aggregate metric values
        model_aggs_path = model_dir / "aggregates.json"
        with open(model_aggs_path, "w", encoding="utf-8") as maf:
            json.dump({"model": model_name, **res["aggregates"]}, maf, indent=2)
        logger.info("[%s] Saved aggregates to: %s", model_name, model_aggs_path)
        # ──────────────────────────────────────────────────────────────────

    # 3. Save Raw Predictions JSONL
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_out_path = args.output_dir / "predictions.jsonl"
    with open(preds_out_path, "w", encoding="utf-8") as pf:
        for p in all_predictions_to_save:
            pf.write(json.dumps(p) + "\n")
    logger.info("Saved %d scored prediction records to: %s", len(all_predictions_to_save), preds_out_path)

    # 4. Consistency Spread Check (Optional 3x Run)
    consistency_data: Dict[str, List[float]] = {}
    if args.consistency_sample > 0:
        c_subset = rows_to_run[: args.consistency_sample]
        logger.info("Running 3x consistency check on %d sample rows per model...", len(c_subset))
        for model_name in args.models:
            scores_3x = []
            for run_i in range(3):
                logger.info("Consistency run %d/3 for '%s'...", run_i + 1, model_name)
                sim_run = benchmark_model(
                    model_name=model_name,
                    dataset_rows=c_subset,
                    base_url=args.base_url,
                    skip_judge=True,
                    max_tokens=args.max_tokens,
                    judge_max_tokens=args.judge_max_tokens,
                )
                scores_3x.append(sim_run["aggregates"]["exact_match"])
            consistency_data[model_name] = scores_3x

    # 5. Build Master Table and Tidy Category Breakdown (FIX B)
    master_df, cat_df = build_master_table(all_results, output_dir=args.output_dir)

    # 6. Generate All Visualizations
    logger.info("Generating visualization plots...")
    plot_bar_per_metric(master_df, output_dir=args.plots_dir)
    plot_radar(master_df, output_dir=args.plots_dir)
    plot_quality_vs_latency(master_df, output_dir=args.plots_dir)
    plot_category_heatmaps(cat_df, output_dir=args.plots_dir)
    if consistency_data:
        plot_consistency_spread(consistency_data, output_dir=args.plots_dir)

    # 7. Statistical Significance Check
    run_statistical_significance(all_results)

    # 8. Print Console Summary Table
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK RESULTS (15 METRICS):")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(master_df.to_string(index=False))
    print("=" * 80)

    if not cat_df.empty:
        print("\n" + "=" * 80)
        print("TIDY CATEGORY ACCURACY BREAKDOWN:")
        print("=" * 80)
        print(cat_df.to_string(index=False))
        print("=" * 80)

    model_dirs_summary = "\n".join(
        f"  - {_safe_model_dir(m)}/  ->  {args.output_dir / _safe_model_dir(m)}"
        for m in args.models
    )
    print(f"\nPipeline complete! Artifacts generated:\n"
          f"  - Master Table:       {args.output_dir / 'master_table.csv'}\n"
          f"  - Category Breakdown: {args.output_dir / 'category_accuracy.csv'}\n"
          f"  - Predictions JSONL:  {preds_out_path}\n"
          f"  - Plots Directory:    {args.plots_dir}/\n"
          f"\nPer-model result folders:\n{model_dirs_summary}\n")



if __name__ == "__main__":
    main()
