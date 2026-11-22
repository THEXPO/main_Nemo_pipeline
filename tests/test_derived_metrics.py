"""
test_derived_metrics.py
=======================
Comprehensive test suite for NeMo Evaluator Derived Metrics Layer.
Tests:
1. Synthetic unit tests per feature (empty inputs, missing fields, extreme values).
2. Golden test on Llama-3.1-8B dataset (all 27 features verified against expected ground truth).
3. Regression & Fail-Safe tests:
   - Exception injection does not crash.
   - DERIVED_METRICS=0 kill switch disables layer.
   - Existing outputs remain unchanged.
4. Idempotency test:
   - Recompute produces byte-identical results across runs.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from derived_metrics import (
    DerivedConfig,
    FEATURE_CATALOG,
    compute_derived,
    process_model_files,
    process_directory,
    is_derived_metrics_enabled,
)


class TestDerivedMetricsSynthetic(unittest.TestCase):
    """Synthetic unit tests for all 27 features, including edge cases and missing fields."""

    def test_empty_rows(self):
        derived, df = compute_derived([])
        self.assertTrue(df.empty)
        # All features should evaluate cleanly to None or valid fallback, without throwing exceptions
        self.assertIn("fn_call_leak_rate", derived)
        self.assertIsNone(derived["fn_call_leak_rate"])
        self.assertIsNone(derived["latency_per_correct"])
        self.assertIsNone(derived["em_ci95_halfwidth"])

    def test_missing_fields_fail_safe(self):
        # Rows with almost no fields populated
        sparse_rows = [
            {"id": "item_1"},
            {"id": "item_2", "output_text": "Random response without standard fields"},
        ]
        derived, df = compute_derived(sparse_rows, aggregates={})
        self.assertEqual(len(df), 2)
        # Verify no crash, features safely return None
        for key in FEATURE_CATALOG:
            self.assertIn(key, derived)

    def test_format_features_synthetic(self):
        rows = [
            {
                "id": "1",
                "output_text": 'Here is reasoning.\n{"name": "lookup", "parameters": {}}',
                "extracted_answer": "42",
                "target": "42",
                "normalized_answer": "42",
                "task_type": "qa_numeric_program",
            },
            {
                "id": "2",
                "output_text": "Final Answer: Yes",
                "extracted_answer": "Yes",
                "target": "Yes",
                "normalized_answer": "Yes",
                "task_type": "qa_label",
            },
            {
                "id": "3",
                "output_text": "No function call here, just plain text",
                "extracted_answer": "",
                "target": "No",
                "normalized_answer": "No",
                "task_type": "qa_label",
            },
        ]
        derived, _ = compute_derived(rows)
        # 1 out of 3 has fn_call leak: ~0.3333
        self.assertEqual(derived["fn_call_leak_rate"], 0.3333)
        # 2 out of 3 have non-empty extracted answer: ~0.6667
        self.assertEqual(derived["extraction_success_rate"], 0.6667)
        # 1 out of 3 has 'Final Answer': ~0.3333
        self.assertEqual(derived["final_answer_line_rate"], 0.3333)

    def test_task_features_synthetic(self):
        rows = [
            {"task_type": "qa_label", "exact_match": 1.0},
            {"task_type": "qa_label", "exact_match": 0.0},
            {"task_type": "qa_numeric_program", "exact_match": 1.0},
            {"task_type": "qa_numeric_program", "exact_match": 1.0},
        ]
        derived, _ = compute_derived(rows)
        self.assertEqual(derived["label_exact_match"], 0.5)
        self.assertEqual(derived["numeric_exact_match"], 1.0)

    def test_judge_features_synthetic(self):
        rows = [
            {"judge_score": 5.0, "completeness": 5.0, "helpfulness": 5.0, "exact_match": 0.0},
            {"judge_score": 1.0, "completeness": 1.0, "helpfulness": 1.0, "exact_match": 0.0},
            {"judge_score": 4.0, "completeness": 4.0, "helpfulness": 4.0, "exact_match": 1.0},
        ]
        derived, _ = compute_derived(rows)
        # mean js = 10/3 = 3.3333; mean em = 1/3 = 0.3333; gap = (3.3333/5) - 0.3333 = 0.6667 - 0.3333 = 0.3333
        self.assertAlmostEqual(derived["judge_em_gap"], 0.3333, places=3)
        self.assertAlmostEqual(derived["judge_composite"], 3.3333, places=3)
        # judge leniency: judge_score >= 4 and em == 0 -> row 1 matches -> 1/3 = 0.3333
        self.assertAlmostEqual(derived["judge_leniency_rate"], 0.3333, places=3)
        # judge fail: judge_score <= 2 -> row 2 matches -> 1/3 = 0.3333
        self.assertAlmostEqual(derived["judge_fail_rate"], 0.3333, places=3)

    def test_reliability_features_synthetic(self):
        rows = [
            {"exact_match": 1.0, "is_hallucinated": False},
            {"exact_match": 1.0, "is_hallucinated": True},
            {"exact_match": 0.0, "is_hallucinated": True},
            {"exact_match": 0.0, "is_hallucinated": False},
        ]
        derived, _ = compute_derived(rows)
        # reliable_accuracy: em == 1 and not H -> row 0 -> 1/4 = 0.25
        self.assertEqual(derived["reliable_accuracy"], 0.25)
        # hallucinated_but_correct: em == 1 and H -> row 1 -> 1/4 = 0.25
        self.assertEqual(derived["hallucinated_but_correct_rate"], 0.25)
        # hallucination_separation: P(H | EM=0) - P(H | EM=1) = 0.5 - 0.5 = 0.0
        self.assertEqual(derived["hallucination_separation"], 0.0)

    def test_efficiency_features_synthetic(self):
        rows = [
            {"latency_seconds": 10.0, "tokens_in": 100, "tokens_out": 50, "exact_match": 1.0, "judge_score": 5.0},
            {"latency_seconds": 20.0, "tokens_in": 100, "tokens_out": 150, "exact_match": 0.0, "judge_score": 3.0},
        ]
        derived, _ = compute_derived(rows)
        # mean EM = 0.5; mean lat = 15.0; lat_per_correct = 15 / 0.5 = 30.0
        self.assertEqual(derived["latency_per_correct"], 30.0)
        # mean tokens_out = 100; tokens_per_correct = 100 / 0.5 = 200.0
        self.assertEqual(derived["tokens_per_correct"], 200.0)
        # quality_per_second = 4.0 / 15.0 = 0.2667
        self.assertEqual(derived["quality_per_second"], 0.2667)
        # output_input_token_ratio = mean(0.5, 1.5) = 1.0
        self.assertEqual(derived["output_input_token_ratio"], 1.0)
        # verbosity_accuracy_corr = correlation between [50, 150] and [1.0, 0.0] = -1.0
        self.assertEqual(derived["verbosity_accuracy_corr"], -1.0)


class TestDerivedMetricsGolden(unittest.TestCase):
    """Golden verification test on Llama-3.1-8B evaluation data."""

    @classmethod
    def setUpClass(cls):
        cls.preds_path = Path("results/meta-llama-3.1-8b-instruct_predictions.jsonl")
        cls.aggs_path = Path("results/meta-llama-3.1-8b-instruct_aggregates.json")
        if not cls.preds_path.exists() or not cls.aggs_path.exists():
            raise unittest.SkipTest("Llama-3.1-8B evaluation results not found in results/")

        with open(cls.preds_path, "r", encoding="utf-8") as f:
            cls.rows = [json.loads(line) for line in f if line.strip()]
        with open(cls.aggs_path, "r", encoding="utf-8") as f:
            cls.aggs = json.load(f)

        cls.derived, cls.df = compute_derived(cls.rows, cls.aggs)

    def test_golden_format_metrics(self):
        self.assertAlmostEqual(self.derived["fn_call_leak_rate"], 0.70, places=2)
        self.assertAlmostEqual(self.derived["extraction_success_rate"], 0.13, places=2)
        self.assertAlmostEqual(self.derived["final_answer_line_rate"], 0.69, places=2)
        self.assertAlmostEqual(self.derived["answer_recoverable_rate"], 0.50, places=2)

    def test_golden_task_metrics(self):
        self.assertAlmostEqual(self.derived["label_exact_match"], 0.06, places=2)
        self.assertAlmostEqual(self.derived["numeric_exact_match"], 0.96, places=2)

    def test_golden_judge_metrics(self):
        self.assertAlmostEqual(self.derived["judge_em_gap"], 0.442, places=3)
        self.assertAlmostEqual(self.derived["judge_composite"], 4.77, places=2)
        self.assertAlmostEqual(self.derived["judge_leniency_rate"], 0.43, places=2)
        self.assertAlmostEqual(self.derived["judge_fail_rate"], 0.06, places=2)

    def test_golden_reliability_metrics(self):
        self.assertAlmostEqual(self.derived["reliable_accuracy"], 0.41, places=2)
        self.assertAlmostEqual(self.derived["hallucinated_but_correct_rate"], 0.10, places=2)
        self.assertAlmostEqual(self.derived["ungrounded_number_ratio"], 0.4174, places=3)
        self.assertAlmostEqual(self.derived["ungrounded_flag_delta"], 0.2527, places=3)
        self.assertAlmostEqual(self.derived["hallucination_separation"], 0.1917, places=3)
        self.assertAlmostEqual(self.derived["label_hedge_rate"], 0.80, places=2)

    def test_golden_efficiency_metrics(self):
        self.assertAlmostEqual(self.derived["latency_per_correct"], 18.4055, places=2)
        self.assertAlmostEqual(self.derived["tokens_per_correct"], 256.39, delta=0.1)
        self.assertAlmostEqual(self.derived["latency_p95"], 16.215, places=2)
        self.assertAlmostEqual(self.derived["latency_cv"], 0.3634, places=3)
        self.assertAlmostEqual(self.derived["quality_per_second"], 0.5071, places=3)
        self.assertAlmostEqual(self.derived["output_input_token_ratio"], 0.4169, places=3)
        self.assertAlmostEqual(self.derived["verbosity_accuracy_corr"], -0.3928, places=3)

    def test_golden_robustness_metrics(self):
        self.assertAlmostEqual(self.derived["macro_exact_match"], 0.3397, places=3)
        self.assertAlmostEqual(self.derived["worst_category_em"], 0.0, places=2)
        self.assertAlmostEqual(self.derived["category_em_std"], 0.3104, places=3)
        self.assertAlmostEqual(self.derived["em_ci95_halfwidth"], 0.095, places=3)


class TestDerivedMetricsModesAndRegression(unittest.TestCase):
    """Tests file generation, Modes A, B, C, Idempotency, and Kill-Switch."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_derived_"))
        # Copy Llama 3.1 sample files into temp dir
        self.src_preds = Path("results/meta-llama-3.1-8b-instruct_predictions.jsonl")
        self.src_aggs = Path("results/meta-llama-3.1-8b-instruct_aggregates.json")
        self.dest_preds = self.test_dir / "meta-llama-3.1-8b-instruct_predictions.jsonl"
        self.dest_aggs = self.test_dir / "meta-llama-3.1-8b-instruct_aggregates.json"

        shutil.copy(self.src_preds, self.dest_preds)
        shutil.copy(self.src_aggs, self.dest_aggs)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mode_a_post_hoc_file_generation(self):
        """Mode A: process existing model files without modifying originals."""
        orig_preds_content = self.dest_preds.read_text(encoding="utf-8")
        orig_aggs_content = self.dest_aggs.read_text(encoding="utf-8")

        res = process_model_files(
            predictions_path=self.dest_preds,
            aggregates_path=self.dest_aggs,
            output_dir=self.test_dir,
            merge_into_aggregates=False,
        )
        self.assertIn("derived_features", res)

        # Check that original files remain byte-identical
        self.assertEqual(self.dest_preds.read_text(encoding="utf-8"), orig_preds_content)
        self.assertEqual(self.dest_aggs.read_text(encoding="utf-8"), orig_aggs_content)

        # Check new files created
        derived_file = self.test_dir / "meta-llama-3.1-8b-instruct_derived_features.json"
        all_metrics_file = self.test_dir / "meta-llama-3.1-8b-instruct_all_metrics.json"
        csv_file = self.test_dir / "meta-llama-3.1-8b-instruct_per_sample_features.csv"

        self.assertTrue(derived_file.exists())
        self.assertTrue(all_metrics_file.exists())
        self.assertTrue(csv_file.exists())

        # Validate all_metrics.json has original 15 + 27 derived
        with open(all_metrics_file, "r", encoding="utf-8") as f:
            all_m = json.load(f)
        self.assertEqual(all_m["model"], "meta-llama-3.1-8b-instruct")
        self.assertEqual(all_m["exact_match"], 0.51)
        self.assertEqual(all_m["label_exact_match"], 0.06)
        self.assertEqual(all_m["fn_call_leak_rate"], 0.70)

    def test_idempotency_mode_c(self):
        """Mode C: Recomputing derived metrics twice produces identical results."""
        process_model_files(
            predictions_path=self.dest_preds,
            aggregates_path=self.dest_aggs,
            output_dir=self.test_dir,
        )
        derived_file = self.test_dir / "meta-llama-3.1-8b-instruct_derived_features.json"
        all_metrics_file = self.test_dir / "meta-llama-3.1-8b-instruct_all_metrics.json"

        content1_derived = derived_file.read_text(encoding="utf-8")
        content1_all = all_metrics_file.read_text(encoding="utf-8")

        # Run again
        process_model_files(
            predictions_path=self.dest_preds,
            aggregates_path=self.dest_aggs,
            output_dir=self.test_dir,
        )
        content2_derived = derived_file.read_text(encoding="utf-8")
        content2_all = all_metrics_file.read_text(encoding="utf-8")

        self.assertEqual(content1_derived, content2_derived)
        self.assertEqual(content1_all, content2_all)

    def test_kill_switch_env_var(self):
        """Kill switch DERIVED_METRICS=0 disables derived processing."""
        os.environ["DERIVED_METRICS"] = "0"
        try:
            cfg = DerivedConfig.from_env_or_dict()
            self.assertFalse(cfg.enabled)
            res = process_model_files(
                predictions_path=self.dest_preds,
                aggregates_path=self.dest_aggs,
                output_dir=self.test_dir,
            )
            self.assertEqual(res, {})
            derived_file = self.test_dir / "meta-llama-3.1-8b-instruct_derived_features.json"
            self.assertFalse(derived_file.exists())
        finally:
            os.environ["DERIVED_METRICS"] = "1"

    def test_merge_into_aggregates_flag(self):
        """Optional flag --merge-into-aggregates adds 'derived' key without touching others."""
        process_model_files(
            predictions_path=self.dest_preds,
            aggregates_path=self.dest_aggs,
            output_dir=self.test_dir,
            merge_into_aggregates=True,
        )
        with open(self.dest_aggs, "r", encoding="utf-8") as f:
            updated_aggs = json.load(f)

        self.assertIn("derived", updated_aggs)
        self.assertEqual(updated_aggs["exact_match"], 0.51)
        self.assertEqual(updated_aggs["derived"]["label_exact_match"], 0.06)

    def test_injected_exception_fail_safe(self):
        """Test that an unexpected exception injected into derived processing is caught and logged, never crashing the pipeline."""
        from unittest.mock import patch

        with patch("derived_metrics.compute_derived", side_effect=RuntimeError("Simulated unexpected crash!")):
            # When calling process_directory or process_model_files, it should catch error and not raise
            try:
                res = process_directory(self.test_dir)
                self.assertEqual(res, [])
            except Exception as e:
                self.fail(f"process_directory raised an unexpected exception: {e}")

    def test_original_results_schema_and_content_unmodified(self):
        """Verify that running process_directory does not alter existing predictions and aggregates."""
        orig_preds_hash = hash(self.dest_preds.read_bytes())
        orig_aggs_hash = hash(self.dest_aggs.read_bytes())

        # Process directory without merge flag
        process_directory(self.test_dir, merge_into_aggregates=False)

        # Confirm hashes are exactly unchanged
        self.assertEqual(hash(self.dest_preds.read_bytes()), orig_preds_hash)
        self.assertEqual(hash(self.dest_aggs.read_bytes()), orig_aggs_hash)


if __name__ == "__main__":
    unittest.main()
