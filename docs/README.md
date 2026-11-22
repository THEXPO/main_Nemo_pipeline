# NeMo Evaluator & LM Studio Benchmark Knowledge Base

Welcome to the comprehensive documentation directory for the **NeMo Evaluator LLM Benchmarking Pipeline**. This knowledge base contains every technical specification, architectural decision, metric definition, dataset schema, execution mode, empirical result, and full-stack implementation detail of the project.

---

## Documentation Index

The documentation is organized into focused, modular technical documents:

| Document | Topic & Scope |
|---|---|
| [**1. System Architecture (`README_ARCHITECTURE.md`)**](README_ARCHITECTURE.md) | High-level system design, modular stage-independent execution pipeline, component map, and fail-safe error isolation principles. |
| [**2. Complete 42-Metric Evaluation Reference (`README_METRICS_AND_EVALUATION.md`)**](README_METRICS_AND_EVALUATION.md) | Exhaustive mathematical formulas, algorithms, regexes, and rubrics for all 15 Core Metrics and 27 Derived Metrics across 6 evaluation dimensions. |
| [**3. Datasets & Schemas (`README_DATASETS.md`)**](README_DATASETS.md) | Unified JSONL benchmark schema, task types (`qa_label`, `qa_numeric_program`), RBI KYC regulatory domain details, and sample structures. |
| [**4. Pipeline Execution & CLI Guide (`README_PIPELINE_AND_USAGE.md`)**](README_PIPELINE_AND_USAGE.md) | Setup prerequisites, LM Studio configuration, CLI flags, the 3 execution modes (Modes A, B, and C), idempotency, and the kill switch. |
| [**5. Benchmark Results & Leaderboard (`README_BENCHMARK_RESULTS.md`)**](README_BENCHMARK_RESULTS.md) | Empirical evaluation analysis across 5 benchmarked LLMs, master 42-metric score comparison, format leakage insights, and compliance findings. |
| [**6. Full-Stack Dashboard & API (`README_FRONTEND_AND_SERVER.md`)**](README_FRONTEND_AND_SERVER.md) | Flask REST API endpoints, Vite + React + TypeScript frontend dashboard, 42 individual metric graphs, head-to-head comparison, and master table view. |

---

## High-Level Project Overview

```
                      ┌────────────────────────────────────────┐
                      │    Local LM Studio Inference Server    │
                      │       (OpenAI /v1/chat API @ 1234)     │
                      └──────────────────▲─────────────────────┘
                                         │ HTTP REST
                      ┌──────────────────┴─────────────────────┐
                      │         llm_benchmark.py               │
                      │  Stage 1: Generation                   │
                      │  Stage 2: NeMo NLP Similarity Scorers  │
                      │  Stage 3: LLM-as-a-Judge Anchor Score  │
                      │  Stage 4: Custom, Safety & Efficiency  │
                      │  Stage 5: Master Table & Static Plots  │
                      └──────────────────┬─────────────────────┘
                                         │ Writes Predictions & Aggregates
                                         ▼
                      ┌────────────────────────────────────────┐
                      │         derived_metrics.py             │
                      │  • Mode A: Post-Hoc Batch Processing   │
                      │  • Mode B: Automatic Pipeline Hook     │
                      │  • Mode C: Deterministic Recompute     │
                      │  • Pure 27-Feature Computation Engine  │
                      └──────────────────┬─────────────────────┘
                                         │ Produces Derived Outputs
                                         ▼
                      ┌────────────────────────────────────────┐
                      │         results/ Directory             │
                      │  • <model>_predictions.jsonl           │
                      │  • <model>_aggregates.json             │
                      │  • <model>_derived_features.json       │
                      │  • <model>_all_metrics.json (42 flat)  │
                      │  • <model>_per_sample_features.csv     │
                      │  • master_table.csv & plots/*.png      │
                      └──────────────────┬─────────────────────┘
                                         │ REST API
                      ┌──────────────────┴─────────────────────┐
                      │         server.py (Flask API @ 5000)   │
                      │  Endpoints for models, table, samples  │
                      └──────────────────┬─────────────────────┘
                                         │ JSON Payload
                      ┌──────────────────┴─────────────────────┐
                      │    React + TypeScript Dashboard        │
                      │  • 42-Metric Interactive Bar Graphs    │
                      │  • Head-to-Head 2-Model Comparison     │
                      │  • Sample Inspector with Tokens & Time │
                      │  • 42-Column Master Table View         │
                      └────────────────────────────────────────┘
```

---

## Project Repository File Structure

```
NeMo Project/
├── docs/                                  # Complete technical documentation knowledge base
│   ├── README.md                          # Documentation index (this file)
│   ├── README_ARCHITECTURE.md             # System architecture & component design
│   ├── README_METRICS_AND_EVALUATION.md   # Mathematical reference for all 42 metrics
│   ├── README_DATASETS.md                 # Dataset specifications & regulatory context
│   ├── README_PIPELINE_AND_USAGE.md       # Pipeline CLI, execution modes, & kill switch
│   ├── README_BENCHMARK_RESULTS.md        # Benchmark evaluation & model leaderboard
│   └── README_FRONTEND_AND_SERVER.md      # Web dashboard & Flask backend reference
│
├── llm_benchmark.py                       # Main 15-metric evaluation pipeline
├── derived_metrics.py                     # 27 derived features computation module & CLI
├── parsers.py                             # Rule-based output parsers and normalizers
├── prompt_templates.py                    # Domain & LLM-as-a-Judge prompt templates
├── run_model.py                           # LM Studio API client connection interface
├── server.py                              # Flask backend serving results & UI build
│
├── frontend/                              # Interactive web analytics dashboard
│   ├── src/
│   │   ├── App.tsx                        # Main UI layout, charts, & master table
│   │   ├── ModelComparison.tsx            # Head-to-head 2-model comparative analytics
│   │   └── main.tsx                       # React application entry point
│   ├── package.json                       # Frontend dependencies & scripts
│   ├── vite.config.ts                     # Vite build configuration
│   └── dist/                              # Compiled production web assets
│
├── newdata/
│   └── kyc_eval_dataset.jsonl             # Gold evaluation dataset (100 RBI KYC cases)
│
├── results/                               # Scored predictions, aggregates, & outputs
│   ├── <model>_predictions.jsonl          # Raw model predictions & sample scores
│   ├── <model>_aggregates.json            # 15 core summary metrics
│   ├── <model>_derived_features.json      # 27 derived metrics + feature catalog
│   ├── <model>_all_metrics.json           # 42 consolidated metrics (flat dictionary)
│   ├── <model>_per_sample_features.csv    # Intermediate per-sample helper columns
│   ├── master_table.csv                   # Master comparison table across models
│   ├── category_accuracy.csv              # Accuracy breakdown by category & task type
│   └── plots/                             # Generated benchmark visualization charts
│
├── tests/
│   ├── __init__.py                        # Test package marker
│   └── test_derived_metrics.py            # Comprehensive 19-test validation suite
│
├── requirements.txt                       # Python dependencies
└── README.md                              # Root repository overview
```

---

## Quick Reference Commands

### 1. Run Complete 15-Metric Benchmark (Fresh Run + Derived Hook)
```powershell
python llm_benchmark.py --sample-size 0
```

### 2. Compute Derived Metrics Offline (Modes A & C)
```powershell
# Recompute all models in results/ deterministically without re-running models:
python derived_metrics.py --recompute --dir results/
```

### 3. Run Automated Test Suite
```powershell
python -m pytest tests/test_derived_metrics.py -v
```

### 4. Launch Web Dashboard
```powershell
python server.py
# Open http://localhost:5000 in your browser
```
