# System Architecture & Technical Design

## 1. Executive Summary & Purpose

The **NeMo Evaluator & LM Studio Benchmark Pipeline** is a modular, dataset-agnostic evaluation framework engineered to benchmark Large Language Models (LLMs) on complex compliance, regulatory reasoning, and arithmetic problem-solving tasks.

It pairs **local OpenAI-compatible inference (via LM Studio)** with evaluation methodologies derived from **NVIDIA NeMo Evaluator**. The system combines 15 standard core NLP and LLM-as-a-judge metrics with an additive 27-metric derived features layer, providing a complete 42-dimensional performance profile for every evaluated model.

---

## 2. Core Architectural Principles

1. **Decoupled Execution & Stage Independence**:
   - Model inference generation (Stage 1) is decoupled from metric scoring (Stages 2–5).
   - Once predictions are saved to disk, all evaluation stages can be run, re-run, or modified offline without re-querying the model or the LLM judge.
2. **Pure Computational Core**:
   - Feature extraction algorithms in `derived_metrics.py` contain zero I/O side-effects.
   - Computations take in-memory dictionaries and return pure dictionaries and DataFrames. File adapters handle I/O independently.
3. **Additive-Only & Schema Invariance**:
   - Adding new metrics never alters, renames, or deletes existing prediction fields or output formats.
   - Existing evaluation outputs (`*_predictions.jsonl`, `*_aggregates.json`, `master_table.csv`) remain 100% byte- and schema-compatible.
4. **Fail-Safe & Graceful Degradation**:
   - Every metric extraction and derived computation runs inside protected `try/except` blocks.
   - Any computation failure or missing data field emits a diagnostic warning and resolves to `None` (`null` in JSON), guaranteeing the evaluation run never crashes or loses previously collected records.
5. **Deterministic Reproducibility**:
   - All stochastic computations (such as bootstrap confidence interval estimations) utilize fixed random seeds (`numpy.random.default_rng(0)`). Running recomputations repeatedly yields byte-identical files.

---

## 3. End-to-End Pipeline Stages

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INFERENCE GENERATION (llm_benchmark.py -> run_model.py)      │
│ • Reads dataset JSONL rows.                                            │
│ • Formats task prompts using prompt_templates.py.                      │
│ • Dispatches requests to LM Studio REST endpoint (/v1/chat/completions)│
│ • Records: output_text, latency_seconds, tokens_in, tokens_out.        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 2: SIMILARITY & EXTRACTION SCORING (NeMo Scorers & Parsers)     │
│ • Extracts normalized candidate answer from output text.               │
│ • Exact Match (EM): Strict normalization matching target label/number. │
│ • Token F1 Score: Harmonic mean of token precision and recall.         │
│ • BLEU & ROUGE-L: N-gram precision & longest common subsequence recall.│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 3: LLM-AS-A-JUDGE SCORING (Anchored Evaluation Rubric)          │
│ • Evaluator model scores response against prompt, context, and target. │
│ • Scores: Correctness (1-5), Completeness (1-5), Helpfulness (1-5).    │
│ • Enforces sanity rules: Caps completeness/helpfulness <= 2 on failure.│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 4: CUSTOM, SAFETY & EFFICIENCY METRICS                           │
│ • Reasoning Faithfulness: Fraction of ground-truth calculation steps.  │
│ • BERTScore: Semantic embedding similarity against reference answer.   │
│ • Hallucination Rate: Whitelist verification of all generated numbers. │
│ • Efficiency: Latency per request, throughput (req/s), tokens/sec.     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 5: ARTIFACT GENERATION & PIPELINE HOOK                           │
│ • Writes results/<model>_predictions.jsonl and <model>_aggregates.json.│
│ • Automatically calls derived_metrics.py pipeline hook (Mode B).       │
│ • Computes 27 derived features across 6 evaluation dimensions.         │
│ • Produces: <model>_derived_features.json, <model>_all_metrics.json,   │
│   and <model>_per_sample_features.csv.                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 6: FULL-STACK VISUALIZATION (server.py & frontend/)             │
│ • Flask backend serves results and 42-metric metadata.                 │
│ • React + Vite UI renders 42 bar charts, head-to-head comparison,     │
│   category accuracy heatmaps, and 42-column master matrix table.       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Architecture & Responsibilities

### `llm_benchmark.py` (Orchestration Engine)
- Serves as the central CLI and pipeline orchestrator.
- Manages dataset ingestion, batch model iteration, stage execution, result serialization, and static chart generation (`matplotlib`).
- Houses the adapter registry (`ModelProfile`) defining model-specific prompt quirks, stop tokens, and context window limits.

### `derived_metrics.py` (Pure Computation Module)
- Completely independent module implementing the 27 derived features.
- Contains the `DerivedConfig` dataclass storing heuristic regexes, task types, and field keys.
- Implements `compute_derived(rows, aggregates, config) -> (derived_dict, per_sample_df)` with zero file I/O inside calculations.
- Exposes thin adapters (`process_model_files`, `process_directory`) for file generation and CLI operation.

### `run_model.py` (Inference Client)
- Direct HTTP client interfacing with LM Studio's OpenAI-compatible REST server (`/v1/chat/completions`).
- Implements connection health checks (`check_lm_studio_server`), timeout handling, and request timing.

### `parsers.py` (Text & Number Parsing)
- Extracts answer substrings from LLM responses (e.g., following `Final Answer:` or closing reasoning tags).
- Standardizes financial numbers (removes currency symbols, commas, percent signs, and parses floating point values).

### `prompt_templates.py` (Instruction Formatting)
- Generates structured compliance and arithmetic prompts.
- Contains the anchored LLM-as-a-Judge system prompt ensuring consistent scoring on a 1.0 to 5.0 scale.

### `server.py` (Flask REST API)
- Serves aggregated and per-sample benchmark data to the web client.
- Exposes REST endpoints (`/api/models`, `/api/master-table`, `/api/derived-catalog`, `/api/predictions`, `/api/sample/<id>`).
- Directly serves the pre-compiled React frontend from `frontend/dist/`.

### `frontend/` (React + TypeScript Web UI)
- Modern SPA built with Vite, React 19, TypeScript, and Recharts.
- Renders high-density visual dashboards, dynamic delta calculations, radar plots, and interactive data tables across all 42 metrics.

---

## 5. Storage Contract & File Formats

For every evaluated model, the pipeline generates 5 standardized output files within `results/`:

| File Pattern | Format | Purpose | Schema Details |
|---|---|---|---|
| `<model>_predictions.jsonl` | JSON Lines | Per-sample raw outputs and scores | 28 fields per row (`id`, `prompt`, `output_text`, `exact_match`, `latency_seconds`, etc.) |
| `<model>_aggregates.json` | JSON | 15 Core aggregate metrics | Flat JSON object with 15 core metrics + `model` identifier |
| `<model>_derived_features.json` | JSON | 27 Derived features + catalog | Contains `model`, `n_samples`, `derived_features` dict, and full `catalog` metadata |
| `<model>_all_metrics.json` | JSON | Consolidated 42-metric record | Flat JSON object combining all 15 core and 27 derived metrics |
| `<model>_per_sample_features.csv` | CSV | Per-sample intermediate analysis | 21 columns (`sample_id`, `has_fn_call_leak`, `is_answer_recoverable`, `ungrounded_number_ratio`, etc.) |
