# Pipeline Execution & CLI Usage Guide

This guide covers all operational workflows for running benchmarks, re-scoring offline, executing the derived metrics layer, configuring kill switches, and running the automated test suite.

---

## 1. Prerequisites & LM Studio Setup

### Local Inference Server Setup
1. Download and open **LM Studio**.
2. Navigate to the **Local Server** tab (icon `<->` on the left sidebar).
3. Search and load your desired model (e.g., `qwen3-8b`, `mistral-7b`, `meta-llama-3.1-8b`, or your fine-tuned model).
4. Verify server settings:
   - Port: `1234`
   - Server URL: `http://127.0.0.1:1234/v1`
   - API Key: `lm-studio` (or any string)
5. Click **Start Server**.

### Python Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Main Benchmark Pipeline (`llm_benchmark.py`)

The main benchmark CLI evaluates models against dataset prompts and computes the 15 core metrics.

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | Path | `newdata/kyc_eval_dataset.jsonl` | Path to benchmark JSONL file. |
| `--models` | List | `["local-model"]` | One or more model identifiers loaded in LM Studio. |
| `--sample-size` | Int | `5` | Number of rows to run (`0` or `-1` runs the full dataset). |
| `--base-url` | String | `http://127.0.0.1:1234/v1` | LM Studio REST API endpoint. |
| `--skip-generation` | Flag | `False` | Skips generation and reads existing predictions to re-score offline. |
| `--predictions` | Path | `None` | Path to existing predictions file when using `--skip-generation`. |
| `--skip-judge` | Flag | `False` | Skips Stage 3 (LLM-as-a-Judge) to accelerate execution. |
| `--judge-model` | String | `local-model` | Model identifier used as the evaluator judge. |
| `--max-tokens` | Int | `None` (auto) | Maximum output tokens per generation request. |
| `--output-dir` | Path | `results` | Directory where output files and logs are saved. |
| `--consistency-sample` | Int | `0` | Number of sample rows to re-run 3x per model for variance analysis. |
| `--no-derived-metrics`| Flag | `False` | Kill switch flag: disables derived metrics computation. |
| `--merge-into-aggregates` | Flag | `False` | Merges derived metrics under `"derived"` in `<model>_aggregates.json`. |

---

### Step-by-Step Usage Examples

#### Quick 5-Sample Sanity Check
```powershell
python llm_benchmark.py --sample-size 5
```
* Runs the first 5 records against the active model.
* Verifies connection, parses responses, runs judge pass, and generates derived metrics.

#### Full Benchmark Run Across All Records
```powershell
python llm_benchmark.py --sample-size 0
```

#### Fast Run Without Judge Pass (Generation + NLP Similarity Only)
```powershell
python llm_benchmark.py --sample-size 0 --skip-judge
```

#### Multi-Model Benchmark In a Single Command
```powershell
python llm_benchmark.py --sample-size 0 --models qwen3-8b meta-llama-3.1-8b ministral-3-8b
```

#### Offline Re-Scoring (No Model Inferences)
Re-evaluates an existing prediction run (e.g., after modifying parsing logic) without re-calling models:
```powershell
python llm_benchmark.py --skip-generation --predictions results/predictions.jsonl
```

---

## 3. The Derived Metrics Layer (`derived_metrics.py`)

The derived metrics layer computes 27 model-level features across Format, Task, Judge, Reliability, Efficiency, and Robustness.

### The Three Execution Modes

#### Mode A: Post-Hoc on Already-Run Models
Run on an existing predictions and aggregates pair:
```powershell
python derived_metrics.py --predictions results/meta-llama-3.1-8b-instruct_predictions.jsonl --aggregates results/meta-llama-3.1-8b-instruct_aggregates.json
```
Or process every model found in a results directory:
```powershell
python derived_metrics.py --dir results/
```

#### Mode B: Fresh Run (Automatic Pipeline Hook)
During a fresh benchmark run, `llm_benchmark.py` automatically calls `derived_metrics.py` right after saving predictions and aggregates.
- Runs silently within `try/except`.
- Generates `<model>_derived_features.json`, `<model>_all_metrics.json`, and `<model>_per_sample_features.csv` alongside the core files.

#### Mode C: Deterministic Recompute
Re-generates derived outputs from saved predictions deterministically:
```powershell
python derived_metrics.py --recompute --dir results/
```
* **Idempotency Guarantee**: Running this command multiple times produces byte-identical results.
* **Deterministic Bootstrap**: Fixed seed `0` with 2,000 resamples ensures reproducible confidence intervals.

---

## 4. Kill Switch Configuration

The derived metrics layer includes a fail-safe kill switch to disable execution in production or legacy environments:

### Via Environment Variable
```powershell
# Disable derived metrics layer
$env:DERIVED_METRICS = "0"

# Re-enable derived metrics layer
$env:DERIVED_METRICS = "1"
```
*(Accepts `0`, `false`, `no`, `off`, `disable`, `disabled`)*

### Via Command-Line Flag
Pass `--no-derived-metrics` to `llm_benchmark.py` or `derived_metrics.py`:
```powershell
python llm_benchmark.py --no-derived-metrics
python derived_metrics.py --no-derived-metrics --dir results/
```

---

## 5. Merging Into Aggregates (Optional)

By default, existing aggregate files (`<model>_aggregates.json`) remain strictly untouched and byte-compatible. If you wish to merge the derived metrics under a single top-level key `"derived"` inside the aggregates JSON:

```powershell
python derived_metrics.py --recompute --dir results/ --merge-into-aggregates
```

---

## 6. Automated Testing

The project includes an automated test suite verifying synthetic edge cases, golden numbers on Llama-3.1-8B, fail-safe isolation, idempotency, and regression safety:

```powershell
python -m pytest tests/test_derived_metrics.py -v
```

### Verified Test Cases
1. **Synthetic Unit Tests**: Empty rows, missing fields, single-row inputs, zero-latency division guards.
2. **Golden Accuracy Test**: Validates that all 27 features on `meta-llama-3.1-8b-instruct` match ground-truth values to 4 decimal places.
3. **Idempotency Test**: Confirms consecutive recomputes generate identical output files.
4. **Exception Injection Test**: Proves an unexpected exception inside derived calculations is logged without crashing the benchmark run.
5. **Kill-Switch Test**: Confirms `DERIVED_METRICS=0` completely bypasses processing.
6. **Regression Hash Test**: Asserts original prediction and aggregate files are unmodified.
