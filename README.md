# LLM 15-Metric Benchmarking Pipeline

End-to-end, dataset-agnostic evaluation pipeline for Large Language Models using **NVIDIA NeMo Evaluator** and **LM Studio**. This framework evaluates any model on any custom evaluation dataset following a unified JSONL schema across 15 comprehensive metrics.

---

## 1. Unified Dataset Schema

Any dataset formatted in JSONL can be benchmarked by this pipeline. Each line represents a test case with the following standard schema:

```json
{
  "id": "sample_001",
  "task_type": "qa_label",
  "input_text": "What is the status or answer to evaluate?",
  "context": "Optional background passage, table, or reference text.",
  "target": "Expected ground-truth answer",
  "normalized_answer": "Cleaned or parsed numeric/label value",
  "metadata": {
    "company": "Domain entity, organization, or category tag",
    "raw_values": ["120", "45.5"],
    "steps": [{"op": "divide", "res": 0.45}]
  }
}
```

* Supported `task_type` values include:
  * `qa_numeric_program`: Numeric calculations & multi-step arithmetic.
  * `qa_label`: Single or multi-class classification and categorical answers.
  * `text_qa`: General free-form domain Q&A.

---

## 2. The 15 Metrics & Their Sources

| # | Metric | Category | Source / Implementation |
|---|---|---|---|
| 1 | **Exact Match (EM)** | Quality | NeMo Evaluator `exact_match` scorer with numeric normalization (strip `$`, `%`, commas, round to 2 decimals) and label cleaning |
| 2 | **F1 Score** | Quality | NeMo Evaluator `f1_token` scorer |
| 3 | **BLEU** | Quality | NeMo Evaluator `bleu` scorer (`bleu_1` / n-gram precision) |
| 4 | **ROUGE** | Quality | NeMo Evaluator `rouge` scorer (`rouge_l` longest common subsequence) |
| 5 | **Judge Score** | LLM Judge | NeMo Evaluator LLM-as-a-Judge (`JUDGE_TEMPLATE`, scale 0.0 to 1.0) |
| 6 | **Completeness** | LLM Judge | NeMo Evaluator LLM-as-a-Judge (step-by-step arithmetic coverage) |
| 7 | **Helpfulness** | LLM Judge | NeMo Evaluator LLM-as-a-Judge (clarity and structure) |
| 8 | **Safety** | LLM Judge | NeMo Evaluator LLM-as-a-Judge (avoids fabricated numbers / misleading advice) |
| 9 | **BERTScore** | Semantic | `bert_score` library (with TF-IDF cosine similarity fallback if torch/bert-score is absent) |
| 10 | **Hallucination Rate** | Factuality | Numbers in output not traceable to `metadata.raw_values`, context table, target, or steps |
| 11 | **Failure Rate** | Reliability | Empty output, API connection error, timeout, or refusal phrase |
| 12 | **Latency (s)** | Efficiency | Mean wall-clock time per request in seconds |
| 13 | **Throughput (req/s)** | Efficiency | Completed requests per total pipeline wall-clock time |
| 14 | **Tokens/sec** | Efficiency | Total generated tokens divided by total generation duration |
| 15 | **Category Accuracy** | Breakdown | Accuracy grouped by `task_type` and entity / category (`metadata.company`) |

---

## 3. Prerequisites & LM Studio Setup

1. **Launch LM Studio**.
2. Go to the **Local Server** tab (the `<->` icon on the left).
3. Select and load your model (e.g., `local-model`, `mistralai/mistral-7b-instruct`, `qwen2.5-7b-instruct`, etc.).
4. Click **Start Server** on port `1234` (`http://127.0.0.1:1234`).

---

## 4. How to Run

### Step 1: Initial Sanity Check (1 Model, Few Runs)
To ensure everything is working correctly with minimal execution time:

```powershell
.\venv\Scripts\python.exe llm_benchmark.py --sample-size 5
```
* Runs on the first 5 records only.
* Benchmarks the currently loaded LM Studio model (`local-model`).
* Evaluates all 15 metrics and writes results to `results/` and charts to `plots/`.

> **Tip (Fastest Test):** If you want to test response generation and similarity metrics without waiting for a second judge pass:
> ```powershell
> .\venv\Scripts\python.exe llm_benchmark.py --sample-size 5 --skip-judge
> ```

---

### Step 2: Full Benchmark Run
Once verified on a sample, run across all records in your dataset:

```powershell
.\venv\Scripts\python.exe llm_benchmark.py --sample-size 0
```
*(Setting `--sample-size 0` runs all rows)*

To specify a custom dataset path:
```powershell
.\venv\Scripts\python.exe llm_benchmark.py --dataset path/to/your_dataset.jsonl --sample-size 0
```

---

### Step 3: Multi-Model Benchmarking
To benchmark multiple models in a single pass:

```powershell
.\venv\Scripts\python.exe llm_benchmark.py --sample-size 0 --models qwen2.5-7b-instruct mistral-7b-instruct llama-3.1-8b-instruct deepseek-coder local-model
```

---

### Step 4: Consistency & Statistical Significance Check
To measure score variance across 3 repeated runs on a subset:

```powershell
.\venv\Scripts\python.exe llm_benchmark.py --sample-size 20 --consistency-sample 20
```

---

### Step 5: Offline / Re-Scoring Mode (No Re-Generation)
If you already generated predictions in `results/predictions.jsonl` and want to re-run scoring or plotting without calling the model again:

```powershell
.\venv\Scripts\python.exe llm_benchmark.py --skip-generation --predictions results/predictions.jsonl
```

---

## 5. Generated Artifacts

### `results/`
- **`master_table.csv`**: Consolidated table with 15 metric columns per model.
- **`category_accuracy.csv`**: Breakdown of model accuracy across task types and categories.
- **`predictions.jsonl`**: Complete per-row log containing model prompts, raw responses, latencies, tokens, and per-metric scores.

### `plots/`
- **`metrics_bar_chart.png`**: Grouped bar chart comparing key quality metrics across models.
- **`radar_chart.png`**: Multi-dimensional radar/spider plot of model capabilities.
- **`quality_vs_latency.png`**: Quality (Exact Match) vs. Mean Latency scatter plot.
- **`heatmap_task_type.png`**: Accuracy heatmap by task type.
- **`heatmap_company.png`**: Accuracy heatmap across categories / entities.
- **`consistency_spread.png`**: Score spread box plot for repeated runs (when `--consistency-sample` is used).

---

## 6. Derived Metrics Layer (27 Model-Level Features)

The pipeline includes an additive, fail-safe derived metrics layer that computes 27 model-level features across 6 evaluation dimensions on top of existing predictions and aggregates without requiring extra model or judge inferences.

### Three Usage Modes

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
During normal evaluation runs, `llm_benchmark.py` automatically invokes the derived metrics layer after writing predictions and aggregates:
```powershell
python llm_benchmark.py --sample-size 0
```
To disable derived metrics during a fresh benchmark run, pass `--no-derived-metrics` or set `DERIVED_METRICS=0`.

#### Mode C: Recompute / Re-run
Recompute derived metrics from saved predictions deterministically without re-running model generation or the judge:
```powershell
python derived_metrics.py --recompute --dir results/
```
Running recompute repeatedly is idempotent and yields identical files.

### Output Files (Per Model)

For each model, the layer produces three non-destructive files in the output directory:
- `<model>_derived_features.json`: The 27 derived metric values and the feature catalog.
- `<model>_all_metrics.json`: The 15 original aggregate metrics and 27 derived metrics flat in one JSON, plus `"model"` and `"n_samples"`.
- `<model>_per_sample_features.csv`: Per-sample intermediate analysis columns (`has_fn_call_leak`, `is_answer_recoverable`, `ungrounded_number_ratio`, `token_ratio`, etc.).
- *(Optional)* `--merge-into-aggregates`: If passed, merges derived metrics under a `"derived"` key in `<model>_aggregates.json`, leaving existing keys untouched.

### Kill Switch

Set the environment variable `DERIVED_METRICS=0` (or `false`, `no`, `off`) to disable the derived metrics layer completely across all scripts:
```powershell
$env:DERIVED_METRICS = "0"
```
Or pass the CLI flag `--no-derived-metrics` when invoking `llm_benchmark.py` or `derived_metrics.py`.

### 27 Derived Features Catalog

| # | Feature Name | Group | Meaning |
|---|---|---|---|
| 1 | `fn_call_leak_rate` | FORMAT | Share of model outputs containing raw function-call JSON syntax or headers |
| 2 | `extraction_success_rate` | FORMAT | Share of predictions with non-empty extracted answer after whitespace stripping |
| 3 | `final_answer_line_rate` | FORMAT | Share of outputs containing an explicit 'Final Answer' line (case-insensitive) |
| 4 | `answer_recoverable_rate` | FORMAT | Share of outputs where reference answer is literally stated in the pre-JSON text |
| 5 | `label_exact_match` | TASK | Mean exact match accuracy for label / classification questions (`qa_label`) |
| 6 | `numeric_exact_match` | TASK | Mean exact match accuracy for numerical / arithmetic questions (`qa_numeric_program`) |
| 7 | `judge_em_gap` | JUDGE | Normalized gap between LLM judge score and exact match: `mean(judge_score)/5 - mean(EM)` |
| 8 | `judge_composite` | JUDGE | Combined mean of judge score, completeness, and helpfulness |
| 9 | `judge_leniency_rate` | JUDGE | Share of predictions awarded high judge score (`>= 4`) despite zero exact match |
| 10 | `judge_fail_rate` | JUDGE | Share of predictions receiving a failing judge score (`<= 2`) |
| 11 | `reliable_accuracy` | RELIABILITY | Share of predictions achieving exact match (`1.0`) without being flagged hallucinated |
| 12 | `hallucinated_but_correct_rate` | RELIABILITY | Share of predictions achieving exact match (`1.0`) while flagged hallucinated |
| 13 | `ungrounded_number_ratio` | RELIABILITY | Average ratio of distinct reasoning numbers not present in prompt, context, or target |
| 14 | `ungrounded_flag_delta` | RELIABILITY | Mean ungrounded ratio of hallucinated samples minus that of non-hallucinated samples |
| 15 | `hallucination_separation` | RELIABILITY | Separation power of hallucination flag: `P(Hallucinated \| EM == 0) - P(Hallucinated \| EM == 1)` |
| 16 | `label_hedge_rate` | RELIABILITY | Share of label questions matching uncertainty/hedging phrases in output text |
| 17 | `latency_per_correct` | EFFICIENCY | Average generation latency in seconds required per unit of exact match accuracy |
| 18 | `tokens_per_correct` | EFFICIENCY | Average generated tokens consumed per unit of exact match accuracy |
| 19 | `latency_p95` | EFFICIENCY | 95th percentile generation latency across samples (seconds) |
| 20 | `latency_cv` | EFFICIENCY | Coefficient of variation of generation latency: `sample std / mean latency` |
| 21 | `quality_per_second` | EFFICIENCY | Mean judge score delivered per second of generation latency |
| 22 | `output_input_token_ratio` | EFFICIENCY | Average ratio of generated output tokens to prompt input tokens |
| 23 | `verbosity_accuracy_corr` | EFFICIENCY | Pearson correlation coefficient between output token count and exact match |
| 24 | `macro_exact_match` | ROBUSTNESS | Macro-averaged exact match across all category subgroups |
| 25 | `worst_category_em` | ROBUSTNESS | Minimum exact match accuracy among categories with at least 5 samples |
| 26 | `category_em_std` | ROBUSTNESS | Sample standard deviation of exact match across categories with at least 5 samples |
| 27 | `em_ci95_halfwidth` | ROBUSTNESS | Half-width of 95% bootstrap confidence interval of mean exact match (2,000 resamples, seed 0) |

