# BFSI LLM 15-Metric Benchmarking Pipeline

End-to-end evaluation pipeline for Large Language Models on the Financial Services & Banking Synthetic Dataset (`data/synthetic_bfsi_train.jsonl`) using **NVIDIA NeMo Evaluator** and **LM Studio**.

---

## 1. The 15 Metrics & Their Sources

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
| 15 | **Category Accuracy** | Breakdown | Accuracy grouped by `task_type` (`qa_numeric_program`, `qa_label`) and `company` |

---

## 2. Prerequisites & LM Studio Setup

1. **Launch LM Studio**.
2. Go to the **Local Server** tab (the `<->` icon on the left).
3. Select and load your model (e.g., `local-model`, `mistralai/mistral-7b-instruct`, `qwen2.5-7b-instruct`, etc.).
4. Click **Start Server** on port `1234` (`http://127.0.0.1:1234`).

---

## 3. How to Run

### Step 1: Initial Sanity Check (1 Model, Few Runs)
To ensure everything is working correctly with minimal execution time:

```powershell
.\venv\Scripts\python.exe bfsi_llm_benchmark.py --sample-size 5
```
* Runs on the first 5 records only.
* Benchmarks the currently loaded LM Studio model (`local-model`).
* Evaluates all 15 metrics and writes results to `results/` and charts to `plots/`.

> **Tip (Fastest Test):** If you want to test response generation and similarity metrics without waiting for a second judge pass:
> ```powershell
> .\venv\Scripts\python.exe bfsi_llm_benchmark.py --sample-size 5 --skip-judge
> ```

---

### Step 2: Full Benchmark Run (All 240 Rows)
Once you have verified the pipeline works on the sample:

```powershell
.\venv\Scripts\python.exe bfsi_llm_benchmark.py --sample-size 0
```
*(Setting `--sample-size 0` runs all 240 dataset rows)*

---

### Step 3: Multi-Model Benchmarking (e.g. 5 Models)
To benchmark multiple models in a single pass (switch or specify their LM Studio identifiers):

```powershell
.\venv\Scripts\python.exe bfsi_llm_benchmark.py --sample-size 0 --models qwen2.5-7b-instruct mistral-7b-instruct llama-3.1-8b-instruct deepseek-coder local-model
```

---

### Step 4: Consistency & Statistical Significance Check
To measure score variance across 3 repeated runs on a sample subset (e.g. 20 rows):

```powershell
.\venv\Scripts\python.exe bfsi_llm_benchmark.py --sample-size 20 --consistency-sample 20
```

---

### Step 5: Offline / Re-Scoring Mode (No Re-Generation)
If you already generated predictions in `results/predictions.jsonl` and want to re-run scoring or plotting without calling the model again:

```powershell
.\venv\Scripts\python.exe bfsi_llm_benchmark.py --skip-generation --predictions results/predictions.jsonl
```

---

## 4. Generated Artifacts

### `results/`
- **`master_table.csv`**: Consolidated table with 15 metric columns per model.
- **`category_accuracy.csv`**: Breakdown of model accuracy across task types (`qa_numeric_program`, `qa_label`) and the 15 BFSI companies.
- **`predictions.jsonl`**: Complete per-row log containing model prompts, raw responses, latencies, tokens, and per-metric scores.

### `plots/`
- **`metrics_bar_chart.png`**: Grouped bar chart comparing key quality metrics across models.
- **`radar_chart.png`**: Multi-dimensional radar/spider plot of model capabilities.
- **`quality_vs_latency.png`**: Quality (Exact Match) vs. Mean Latency scatter plot.
- **`heatmap_task_type.png`**: Accuracy heatmap by task type.
- **`heatmap_company.png`**: Accuracy heatmap across financial entities.
- **`consistency_spread.png`**: Score spread box plot for repeated runs (when `--consistency-sample` is used).
