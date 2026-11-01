# Comprehensive System Guide, Architecture & Benchmark Results

This document provides a complete technical analysis of the **LLM 15-Metric Benchmarking Pipeline**, covering the architecture, dataset schemas, prompt design, mathematical and algorithmic metric definitions, backend/frontend dashboard implementation, and empirical benchmark results across evaluated models.

---

## 1. Executive Summary & Purpose

The pipeline is an **end-to-end, dataset-agnostic evaluation framework** designed for Large Language Models (LLMs). It evaluates model performance on domain-specific compliance, reasoning, and numerical reasoning tasks using **local inference via LM Studio** (OpenAI-compatible API) and evaluation methodologies compatible with **NVIDIA NeMo Evaluator**.

### Key Architectural Strengths
- **Dataset-Agnostic Engine**: Evaluates any dataset conforming to a clean, unified JSONL schema.
- **15 Multi-Dimensional Metrics**: Spans exact token match, semantic overlap (BLEU/ROUGE/BERTScore), LLM-as-a-judge (correctness, completeness, helpfulness), reasoning faithfulness, factual hallucination detection, operational efficiency (latency, throughput, tokens/sec), and category-level accuracy breakdowns.
- **Strict Separation of Concerns**: Generation and evaluation are decoupled. An offline re-scoring mode enables re-evaluating existing prediction runs without re-invoking models.
- **Full-Stack Visualization**: A Flask/Python REST API serves structured results and charts to a modern React + Vite + TypeScript dashboard.

---

## 2. End-to-End System Architecture

```
                       ┌─────────────────────────┐
                       │   Local LM Studio       │
                       │   (OpenAI /v1/chat API) │
                       └───────────▲─────────────┘
                                   │
                             HTTP Requests
                                   │
┌────────────────────────┐         │         ┌────────────────────────┐
│  Dataset (JSONL)       │─────────┴────────▶│  Pipeline Engine       │
│  e.g. kyc_eval_dataset │                   │  (llm_benchmark.py)    │
└────────────────────────┘                   └───────────┬────────────┘
                                                         │
                               Produces Logs, Scores, Plots
                                                         │
                                                         ▼
                                             ┌────────────────────────┐
                                             │ results/ Directory     │
                                             │ • master_table.csv     │
                                             │ • category_accuracy.csv│
                                             │ • predictions.jsonl    │
                                             │ • plots/*.png          │
                                             └───────────┬────────────┘
                                                         │
                                                    REST API
                                                         │
                                                         ▼
                                             ┌────────────────────────┐
                                             │ Flask Server (server.py)│
                                             └───────────┬────────────┘
                                                         │
                                                   JSON Endpoints
                                                         │
                                                         ▼
                                             ┌────────────────────────┐
                                             │ React Web Dashboard    │
                                             │ (frontend/src/)        │
                                             └────────────────────────┘
```

### Execution Lifecycle:
1. **Data Ingestion**: Standardized JSONL rows are loaded and validated.
2. **Dynamic Prompt Assembly**: Task-specific prompt templates are constructed with domain compliance constraints and few-shot formatting rules.
3. **Inference Execution**: Asynchronous or batch HTTP calls are dispatched to LM Studio (`http://localhost:1234/v1`).
4. **Answer Extraction**: Regex and NLP parsers extract `Final Answer: ...` candidate numbers and categorical labels.
5. **Metric Computation**: All 15 metrics are scored deterministically or through LLM judge evaluation.
6. **Persistence**: Detailed per-row outputs are written to `results/<model>_predictions.jsonl`, summary numbers to `results/master_table.csv`, and radar/bar charts to `results/plots/`.
7. **Interactive Dashboard**: `server.py` serves the REST API consumed by the React UI.

---

## 3. Dataset Schema & Data Breakdown

### Standard JSONL Schema
Every record in the evaluation pipeline conforms to the following schema:
```json
{
  "sample_id": "item_001",
  "task_type": "qa_label | qa_numeric_program | text_qa",
  "input_text": "What is the status or answer to evaluate?",
  "context": "Reference background passage, rule, or table",
  "target": "Ground-truth answer",
  "normalized_answer": "Cleaned or parsed numeric/label value",
  "raw_values": [28000, 11500, 8500],
  "steps": ["28000 + 11500 + 8500"],
  "constants": [50000],
  "company": "Domain entity or compliance category tag"
}
```

### Active Dataset: `newdata/kyc_eval_dataset.jsonl`
- **Total Records**: 100 test items.
- **Task Types**:
  - `qa_label` (50 records): Single or categorical verification questions based on Indian banking compliance (RBI Master Direction on KYC & PMLA Rules).
  - `qa_numeric_program` (50 records): Multi-step arithmetic problems calculating transaction aggregation, KYC threshold breaches, and periodic updation timelines.
- **Category / Regulatory Tag Breakdown (`company`)**:
  - `Banking_KYC_Periodic_Updation`: 27 records
  - `Banking_KYC_Applicability`: 18 records
  - `Banking_KYC_Documents`: 18 records
  - `Banking_KYC_Instruments`: 14 records
  - `Banking_KYC_Onboarding`: 5 records
  - `Banking_KYC_VCIP`: 5 records
  - `Banking_KYC_Inoperative_Accounts`: 5 records
  - `Banking_KYC_CKYC`: 4 records
  - `Banking_KYC_Small_Account`: 3 records
  - `Banking_KYC_Fraud_Awareness`: 1 record

### Preprocessing Legacy Datasets (`scripts/normalize_datasets.py`)
The pipeline also supports transforming 5 raw benchmark datasets into the unified schema:
1. **Banking77**: 77-class customer intent classification.
2. **Financial PhraseBank**: 3-class sentiment analysis (Positive / Neutral / Negative) with agreement tier deduplication.
3. **FinanceBench**: SEC filing (10-K/10-Q) open-book question answering.
4. **FinQA**: Deep numerical reasoning over corporate disclosures and financial tables.
5. **ConvFinQA**: Multi-turn dialogue financial question answering.

---

## 4. Prompt Engineering & Answer Parsing

### 1. `qa_numeric_program` Prompt
Instructs the model to act as a financial auditor performing step-by-step arithmetic:
```
You are an expert financial and banking compliance auditor performing numerical calculations.
Reference Context / Rules: <context>
Question: <question>
Instructions:
1. Examine the provided context and perform the required calculation step-by-step.
2. Show the arithmetic in one line, then give the result.
3. State your final numeric answer clearly at the end (e.g., Final Answer: <number>).
```

### 2. `qa_label` Compliance Prompt
Constrains the model to RBI / Indian banking law jurisdiction:
```
You are an expert compliance analyst specializing in Indian banking regulation.
Jurisdiction: India only. Base your answer strictly on the RBI Master Direction on KYC and the PMLA.
Never cite foreign laws (BSA, AMLD4, OFAC, FATF, etc.).
Context: <context>
Question: <question>
Rules:
1. Apply any threshold or condition in the question or context exactly.
2. For calculations, show arithmetic in one line.
3. Never invent section names, numbers, or thresholds.
4. Keep reasoning to 2-4 sentences.
5. Match the answer to question type (Yes/No or direct fact).
6. End response with: Final Answer: <your answer>
```

### Answer Extraction (`parsers.py` & `llm_benchmark.py`)
- Searches for explicit anchors (`Final Answer:`, `Answer:`, `Therefore,`).
- For numeric questions: Extracts floating-point numbers, strips currency symbols (`₹`, `$`, `€`), removes thousand separators (`,`), and standardizes percentages.
- For labels: Normalizes whitespace, lowercases, and maps to canonical forms (`yes`, `no`, category strings).

---

## 5. Mathematical & Algorithmic Metric Definitions (All 15 Metrics)

| # | Metric | Category | Implementation & Formula |
|---|---|---|---|
| 1 | **Exact Match (EM)** | Quality | Binary score (0.0 or 1.0). For numeric: $|y_{pred} - y_{gold}| \le \max(0.02, 0.01 \cdot |y_{gold}|)$. For labels: canonical string match. |
| 2 | **F1 Score** | Quality | Token-level harmonic mean: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. For numeric, equal to Exact Match. |
| 3 | **BLEU** | Quality | Evaluates extracted final-answer n-gram overlap ($BLEU\text{-}1$ precision) against reference target string. |
| 4 | **ROUGE-L** | Quality | Longest common subsequence (LCS) overlap between model extraction and gold reference. |
| 5 | **Judge Score** | LLM Judge | Correctness scored by LLM judge on 1.0 to 5.0 scale (mapped from binary correctness 0 $\rightarrow$ 1.0, 1 $\rightarrow$ 5.0). |
| 6 | **Completeness** | LLM Judge | Scored 1.0 to 5.0 evaluating reasoning validity and intermediate calculation coverage. |
| 7 | **Helpfulness** | LLM Judge | Scored 1.0 to 5.0 measuring clarity, adherence to format rules, and lack of fabricated claims. |
| 8 | **Reasoning Faithfulness** | Reliability | Fraction of ground-truth step values that appear in the model's intermediate explanation: $\frac{\sum \mathbb{I}(step\_val \in output)}{\text{Total Steps}}$. |
| 9 | **BERTScore** | Semantic | Cosine similarity of contextual embeddings ($F_1$). Fallback to LM Studio embedding endpoint or TF-IDF semantic cosine similarity. |
| 10 | **Hallucination Rate** | Factuality | Percentage of generated outputs containing numbers not present in the expanded whitelist (raw values, context, target, intermediate step outputs, scale factors 100, 1..12). |
| 11 | **Failure Rate** | Reliability | Percentage of runs failing due to empty response, network errors, timeouts, or refusal phrases (`"I cannot"`, `"as an AI"`). |
| 12 | **Latency (s)** | Efficiency | Wall-clock time in seconds per request: $t_{\text{end}} - t_{\text{start}}$. |
| 13 | **Throughput (req/s)** | Efficiency | Number of completed evaluations divided by total execution duration: $\frac{N}{\sum t_{\text{total}}}$. |
| 14 | **Tokens/sec** | Efficiency | Generated tokens divided by generation latency: $\frac{\text{tokens\_out}}{t_{\text{gen}}}$. |
| 15 | **Category Accuracy** | Granular | Subgroup accuracy computed by slicing predictions across `task_type` and `company` category tags. |

---

## 6. Comprehensive Benchmark Results

The pipeline evaluated 5 Large Language Models over the complete 100-sample KYC benchmark dataset.

### 6.1 Master Comparison Table (100-Sample Benchmark)

| Model Name | Exact Match | F1 Score | BLEU | ROUGE | Judge Score | Completeness | Helpfulness | Faithfulness | BERTScore | Hallucination Rate | Failure Rate | Latency (s) | Throughput (req/s) | Tokens / sec | Overall Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-8B-GGUF** | **0.7100** | **0.7413** | **0.4902** | **0.7389** | 4.44 | 2.28 | 4.88 | 0.9930 | 0.3168 | **0.1400** | **0.00%** | **4.5864s** | **0.2180** | 15.55 | **71.0%** |
| **Ministral-3-8B-Reasoning-2512-GGUF** | 0.6800 | 0.7210 | 0.4130 | 0.7197 | 4.68 | **4.32** | 4.92 | **1.0000** | **0.7187** | 0.4800 | 1.00% | 19.3649s | 0.0516 | 18.66 | 68.0% |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | 0.6300 | 0.6523 | 0.3301 | 0.6512 | **4.96** | 1.48 | **5.00** | 0.9350 | 0.6654 | 0.3300 | **0.00%** | 17.1175s | 0.0584 | **24.42** | 63.0% |
| **ministral-3-8b-instruct-2512** | 0.6300 | 0.6300 | 0.1144 | 0.6300 | 4.48 | 4.30 | 4.76 | **1.0000** | 0.4450 | 0.4900 | **0.00%** | 24.2293s | 0.0413 | 4.47 | 63.0% |
| **meta-llama-3.1-8b-instruct** | 0.5100 | 0.5100 | 0.2482 | 0.5100 | 4.76 | 4.71 | 4.84 | **1.0000** | 0.2373 | 0.2900 | **0.00%** | 9.3868s | 0.1065 | 13.93 | 51.0% |

---

## 7. Deep-Dive Category & Task Breakdown

### 7.1 Accuracy by Task Type
| Model | `qa_numeric_program` (50 items) | `qa_label` (50 items) |
|---|:---:|:---:|
| **Qwen3-8B-GGUF** | **100.0% (50/50)** | **42.0% (21/50)** |
| **Ministral-3-8B-Reasoning-2512-GGUF** | **100.0% (50/50)** | 36.0% (18/50) |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | 84.0% (42/50) | **42.0% (21/50)** |
| **ministral-3-8b-instruct-2512** | 98.0% (49/50) | 28.0% (14/50) |
| **meta-llama-3.1-8b-instruct** | 96.0% (48/50) | 6.0% (3/50) |

> **Critical Finding:**  
> All top models excel at numerical calculations (84% to 100%), but compliance label verification (`qa_label`) is dramatically harder. `meta-llama-3.1-8b-instruct` scored only 6% on `qa_label` due to formatting divergence (generating markdown/tool-call tokens instead of plain text answers), whereas `Qwen3-8B` and `DeepSeek-R1-Distill` reached 42%.

### 7.2 Accuracy by Regulatory Subcategory (`company`)
| Category | Samples | Qwen3-8B | Ministral-Reasoning | DeepSeek-R1-Distill | Llama-3.1-8B |
|---|:---:|:---:|:---:|:---:|:---:|
| **Banking_KYC_Inoperative_Accounts** | 5 | **100.0%** | 80.0% | 80.0% | 80.0% |
| **Banking_KYC_Instruments** | 14 | **92.86%** | 85.71% | 71.43% | 78.57% |
| **Banking_KYC_Applicability** | 18 | **94.44%** | 88.89% | 55.56% | 61.11% |
| **Banking_KYC_Periodic_Updation** | 27 | **77.78%** | 74.07% | 70.37% | 66.67% |
| **Banking_KYC_Onboarding** | 5 | 60.0% | 40.0% | **80.0%** | 20.0% |
| **Banking_KYC_VCIP** | 5 | 40.0% | 40.0% | **60.0%** | 0.0% |
| **Banking_KYC_Documents** | 18 | 50.0% | 55.56% | **61.11%** | 33.33% |
| **Banking_KYC_CKYC** | 4 | 25.0% | **50.0%** | **50.0%** | 0.0% |
| **Banking_KYC_Small_Account** | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| **Banking_KYC_Fraud_Awareness** | 1 | 0.0% | 0.0% | 0.0% | 0.0% |

---

## 8. Efficiency, Latency & Generation Speed Analysis

```
Speed vs Accuracy Trade-off:
┌──────────────────────────────────────────────────────────┐
│ High                                                     │
│ Acc  ▲        [Ministral-Reasoning]      [Qwen3-8B]      │
│      │        (68%, 19.36s)              (71%, 4.58s) ★  │
│      │                                                   │
│      │   [DeepSeek-R1-Distill]                           │
│      │   (63%, 17.12s)                                   │
│      │                                                   │
│      │   [ministral-instruct]        [Llama-3.1-8B]      │
│      │   (63%, 24.23s)               (51%, 9.39s)        │
│ Low  └───────────────────────────────────────────────────▶
│       Slow Latency (s)                     Fast Latency  │
└──────────────────────────────────────────────────────────┘
```

1. **Overall Champion — `Qwen3-8B-GGUF`**:
   - **Highest accuracy**: 71.0% overall (100% math, 42% compliance labels).
   - **Lowest latency**: **4.59s** per request (over 4x faster than reasoning models).
   - **Lowest hallucination rate**: Only **14.0%**.
   - **Zero API failures**: 100% completion rate.

2. **Highest Generation Throughput — `OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF`**:
   - Reached **24.42 tokens/sec** raw generation speed.
   - Long thinking traces increase total wall-clock time per item (17.12s), but arithmetic and document parsing remain solid.

3. **Highest Reasoning Completeness — `Ministral-3-8B-Reasoning-2512-GGUF`**:
   - Achieved the highest reasoning completeness score (**4.32 / 5.0**) and **100% math accuracy**.
   - Traded execution speed (19.36s) for detailed proof steps.

---

## 9. Web Application & Dashboard Architecture

The project features a full dashboard connecting the backend to interactive UI components:

### Backend: `server.py`
Implemented in Flask with CORS enabled:
- **`GET /api/models`**: Scans `results/` for all available model aggregate and prediction files.
- **`GET /api/master-table`**: Returns the consolidated 15-metric comparison matrix as JSON.
- **`GET /api/category-accuracy`**: Slices accuracy by `task_type` and `company`.
- **`GET /api/predictions?model=<name>&limit=500`**: Streams individual prediction records with input questions, ground truth, generated responses, and metric scores.
- **`GET /api/sample/<sample_id>`**: Cross-model comparative view: fetches how each model answered a specific test question side-by-side.
- **`GET /api/plots/<plot_name>`**: Serves generated visualizations (`radar_chart.png`, `metrics_bar_chart.png`, `quality_vs_latency.png`, `heatmap_*.png`).

### Frontend: `frontend/`
- **Tech Stack**: React 18, Vite, TypeScript, Lucide Icons, Chart.js / Recharts.
- **Views**:
  - **Overview / Leaderboard**: Ranked model cards with win badges and headline metrics.
  - **15-Metric Master Table**: Interactive tabular matrix with metric tooltips and column sort.
  - **Model Deep-Dive & Comparison**: Side-by-side comparative inspection of reasoning traces for any question ID.
  - **Charts Gallery**: Displays radar, scatter (quality vs latency), and category heatmaps.

---

## 10. How to Run the System

### 1. Start LM Studio Server
1. Open **LM Studio**.
2. Load any model (e.g. `Qwen3-8B-GGUF` or `Meta-Llama-3.1-8B-Instruct`).
3. Start the local server on `http://localhost:1234`.

### 2. Run the Benchmark
- **Quick Smoke Test (5 samples, single model)**:
  ```powershell
  .\venv\Scripts\python.exe llm_benchmark.py --sample-size 5
  ```
- **Full Benchmark (100 samples, all rows)**:
  ```powershell
  .\venv\Scripts\python.exe llm_benchmark.py --sample-size 0
  ```
- **Offline Rescoring (Recalculate metrics without model calls)**:
  ```powershell
  .\venv\Scripts\python.exe llm_benchmark.py --skip-generation --predictions results/predictions.jsonl
  ```

### 3. Launch Dashboard
- **Backend**:
  ```powershell
  .\venv\Scripts\python.exe server.py
  ```
  *(Server listens on `http://127.0.0.1:5000`)*
- **Frontend**:
  ```powershell
  cd frontend
  npm run dev
  ```
  *(Opens on `http://localhost:5173`)*
