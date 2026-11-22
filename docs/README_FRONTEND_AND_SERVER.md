# Full-Stack Dashboard & API Reference (`README_FRONTEND_AND_SERVER.md`)

This document provides the exhaustive architectural and operational reference for the full-stack visualization layer of the NeMo Evaluator benchmark: the Flask backend API (`server.py`) and the React 19 + TypeScript analytics dashboard (`frontend/`).

---

## 1. Architecture Overview

The visualization suite decouples model evaluation from analytics rendering:
1. **Benchmark Engine (`llm_benchmark.py` + `derived_metrics.py`)**: Runs headless inference and statistical evaluation, persisting immutable structured artifacts (`*_predictions.jsonl`, `*_aggregates.json`, `*_derived_features.json`, `*_all_metrics.json`) into `results/`.
2. **Flask REST API (`server.py`)**: A lightweight, stateless Python backend running on port 5000. It reads, parses, caches, and joins benchmark artifacts on demand and exposes structured JSON endpoints.
3. **React Dashboard (`frontend/`)**: A client-side Single Page Application (SPA) built with React 19, TypeScript, and Vite. It renders interactive multi-model bar charts, head-to-head comparison matrices with radar graphs, a sample-level inspector, and a comprehensive 42-column master table.

```
┌────────────────────────────────────────────────────────┐
│           File System Benchmark Artifacts              │
│  results/<model>_predictions.jsonl                     │
│  results/<model>_all_metrics.json (42 metrics flat)    │
│  results/category_accuracy.csv                         │
└───────────────────────────┬────────────────────────────┘
                            │ Read & Parse on Request
                            ▼
┌────────────────────────────────────────────────────────┐
│          Flask REST API Server (server.py:5000)        │
│  • CORS Enabled (*), JSON Payloads                     │
│  • Endpoints: /api/models, /api/master-table,          │
│    /api/derived-catalog, /api/sample/<id>, etc.        │
│  • Serves Production Static Assets from frontend/dist/ │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP JSON / SPA Routing
                            ▼
┌────────────────────────────────────────────────────────┐
│        React 19 + TypeScript SPA (frontend/)           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Navigation & View Switcher (Overview/Comp/Table) │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Tab 1: Overview Dashboard                        │  │
│  │   • Quick Stats Summary Cards                    │  │
│  │   • 42 Metric Recharts Bar Graphs                │  │
│  │   • Filter Pills: All(42), Core(15), Format(4)...│  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Tab 2: Head-to-Head Comparison                   │  │
│  │   • Model A vs Model B Selector                  │  │
│  │   • 42-Metric Delta Table & Victory Share        │  │
│  │   • Multi-Dimensional Radar Comparison Chart     │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Tab 3: Sample-Level Inspector                    │  │
│  │   • Prompt, Context, & Reference Target Viewer   │  │
│  │   • Multi-Model Outputs, Latencies & Tokens      │  │
│  │   • Random Sample Picker & ID Filter             │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Tab 4: 42-Column Master Table                    │  │
│  │   • Sortable, Filterable Unified Matrix          │  │
│  │   • Color-Coded Performance Cells                │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 2. Flask REST Backend (`server.py`)

### 2.1 Server Configuration
- **Host**: `0.0.0.0`
- **Port**: `5000`
- **CORS Policy**: Configured via `@app.after_request` to inject:
  - `Access-Control-Allow-Origin: *`
  - `Access-Control-Allow-Headers: Content-Type, Authorization`
  - `Access-Control-Allow-Methods: GET, PUT, POST, DELETE, OPTIONS`
- **Static Assets Directory**: `frontend/dist/`

### 2.2 Endpoint Reference

#### `GET /api/models`
Discovers all evaluated models available in `results/`.
- **Discovery Strategy**:
  1. Inspects subdirectories of `results/` containing `aggregates.json`.
  2. Inspects flat files matching `results/*_all_metrics.json` and strips the suffix.
  3. Inspects flat files matching `results/*_aggregates.json` and strips the suffix.
- **Response Format**:
  ```json
  {
    "models": [
      "meta-llama-3_1-8b-instruct",
      "ministral-3-8b-instruct-2512",
      "Ministral-3-8B-Reasoning-2512-GGUF",
      "OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF",
      "Qwen3-8B-GGUF"
    ]
  }
  ```

---

#### `GET /api/derived-catalog`
Exposes the metadata catalog of all 27 derived metrics directly from `derived_metrics.FEATURE_CATALOG`.
- **Response Format**:
  ```json
  {
    "fn_call_leak_rate": {
      "group": "format",
      "name": "Function Call Leak Rate",
      "description": "Share of model outputs containing raw JSON/tool call structures",
      "higher_is_better": false
    },
    "reliable_accuracy": {
      "group": "reliability",
      "name": "Reliable Accuracy",
      "description": "Share of samples with exact_match==1 and is_hallucinated==False",
      "higher_is_better": true
    }
  }
  ```

---

#### `GET /api/master-table`
Returns a consolidated array of all evaluated models across all 42 metrics (15 Core + 27 Derived).
- **Processing Logic**:
  1. Looks for all `results/*_all_metrics.json` files.
  2. For each file, extracts the `model` identifier and maps every metric key into dual representations:
     - Formatted title key (e.g., `"Exact Match"`, `"Latency (s)"`, `"FN Call Leak Rate"`).
     - Original raw key (e.g., `"exact_match"`, `"latency"`, `"fn_call_leak_rate"`).
  3. Fallback: If `*_all_metrics.json` is missing, reads `results/master_table.csv` or merges `results/*_aggregates.json`.
- **Response Format**:
  ```json
  [
    {
      "Model": "meta-llama-3_1-8b-instruct",
      "model": "meta-llama-3_1-8b-instruct",
      "Exact Match": 0.51,
      "exact_match": 0.51,
      "F1": 0.54,
      "f1": 0.54,
      "Judge Score": 4.76,
      "judge_score": 4.76,
      "Latency (s)": 9.3868,
      "latency": 9.3868,
      "FN Call Leak Rate": 0.70,
      "fn_call_leak_rate": 0.70,
      "Reliable Accuracy": 0.41,
      "reliable_accuracy": 0.41,
      "Worst Category EM": 0.0,
      "worst_category_em": 0.0
    }
  ]
  ```

---

#### `GET /api/category-accuracy`
Returns the breakdown of accuracy across KYC sub-categories.
- **Source**: `results/category_accuracy.csv`
- **Response Format**:
  ```json
  [
    {
      "category": "Customer Identification",
      "meta-llama-3_1-8b-instruct": 0.60,
      "ministral-3-8b-instruct-2512": 0.55
    }
  ]
  ```

---

#### `GET /api/predictions`
Fetches individual sample predictions for a given model with optional ID filtering.
- **Query Parameters**:
  - `model` (string, required): Model identifier (fuzzy matches `results/<model>_predictions.jsonl`).
  - `id` (string, optional): Substring filter for sample ID.
  - `limit` (integer, optional, default: 500): Maximum rows to return.
- **Response Format**:
  ```json
  {
    "model": "meta-llama-3_1-8b-instruct",
    "count": 100,
    "predictions": [
      {
        "id": "kyc_sample_001",
        "task_type": "qa_label",
        "company": "Customer Identification",
        "input_text": "...",
        "target": "Officially Valid Document",
        "output_text": "...",
        "exact_match": 1,
        "judge_score": 5.0,
        "latency_seconds": 6.84,
        "tokens_out": 142
      }
    ]
  }
  ```

---

#### `GET /api/sample/<sample_id>`
Multi-model alignment inspector endpoint. Compares how all evaluated models performed on the exact same sample.
- **Path Parameter**: `sample_id` (e.g., `kyc_sample_042`)
- **Processing Logic**:
  1. Searches all `results/*_predictions.jsonl` files for the specified sample ID.
  2. Extracts the canonical prompt, input text, context, target, and metadata.
  3. Builds a `models` mapping containing each model's generated text, exact match, judge score, token counts, latency, and hallucination flags.
- **Response Format**:
  ```json
  {
    "id": "kyc_sample_042",
    "input_text": "Calculate the applicable periodic review timeline...",
    "context": "High-risk customers require review every 2 years...",
    "target": "2 years",
    "prompt": "...",
    "metadata": { "task_type": "qa_numeric_program", "category": "Periodic Review" },
    "models": {
      "meta-llama-3_1-8b-instruct": {
        "output_text": "The periodic review timeline is 2 years.",
        "exact_match": 1,
        "f1": 1.0,
        "judge_score": 5.0,
        "completeness": 5.0,
        "helpfulness": 5.0,
        "bert_score": 0.98,
        "latency_seconds": 4.12,
        "tokens_out": 88,
        "is_hallucinated": false,
        "is_failure": false
      },
      "Qwen3-8B-GGUF": {
        "output_text": "Review must be conducted every 3 years.",
        "exact_match": 0,
        "f1": 0.5,
        "judge_score": 2.0,
        "completeness": 3.0,
        "helpfulness": 2.0,
        "bert_score": 0.81,
        "latency_seconds": 3.85,
        "tokens_out": 76,
        "is_hallucinated": true,
        "is_failure": false
      }
    }
  }
  ```

---

#### `GET /api/random-id`
Returns a randomly selected sample ID from available prediction files.
- **Response Format**:
  ```json
  {
    "id": "kyc_sample_073",
    "all_ids": ["kyc_sample_001", "kyc_sample_002", "..."]
  }
  ```

---

#### `GET /` and `GET /<path:path>`
Single Page Application (SPA) catch-all handler.
- If the requested path exists inside `frontend/dist/` (e.g., `assets/index.js`), it is served directly.
- If the path is not a static file, it serves `frontend/dist/index.html`, allowing React Router / client-side state navigation to function seamlessly.

---

## 3. Frontend Architecture (`frontend/`)

### 3.1 Technology Stack
- **Framework**: React 19 (`^19.2.8`)
- **Language**: TypeScript (`~6.0.2`)
- **Build Tool**: Vite (`^8.3.0`)
- **Charting Engine**: Recharts (`^3.10.1`)
- **Iconography**: Lucide React (`^1.48.0`)
- **Linter**: Oxlint (`^1.81.0`)

### 3.2 Component Breakdown

#### `src/App.tsx` (Main Dashboard Controller)
- **State Management**:
  - `activeTab`: Current view (`'overview' | 'comparison' | 'samples' | 'table'`).
  - `masterData`: Array of all model rows fetched from `/api/master-table`.
  - `selectedMetricGroup`: Active category filter pill (`'all' | 'core' | 'format' | 'task' | 'judge' | 'reliability' | 'efficiency' | 'robustness'`).
  - `searchQuery`: Filter string for searching models or metrics.
- **Key Sub-Views**:
  1. **Quick Stats Ribbon**: Displays total models evaluated, leader in Exact Match, leader in Judge Score, and lowest latency model.
  2. **42-Metric Overview**: Dynamically renders individual Recharts `<ResponsiveContainer>` and `<BarChart>` cards for all 42 metrics. Each card displays:
     - Metric title and category badge.
     - Comparative horizontal or vertical bars across all evaluated models.
     - Formatted tooltips and value labels.
  3. **Master Table View**: Full 42-column tabular view with sticky model name column, color-coded metric intensity cells, search filtering, and raw data export.
  4. **Sample Inspector**: Deep dive into individual regulatory test cases. Allows users to inspect reference answers versus model generations side-by-side.

#### `src/ModelComparison.tsx` (Head-to-Head Comparative Suite)
- **Features**:
  - **Dual Model Selectors**: Compare Model A against Model B.
  - **Head-to-Head Delta Table**: Lists all 42 metrics with Model A value, Model B value, absolute difference (`Δ`), percentage difference (`%`), and a visual winner badge.
  - **Radar Chart**: Normalized spider-web diagram comparing core capabilities across 6 normalized axes:
    - Task Accuracy (`exact_match`)
    - Judge Quality (`judge_score` / 5.0)
    - Semantic Precision (`bert_score`)
    - Factual Grounding (`1 - hallucination_rate`)
    - Reliability (`reliable_accuracy`)
    - Extraction Robustness (`extraction_success_rate`)
  - **Victory Share Card**: Real-time counter displaying how many metrics Model A won vs. Model B (e.g., *Model A won 28/42 metrics*).

---

## 4. Development & Build Commands

### 4.1 Running in Development Mode
To develop the frontend with Hot Module Replacement (HMR):
```bash
# Terminal 1: Start Flask Backend API
python server.py

# Terminal 2: Start Vite Dev Server
cd frontend
npm run dev
```
Vite will start on `http://localhost:5173` with automatic API proxying or CORS support to `http://localhost:5000`.

### 4.2 Building Production Assets
To compile the TypeScript source into optimized static production bundles:
```bash
cd frontend
npm run build
```
This runs `tsc -b && vite build`. The output artifacts are emitted to `frontend/dist/`:
- `frontend/dist/index.html`: Entrypoint HTML.
- `frontend/dist/assets/*.js`: Minified JavaScript chunks (React, Recharts, Lucide).
- `frontend/dist/assets/*.css`: Bundled and autoprefixed stylesheets.

### 4.3 Serving the Full Stack via Flask
Once `npm run build` has run, the entire application can be hosted from a single Python command without Node.js:
```bash
python server.py
```
Open your browser to:
```
http://localhost:5000
```
Flask will serve both the static React frontend at `/` and the REST API endpoints under `/api/*`.
