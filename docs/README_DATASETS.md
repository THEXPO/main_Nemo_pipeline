# Dataset Specification & Regulatory Knowledge Base

## 1. Unified Dataset JSONL Schema

The benchmarking framework is completely dataset-agnostic. Any evaluation dataset formatted as JSONL with the following schema can be evaluated:

```json
{
  "id": "sample_001",
  "task_type": "qa_label",
  "input_text": "Is KYC mandatory when opening any type of account with a Regulated Entity?",
  "context": "(Optional) Authoritative regulatory passage, policy extract, or reference context.",
  "target": "Yes",
  "normalized_answer": "Yes",
  "company": "Banking_KYC_Applicability",
  "raw_values": [],
  "steps": [],
  "constants": []
}
```

---

## 2. Field Definitions

| Field Name | Type | Description |
|---|---|---|
| `id` / `sample_id` | String | Unique identifier for the test item (e.g., `item_001`). |
| `task_type` | String | Determines parsing and evaluation logic (`qa_label` or `qa_numeric_program`). |
| `input_text` | String | The query, compliance scenario, or arithmetic problem presented to the model. |
| `context` | String | Optional background passage or regulatory text. If empty, prompt directs model to use authoritative domain knowledge. |
| `target` | String | Exact ground-truth expected answer (e.g., `"Yes"`, `"₹50,000"`, `"12"`). |
| `normalized_answer` | String | Cleaned reference value without commas or extra symbols used for strict matching. |
| `company` | String | Category tag or domain subgroup used for macro and category breakdowns. |
| `raw_values` | List / Dict | Numeric constants present in the question context used by hallucination detector. |
| `steps` | List | Intermediate steps and expected calculation results used by reasoning faithfulness scorer. |
| `constants` | List | Whitelisted numerical thresholds relevant to the question. |

---

## 3. Supported Task Types

### A. Label / Categorical QA (`qa_label`)
- **Scope**: Binary classification (Yes/No), regulatory determination, KYC applicability, and document validation.
- **Evaluation**: Evaluated using strict string equality, label token F1, and ROUGE-L.
- **Example Prompt Constraint**:
  ```
  Question starts with Is/Does/Can/Must/Are/Should: answer "Yes" or "No".
  End your response with: Final Answer: <your answer>
  ```

### B. Numeric / Program QA (`qa_numeric_program`)
- **Scope**: Threshold calculations, transaction limits, periodic updation timeframes, penalty computations, and statutory limits.
- **Evaluation**: Normalized floating point comparison with tolerance ($|y_{pred} - y_{gold}| \le 0.01$). F1 and ROUGE are aligned to Exact Match.
- **Example Prompt Constraint**:
  ```
  For calculations, show the arithmetic in one line, then give the result.
  End your response with: Final Answer: <your answer>
  ```

---

## 4. Benchmark Dataset: RBI KYC Regulatory Compliance

The primary production benchmark dataset included in the repository is [`newdata/kyc_eval_dataset.jsonl`](file:///d:/Sem%207/Edi/NeMo%20project/Project/newdata/kyc_eval_dataset.jsonl).

### Regulatory Domain
- **Jurisdiction**: India (Reserve Bank of India regulatory regime).
- **Primary Source Texts**:
  1. **RBI Master Direction - Know Your Customer (KYC) Direction, 2016** (Updated periodically).
  2. **Prevention of Money Laundering Act, 2002 (PMLA)** and Prevention of Money-Laundering (Maintenance of Records) Rules, 2005.
- **Regulated Entities (REs)**: Scheduled Commercial Banks, Regional Rural Banks, Small Finance Banks, NBFCs, and Payment System Providers.

### Sub-Category Breakdown (100 Test Cases)

| Category Subgroup | Sample Count | Primary Regulatory Focus |
|---|:---:|---|
| `Banking_KYC_Periodic_Updation` | 27 | Updation frequency: 2 years (high-risk), 8 years (medium-risk), 10 years (low-risk); non-face-to-face updation. |
| `Banking_KYC_Applicability` | 18 | Account opening triggers, walk-in customers, occasional transactions ($\ge ₹50,000$), international transfers. |
| `Banking_KYC_Documents` | 18 | Officially Valid Documents (OVDs): Passport, Driving Licence, Voter ID, Aadhaar, Job Card issued by NREGA, NPR letter. |
| `Banking_KYC_Instruments` | 14 | Pre-paid instruments (PPI), demand drafts, wire transfers, traveler cheques, cross-border remittance limits. |
| `Banking_KYC_Onboarding` | 5 | Customer Identification Procedure (CIP), Simplified Due Diligence, Beneficial Ownership determination ($>10\%$ or $>15\%$). |
| `Banking_KYC_VCIP` | 5 | Video-based Customer Identification Process (VCIP): live geofencing, facial matching, concurrent audit rules. |
| `Banking_KYC_Inoperative_Accounts` | 5 | Accounts with no customer-mandated transactions for $>2$ years, handling of interest credits, activation without charges. |
| `Banking_KYC_CKYC` | 4 | Central KYC Records Registry upload mandates (within 10 days), 14-digit CKYC Identifier usage. |
| `Banking_KYC_Small_Account` | 3 | Small account caps: aggregate credits $\le ₹1,00,000$/year, maximum balance $\le ₹50,000$, monthly debits $\le ₹10,000$. |
| `Banking_KYC_Fraud_Awareness` | 1 | Suspicious Transaction Reports (STRs), Cash Transaction Reports (CTRs), reporting within 7 working days to FIU-IND. |

---

## 5. Concrete Data Examples

### Example 1: `qa_label` (Binary Determination)
```json
{
  "id": "item_001",
  "task_type": "qa_label",
  "input_text": "Is KYC mandatory when opening any type of account with a Regulated Entity?",
  "context": "",
  "target": "Yes",
  "normalized_answer": "Yes",
  "company": "Banking_KYC_Applicability",
  "raw_values": [],
  "steps": [],
  "constants": []
}
```

### Example 2: `qa_numeric_program` (Threshold Calculation)
```json
{
  "id": "item_052",
  "task_type": "qa_numeric_program",
  "input_text": "What is the maximum aggregate of all withdrawals and transfers in a month for a Small Account?",
  "context": "As per RBI Master Direction on KYC, a 'Small Account' has specific operational limitations including balance and withdrawal ceilings.",
  "target": "₹10,000",
  "normalized_answer": "10000",
  "company": "Banking_KYC_Small_Account",
  "raw_values": ["10000"],
  "steps": [],
  "constants": ["10000", "50000", "100000"]
}
```

---

## 6. How to Evaluate Custom Datasets

To evaluate any new dataset, simply structure it in JSONL adhering to the fields above, then run:

```powershell
python llm_benchmark.py --dataset path/to/your_dataset.jsonl --sample-size 0
```

The pipeline automatically handles label matching, numerical program comparison, metric aggregation, and derived feature generation without needing code changes.
