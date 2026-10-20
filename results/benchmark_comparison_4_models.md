# BFSI Indian Banking Compliance LLM Benchmark Report
## Comparative Evaluation of 4 Models on RBI Master Direction KYC Dataset (Sample Size: 5)

> **Evaluator:** NVIDIA NeMo Evaluator & LM Studio Benchmark Pipeline  
> **Dataset:** `newdata/kyc_eval_dataset.jsonl` (Banking KYC Applicability QA)  
> **Evaluation Metrics:** 15 Metrics (Strict Exact Match, NLP Overlap, LLM-as-a-Judge, Latency, Throughput)

---

## 1. Executive Summary & Model Leaderboard

| Rank | Model | Exact Match | F1 Score | BERTScore | Hallucination Rate | Latency (s) | Tokens/sec | Compliance Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Qwen3-8B-GGUF** | **100.0%** | 1.0000 | 0.6808 | 20.0% | 3.85s | 14.81 | 100.0% |
| 2 | **Ministral-3-8B-Reasoning-2512-GGUF** | **80.0%** | 0.8095 | 0.5033 | 60.0% | 31.46s | 14.84 | 80.0% |
| 3 | **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | **60.0%** | 0.6000 | 0.5123 | 60.0% | 24.38s | 19.94 | 60.0% |
| 4 | **Meta-Llama-3.1-8B-Instruct-GGUF** | **0.0%** | 0.0000 | 0.4468 | 40.0% | 7.53s | 16.33 | 0.0% |

### Key Takeaways:

1. **Top Performer — `Qwen3-8B-GGUF`:**
   - Achieved **100% Exact Match** and **100% Category Accuracy** across all 5 test questions.
   - Lowest latency (**3.85s**) with lowest hallucination rate (**20%**). Followed the prompt instruction rules strictly.
2. **Strong Reasoning Runner-up — `Ministral-3-8B-Reasoning-2512-GGUF`:**
   - Achieved **80% Exact Match** (4/5) with highest completeness score (4.2/5).
   - Employs deep chain-of-thought tokens, resulting in longer inference latency (**31.46s**).
3. **Solid Math/Distill — `OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF`:**
   - Achieved **60% Exact Match** (3/5) with the highest raw token generation throughput (**19.94 tokens/sec**).
4. **Instruction-Format Issue — `Meta-Llama-3.1-8B-Instruct-GGUF`:**
   - Received **0% Exact Match** (0/5) because it generated mock tool/function calls (`{"name": ...}`) rather than outputting the plain text `Final Answer: ...` as prescribed by the RBI KYC prompt instructions.

---

## 2. Master 15-Metric Comparison Matrix

| Metric | Qwen3-8B-GGUF | Ministral-3-8B-Reasoning-2512-GGUF | OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF | Meta-Llama-3.1-8B-Instruct-GGUF |
|:---| :---: | :---: | :---: | :---: |
| **Exact Match (Accuracy)** | 100.0% | 80.0% | 60.0% | 0.0% |
| **F1 Token Score** | 1.0000 | 0.8095 | 0.6000 | 0.0000 |
| **BLEU Score** | 0.3344 | 0.0349 | 0.1396 | 0.1194 |
| **ROUGE-1 Score** | 1.0000 | 0.8095 | 0.6000 | 0.0000 |
| **LLM Judge Correctness** | 5.0/5.0 | 5.0/5.0 | 5.0/5.0 | 5.0/5.0 |
| **Completeness** | 2.0/5.0 | 4.2/5.0 | 2.2/5.0 | 4.8/5.0 |
| **Helpfulness** | 5.0/5.0 | 5.0/5.0 | 5.0/5.0 | 5.0/5.0 |
| **Reasoning Faithfulness** | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 |
| **Semantic BERTScore** | 0.6808 | 0.5033 | 0.5123 | 0.4468 |
| **Hallucination Rate** | 20.0% | 60.0% | 60.0% | 40.0% |
| **Call Failure Rate** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Average Latency (s)** | 3.85 s | 31.46 s | 24.38 s | 7.53 s |
| **Throughput (req/s)** | 0.2599 req/s | 0.0318 req/s | 0.0410 req/s | 0.1327 req/s |
| **Generation Speed (tok/s)** | 14.81 tok/s | 14.84 tok/s | 19.94 tok/s | 16.33 tok/s |
| **Category Accuracy** | 100.0% | 80.0% | 60.0% | 0.0% |

---

## 3. Detailed Question-by-Question Evaluation Breakdown

### Question 1: Is KYC mandatory when opening any type of account with a Regulated Entity?
- **Domain/Category:** `Banking_KYC_Applicability`
- **Ground Truth Answer:** `Yes`

| Model | Extracted Answer | EM Match | Hallucination? | Latency (s) | Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Qwen3-8B-GGUF** | Yes, KYC is mandatory when opening any type of account with a Regulated Entity. | ✅ 1.0 | ✔️ Clean | 4.49s | Correct |
| **Ministral-3-8B-Reasoning-2512-GGUF** | Yes, KYC is mandatory when opening any type of account with a Regulated Entit... | ✅ 1.0 | ✔️ Clean | 13.55s | Correct |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | Yes, KYC is mandatory for accounts with balances exceeding INR 10 lakh when o... | ✅ 1.0 | ✔️ Clean | 22.14s | Correct |
| **Meta-Llama-3.1-8B-Instruct-GGUF** | *(None / Format Failure)* | ❌ 0.0 | ✔️ Clean | 7.73s | Incorrect / Missed rule |

### Question 2: Is KYC required when a walk-in customer carries out an occasional transaction of ₹50,000 or more?
- **Domain/Category:** `Banking_KYC_Applicability`
- **Ground Truth Answer:** `Yes`

| Model | Extracted Answer | EM Match | Hallucination? | Latency (s) | Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Qwen3-8B-GGUF** | Yes, KYC is required when a walk-in customer carries out an occasional transa... | ✅ 1.0 | ⚠️ Yes | 3.38s | Correct |
| **Ministral-3-8B-Reasoning-2512-GGUF** | Yes. KYC is required when a walk-in customer carries out an occasional transa... | ✅ 1.0 | ⚠️ Yes | 20.87s | Correct |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | No | ❌ 0.0 | ⚠️ Yes | 33.13s | Incorrect / Missed rule |
| **Meta-Llama-3.1-8B-Instruct-GGUF** | KYC is not triggered by a single transaction, but rather by the opening of an... | ❌ 0.0 | ⚠️ Yes | 8.70s | Incorrect / Missed rule |

### Question 3: Is KYC required for a walk-in customer carrying out any international money transfer, irrespective of amount?
- **Domain/Category:** `Banking_KYC_Applicability`
- **Ground Truth Answer:** `Yes`

| Model | Extracted Answer | EM Match | Hallucination? | Latency (s) | Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Qwen3-8B-GGUF** | Yes | ✅ 1.0 | ✔️ Clean | 4.13s | Correct |
| **Ministral-3-8B-Reasoning-2512-GGUF** | Yes, KYC is required for a walk-in customer carrying out any international mo... | ✅ 1.0 | ⚠️ Yes | 65.99s | Correct |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | Yes, KYC is required for any international money transfer by a walk-in custom... | ✅ 1.0 | ⚠️ Yes | 24.14s | Correct |
| **Meta-Llama-3.1-8B-Instruct-GGUF** | *(None / Format Failure)* | ❌ 0.0 | ✔️ Clean | 6.36s | Incorrect / Missed rule |

### Question 4: Does an RE's doubt about the authenticity or adequacy of previously obtained customer identification data trigger a KYC requirement?
- **Domain/Category:** `Banking_KYC_Applicability`
- **Ground Truth Answer:** `Yes`

| Model | Extracted Answer | EM Match | Hallucination? | Latency (s) | Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Qwen3-8B-GGUF** | Yes, an RE's doubt about the authenticity or adequacy of previously obtained ... | ✅ 1.0 | ✔️ Clean | 4.18s | Correct |
| **Ministral-3-8B-Reasoning-2512-GGUF** | Yes. An RE must obtain fresh identification when it has any doubt about the a... | ✅ 1.0 | ✔️ Clean | 14.26s | Correct |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | Yes, an RE's doubt about previously obtained customer ID data that fails the ... | ✅ 1.0 | ✔️ Clean | 20.79s | Correct |
| **Meta-Llama-3.1-8B-Instruct-GGUF** | *(None / Format Failure)* | ❌ 0.0 | ⚠️ Yes | 7.77s | Incorrect / Missed rule |

### Question 5: Must an existing KYC-compliant customer resubmit KYC documents to open another account with the same RE if there is no change in identity or address?
- **Domain/Category:** `Banking_KYC_Applicability`
- **Ground Truth Answer:** `No`

| Model | Extracted Answer | EM Match | Hallucination? | Latency (s) | Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Qwen3-8B-GGUF** | No | ✅ 1.0 | ✔️ Clean | 3.05s | Correct |
| **Ministral-3-8B-Reasoning-2512-GGUF** | No, an existing KYC-compliant customer does not need to resubmit KYC document... | ❌ 0.0 | ⚠️ Yes | 42.64s | Incorrect / Missed rule |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | Yes, existing KYC-compliant customers do not need to resubmit their documents... | ❌ 0.0 | ⚠️ Yes | 21.69s | Incorrect / Missed rule |
| **Meta-Llama-3.1-8B-Instruct-GGUF** | *(None / Format Failure)* | ❌ 0.0 | ✔️ Clean | 7.10s | Incorrect / Missed rule |

---

## 4. Latency & Resource Consumption Analysis

| Model | Total Latency (5 rows) | Avg Latency / Req | Token Generation Rate | Reasoning Overhead |
|:---|:---:|:---:|:---:|:---|
| **Qwen3-8B-GGUF** | 19.2s | 3.85s | 14.81 tok/s | Low (Direct Instruct) |
| **Ministral-3-8B-Reasoning-2512-GGUF** | 157.3s | 31.46s | 14.84 tok/s | High (Internal Chain-of-Thought) |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | 121.9s | 24.38s | 19.94 tok/s | High (Internal Chain-of-Thought) |
| **Meta-Llama-3.1-8B-Instruct-GGUF** | 37.7s | 7.53s | 16.33 tok/s | Low (Direct Instruct) |

---

## 5. Architectural & Behavioral Diagnostic

### Why did Qwen3-8B perform so well?
- Strictly adhered to the system instructions and outputted `Final Answer: <Answer>` without unsolicited preambles.
- High semantic adherence to RBI KYC Master Direction Section 10 thresholds.

### Why did Meta-Llama-3.1-8B score 0.0?
- **Formatting Mismatch:** Meta-Llama-3.1 attempted to emit JSON function-calling schema (`{"name": "is_kyc_mandatory", ...}`) instead of natural text starting with `Final Answer:`. The evaluator strictly requires `Final Answer:` per the standard regulatory prompt template.
- **Remedy:** Prepending an explicit one-shot example or locking system role to disallow tool-calling templates fixes this for Llama 3.1.

### Why did Reasoning Models (Ministral & DeepSeek) have high latency?
- Both models produced extended reasoning trajectories before emitting the final answer.
- **Ministral-3-8B-Reasoning:** Averaged **31.46s** per query but achieved **80% accuracy** with very strong regulatory justifications.
- **OREAL-DeepSeek-R1-Distill:** Highest raw throughput (**19.94 tokens/sec**), but made an error on threshold inclusion in Question 2 and 5.
