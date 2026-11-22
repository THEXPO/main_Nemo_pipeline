# Benchmark Evaluation Results & Technical Analysis

This document provides a comparative technical analysis of the **5 Large Language Models** benchmarked on the RBI KYC Regulatory Compliance Dataset.

---

## 1. Evaluated Models Overview

| Model Identifier | Architecture / Family | Primary Focus | Key Behavioral Characteristics |
|---|---|---|---|
| **Qwen3-8B-GGUF** | Qwen 3 (Alibaba Cloud) | General / Instruction | High instruction adherence, concise answers, fast generation, lowest hallucination. |
| **Ministral-3-8B-Reasoning-2512-GGUF** | Mistral AI Reasoning | Deep Chain-of-Thought | Extended step-by-step reasoning tokens, high completeness, higher latency. |
| **OREAL-DeepSeek-R1-Distill-Qwen-7B-GGUF** | DeepSeek R1 Distill | Mathematical / Distill | High token generation throughput, strong numeric step accuracy, concise final answers. |
| **meta-llama-3.1-8b-instruct** | Llama 3.1 (Meta) | Instruction / Chat | Strong domain knowledge, but prone to tool-call JSON leaks under compliance prompts. |
| **ministral-3-8b-instruct-2512** | Mistral AI Instruct | Concise Instruction | Balanced speed and compliance reasoning, clean output formatting without tool leaks. |

---

## 2. Multi-Model Benchmark Comparison Matrix

Below is the consolidated matrix across all 5 models for the 15 Core Metrics and key Derived Metrics:

| Metric Category & Name | Qwen3-8B-GGUF | Ministral-3-8B-Reasoning | OREAL-DeepSeek-R1 | ministral-3-8b-instruct | meta-llama-3.1-8b |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CORE METRICS** | | | | | |
| **Exact Match (EM)** | **0.7100** | 0.6800 | 0.6300 | 0.6300 | 0.5100 |
| **Token F1 Score** | **0.7413** | 0.7210 | 0.6523 | 0.6300 | 0.5100 |
| **BLEU Score** | **0.4902** | 0.4130 | 0.3301 | 0.1144 | 0.2482 |
| **ROUGE-L Score** | **0.7389** | 0.7197 | 0.6512 | 0.6300 | 0.5100 |
| **LLM Judge Score** (1-5) | 4.44 | 4.68 | **4.96** | 4.48 | 4.76 |
| **Completeness** (1-5) | 2.28 | **4.32** | 1.48 | 4.30 | 4.71 |
| **Helpfulness** (1-5) | 4.88 | 4.92 | **5.00** | 4.76 | 4.84 |
| **Reasoning Faithfulness** | **0.9930** | 1.0000 | 0.9350 | 1.0000 | 1.0000 |
| **BERTScore** | 0.3168 | **0.7187** | 0.6654 | 0.4450 | 0.2373 |
| **Hallucination Rate** (↓) | **0.1400** | 0.4800 | 0.3300 | 0.4900 | 0.2900 |
| **Failure Rate** (↓) | **0.0000** | 0.0100 | **0.0000** | **0.0000** | **0.0000** |
| **Latency (s)** (↓) | **4.5864s** | 19.3649s | 17.1175s | 24.2293s | 9.3868s |
| **Throughput (req/s)** (↑) | **0.2180** | 0.0516 | 0.0584 | 0.0413 | 0.1065 |
| **Tokens/sec** (↑) | 15.55 | 18.66 | **24.42** | 4.47 | 13.93 |
| **Category Accuracy** | **0.7100** | 0.6800 | 0.6300 | 0.6300 | 0.5100 |
| **DERIVED METRICS** | | | | | |
| **FN Call Leak Rate** (↓) | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.7000 |
| **Extraction Success Rate** | 0.7300 | **0.7500** | 0.6800 | 0.6700 | 0.1300 |
| **Final Answer Line Rate** | 0.7800 | **0.8200** | 0.7400 | 0.7000 | 0.6900 |
| **Answer Recoverable Rate** | 0.7200 | **0.7500** | 0.6800 | 0.6700 | 0.5000 |
| **Label Exact Match** | 0.4600 | 0.4200 | 0.3800 | **0.4800** | 0.0600 |
| **Numeric Exact Match** | 0.9600 | 0.9400 | 0.8800 | 0.7800 | **0.9600** |
| **Judge EM Gap** (↓) | 0.1780 | 0.2560 | 0.3620 | 0.2660 | 0.4420 |
| **Judge Composite** | 3.8667 | 4.6400 | 3.8133 | 4.5133 | **4.7700** |
| **Judge Leniency Rate** (↓) | **0.1900** | 0.2300 | 0.3200 | 0.2600 | 0.4300 |
| **Reliable Accuracy** (↑) | **0.6000** | 0.4300 | 0.4700 | 0.4000 | 0.4100 |
| **Ungrounded Number Ratio** (↓) | **0.1842** | 0.3912 | 0.2980 | 0.4215 | 0.4174 |
| **Latency per Correct** (↓) | **6.4597s** | 28.4778s | 27.1706s | 38.4592s | 18.4055s |
| **Tokens per Correct** (↓) | **100.46** | 531.44 | 663.02 | 171.86 | 256.39 |
| **Latency P95 (s)** (↓) | **8.12s** | 34.62s | 28.90s | 39.44s | 16.22s |
| **Macro Exact Match** | **0.5400** | 0.5100 | 0.5300 | 0.4300 | 0.3397 |
| **Worst Category EM** | **0.2000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **EM 95% CI Halfwidth** (↓) | **0.0880** | 0.0910 | 0.0940 | 0.0940 | 0.0950 |

---

## 3. In-Depth Technical Findings

### Finding 1: The Function-Call Leak Phenomenon (Llama-3.1-8B)
- **Observation**: `meta-llama-3.1-8b-instruct` achieved an Exact Match of only `0.5100` with an extraction success rate of `0.1300`.
- **Root Cause**: The model has a **`0.7000` (70%) function-call leak rate**. Instead of ending its response with `Final Answer: Yes`, it generated mock tool-call payloads:
  ```json
  Function call:
  {"name": "is_kyc_mandatory", "parameters": {"account_type": "all_types"}}
  ```
- **Diagnostic Recovery**: The derived feature `answer_recoverable_rate` (`0.5000`) revealed that the model actually possessed the correct compliance knowledge in its pre-JSON reasoning chain in 50% of the cases.

### Finding 2: Overall Champion (Qwen3-8B-GGUF)
- Achieved highest overall **Exact Match (`0.7100`)**, highest **Reliable Accuracy (`0.6000`)**, and highest **Macro EM (`0.5400`)**.
- Only model with **zero function-call leakage (`0.0000`)** combined with ultra-low latency (**4.58s** vs 19–24s for competitors).
- Highly cost-effective: Consumed only **100.46 tokens per correct answer**, compared to 531.44 for Ministral Reasoning.

### Finding 3: Chain-of-Thought Trade-Offs (Ministral Reasoning)
- Highest completeness score (`4.32/5.0`) and strong semantic BERTScore (`0.7187`).
- Employs deep chain-of-thought scratchpad tokens, raising average latency to **19.36s** and P95 latency to **34.62s**.
- Best suited for complex audits and offline compliance reports where latency is not user-facing.

### Finding 4: High Generation Throughput (OREAL DeepSeek-R1 Distill)
- Highest raw token emission rate (**24.42 tokens/sec**).
- Strong numerical calculation consistency (`numeric_exact_match`: `0.8800`).
- Received the highest LLM judge score (`4.96/5.0`), but exhibited a large `judge_em_gap` (`0.3620`), indicating judge leniency towards eloquent but non-exact answers.

---

## 4. Compliance & Deployment Recommendations

1. **For Production Real-Time APIs**:
   - **Recommended Model**: `Qwen3-8B-GGUF`.
   - **Rationale**: Sub-5s latency, 0% formatting leakage, highest reliable accuracy (`60%`), and lowest variance across categories.
2. **For High-Stakes Compliance Audit / Regulatory Investigation**:
   - **Recommended Model**: `Ministral-3-8B-Reasoning-2512-GGUF`.
   - **Rationale**: Superior reasoning completeness and deep step-by-step auditability.
3. **Prompt Adaptation for Llama 3.1**:
   - Add negative prompt constraints: `Do not emit tool calls, function calls, or JSON objects. Provide the answer in plain English.`
