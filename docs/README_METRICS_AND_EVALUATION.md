# Complete 42-Metric Evaluation Framework

This document provides the definitive mathematical and algorithmic reference for all **42 evaluation metrics** (15 Core + 27 Derived) implemented in the benchmark pipeline.

---

## 1. The 15 Core Metrics

The core pipeline evaluates 15 foundational dimensions across NLP accuracy, LLM-as-a-judge scoring, reasoning faithfulness, factual hallucination, and operational performance.

### 1. Exact Match (EM)
* **Category**: Accuracy & Match
* **Range**: $0.0$ to $1.0$ (Higher is better)
* **Algorithm**:
  - For label tasks (`qa_label`): Lowercase, strip punctuation/whitespace, check strict string equality against reference target.
  - For numeric tasks (`qa_numeric_program`): Parse numbers, strip commas, currency (`₹`, `$`) and percent symbols (`%`), round to 2 decimal places, and check absolute equality ($|y_{pred} - y_{gold}| \le 0.01$).

### 2. Token F1 Score
* **Category**: Accuracy & Match
* **Range**: $0.0$ to $1.0$ (Higher is better)
* **Algorithm**:
  - Computes harmonic mean of precision and recall over token sets:
    $$\text{Precision} = \frac{|T_{pred} \cap T_{gold}|}{|T_{pred}|}, \quad \text{Recall} = \frac{|T_{pred} \cap T_{gold}|}{|T_{gold}|}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  - For numeric tasks, token F1 is dynamically aligned to Exact Match.

### 3. BLEU Score
* **Category**: Accuracy & Match
* **Range**: $0.0$ to $1.0$ (Higher is better)
* **Algorithm**: Unigram overlap precision ($BLEU_1$) computed using NeMo Evaluator / standard BLEU scoring over extracted final answer strings against the reference.

### 4. ROUGE-L Score
* **Category**: Accuracy & Match
* **Range**: $0.0$ to $1.0$ (Higher is better)
* **Algorithm**: Longest Common Subsequence (LCS) recall measure. For numeric tasks, ROUGE is aligned to Exact Match.

### 5. LLM Judge Score
* **Category**: LLM Judge & Quality
* **Range**: $1.0$ to $5.0$ (Higher is better)
* **Algorithm**: An evaluator model (`local-model` or specified judge) evaluates the model's extracted answer and reasoning validity against an anchored compliance rubric:
  - `5.0`: Correct answer, fully sound regulatory reasoning.
  - `3.0`–`4.0`: Partially correct or correct with ambiguous reasoning.
  - `1.0`–`2.0`: Factually incorrect, contradicting RBI directions or thresholds.

### 6. Completeness Score
* **Category**: LLM Judge & Quality
* **Range**: $1.0$ to $5.0$ (Higher is better)
* **Algorithm**: Evaluates whether all aspects of the compliance scenario or multi-step arithmetic were addressed. Capped at $\le 2.0$ whenever the answer is incorrect.

### 7. Helpfulness Score
* **Category**: LLM Judge & Quality
* **Range**: $1.0$ to $5.0$ (Higher is better)
* **Algorithm**: Evaluates clarity, structure, and directness of the output. Capped at $\le 2.0$ if the core answer is wrong.

### 8. Reasoning Faithfulness
* **Category**: Faithfulness & Safety
* **Range**: $0.0$ to $1.0$ (Higher is better)
* **Algorithm**: Rule-based verification measuring the fraction of ground-truth intermediate calculation steps present in the generated reasoning text:
  $$\text{Faithfulness} = \frac{\sum_{s \in \text{steps}} \mathbb{I}(\text{result}(s) \in \text{reasoning})}{|\text{steps}|}$$

### 9. BERTScore
* **Category**: Faithfulness & Safety
* **Range**: $-1.0$ to $1.0$ (Higher is better)
* **Algorithm**: Semantic cosine similarity between contextual embeddings of candidate answer substring and reference target (with TF-IDF cosine fallback if torch is unavailable).

### 10. Hallucination Rate
* **Category**: Faithfulness & Safety
* **Range**: $0.0$ to $1.0$ (Lower is better)
* **Algorithm**: Scans generated response for all numbers. Flags as hallucinated ($H=1$) if any number appears that is NOT in the expanded whitelist:
  - Whitelist: `raw_values`, `target`, `normalized_answer`, intermediate step values (`metadata.steps`), integers $1 \dots 12$, and scaling variants ($v \times 100$, $v / 100$).

### 11. Failure Rate
* **Category**: Faithfulness & Safety
* **Range**: $0.0$ to $1.0$ (Lower is better)
* **Algorithm**: Share of calls that resulted in HTTP API errors, timeouts, empty responses, or standard refusal phrases ("I cannot assist", "as an AI").

### 12. Latency (s)
* **Category**: Performance & Speed
* **Unit**: Seconds (Lower is better)
* **Algorithm**: Mean wall-clock time elapsed from dispatch of HTTP POST request to completion of response streaming.

### 13. Throughput (req/s)
* **Category**: Performance & Speed
* **Unit**: Requests / second (Higher is better)
* **Algorithm**: Total completed requests divided by total pipeline wall-clock time:
  $$\text{Throughput} = \frac{N_{\text{samples}}}{T_{\text{total}}}$$

### 14. Tokens/sec
* **Category**: Performance & Speed
* **Unit**: Tokens / second (Higher is better)
* **Algorithm**: Total generated completion tokens divided by total generation duration:
  $$\text{Tokens/sec} = \frac{\sum \text{tokens\_out}}{\sum \text{latency\_seconds}}$$

### 15. Category Accuracy
* **Category**: Accuracy & Match
* **Range**: $0.0$ to $1.0$ (Higher is better)
* **Algorithm**: Overall accuracy aggregated across task types and company/category sub-disclosures.

---

## 2. The 27 Derived Features

The derived metrics layer computes 27 diagnostic features across 6 specialized dimensions purely from saved predictions and aggregates without extra model calls.

### Notation & Primitives
- **EM**: Exact Match indicator per sample ($EM \in \{0, 1\}$).
- **H**: Hallucination indicator per sample ($H \in \{0, 1\}$).
- **pre-JSON text**: The substring of `output_text` preceding the first function-call blob (regex `\{\s*"name"\s*:\s*"`) or line starting with `JSON for` / `Function call`.

---

### Group 1: Format & Leakage (Features 1–4)

#### 1. `fn_call_leak_rate`
* **Definition**: Share of responses containing raw function-call JSON syntax or function call headers:
  $$\text{fn\_call\_leak\_rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{regex}(\text{output\_text}_i))$$
* **Regex**: `\{\s*"name"\s*:\s*"`
* **Interpretation**: Measures prompt instruction adherence failure where models leak internal tool-call syntax.

#### 2. `extraction_success_rate`
* **Definition**: Share of samples where an answer could be parsed and `extracted_answer.strip()` is non-empty:
  $$\text{extraction\_success\_rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{len}(\text{extracted\_answer}_i.\text{strip}()) > 0)$$

#### 3. `final_answer_line_rate`
* **Definition**: Share of outputs containing an explicit case-insensitive `Final Answer` header line.

#### 4. `answer_recoverable_rate`
* **Definition**: Share of samples where the ground truth is literally stated in the pre-JSON reasoning text, even if final extraction failed:
  - **Numeric Tasks**: Normalized answer (commas stripped) appears as a standalone number.
  - **Yes/No Targets**: The last Yes/No token occurring within 25 characters after `"answer"` or at line start equals target.
  - **Other Targets**: Normalized substring containment (lowercase, stripped punctuation and currency).

---

### Group 2: Task Granularity (Features 5–6)

#### 5. `label_exact_match`
* **Definition**: Mean Exact Match restricted to classification tasks (`task_type == "qa_label"`).

#### 6. `numeric_exact_match`
* **Definition**: Mean Exact Match restricted to numerical arithmetic tasks (`task_type == "qa_numeric_program"`).

---

### Group 3: Judge Alignment & Bias (Features 7–10)

#### 7. `judge_em_gap`
* **Definition**: Normalized delta between LLM judge score (scaled 0 to 1) and actual exact match:
  $$\text{judge\_em\_gap} = \frac{\overline{\text{judge\_score}}}{5.0} - \overline{\text{EM}}$$
* **Interpretation**: High positive values indicate judge hallucination or leniency toward incorrect answers.

#### 8. `judge_composite`
* **Definition**: Unweighted arithmetic mean of average judge score, completeness, and helpfulness:
  $$\text{judge\_composite} = \frac{\overline{\text{judge\_score}} + \overline{\text{completeness}} + \overline{\text{helpfulness}}}{3}$$

#### 9. `judge_leniency_rate`
* **Definition**: Proportion of samples awarded high judge rating ($\ge 4.0$) despite having zero exact match ($EM = 0$):
  $$\text{judge\_leniency\_rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{judge\_score}_i \ge 4.0 \;\land\; EM_i = 0)$$

#### 10. `judge_fail_rate`
* **Definition**: Proportion of samples receiving failing judge rating ($\le 2.0$):
  $$\text{judge\_fail\_rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{judge\_score}_i \le 2.0)$$

---

### Group 4: Reliability & Grounding (Features 11–16)

#### 11. `reliable_accuracy`
* **Definition**: Proportion of samples that are both Exact Match correct AND completely non-hallucinated:
  $$\text{reliable\_accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(EM_i = 1 \;\land\; \neg H_i)$$

#### 12. `hallucinated_but_correct_rate`
* **Definition**: Proportion of samples achieving exact match despite containing ungrounded reasoning numbers:
  $$\text{hallucinated\_but\_correct\_rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(EM_i = 1 \;\land\; H_i)$$

#### 13. `ungrounded_number_ratio`
* **Definition**: Per-sample set cardinality ratio of distinct reasoning numbers not present in prompt, context, or target, averaged over samples containing at least one number:
  $$\text{ratio}_i = \frac{|S_{\text{output}, i} \setminus S_{\text{source}, i}|}{|S_{\text{output}, i}|}, \quad \text{ungrounded\_number\_ratio} = \frac{1}{|K|} \sum_{i \in K} \text{ratio}_i$$
  * Leading list numerals (regex `^\s*\d+[.)]\s`, multiline) and thousands commas are stripped prior to extraction.

#### 14. `ungrounded_flag_delta`
* **Definition**: Difference in mean ungrounded number ratio between hallucinated and non-hallucinated rows:
  $$\text{ungrounded\_flag\_delta} = \overline{\text{ratio}}_{H=1} - \overline{\text{ratio}}_{H=0}$$

#### 15. `hallucination_separation`
* **Definition**: Statistical separation power of the hallucination flag between failures and successes:
  $$\text{hallucination\_separation} = P(H=1 \mid EM=0) - P(H=1 \mid EM=1)$$

#### 16. `label_hedge_rate`
* **Definition**: Share of label question responses matching uncertainty or hedging patterns:
  * Regex: `does not specify|not specif|cannot definitively|cannot provide a definitive|no specific provision|not explicit|definitive answer|not clear|unsure` (case-insensitive).

---

### Group 5: Efficiency & Scaling (Features 17–23)

#### 17. `latency_per_correct`
* **Definition**: Average seconds of generation time required per unit of Exact Match accuracy:
  $$\text{latency\_per\_correct} = \frac{\overline{\text{latency\_seconds}}}{\overline{EM}} \quad (\text{null if } \overline{EM} = 0)$$

#### 18. `tokens_per_correct`
* **Definition**: Average generated output tokens consumed per unit of Exact Match accuracy:
  $$\text{tokens\_per\_correct} = \frac{\overline{\text{tokens\_out}}}{\overline{EM}} \quad (\text{null if } \overline{EM} = 0)$$

#### 19. `latency_p95`
* **Definition**: 95th percentile generation latency across sample requests (seconds).

#### 20. `latency_cv`
* **Definition**: Coefficient of variation of generation latency (sample standard deviation with $ddof=1$ divided by mean latency):
  $$\text{latency\_cv} = \frac{s_{\text{latency}}}{\overline{\text{latency}}}$$

#### 21. `quality_per_second`
* **Definition**: Mean judge score delivered per second of inference latency:
  $$\text{quality\_per\_second} = \frac{\overline{\text{judge\_score}}}{\overline{\text{latency\_seconds}}}$$

#### 22. `output_input_token_ratio`
* **Definition**: Average of per-sample generation expansion ratios:
  $$\text{output\_input\_token\_ratio} = \frac{1}{N} \sum_{i=1}^N \frac{\text{tokens\_out}_i}{\text{tokens\_in}_i}$$

#### 23. `verbosity_accuracy_corr`
* **Definition**: Pearson correlation coefficient between output token length (`tokens_out`) and exact match ($EM$).

---

### Group 6: Robustness & Confidence Intervals (Features 24–27)

#### 24. `macro_exact_match`
* **Definition**: Unweighted mean of category-level Exact Match averages across all unique categories ($C$):
  $$\text{macro\_exact\_match} = \frac{1}{|C|} \sum_{c \in C} \overline{EM}_c$$

#### 25. `worst_category_em`
* **Definition**: Minimum category Exact Match among categories with sample size $n \ge 5$.

#### 26. `category_em_std`
* **Definition**: Sample standard deviation ($ddof=1$) of category Exact Match among categories with $n \ge 5$.

#### 27. `em_ci95_halfwidth`
* **Definition**: Half-width of 95% bootstrap confidence interval of mean Exact Match:
  - 2,000 resamples with replacement using fixed random seed (`numpy.random.default_rng(0)`):
  $$\text{em\_ci95\_halfwidth} = \frac{P_{97.5} - P_{2.5}}{2}$$

---

## 3. Master Reference Catalog (All 42 Metrics)

| # | Metric Identifier | Display Title | Group | Type | Optimal Direction |
|---|---|---|---|---|---|
| 1 | `exact_match` | Exact Match (EM) | Quality | Core | Higher |
| 2 | `f1` | Token F1 Score | Quality | Core | Higher |
| 3 | `bleu` | BLEU Score | Quality | Core | Higher |
| 4 | `rouge` | ROUGE-L Score | Quality | Core | Higher |
| 5 | `judge_score` | LLM Judge Score | Judge | Core | Higher |
| 6 | `completeness` | Completeness Score | Judge | Core | Higher |
| 7 | `helpfulness` | Helpfulness Score | Judge | Core | Higher |
| 8 | `reasoning_faithfulness` | Reasoning Faithfulness | Faithfulness | Core | Higher |
| 9 | `bert_score` | BERTScore | Semantic | Core | Higher |
| 10 | `hallucination_rate` | Hallucination Rate | Factuality | Core | Lower |
| 11 | `failure_rate` | Failure Rate | Reliability | Core | Lower |
| 12 | `latency` | Latency (s) | Efficiency | Core | Lower |
| 13 | `throughput` | Throughput (req/s) | Efficiency | Core | Higher |
| 14 | `tokens_per_sec` | Tokens/sec | Efficiency | Core | Higher |
| 15 | `category_accuracy` | Category Accuracy | Breakdown | Core | Higher |
| 16 | `fn_call_leak_rate` | FN Call Leak Rate | FORMAT | Derived | Lower |
| 17 | `extraction_success_rate` | Extraction Success Rate | FORMAT | Derived | Higher |
| 18 | `final_answer_line_rate` | Final Answer Line Rate | FORMAT | Derived | Higher |
| 19 | `answer_recoverable_rate` | Answer Recoverable Rate | FORMAT | Derived | Higher |
| 20 | `label_exact_match` | Label Exact Match | TASK | Derived | Higher |
| 21 | `numeric_exact_match` | Numeric Exact Match | TASK | Derived | Higher |
| 22 | `judge_em_gap` | Judge EM Gap | JUDGE | Derived | Lower |
| 23 | `judge_composite` | Judge Composite | JUDGE | Derived | Higher |
| 24 | `judge_leniency_rate` | Judge Leniency Rate | JUDGE | Derived | Lower |
| 25 | `judge_fail_rate` | Judge Fail Rate | JUDGE | Derived | Lower |
| 26 | `reliable_accuracy` | Reliable Accuracy | RELIABILITY | Derived | Higher |
| 27 | `hallucinated_but_correct_rate` | Hallucinated But Correct | RELIABILITY | Derived | Lower |
| 28 | `ungrounded_number_ratio` | Ungrounded Number Ratio | RELIABILITY | Derived | Lower |
| 29 | `ungrounded_flag_delta` | Ungrounded Flag Delta | RELIABILITY | Derived | Higher |
| 30 | `hallucination_separation` | Hallucination Separation | RELIABILITY | Derived | Higher |
| 31 | `label_hedge_rate` | Label Hedge Rate | RELIABILITY | Derived | Lower |
| 32 | `latency_per_correct` | Latency per Correct | EFFICIENCY | Derived | Lower |
| 33 | `tokens_per_correct` | Tokens per Correct | EFFICIENCY | Derived | Lower |
| 34 | `latency_p95` | Latency P95 (s) | EFFICIENCY | Derived | Lower |
| 35 | `latency_cv` | Latency CV | EFFICIENCY | Derived | Lower |
| 36 | `quality_per_second` | Quality per Second | EFFICIENCY | Derived | Higher |
| 37 | `output_input_token_ratio` | Output/Input Token Ratio | EFFICIENCY | Derived | Informative |
| 38 | `verbosity_accuracy_corr` | Verbosity-Accuracy Corr | EFFICIENCY | Derived | Informative |
| 39 | `macro_exact_match` | Macro Exact Match | ROBUSTNESS | Derived | Higher |
| 40 | `worst_category_em` | Worst Category EM | ROBUSTNESS | Derived | Higher |
| 41 | `category_em_std` | Category EM Std | ROBUSTNESS | Derived | Lower |
| 42 | `em_ci95_halfwidth` | EM 95% CI Halfwidth | ROBUSTNESS | Derived | Lower |
