# BFSI Benchmark Datasets: Overview & Sample Examples

This document provides a comprehensive guide to the 5 Banking, Financial Services, and Insurance (BFSI) benchmark datasets stored in [`data/raw/`](file:///d:/Sem%207/Edi/NeMo%20project/Project/data/raw/). It is designed so that anyone reviewing this project can immediately understand what each dataset contains, how it is structured, and what tasks it supports.

---

## 1. Quick Reference & Dataset Matrix

| Dataset | Primary Task | Input Modality | Target Output | Total Size / Splits | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Banking77** | Intent Classification | Single customer utterance (text) | 1 of 77 fine-grained banking intents | 10,003 train / 3,080 test | PolyAI (`GitHub`) |
| **Financial PhraseBank** | Sentiment Analysis | Financial news sentence | 3 classes: Negative (0), Neutral (1), Positive (2) | 2,264 to 4,846 sentences (by agreement tier) | Malo et al. (`HuggingFace`) |
| **FinanceBench** | Open-Book Financial QA & Retrieval | Financial question + SEC filing context (10-K/10-Q/8-K) | Ground-truth numerical answer + justification + page evidence | 150 train examples across top public companies | PatronusAI (`HuggingFace`) |
| **FinQA** | Deep Numerical Reasoning & Program Synthesis | Financial table + unstructured pre/post context text | Executable multi-step arithmetic program + numerical answer | 6,251 train / 883 dev / 1,147 test | EMNLP 2021 (`GitHub`) |
| **ConvFinQA** | Multi-Turn Conversational QA | Dialogue history + financial table + context prose | Sequential arithmetic operations per dialogue turn | 421 dev dialogues / 3,000+ train dialogues | EMNLP 2022 (`GitHub`) |

---

## 2. Dataset Deep Dives & Real Examples

---

### Dataset 1: Banking77

* **Directory**: [`data/raw/banking77/`](file:///d:/Sem%207/Edi/NeMo%20project/Project/data/raw/banking77/)
* **Files**: `train.jsonl`, `test.jsonl`, `train.csv`, `test.csv`, `categories.json`
* **Domain**: Retail banking customer service queries.
* **Task**: Classify a customer's inquiry into exactly one of 77 specific banking intent categories.

#### Schema
```json
{
  "text": "string (the user's message/query)",
  "label": "string (one of 77 intent category strings)"
}
```

#### Real Examples

##### Example 1: Card Arrival Inquiry
* **User Query**: `"I am still waiting on my card?"`
* **Target Intent Label**: `card_arrival`
* **Context / Scope**: Inquiries regarding pending physical debit/credit card delivery times and status.

##### Example 2: Exchange Rates & International Spending
* **User Query**: `"What is my money worth in other countries?"`
* **Target Intent Label**: `exchange_rate`
* **Context / Scope**: Inquiries regarding foreign currency conversion rates and cross-border currency values.

##### Example 3: Security & Fraud Alert
* **User Query**: `"Has there been any activity on my card today?"`
* **Target Intent Label**: `lost_or_stolen_card`
* **Context / Scope**: Queries triggered when a card is misplaced, compromised, or experiencing suspicious charges.

##### Example 4: Unexpected Fee Complaint
* **User Query**: `"I was transferring some money to a friend, which I thought would be free, but I see there is a charge. What's up with that?"`
* **Target Intent Label**: `transfer_fee_charged`
* **Context / Scope**: Fee dispute and explanation requests for wire or peer-to-peer transfers.

#### Why it's challenging:
Unlike coarse 3-to-5 class intent sets, Banking77 has 77 overlapping categories (e.g., `card_arrival` vs. `card_linking` vs. `card_delivery_estimate` vs. `order_physical_card`), requiring fine-grained semantic boundaries.

---

### Dataset 2: Financial PhraseBank

* **Directory**: [`data/raw/financial_phrasebank/`](file:///d:/Sem%207/Edi/NeMo%20project/Project/data/raw/financial_phrasebank/)
* **Sub-configurations**: 
  - `sentences_allagree/` (100% annotator agreement, 2,264 samples)
  - `sentences_75agree/` (≥75% annotator agreement, 3,453 samples)
  - `sentences_66agree/` (≥66% annotator agreement, 4,217 samples)
  - `sentences_50agree/` (≥50% annotator agreement, 4,846 samples)
* **Domain**: Financial news, quarterly earnings releases, and stock market disclosures.
* **Task**: Sentence-level financial sentiment classification into **Negative (0)**, **Neutral (1)**, or **Positive (2)**.

#### Schema
```json
{
  "sentence": "string (financial statement or news headline)",
  "label": 0 | 1 | 2,
  "label_text": "negative" | "neutral" | "positive"
}
```

#### Real Examples

##### Example 1: Neutral Operational Fact (Label = 1)
```json
{
  "sentence": "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .",
  "label": 1,
  "sentiment": "Neutral"
}
```
* **Explanation**: The sentence presents factual corporate strategy without expressing upside growth surprises or downside warnings.

##### Example 2: Positive Earnings & Sales Growth (Label = 2)
```json
{
  "sentence": "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .",
  "label": 2,
  "sentiment": "Positive"
}
```
* **Explanation**: Net sales doubled and pre-tax loss turned around to zero pre-tax profit (financial recovery).

##### Example 3: Strategic Profitability Outlook (Label = 2)
```json
{
  "sentence": "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability .",
  "label": 2,
  "sentiment": "Positive"
}
```
* **Explanation**: Forward-looking statement signaling higher plant capacity, efficiency, and margins.

#### Why it's challenging:
General NLP sentiment models fail because standard words (e.g. "liability", "depreciation", "freeze", "cut") can be neutral accounting terms or positive indicators (e.g., "cost cuts" = positive for earnings).

---

### Dataset 3: FinanceBench

* **Directory**: [`data/raw/financebench/`](file:///d:/Sem%207/Edi/NeMo%20project/Project/data/raw/financebench/)
* **Files**: `train.jsonl` (150 in-depth curated evaluation questions)
* **Domain**: Public SEC Filings (Form 10-K, Form 10-Q, Form 8-K) from companies such as 3M, Apple, Microsoft, Pfizer, etc.
* **Task**: Complex Question Answering, Financial Retrieval-Augmented Generation (RAG), and Financial Ratio Analysis with page-level ground truth evidence.

#### Schema
```json
{
  "financebench_id": "string (e.g. financebench_id_03029)",
  "doc_name": "string (e.g. 3M_2018_10K)",
  "question": "string (natural language financial analyst question)",
  "answer": "string (verified answer)",
  "justification": "string (methodology / line item source)",
  "evidence": [
    {
      "evidence_text": "string (extracted raw text / table snippet)",
      "doc_name": "string",
      "evidence_page_num": 59,
      "evidence_text_full_page": "string"
    }
  ]
}
```

#### Real Examples

##### Example 1: Direct Financial Metric Extraction (Cash Flow Statement)
* **Target Document**: `3M_2018_10K`
* **Question**: 
  > *"What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."*
* **Ground Truth Answer**: `"$1577.00"`
* **Analyst Justification**: 
  > *"The metric capital expenditures was directly extracted from the company 10K. The line item name, as seen in the 10K, was: Purchases of property, plant and equipment (PP&E)."*
* **Source Evidence Snippet (Page 59)**:
  ```text
  Cash Flows from Investing Activities:
  Purchases of property, plant and equipment (PP&E): (1,577) [in Millions]
  ```

##### Example 2: Qualitative Financial Analysis & Ratio Derivation
* **Target Document**: `3M_2022_10K`
* **Question**: 
  > *"Is 3M a capital-intensive business based on FY2022 data?"*
* **Ground Truth Answer**: 
  > *"No, the company is managing its CAPEX and Fixed Assets pretty efficiently, which is evident from below key metrics:*  
  > *- CAPEX/Revenue Ratio: 5.1%*  
  > *- Fixed assets/Total Assets: 20%*  
  > *- Return on Assets = 12.4%"*
* **Analyst Justification**: Requires computing `CAPEX / Revenue`, `Fixed Assets / Total Assets`, and `ROA = Net Income / Total Assets` across the Income Statement and Balance Sheet.

#### Why it's challenging:
Unlike simple search tasks, questions often require synthesizing across multiple financial statements (e.g., finding net income on the Income Statement and total assets on the Balance Sheet to compute ROA).

---

### Dataset 4: FinQA

* **Directory**: [`data/raw/finqa/`](file:///d:/Sem%207/Edi/NeMo%20project/Project/data/raw/finqa/)
* **Files**: `train.json` (6,251 items), `dev.json` (883 items), `test.json` (1,147 items)
* **Domain**: Earnings reports and 10-K/10-Q reports from S&P 500 companies.
* **Task**: Numerical reasoning over hybrid documents (financial tables + unstructured preceding/succeeding text) that generates executable mathematical programs.

#### Schema
```json
{
  "id": "string (e.g., V/2008/page_17.pdf-1)",
  "pre_text": ["array of sentences preceding the table"],
  "table": [["row1_col1", "row1_col2", "..."], ["..."]],
  "post_text": ["array of sentences following the table"],
  "qa": {
    "question": "string (numerical financial question)",
    "answer": "string (e.g., '127.40')",
    "program": "string (executable DSL program, e.g., 'divide(637, const_5)')",
    "steps": [
      {
        "op": "string (e.g. 'divide1-1')",
        "arg1": "string (value or step reference)",
        "arg2": "string",
        "res": "string"
      }
    ]
  }
}
```

#### Real Examples

##### Example 1: Multi-Provider Payment Volume Calculation
* **Context Table (excerpt from Visa 2008 filing)**:
  | Company | Payments Volume ($ billions) | Total Volume ($ billions) | Total Transactions (billions) | Cards (millions) |
  | :--- | :--- | :--- | :--- | :--- |
  | **Visa Inc.** | $2,457 | $3,822 | 50.3 | 1,592 |
  | **MasterCard** | 1,697 | 2,276 | 27.0 | 916 |
  | **American Express** | 637 | 647 | 5.0 | 86 |

* **Context Text**:
  > *"The largest operators of open-loop and closed-loop retail electronic payments networks are Visa, MasterCard, American Express, Discover, JCB and Diners Club."*

* **Question**:
  > *"what is the average payment volume per transaction for american express?"*

* **Reasoning Program**:
  ```text
  divide(637, 5.0)
  ```
* **Step-by-Step Execution**:
  - `Arg 1`: American Express Payments Volume = `637` ($ billions)
  - `Arg 2`: American Express Total Transactions = `5.0` (billions)
  - `Result`: `127.40` dollars per transaction

##### Example 2: Multi-Year Percentage Return Calculation
* **Question**:
  > *"what was the percentage cumulative total return for the five year period ended 31-dec-2017 of citi common stock?"*
* **Program**:
  ```text
  subtract(193.5, const_100), divide(#0, const_100)
  ```
* **Executed Answer**: `93.5%`

#### Why it's challenging:
Models cannot just memorize or copy text spans; they must locate the relevant numbers across both structured rows and unstructured text blocks, select the correct arithmetic operations (`add`, `subtract`, `multiply`, `divide`), and chain intermediate results (`#0`, `#1`).

---

### Dataset 5: ConvFinQA

* **Directory**: [`data/raw/convfinqa/`](file:///d:/Sem%207/Edi/NeMo%20project/Project/data/raw/convfinqa/)
* **Files**: `train.json`, `dev.json`, `train_turn.json`, `dev_turn.json`, `test_private.json`
* **Domain**: Financial reports with multi-turn conversational interactions.
* **Task**: Conversational numerical reasoning. Users ask a sequence of follow-up questions where later turns depend on answers, pronouns, or values calculated in earlier turns.

#### Schema
```json
{
  "id": "string",
  "pre_text": ["introductory narrative paragraphs"],
  "table": [["table rows and columns"]],
  "annotation": {
    "dialogue_break": ["Turn 1 Question", "Turn 2 Question", "Turn 3 Question", "..."],
    "exe_ans_list": [60.94, 25.14, 35.80, ...],
    "turn_program": ["60.94", "25.14", "subtract(60.94, 25.14)", ...]
  }
}
```

#### Real Examples

##### Multi-Turn Dialogue Sample: Marathon Oil Stock Option Analysis
* **Context Table (Stock Option Awards Grant)**:
  | Metric | 2007 | 2006 | 2005 |
  | :--- | :--- | :--- | :--- |
  | Weighted average exercise price per share | **$60.94** | $37.84 | **$25.14** |
  | Expected annual dividends per share | $0.96 | $0.80 | $0.66 |
  | Expected life in years | 5.0 | 5.1 | 5.5 |
  | Expected volatility | 27% | 28% | 28% |

* **Conversational Sequence**:

  * **Turn 1**:
    - **User**: *"what was the weighted average exercise price per share in 2007?"*
    - **Program**: `60.94`
    - **Output**: **`60.94`**

  * **Turn 2**:
    - **User**: *"and what was it in 2005?"*  *(Coreference / Implicit entity)*
    - **Program**: `25.14`
    - **Output**: **`25.14`**

  * **Turn 3**:
    - **User**: *"what was, then, the change over the years?"*  *(Context-dependent calculation)*
    - **Program**: `subtract(60.94, 25.14)`
    - **Output**: **`35.80`**

  * **Turn 4**:
    - **User**: *"and how much does that change represent in relation to this 2005 weighted average exercise price?"*
    - **Program**: `divide(35.80, 25.14)`
    - **Output**: **`1.42403`** (or **`142.4%`**)

#### Why it's challenging:
It combines **conversational coreference** (resolving "it", "that change", "the previous year") with **stateful numerical calculation chaining** over tabular SEC disclosures.

---

## 3. How to Load and Inspect Each Dataset in Python

Here is a quick Python snippet demonstrating how to load and read rows from each dataset directly from `data/raw/`:

```python
import json
import csv
from pathlib import Path

DATA_DIR = Path("data/raw")

# 1. Banking77 (JSONL or CSV)
with open(DATA_DIR / "banking77/train.jsonl", encoding="utf-8") as f:
    sample_b77 = json.loads(f.readline())
    print("[Banking77]", sample_b77)

# 2. Financial PhraseBank (JSONL)
with open(DATA_DIR / "financial_phrasebank/sentences_allagree/train.jsonl", encoding="utf-8") as f:
    sample_fpb = json.loads(f.readline())
    print("[PhraseBank]", sample_fpb)

# 3. FinanceBench (JSONL)
with open(DATA_DIR / "financebench/train.jsonl", encoding="utf-8") as f:
    sample_fb = json.loads(f.readline())
    print("[FinanceBench]", sample_fb["question"], "->", sample_fb["answer"])

# 4. FinQA (JSON)
with open(DATA_DIR / "finqa/dev.json", encoding="utf-8") as f:
    sample_finqa = json.load(f)[0]
    print("[FinQA]", sample_finqa["qa"]["question"], "->", sample_finqa["qa"]["program"])

# 5. ConvFinQA (JSON)
with open(DATA_DIR / "convfinqa/dev.json", encoding="utf-8") as f:
    sample_conv = json.load(f)[0]
    print("[ConvFinQA Dialog]", sample_conv["annotation"]["dialogue_break"][:2])
```

---

## 4. Summary & Recommended Evaluation Plan

If you are building or fine-tuning financial AI assistants, the 5 datasets cover the entire cognitive hierarchy:

1. **Foundational Understanding**:
   - `Banking77`: Understanding customer intent & routing.
   - `Financial PhraseBank`: Discerning market sentiment from corporate phraseology.
2. **Document Level QA & Extraction**:
   - `FinanceBench`: Locating real numbers and footnotes in SEC 10-K filings.
3. **Advanced Numerical Reasoning**:
   - `FinQA`: Performing multi-step arithmetic over combined tables + text.
   - `ConvFinQA`: Maintaining mathematical context across interactive multi-turn chats.
