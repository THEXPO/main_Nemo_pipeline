"""
prompt_templates.py
Generates simple, clear prompts for the 5 BFSI benchmark datasets.
Prints the built prompt for item #1 of each dataset to inspect prompt quality.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

_BANKING77_CATEGORIES_PATH = Path("data/raw/banking77/categories.json")
if _BANKING77_CATEGORIES_PATH.exists():
    with open(_BANKING77_CATEGORIES_PATH, "r", encoding="utf-8") as _f:
        BANKING77_CATEGORIES = json.load(_f)
else:
    _alt_path = Path(__file__).resolve().parent / "data" / "raw" / "banking77" / "categories.json"
    if _alt_path.exists():
        with open(_alt_path, "r", encoding="utf-8") as _f:
            BANKING77_CATEGORIES = json.load(_f)
    else:
        BANKING77_CATEGORIES = []


def format_table_as_text(table: Any) -> str:
    """Formats a 2D list into a readable text table."""
    if not table:
        return "[No table provided]"
    if isinstance(table, str):
        return table
    if not isinstance(table, list):
        return str(table)

    rows = []
    for row in table:
        if isinstance(row, list):
            rows.append(" | ".join(str(cell).strip() for cell in row))
        else:
            rows.append(str(row).strip())
    return "\n".join(rows)


def build_banking77_prompt(item: Dict[str, Any], print_if_first: bool = False) -> str:
    """
    Banking77 Prompt:
    Lists all 77 valid canonical categories, presents customer message,
    instructs the model to output ONLY the exact label, and ends with a trailing Category: cue.
    """
    text = item.get("text", "").strip()
    categories_list = "\n".join(f"- {cat}" for cat in BANKING77_CATEGORIES)

    prompt = (
        "You are an AI assistant for a retail bank. "
        "Classify the customer's inquiry into exactly ONE valid intent category from the list below.\n\n"
        f"Valid Intent Categories:\n{categories_list}\n\n"
        f"Customer Message: \"{text}\"\n\n"
        "Instructions: Respond with ONLY the exact category label from the list above. "
        "Do not include any explanation, extra text, quotation marks, or punctuation.\n\n"
        "Category:"
    )
    if print_if_first:
        print("\n" + "=" * 60)
        print("PROMPT #1 [Banking77]:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    return prompt


def build_phrasebank_prompt(item: Dict[str, Any], print_if_first: bool = False) -> str:
    """
    Financial PhraseBank Prompt:
    Presents financial sentence and asks for sentiment: negative, neutral, or positive.
    """
    sentence = item.get("sentence", "").strip()
    prompt = (
        "You are a financial sentiment analysis expert. "
        "Analyze the sentiment of the following sentence from a financial disclosure or news report.\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        "Question: Is the sentiment of this sentence negative, neutral, or positive? "
        "State your answer clearly as one of: negative, neutral, or positive."
    )
    if print_if_first:
        print("\n" + "=" * 60)
        print("PROMPT #1 [Financial PhraseBank]:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    return prompt


def build_financebench_prompt(item: Dict[str, Any], print_if_first: bool = False) -> str:
    """
    FinanceBench Prompt:
    Presents question + joined evidence snippets, asks for a direct answer.
    """
    question = item.get("question", "").strip()
    doc_name = item.get("doc_name", "SEC Filing")
    evidence_list = item.get("evidence", [])

    evidence_texts = []
    if isinstance(evidence_list, list):
        for ev in evidence_list:
            if isinstance(ev, dict):
                text = ev.get("evidence_text", "").strip()
                page = ev.get("evidence_page_num")
                header = f"[Page {page}]" if page is not None else ""
                if text:
                    evidence_texts.append(f"{header}\n{text}".strip())
            elif isinstance(ev, str):
                evidence_texts.append(ev.strip())
    elif isinstance(evidence_list, str):
        evidence_texts.append(evidence_list.strip())

    joined_evidence = "\n\n---\n\n".join(evidence_texts) if evidence_texts else "[No context snippet provided]"

    prompt = (
        f"You are a financial analyst reviewing SEC filing '{doc_name}'. "
        "Answer the following question relying directly on the provided context evidence.\n\n"
        f"Context Evidence:\n{joined_evidence}\n\n"
        f"Question: {question}\n\n"
        "Provide a direct, factual answer with the exact numeric value or conclusion based on the filing."
    )
    if print_if_first:
        print("\n" + "=" * 60)
        print("PROMPT #1 [FinanceBench]:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    return prompt


def build_finqa_prompt(item: Dict[str, Any], print_if_first: bool = False) -> str:
    """
    FinQA Prompt:
    Presents pre_text + table + post_text + question.
    Asks for the numeric answer and the calculation steps.
    """
    pre_text = item.get("pre_text", [])
    if isinstance(pre_text, list):
        pre_str = " ".join(str(s) for s in pre_text).strip()
    else:
        pre_str = str(pre_text).strip()

    post_text = item.get("post_text", [])
    if isinstance(post_text, list):
        post_str = " ".join(str(s) for s in post_text).strip()
    else:
        post_str = str(post_text).strip()

    table_str = format_table_as_text(item.get("table", []))
    qa_block = item.get("qa", {})
    question = qa_block.get("question", "") if isinstance(qa_block, dict) else ""

    context_parts = []
    if pre_str:
        context_parts.append(f"Context Introduction:\n{pre_str}")
    if table_str:
        context_parts.append(f"Financial Table:\n{table_str}")
    if post_str:
        context_parts.append(f"Footnotes / Notes:\n{post_str}")

    full_context = "\n\n".join(context_parts)

    prompt = (
        "You are an expert financial auditor performing numerical reasoning over corporate disclosures.\n\n"
        f"{full_context}\n\n"
        f"Question: {question}\n\n"
        "Please provide:\n"
        "1. Step-by-step arithmetic calculations showing where each number came from.\n"
        "2. The final numerical answer."
    )
    if print_if_first:
        print("\n" + "=" * 60)
        print("PROMPT #1 [FinQA]:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    return prompt


def build_convfinqa_prompt(item: Dict[str, Any], print_if_first: bool = False) -> str:
    """
    ConvFinQA Prompt:
    Presents pre_text + table + all dialogue turns so far, asks for the answer to the final turn.
    """
    pre_text = item.get("pre_text", [])
    if isinstance(pre_text, list):
        pre_str = " ".join(str(s) for s in pre_text).strip()
    else:
        pre_str = str(pre_text).strip()

    table_str = format_table_as_text(item.get("table", []))

    annotation = item.get("annotation", {})
    dialogue_questions = annotation.get("dialogue_break", []) if isinstance(annotation, dict) else []
    dialogue_answers = annotation.get("exe_ans_list", []) if isinstance(annotation, dict) else []

    dialogue_history = []
    total_turns = len(dialogue_questions)

    # All prior turns with their answers
    for i in range(max(0, total_turns - 1)):
        ans = dialogue_answers[i] if i < len(dialogue_answers) else "[Answered]"
        dialogue_history.append(f"User: {dialogue_questions[i]}\nAssistant: {ans}")

    final_question = dialogue_questions[-1] if dialogue_questions else "[No question]"

    history_str = "\n".join(dialogue_history) if dialogue_history else "(Beginning of conversation)"

    context_parts = []
    if pre_str:
        context_parts.append(f"Report Excerpt:\n{pre_str}")
    if table_str:
        context_parts.append(f"Financial Table:\n{table_str}")

    context_str = "\n\n".join(context_parts)

    prompt = (
        "You are an interactive financial analyst participating in a multi-turn dialogue based on the following financial report.\n\n"
        f"{context_str}\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Current Turn Question: {final_question}\n\n"
        "Answer the current turn question directly based on the table, report context, and preceding dialogue."
    )
    if print_if_first:
        print("\n" + "=" * 60)
        print("PROMPT #1 [ConvFinQA]:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    return prompt


# Dispatcher dictionary mapping dataset name to prompt builder
PROMPT_BUILDERS = {
    "banking77": build_banking77_prompt,
    "financial_phrasebank": build_phrasebank_prompt,
    "financebench": build_financebench_prompt,
    "finqa": build_finqa_prompt,
    "convfinqa": build_convfinqa_prompt,
}


def build_prompt_for_dataset(dataset_name: str, item: Dict[str, Any], is_first: bool = False) -> str:
    """Builds a prompt using the appropriate dataset prompt template."""
    builder = PROMPT_BUILDERS.get(dataset_name)
    if not builder:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Must be one of: {list(PROMPT_BUILDERS.keys())}")
    return builder(item, print_if_first=is_first)


if __name__ == "__main__":
    from load_samples import load_all_samples

    samples = load_all_samples(n=1)
    print("\nEyeballing Item #1 prompts for all 5 datasets:")
    for ds_name, items in samples.items():
        if items:
            build_prompt_for_dataset(ds_name, items[0], is_first=True)
