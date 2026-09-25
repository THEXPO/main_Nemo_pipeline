"""
smoke_test.py
End-to-end smoke test for BFSI LLM benchmarking pipeline across all 5 datasets.
Connects to an LM Studio local server, processes 5 items per dataset,
and writes comprehensive results to smoke_test_results.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from load_samples import (
    load_banking77,
    load_convfinqa,
    load_financebench,
    load_financial_phrasebank,
    load_finqa,
)
from parsers import parse_banking77_output
from prompt_templates import build_prompt_for_dataset
from run_model import MODEL_NAME, check_lm_studio_server, generate


DATASET_LOADERS = [
    ("banking77", "Banking77 (Customer Intent Classification)", load_banking77),
    ("financial_phrasebank", "Financial PhraseBank (Sentiment Analysis)", load_financial_phrasebank),
    ("financebench", "FinanceBench (SEC Filing Question Answering)", load_financebench),
    ("finqa", "FinQA (Numerical Reasoning & Program Synthesis)", load_finqa),
    ("convfinqa", "ConvFinQA (Conversational Financial QA)", load_convfinqa),
]


def run_smoke_test(
    data_dir: str = "data/raw",
    model_name: str = MODEL_NAME,
    samples_per_dataset: int = 5,
    output_json_path: str = "smoke_test_results.json",
) -> None:
    print("\n" + "=" * 75)
    print("      BFSI LLM BENCHMARKING PIPELINE: END-TO-END SMOKE TEST")
    print("=" * 75)

    # 1. Check LM Studio server connectivity
    print(f"\n[Step 1/3] Pinging LM Studio local server at http://localhost:1234/v1/models...")
    is_connected, status_msg, available_models = check_lm_studio_server()

    if not is_connected:
        print("\n" + "!" * 75)
        print("[!] WARNING: LM Studio local server is currently UNREACHABLE.")
        print(f"    Details: {status_msg}")
        print("\n    To run inference:")
        print("    1. Open LM Studio on your computer.")
        print("    2. Load your desired model (e.g. Mistral, Qwen, Llama).")
        print("    3. Click the '<->' Local Server icon in LM Studio and hit 'Start Server'.")
        print("    4. Ensure it is serving at port 1234 (http://localhost:1234/v1).")
        print("!" * 75 + "\n")
        print("Proceeding in plumbing test mode (pipeline validation & error recovery check)...\n")
    else:
        print(f"[OK] LM Studio server is ACTIVE!")
        if available_models:
            print(f"     Loaded models: {', '.join(available_models)}")
            if model_name not in available_models and available_models:
                print(f"     [Note] Configured MODEL_NAME is '{model_name}'. Active model in LM Studio is '{available_models[0]}'.")
                print(f"            Auto-selecting active model: '{available_models[0]}'")
                model_name = available_models[0]
        else:
            print(f"     No model currently loaded in LM Studio. Using target identifier: '{model_name}'")

    print(f"Target Model: '{model_name}'")
    print(f"Input Directory: '{data_dir}' (5 samples per dataset)\n")

    all_dataset_results = {}
    overall_summary = []

    # 2. Iterate through all 5 datasets
    for key, display_name, loader_fn in DATASET_LOADERS:
        print("-" * 75)
        print(f"TESTING DATASET: {display_name}")
        print("-" * 75)

        # Load samples
        items = loader_fn(data_dir=data_dir, n=samples_per_dataset)
        if not items:
            print(f"[SKIP] No items loaded for {key}. Skipping dataset.\n")
            overall_summary.append({
                "dataset": key,
                "display_name": display_name,
                "loaded": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": "0.0%",
                "avg_latency": 0.0,
                "avg_tokens": 0.0,
            })
            continue

        dataset_runs = []
        errors = []
        latencies = []
        token_counts = []

        item_1_prompt = ""
        item_1_output = ""
        item_1_error = None
        item_1_parsed_label = None

        for idx, item in enumerate(items):
            is_first = (idx == 0)
            try:
                prompt = build_prompt_for_dataset(key, item, is_first=False)
            except Exception as pe:
                err_str = f"Prompt building failed for item {idx + 1}: {pe}"
                errors.append(err_str)
                dataset_runs.append({
                    "item_index": idx + 1,
                    "prompt": "",
                    "output_text": "",
                    "latency_seconds": 0.0,
                    "tokens_out": 0,
                    "error": err_str,
                })
                continue

            # Model generation call
            res = generate(prompt=prompt, model_name=model_name)
            output_text = res.get("output_text", "")
            latency = res.get("latency_seconds", 0.0)
            tokens = res.get("tokens_out", 0)
            err = res.get("error")

            latencies.append(latency)
            if err:
                errors.append(err)
            else:
                token_counts.append(tokens)

            # Banking77 specific output parsing and comparison
            parsed_label = None
            if key == "banking77":
                gold_label = item.get("label", "")
                if not err:
                    parsed_label = parse_banking77_output(output_text)
                print(f"  [Item {idx + 1}] Gold: {gold_label!r:<25} | Parsed: {parsed_label!r:<25} | Raw Output: {output_text!r}")

            record = {
                "item_index": idx + 1,
                "prompt": prompt,
                "output_text": output_text,
                "latency_seconds": latency,
                "tokens_out": tokens,
                "error": err,
                "raw_item": item,
            }
            if key == "banking77":
                record["gold_label"] = item.get("label")
                record["parsed_label"] = parsed_label

            dataset_runs.append(record)

            if is_first:
                item_1_prompt = prompt
                item_1_output = output_text
                item_1_error = err
                if key == "banking77":
                    item_1_parsed_label = parsed_label

        success_count = len(items) - len(errors)
        avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        avg_tokens = round(sum(token_counts) / len(token_counts), 1) if token_counts else 0.0
        success_rate = f"{(success_count / len(items)) * 100:.1f}%"

        all_dataset_results[key] = {
            "display_name": display_name,
            "samples_evaluated": len(items),
            "successful_generations": success_count,
            "errors": errors,
            "avg_latency_seconds": avg_latency,
            "avg_tokens_out": avg_tokens,
            "runs": dataset_runs,
        }

        overall_summary.append({
            "dataset": key,
            "display_name": display_name,
            "loaded": len(items),
            "success_count": success_count,
            "error_count": len(errors),
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
        })

        # Summary block after each dataset
        print("\n>>> DATASET SUMMARY BLOCK:")
        print(f"    Dataset Name:              {display_name}")
        print(f"    Successful Generations:    {success_count} / {len(items)}")
        print(f"    Errors:                    {len(errors)}")
        if errors:
            unique_errors = list(set(errors))
            print(f"    Error Details:             {unique_errors[0]}")
        print(f"    Average Latency:           {avg_latency:.3f} s")
        print("\n--- ITEM #1 INSPECTION ---")
        print(f"[Full Prompt for Item #1]:\n{item_1_prompt}")
        print("-" * 40)
        if item_1_error:
            print(f"[Raw Model Output for Item #1]: (Failed with error: {item_1_error})")
        else:
            print(f"[Raw Model Output for Item #1]:\n{item_1_output if item_1_output else '(Empty response)'}")
        if key == "banking77":
            print("-" * 40)
            print(f"[Gold Label for Item #1]:   {items[0].get('label')}")
            print(f"[Parsed Label for Item #1]: {item_1_parsed_label}")
        print("-" * 75 + "\n")

    # 3. Final Summary Table
    print("\n" + "=" * 85)
    print("                    FINAL PIPELINE SMOKE TEST SUMMARY")
    print("=" * 85)
    print(f"{'Dataset':<24} | {'Success Rate':<14} | {'Avg Latency (s)':<17} | {'Avg Tokens Out':<14}")
    print("-" * 85)
    for row in overall_summary:
        print(
            f"{row['dataset']:<24} | "
            f"{row['success_count']}/{row['loaded']} ({row['success_rate']}){'':<4} | "
            f"{row['avg_latency']:<17.3f} | "
            f"{row['avg_tokens']:<14.1f}"
        )
    print("=" * 85)

    # 4. Save results to JSON
    out_path = Path(output_json_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_model": model_name,
                "lm_studio_connected": is_connected,
                "summary": overall_summary,
                "datasets": all_dataset_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[Saved] All raw prompts, outputs, latencies, and metadata saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BFSI LLM smoke test on 5 datasets via LM Studio.")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw datasets folder (default: data/raw)")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Model identifier in LM Studio (default: {MODEL_NAME})")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per dataset (default: 5)")
    parser.add_argument("--out", default="smoke_test_results.json", help="Output JSON path (default: smoke_test_results.json)")
    args = parser.parse_args()

    run_smoke_test(
        data_dir=args.data_dir,
        model_name=args.model,
        samples_per_dataset=args.n,
        output_json_path=args.out,
    )
