import os
import json
import random
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")

PROJECT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_DIR / "results"

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/models', methods=['GET'])
def get_models():
    models = set()
    if RESULTS_DIR.exists():
        # Check subdirectories
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and (d / "aggregates.json").exists():
                models.add(d.name)
        # Check flat files
        for f in RESULTS_DIR.glob("*_aggregates.json"):
            model_name = f.name[:-len("_aggregates.json")]
            models.add(model_name)
    return jsonify({"models": sorted(list(models))})

@app.route('/api/master-table', methods=['GET'])
def get_master_table():
    master_csv = RESULTS_DIR / "master_table.csv"
    if master_csv.exists():
        import pandas as pd
        df = pd.read_csv(master_csv)
        if len(df) > 1:
            return jsonify(df.to_dict(orient="records"))
    
    # Consolidate from all aggregates
    rows = []
    if RESULTS_DIR.exists():
        # Subdirectories
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and (d / "aggregates.json").exists():
                with open(d / "aggregates.json", "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
        # Flat files
        for f in sorted(RESULTS_DIR.glob("*_aggregates.json")):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                rows.append({
                    "Model": data.get("model", f.name[:-len("_aggregates.json")]),
                    "Exact Match": data.get("exact_match", 0.0),
                    "F1": data.get("f1", 0.0),
                    "BLEU": data.get("bleu", 0.0),
                    "ROUGE": data.get("rouge", 0.0),
                    "Judge Score": data.get("judge_score", 0.0),
                    "Completeness": data.get("completeness", 0.0),
                    "Helpfulness": data.get("helpfulness", 0.0),
                    "Reasoning Faithfulness": data.get("reasoning_faithfulness", 0.0),
                    "BERTScore": data.get("bert_score", 0.0),
                    "Hallucination Rate": data.get("hallucination_rate", 0.0),
                    "Failure Rate": data.get("failure_rate", 0.0),
                    "Latency (s)": data.get("latency", 0.0),
                    "Throughput (req/s)": data.get("throughput", 0.0),
                    "Tokens/sec": data.get("tokens_per_sec", 0.0),
                    "Category Accuracy": data.get("category_accuracy", 0.0)
                })
    return jsonify(rows)

@app.route('/api/category-accuracy', methods=['GET'])
def get_category_accuracy():
    cat_csv = RESULTS_DIR / "category_accuracy.csv"
    if cat_csv.exists():
        import pandas as pd
        df = pd.read_csv(cat_csv)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

def find_pred_file(model_name):
    # Try directory
    d_file = RESULTS_DIR / model_name / "predictions.jsonl"
    if d_file.exists():
        return d_file
    # Try flat file exact match
    f_file = RESULTS_DIR / f"{model_name}_predictions.jsonl"
    if f_file.exists():
        return f_file
    # Try case-insensitive / partial match
    for f in RESULTS_DIR.glob("*_predictions.jsonl"):
        clean_name = f.name[:-len("_predictions.jsonl")]
        if clean_name.lower() == model_name.lower() or model_name.lower() in clean_name.lower():
            return f
    # Fallback to any predictions file
    any_preds = list(RESULTS_DIR.glob("*predictions*.jsonl"))
    if any_preds:
        return any_preds[0]
    return None

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    model = request.args.get('model', '')
    search_id = request.args.get('id', '')
    limit = int(request.args.get('limit', '500'))
    
    pred_file = find_pred_file(model) if model else None
    if not pred_file:
        any_preds = list(RESULTS_DIR.glob("*_predictions.jsonl"))
        if any_preds:
            pred_file = any_preds[0]
            
    results = []
    if pred_file and pred_file.exists():
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                if search_id:
                    if search_id.lower() in item.get('id', '').lower():
                        results.append(item)
                else:
                    results.append(item)
                if len(results) >= limit:
                    break
                    
    model_ret = pred_file.name.replace("_predictions.jsonl", "") if pred_file else model
    return jsonify({"model": model_ret, "count": len(results), "predictions": results})

@app.route('/api/sample/<sample_id>', methods=['GET'])
def get_sample_comparison(sample_id):
    sample_data = {
        "id": sample_id,
        "input_text": "",
        "context": "",
        "target": "",
        "metadata": {},
        "models": {}
    }
    
    found_info = False
    if RESULTS_DIR.exists():
        # Check subdirectories
        pred_sources = []
        for d in RESULTS_DIR.iterdir():
            if d.is_dir():
                p = d / "predictions.jsonl"
                if p.exists():
                    pred_sources.append((d.name, p))
        # Check flat files
        for f in RESULTS_DIR.glob("*_predictions.jsonl"):
            m_name = f.name[:-len("_predictions.jsonl")]
            pred_sources.append((m_name, f))
            
        for m_name, pfile in pred_sources:
            try:
                with open(pfile, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line.strip())
                        if item.get("id") == sample_id or item.get("sample_id") == sample_id:
                            if not found_info:
                                sample_data["input_text"] = item.get("input_text", "")
                                sample_data["context"] = item.get("context", "")
                                sample_data["target"] = item.get("target", "")
                                sample_data["metadata"] = item.get("metadata", {})
                                sample_data["prompt"] = item.get("prompt", "")
                                found_info = True
                            sample_data["models"][m_name] = {
                                "output_text": item.get("output_text", ""),
                                "exact_match": item.get("exact_match", 0),
                                "f1": item.get("f1", 0),
                                "bleu": item.get("bleu", 0),
                                "rouge": item.get("rouge", 0),
                                "judge_score": item.get("judge_score", 0),
                                "completeness": item.get("completeness", 0),
                                "helpfulness": item.get("helpfulness", 0),
                                "bert_score": item.get("bert_score", 0),
                                "reasoning_faithfulness": item.get("reasoning_faithfulness", 0),
                                "latency_seconds": item.get("latency_seconds", 0),
                                "tokens_out": item.get("tokens_out", 0),
                                "is_hallucinated": item.get("is_hallucinated", False),
                                "is_failure": item.get("is_failure", False)
                            }
                            break
            except Exception as e:
                print(f"Error reading {pfile}: {e}")
                        
    return jsonify(sample_data)

@app.route('/api/random-id', methods=['GET'])
def get_random_id():
    ids = []
    # Pick from any available predictions file
    for pfile in RESULTS_DIR.glob("*_predictions.jsonl"):
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        sid = item.get("id") or item.get("sample_id")
                        if sid and sid not in ids:
                            ids.append(sid)
            if ids:
                break
        except Exception:
            pass
            
    if ids:
        return jsonify({"id": random.choice(ids), "all_ids": ids})
    return jsonify({"id": "item_001", "all_ids": ["item_001"]})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    dist_dir = PROJECT_DIR / "frontend" / "dist"
    if dist_dir.exists() and (dist_dir / path).exists() and path != "":
        return send_from_directory(dist_dir, path)
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        return send_from_directory(dist_dir, "index.html")
    return jsonify({"status": "Backend running. Build frontend to view interface at /."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
