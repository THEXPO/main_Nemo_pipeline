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
    models = []
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and (d / "aggregates.json").exists():
                models.append(d.name)
    return jsonify({"models": sorted(models)})

@app.route('/api/master-table', methods=['GET'])
def get_master_table():
    master_csv = RESULTS_DIR / "master_table.csv"
    if master_csv.exists():
        import pandas as pd
        df = pd.read_csv(master_csv)
        return jsonify(df.to_dict(orient="records"))
    
    # Fallback by loading aggregates.json from directories
    rows = []
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and (d / "aggregates.json").exists():
                with open(d / "aggregates.json", "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
    return jsonify(rows)

@app.route('/api/category-accuracy', methods=['GET'])
def get_category_accuracy():
    cat_csv = RESULTS_DIR / "category_accuracy.csv"
    if cat_csv.exists():
        import pandas as pd
        df = pd.read_csv(cat_csv)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    model = request.args.get('model', 'llama3.1-8b')
    search_id = request.args.get('id', '')
    limit = int(request.args.get('limit', '500'))
    
    model_dir = RESULTS_DIR / model
    if not model_dir.exists():
        # Fallback to first available directory
        dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists()]
        if dirs:
            model_dir = dirs[0]
            
    pred_file = model_dir / "predictions.jsonl"
    if not pred_file.exists():
        # Check deepseek old file as fallback
        old_file = model_dir / "predictions_240_old.jsonl"
        if old_file.exists():
            pred_file = old_file
            
    results = []
    if pred_file.exists():
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
                    
    return jsonify({"model": model_dir.name, "count": len(results), "predictions": results})

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
        for d in RESULTS_DIR.iterdir():
            if not d.is_dir():
                continue
            pfile = d / "predictions.jsonl"
            if not pfile.exists():
                pfile = d / "predictions_240_old.jsonl"
            if not pfile.exists():
                continue
                
            with open(pfile, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line.strip())
                    if item.get("id") == sample_id:
                        if not found_info:
                            sample_data["input_text"] = item.get("input_text", "")
                            sample_data["context"] = item.get("context", "")
                            sample_data["target"] = item.get("target", "")
                            sample_data["metadata"] = item.get("metadata", {})
                            sample_data["prompt"] = item.get("prompt", "")
                            found_info = True
                        sample_data["models"][d.name] = {
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
                        
    return jsonify(sample_data)

@app.route('/api/random-id', methods=['GET'])
def get_random_id():
    ids = []
    pfile = RESULTS_DIR / "phi4-mini" / "predictions.jsonl"
    if not pfile.exists():
        pfile = RESULTS_DIR / "llama3.1-8b" / "predictions.jsonl"
        
    if pfile.exists():
        with open(pfile, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    ids.append(item.get("id"))
    if ids:
        return jsonify({"id": random.choice(ids), "all_ids": ids})
    return jsonify({"id": "synthetic_bfsi_000284", "all_ids": ["synthetic_bfsi_000284"]})

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
