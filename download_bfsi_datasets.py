#!/usr/bin/env python3
"""
Download 5 BFSI benchmark datasets into ./data/raw/

  1. Banking77            (GitHub: PolyAI-LDN/task-specific-datasets)  intent classification
  2. Financial PhraseBank (HF: takala/financial_phrasebank, w/ fallback) sentiment
  3. FinanceBench         (HF: PatronusAI/financebench)                doc QA / retrieval
  4. FinQA                (GitHub: czyssrs/FinQA)                      numeric reasoning
  5. ConvFinQA            (GitHub: czyssrs/ConvFinQA)                  multi-turn numeric reasoning

Usage:
    pip install datasets huggingface_hub requests
    python download_bfsi_datasets.py                 # download everything
    python download_bfsi_datasets.py --only banking77 financial_phrasebank
    python download_bfsi_datasets.py --out my_data/raw

A manifest.json in the output folder records what was downloaded (for reproducibility).
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

GITHUB_RAW = "https://raw.githubusercontent.com/czyssrs"
BANKING77_RAW = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def http_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                f.write(part)
    print(f"    saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"    saved {path} ({n} rows)")
    return n


def save_hf_dataset_dict(dsd, out_dir: Path) -> dict:
    """Write every split of a HF DatasetDict to <out_dir>/<split>.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, ds in dsd.items():
        path = out_dir / f"{split}.jsonl"
        ds.to_json(str(path), lines=True, force_ascii=False)
        counts[split] = len(ds)
        print(f"    saved {path} ({len(ds)} rows)")
    return counts


# --------------------------------------------------------------------------- #
# downloaders (each returns a dict of info for the manifest)
# --------------------------------------------------------------------------- #
def get_banking77(out: Path) -> dict:
    """Original PolyAI CSVs from GitHub (the HF version is a loading script, which
    datasets>=4 can't run). Saved as JSONL: {text, label (str)}."""
    d = out / "banking77"
    counts = {}
    for split in ["train", "test"]:
        raw = d / f"{split}.csv"
        http_download(f"{BANKING77_RAW}/{split}.csv", raw)
        with open(raw, newline="", encoding="utf-8") as f:
            rows = [{"text": r["text"], "label": r["category"]} for r in csv.DictReader(f)]
        counts[split] = write_jsonl(rows, d / f"{split}.jsonl")
    http_download(f"{BANKING77_RAW}/categories.json", d / "categories.json")
    return {"source": "github:PolyAI-LDN/task-specific-datasets", "splits": counts}


FPB_LABELS = {"negative": 0, "neutral": 1, "positive": 2}  # same ids as the HF version
FPB_FILES = {  # config name -> file in the original release
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}


def _fpb_from_raw_txt(out: Path) -> dict:
    """Fallback: original release files (format 'sentence@label', latin-1) hosted
    in a public HF dataset repo that bundles FinancialPhraseBank-v1.0."""
    from huggingface_hub import hf_hub_download

    info = {}
    for cfg, fname in FPB_FILES.items():
        p = hf_hub_download("datht/FINDSum", f"FinancialPhraseBank-v1.0/{fname}", repo_type="dataset")
        rows = []
        with open(p, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line or "@" not in line:
                    continue
                sent, lab = line.rsplit("@", 1)
                rows.append({"sentence": sent.strip(), "label": FPB_LABELS[lab.strip()], "label_text": lab.strip()})
        info[cfg] = {"route": "raw-txt:datht/FINDSum", "splits": {"train": write_jsonl(rows, out / "financial_phrasebank" / cfg / "train.jsonl")}}
    return info


def get_financial_phrasebank(out: Path) -> dict:
    """
    takala/financial_phrasebank is still a loading-script dataset on the Hub main
    branch, so `datasets>=4` refuses it. Routes tried in order:
      1. PR #10 on the Hub (parquet conversion, revision refs/pr/10)
      2. original raw .txt files from a public mirror repo
    License: CC BY-NC-SA 3.0 (non-commercial).
    """
    from datasets import load_dataset

    info = {"source": "hf:takala/financial_phrasebank", "configs": {}}
    try:
        for cfg in FPB_FILES:
            dsd = load_dataset("takala/financial_phrasebank", cfg, revision="refs/pr/10")
            info["configs"][cfg] = {
                "route": "hf-pr-10-parquet",
                "splits": save_hf_dataset_dict(dsd, out / "financial_phrasebank" / cfg),
            }
        return info
    except Exception as e:
        print(f"    parquet PR route failed ({type(e).__name__}: {str(e)[:100]}); using raw-txt fallback")

    info["configs"] = _fpb_from_raw_txt(out)
    return info


def get_financebench(out: Path) -> dict:
    from datasets import load_dataset

    dsd = load_dataset("PatronusAI/financebench")
    return {"source": "hf:PatronusAI/financebench", "splits": save_hf_dataset_dict(dsd, out / "financebench")}


def get_finqa(out: Path) -> dict:
    d = out / "finqa"
    files = ["train.json", "dev.json", "test.json"]
    for fn in files:
        http_download(f"{GITHUB_RAW}/FinQA/main/dataset/{fn}", d / fn)
    counts = {fn.replace(".json", ""): len(json.loads((d / fn).read_text(encoding="utf-8"))) for fn in files}
    print(f"    example counts: {counts}")
    return {"source": "github:czyssrs/FinQA", "splits": counts}


def get_convfinqa(out: Path) -> dict:
    d = out / "convfinqa"
    zip_path = d / "data.zip"
    http_download(f"{GITHUB_RAW}/ConvFinQA/main/data.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        z.extractall(d)
    print(f"    extracted {len(names)} entries into {d}")
    return {"source": "github:czyssrs/ConvFinQA", "files": [n for n in names if not n.endswith("/")][:20]}


DATASETS = {
    "banking77": get_banking77,
    "financial_phrasebank": get_financial_phrasebank,
    "financebench": get_financebench,
    "finqa": get_finqa,
    "convfinqa": get_convfinqa,
}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Download BFSI benchmark datasets")
    ap.add_argument("--out", default="data/raw", help="output directory (default: data/raw)")
    ap.add_argument("--only", nargs="+", choices=DATASETS.keys(), help="download only these datasets")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(DATASETS)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failures = []

    for name in selected:
        print(f"\n[{name}]")
        try:
            info = DATASETS[name](out)
            info["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            manifest[name] = info
            print("  OK")
        except Exception as e:
            failures.append(name)
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest written to {manifest_path}")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())