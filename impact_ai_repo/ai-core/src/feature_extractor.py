#!/usr/bin/env python3
"""
feature_extractor.py

Extract features from curated ACE NDJSON files.
Produces features.csv and features.parquet.

Usage:
    python feature_extractor.py \
        --curated-dir ai-core/src/datasets/curated \
        --out-dir ai-core/src/analysis_outputs
"""

from __future__ import annotations
import argparse
import os
import json
import hashlib
import glob
import pandas as pd


def stable_id(pair_id: str, ace: dict) -> str:
    key = f"{pair_id}|{ace.get('type','')}|{ace.get('path','')}|{ace.get('method','')}|{ace.get('detail','')}"
    return "ACE_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def extract_one(ace: dict) -> dict:
    row = {}

    pair_id = ace.get("pair_id") or ace.get("pair") or ace.get("pairId") or None
    ace_id = ace.get("ace_id") or ace.get("id") or ace.get("aceId")

    if not ace_id:
        ace_id = stable_id(pair_id or "nopair", ace)

    row["ace_id"] = ace_id
    row["pair_id"] = pair_id
    row["type"] = ace.get("type") or ace.get("change_type") or "unknown"
    row["method"] = (ace.get("method") or "").lower()

    path = ace.get("path") or ""
    if not path and isinstance(ace.get("detail",""), str):
        parts = ace.get("detail","").split(" ")
        if len(parts) > 1:
            path = parts[1]

    row["path"] = path
    row["path_depth"] = path.count("/") if isinstance(path, str) else 0

    row["breaking_change"] = 1 if ace.get("breaking_change") or ace.get("breaking") else 0
    row["shared_schemas_count"] = len(ace.get("shared_schemas", []))
    row["calls_services_count"] = len(ace.get("calls_services", []))
    row["has_request_body"] = 1 if ace.get("request") else 0
    row["has_response_body"] = 1 if ace.get("responses") else 0

    detail = ace.get("detail", "") or ""
    row["detail_len"] = len(detail)
    row["is_add"] = 1 if "add" in detail.lower() or "added" in detail.lower() else 0
    row["is_remove"] = 1 if "remove" in detail.lower() or "removed" in detail.lower() else 0

    row["detail"] = detail
    row["raw_type"] = ace.get("type")

    return row


def load_ndjson_files(curated_dir: str):
    pattern = os.path.join(curated_dir, "ndjson", "*.ndjson")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    try:
                        yield json.loads(line.rstrip(","))
                    except Exception:
                        continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for ace in load_ndjson_files(args.curated_dir):
        try:
            rows.append(extract_one(ace))
        except Exception:
            continue

    if not rows:
        raise SystemExit("No ACEs found in curated dataset.")

    df = pd.DataFrame(rows)

    keep_cols = [
        "ace_id","pair_id","type","method","path","path_depth",
        "breaking_change","shared_schemas_count","calls_services_count",
        "has_request_body","has_response_body","detail_len",
        "is_add","is_remove","detail","raw_type"
    ]

    df = df.reindex(columns=[c for c in keep_cols if c in df.columns])

    csv_path = os.path.join(args.out_dir, "features.csv")
    parquet_path = os.path.join(args.out_dir, "features.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"Extracted {len(df)} ACEs → features.csv")
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()
