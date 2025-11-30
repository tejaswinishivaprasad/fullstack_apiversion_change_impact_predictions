#!/usr/bin/env python3
"""
feature_extractor.py - patched

Usage:
    python feature_extractor.py \
        --curated-dir ai-core/src/datasets/curated \
        --out-dir ai-core/src/analysis_outputs \
        [--sample N] [--verbose]

Produces:
    - features.csv
    - features.parquet
    - feature_stats.json (row counts, nan counts, min/max, sha1)
"""
from __future__ import annotations
import argparse
import os
import json
import hashlib
import glob
import pandas as pd
import sys
from typing import Iterator, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("feature_extractor")


def stable_id(pair_id: str, ace: dict) -> str:
    key = f"{pair_id}|{ace.get('type','')}|{ace.get('path','')}|{ace.get('method','')}|{ace.get('detail','')}"
    return "ACE_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def normalize_str(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            x = str(x)
        except Exception:
            return ""
    return x.strip()


def truthy(val):
    """Normalize many truthy representations to boolean True/False"""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = normalize_str(val).lower()
    return s in ("1", "true", "yes", "y", "t")


def extract_one(ace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a stable, cleaned feature dict from a single ACE (ndjson object).
    This version is robust to multiple ACE schemas: it checks several common
    key names for breaking flags, request/response bodies and provenance.
    """
    row: Dict[str, Any] = {}

    # canonical ids
    pair_id = ace.get("pair_id") or ace.get("pair") or ace.get("pairId") or None
    pair_id = normalize_str(pair_id) if pair_id else None

    ace_id = ace.get("ace_id") or ace.get("id") or ace.get("aceId")
    if not ace_id:
        ace_id = stable_id(pair_id or "nopair", ace)
    ace_id = normalize_str(ace_id)

    row["ace_id"] = ace_id
    row["pair_id"] = pair_id

    # types and raw_type
    raw_type = ace.get("type") or ace.get("change_type") or "unknown"
    row["raw_type"] = normalize_str(raw_type)
    row["type"] = row["raw_type"].lower()

    # method and path
    method = ace.get("method") or ace.get("http_method") or ""
    row["method"] = normalize_str(method).lower()

    path = ace.get("path") or ace.get("endpoint") or ""
    if not path and isinstance(ace.get("detail", ""), str):
        parts = ace.get("detail", "").split(" ")
        if len(parts) > 1 and parts[1].startswith("/"):
            path = parts[1]
    row["path"] = normalize_str(path)
    row["path_depth"] = sum(1 for p in row["path"].split("/") if p.strip()) if row["path"] else 0

    # --- Breaking change detection: check multiple possible keys ---
    breaking_keys = ["breaking_change", "breaking", "is_breaking", "breakingChange", "isBreaking"]
    row["breaking_change"] = 1 if any(truthy(ace.get(k)) for k in breaking_keys) else 0

    # --- shared schemas and calls/services counts ---
    # allow both list and dict forms (dict -> count keys)
    shared = ace.get("shared_schemas") or ace.get("sharedSchemaRefs") or ace.get("shared")
    if isinstance(shared, dict):
        shared_count = len(shared.keys())
    elif isinstance(shared, list):
        shared_count = len(shared)
    else:
        shared_count = 0
    row["shared_schemas_count"] = int(shared_count)

    calls = ace.get("calls_services") or ace.get("consumers") or ace.get("calls")
    if isinstance(calls, dict):
        calls_count = len(calls.keys())
    elif isinstance(calls, list):
        calls_count = len(calls)
    else:
        calls_count = 0
    row["calls_services_count"] = int(calls_count)

    # --- Request / response body detection: cover common variants ---
    # Many ACEs use "request", "requestBody", or nested objects; same for responses.
    has_request = False
    if "request" in ace and ace.get("request"):
        has_request = True
    elif "requestBody" in ace and ace.get("requestBody"):
        has_request = True
    else:
        # sometimes detail contains markers like "requestBody" or "body"
        det = normalize_str(ace.get("detail", "") or "")
        if "requestbody" in det.lower() or "request body" in det.lower() or "body" in det.lower():
            has_request = True

    has_response = False
    if "responses" in ace and ace.get("responses"):
        has_response = True
    elif "response" in ace and ace.get("response"):
        has_response = True
    else:
        det = normalize_str(ace.get("detail", "") or "")
        if "responses" in det.lower() or "response body" in det.lower():
            has_response = True

    row["has_request_body"] = 1 if has_request else 0
    row["has_response_body"] = 1 if has_response else 0

    # detail text and derived flags
    detail = normalize_str(ace.get("detail", "") or "")
    row["detail"] = detail
    row["detail_len"] = len(detail)
    low_detail = detail.lower()
    row["is_add"] = 1 if ("add" in low_detail or "added" in low_detail or "adding" in low_detail) else 0
    row["is_remove"] = 1 if ("remove" in low_detail or "removed" in low_detail or "removing" in low_detail) else 0

    # provenance / versions (optional, helpful)
    row["project"] = normalize_str(ace.get("project") or ace.get("repo") or ace.get("source") or "")
    row["old_version"] = normalize_str(ace.get("old_version") or ace.get("v1") or ace.get("from_version") or "")
    row["new_version"] = normalize_str(ace.get("new_version") or ace.get("v2") or ace.get("to_version") or "")

    return row



def load_ndjson_files(curated_dir: str) -> Iterator[Dict[str, Any]]:
    # recursively find any .ndjson file under curated_dir
    pattern = os.path.join(curated_dir, "**", "*.ndjson")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        logger.warning("No ndjson files found with pattern: %s", pattern)
    for path in paths:
        logger.info("Reading NDJSON file: %s", path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    try:
                        yield json.loads(line.rstrip(","))
                    except Exception:
                        logger.debug("Skipping line %d in %s - invalid JSON", i + 1, path)
                        continue


def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_feature_stats(df: pd.DataFrame, out_dir: str, csv_path: str):
    stats = {}
    stats["rows"] = int(len(df))
    stats["columns"] = list(df.columns)
    stats["nan_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats["numeric_summary"] = {}
    for c in numeric_cols:
        stats["numeric_summary"][c] = {
            "min": None if df[c].dropna().empty else float(df[c].min()),
            "max": None if df[c].dropna().empty else float(df[c].max())
        }
    try:
        stats["features_csv_sha1"] = sha1_of_file(csv_path)
    except Exception:
        stats["features_csv_sha1"] = None

    stats_path = os.path.join(out_dir, "feature_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Wrote feature stats to %s", stats_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", required=True, help="Root curated dataset dir (contains ndjson/)")
    parser.add_argument("--out-dir", required=True, help="Where to write features.csv and statistics")
    parser.add_argument("--sample", type=int, default=0, help="Optional sample size for quick runs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    curated = args.curated_dir
    out_dir = args.out_dir
    sample_n = int(args.sample or 0)

    if not os.path.isdir(curated):
        logger.error("Curated directory does not exist: %s", curated)
        sys.exit(2)

    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total_read = 0
    for ace in load_ndjson_files(curated):
        total_read += 1
        try:
            rows.append(extract_one(ace))
        except Exception as e:
            logger.debug("Failed to extract ACE: %s", e)
            continue
        if sample_n and len(rows) >= sample_n:
            break

    if not rows:
        logger.error("No ACEs found in curated dataset. Checked path: %s", os.path.join(curated, "ndjson", "*.ndjson"))
        sys.exit(3)

    df = pd.DataFrame(rows)

    # Deduplicate by ace_id, keep first occurrence (stable)
    before = len(df)
    df = df.drop_duplicates(subset=["ace_id"], keep="first")
    after = len(df)
    if before != after:
        logger.warning("Dropped %d duplicate ACEs (by ace_id).", before - after)

    # Deterministic ordering
    sort_cols = []
    if "pair_id" in df.columns:
        sort_cols.append("pair_id")
    if "ace_id" in df.columns:
        sort_cols.append("ace_id")
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Ensure expected columns and types
    keep_cols = [
        "ace_id", "pair_id", "project", "old_version", "new_version",
        "type", "raw_type", "method", "path", "path_depth",
        "breaking_change", "shared_schemas_count", "calls_services_count",
        "has_request_body", "has_response_body", "detail_len",
        "is_add", "is_remove", "detail"
    ]
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df.reindex(columns=cols_present)

    # Fill numeric NaNs with 0 and cast ints
    for c in ["path_depth", "breaking_change", "shared_schemas_count", "calls_services_count",
              "has_request_body", "has_response_body", "detail_len", "is_add", "is_remove"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Normalize detail to str
    if "detail" in df.columns:
        df["detail"] = df["detail"].fillna("").astype(str)

    csv_path = os.path.join(out_dir, "features.csv")
    parquet_path = os.path.join(out_dir, "features.parquet")

    # Save CSV (no index)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logger.info("Extracted %d ACEs (read %d lines) → features.csv", len(df), total_read)
    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved Parquet: %s", parquet_path)

    # Write feature stats and checksum
    try:
        write_feature_stats(df, out_dir, csv_path)
    except Exception as e:
        logger.warning("Failed to write feature stats: %s", e)

    logger.info("Done.")

if __name__ == "__main__":
    main()

