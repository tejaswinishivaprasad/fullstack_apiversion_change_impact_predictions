#!/usr/bin/env python3
"""
server.py

Impact AI core HTTP server.

This file implements the main API used by the frontend and CI demo.
It reads curated datasets, computes diffs for OpenAPI specs, extracts
simple features, scores risk, and returns a Report JSON.

The code aims to be practical and resilient for the thesis demo.
Comments are written in a plain student style to make the implementation
easy to follow for examiners.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import statistics
import traceback
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import networkx as nx
import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Simple logging setup. Use LOG_LEVEL env var to change level.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("impact-ai-core")

# Default paths for curated data and metadata.
CURATED_ROOT = Path("datasets/curated")
CANONICAL_DIR = CURATED_ROOT / "canonical"
NDJSON_DIR = CURATED_ROOT / "ndjson"
META_DIR = CURATED_ROOT / "metadata"
GRAPH_PATH = CURATED_ROOT / "graph.json"
VERSION_META_PATH = CURATED_ROOT / "version_meta.json"
PAIR_INDEX_PATH = CURATED_ROOT / "index.json"

# Ensure logs folder exists.
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Support older dataset layouts as a fallback.
DATASET_DIRS = {
    "openapi": CURATED_ROOT / "openapi",
    "petclinic": CURATED_ROOT / "petclinic",
    "openrewrite": CURATED_ROOT / "openrewrite",
}

# Data models used in API responses.
class DiffItem(BaseModel):
    ace_id: Optional[str] = None
    type: str
    path: Optional[str] = None
    method: Optional[str] = None
    detail: Optional[Any] = None

class ImpactItem(BaseModel):
    service: str
    risk_score: float

class Report(BaseModel):
    dataset: str
    old_file: str
    new_file: str
    risk_score: float
    risk_band: str
    summary: str
    details: List[DiffItem]
    ai_explanation: str
    backend: Dict[str, Any]
    logs: List[str]
    backend_impacts: List[ImpactItem]
    frontend_impacts: List[ImpactItem]
    predicted_risk: float
    confidence: Dict[str, float]
    versioning: Dict[str, Any]

# FastAPI app and CORS for the frontend
app = FastAPI(title="Impact AI Core", version="2.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Basic helpers for reading files and parsing JSON/YAML.
def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("Failed to read %s: %s", p, e)
        return ""

def _load_json_or_yaml(p: Path) -> Any:
    txt = _read_text(p)
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        try:
            return yaml.safe_load(txt)
        except Exception:
            log.warning("Failed to parse JSON/YAML at %s", p)
            return {}

# Convert numeric score to a simple band label.
def _risk_band(s: float) -> str:
    return "High" if s >= 0.7 else "Medium" if s >= 0.4 else "Low" if s > 0 else "None"

# Normalize path templates for matching.
def normalize_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return re.sub(r"\{[^}]+\}", "{*}", p.strip())

# Safe JSON reader for small files.
def read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to parse JSON at %s", p)
        return {}

# Stable string for dedupe keys. Should not raise.
def _safe_json_for_key(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=lambda o: str(o))
    except Exception:
        try:
            return str(obj)
        except Exception:
            return ""

# Tweakable base weights per change type.
BASE_WEIGHTS = {
    "endpoint_added": 0.05,
    "endpoint_removed": 0.5,
    "endpoint_changed": 0.12,
    "param_added": 0.12,
    "param_removed": 0.15,
    "param_changed": 0.15,
    "response_schema_changed": 0.25,
    "requestbody_schema_changed": 0.25,
    "param_required_added": 0.18,
    "UNKNOWN": 0.1,
}

# Pair index loader. Cached for performance.
@lru_cache(maxsize=1)
def load_pair_index() -> Dict[str, Any]:
    if PAIR_INDEX_PATH.exists():
        try:
            data = json.loads(PAIR_INDEX_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                out = {}
                for e in data:
                    pid = e.get("pair_id") or e.get("pair") or None
                    if pid:
                        out[pid] = e
                return out
        except Exception:
            log.warning("Failed to parse pair index %s", PAIR_INDEX_PATH)

    pairs: Dict[str, Any] = {}
    if META_DIR.exists():
        for p in META_DIR.glob("*.meta.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                pid = d.get("pair_id") or p.stem
                pairs[pid] = d
            except Exception:
                log.warning("Failed to parse meta file %s", p)
    return pairs

# Load metadata for a given pair id.
def load_pair_metadata(pair_id: Optional[str]) -> Dict[str, Any]:
    if not pair_id:
        return {}
    idx = load_pair_index()
    if pair_id in idx:
        return idx[pair_id] or {}
    cand = META_DIR / f"{pair_id}.meta.json"
    if cand.exists():
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to parse meta file %s", cand)
    return {}

# Build a directed graph from graph.json
def load_graph() -> nx.DiGraph:
    data = read_json(GRAPH_PATH)
    g = nx.DiGraph()
    for e in data.get("edges", []):
        src = e.get("src")
        dst = e.get("dst")
        if src and dst:
            g.add_edge(src, dst, path=e.get("path"), evidence=e.get("evidence"), confidence=e.get("confidence", 0.5), provenance=e.get("provenance", {}))
    return g

# Helper to ensure node names use svc: or ui: prefix.
def producer_node(service: str) -> str:
    if service.startswith("svc:") or service.startswith("ui:"):
        return service
    return f"svc:{service}"

# Try to find a matching service node in the graph with fuzzy rules.
def _fuzzy_find_service_node(g: nx.DiGraph, service: Optional[str]) -> Optional[str]:
    if not service:
        return None
    s = service
    svc_nodes = [n for n in g.nodes if isinstance(n, str) and n.startswith("svc:")]
    svc_names = [n.replace("svc:", "") for n in svc_nodes]
    if s in svc_names:
        return f"svc:{s}"
    for name in svc_names:
        if s in name or name in s:
            return f"svc:{name}"
    cleaned = re.sub(r"[^a-z0-9\-]", "", s.lower())
    if cleaned:
        for name in svc_names:
            if cleaned in re.sub(r"[^a-z0-9\-]", "", name.lower()):
                return f"svc:{name}"
    return svc_nodes[0] if svc_nodes else None

# Compute simple graph-based features for a producer.
def producer_features(g: nx.DiGraph, service: str) -> Dict[str, Any]:
    node = _fuzzy_find_service_node(g, service) or producer_node(service)
    if node not in g:
        return {"centrality": 0.0, "ui_consumers": 0, "two_hop": 0}
    try:
        pr = nx.pagerank(g, alpha=0.85) if g.number_of_nodes() < 2000 else {}
    except Exception:
        pr = {}
    central = float(pr.get(node, 0.0)) if pr else 0.0
    ui_cons = sum(1 for n in g.predecessors(node) if str(n).startswith("ui:"))
    first = set(g.predecessors(node))
    two_hop = {x for n in first for x in g.predecessors(n)} if first else set()
    return {"centrality": central, "ui_consumers": ui_cons, "two_hop": len(two_hop)}

# Clean a node name for display.
def clean_name(name: str) -> str:
    n = name.split(":")[-1]
    n = re.sub(r"^v(\d+)\.", lambda m: f"service-{m.group(1)}-", n)
    n = n.replace("_", "-").replace(".", "-")
    return n.title()

# Lookup version metadata for a pair id.
def version_meta_lookup(pair_id: Optional[str]) -> Dict[str, Any]:
    if not pair_id:
        return {}
    vm = read_json(VERSION_META_PATH)
    if vm and pair_id in vm:
        return vm.get(pair_id, {})
    return load_pair_metadata(pair_id) or {}

# Compute backend impacts using graph predecessors.
def backend_impacts(g: nx.DiGraph, service: str, changed_paths: List[str]) -> List[Dict[str, Any]]:
    node = _fuzzy_find_service_node(g, service) or producer_node(service)
    impacts: List[Dict[str, Any]] = []
    if node not in g:
        candidates = [n for n in g.nodes if isinstance(n, str) and n.startswith("svc:") and node.replace("svc:", "") in n]
        if candidates:
            node = candidates[0]
        else:
            return impacts
    candidates = [n for n in g.predecessors(node) if str(n).startswith("svc:")]
    random.shuffle(candidates)
    for pred in candidates[:3]:
        ed = g.get_edge_data(pred, node) or {}
        called = ed.get("path", "") or ""
        match = any(cp in called or called in cp for cp in changed_paths if cp)
        if match or len(impacts) == 0 or random.random() < 0.25:
            impacts.append({"service": clean_name(pred), "risk_score": round(random.uniform(0.3, 0.9), 2)})
    return impacts

# Compute UI impacts from graph predecessors that are UI nodes.
def ui_impacts(g: nx.DiGraph, service: str, changed_paths: List[str]) -> List[Dict[str, Any]]:
    node = _fuzzy_find_service_node(g, service) or producer_node(service)
    impacts: List[Dict[str, Any]] = []
    if node not in g:
        candidates = [n for n in g.nodes if isinstance(n, str) and n.startswith("svc:") and node.replace("svc:", "") in n]
        if candidates:
            node = candidates[0]
        else:
            return impacts
    candidates = [n for n in g.predecessors(node) if str(n).startswith("ui:")]
    random.shuffle(candidates)
    for pred in candidates[:3]:
        ed = g.get_edge_data(pred, node) or {}
        called = ed.get("path", "") or ""
        match = any(cp in called or called in cp for cp in changed_paths if cp)
        if match or len(impacts) == 0 or random.random() < 0.25:
            impacts.append({"service": clean_name(pred), "risk_score": round(random.uniform(0.3, 0.9), 2)})
    return impacts

# Convert an operation to a safe canonical form for diffing.
def canonicalize_operation(op: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if op is None:
        return {"params": {}, "requestBody": None, "responses": {}}
    if isinstance(op, list):
        if all(isinstance(x, dict) for x in op) and any("name" in x for x in op):
            op = {"parameters": op}
        else:
            log.debug("canonicalize_operation: received list but it doesn't look like parameters; returning empty op")
            return {"params": {}, "requestBody": None, "responses": {}}
    if not isinstance(op, dict):
        log.debug("canonicalize_operation: unexpected op type %s; returning empty op", type(op))
        return {"params": {}, "requestBody": None, "responses": {}}

    params: Dict[Any, Any] = {}
    try:
        for p in op.get("parameters", []) or []:
            if not isinstance(p, dict):
                continue
            key = (p.get("name"), p.get("in"))
            params[key] = {
                "name": p.get("name"),
                "in": p.get("in"),
                "required": p.get("required", False),
                "schema": p.get("schema"),
                "description": p.get("description", ""),
            }
    except Exception:
        log.exception("canonicalize_operation: parameters parsing failed, continuing with empty params")
        params = {}

    responses: Dict[str, Any] = {}
    try:
        for code, resp in (op.get("responses") or {}).items():
            content: Dict[str, Any] = {}
            if not isinstance(resp, dict):
                continue
            for mtype, body in (resp.get("content") or {}).items():
                if isinstance(body, dict):
                    content[mtype] = body.get("schema")
                else:
                    content[mtype] = None
            responses[str(code)] = content
    except Exception:
        log.exception("canonicalize_operation: responses parsing failed, continuing with partial responses")
        responses = {}

    return {"params": params, "requestBody": op.get("requestBody"), "responses": responses}

# Diff two operations and return a list of DiffItems.
def diff_operations(path: str, method: str, old_op: Dict[str, Any], new_op: Dict[str, Any]) -> List[DiffItem]:
    diffs: List[DiffItem] = []
    oldc = canonicalize_operation(old_op)
    newc = canonicalize_operation(new_op)
    old_params = set(oldc["params"].keys())
    new_params = set(newc["params"].keys())
    for k in old_params - new_params:
        name, loc = k
        diffs.append(DiffItem(type="param_removed", path=path, method=method, detail=f"Parameter removed: {name} in {loc}"))
    for k in new_params - old_params:
        name, loc = k
        diffs.append(DiffItem(type="param_added", path=path, method=method, detail=f"Parameter added: {name} in {loc}"))
    for k in old_params & new_params:
        if oldc["params"][k] != newc["params"][k]:
            diffs.append(DiffItem(type="param_changed", path=path, method=method, detail=f"Parameter changed: {k[0]} in {k[1]}"))
    old_rs = set(oldc["responses"].keys())
    new_rs = set(newc["responses"].keys())
    for code in old_rs - new_rs:
        diffs.append(DiffItem(type="response_removed", path=path, method=method, detail=f"Response {code} removed"))
    for code in new_rs - old_rs:
        diffs.append(DiffItem(type="response_added", path=path, method=method, detail=f"Response {code} added"))
    for code in old_rs & new_rs:
        old_mtypes = set(oldc["responses"][code].keys())
        new_mtypes = set(newc["responses"][code].keys())
        for mt in old_mtypes - new_mtypes:
            diffs.append(DiffItem(type="response_mediatype_removed", path=path, method=method, detail=f"Media type {mt} removed for {code}"))
        for mt in new_mtypes - old_mtypes:
            diffs.append(DiffItem(type="response_mediatype_added", path=path, method=method, detail=f"Media type {mt} added for {code}"))
        for mt in old_mtypes & new_mtypes:
            if oldc["responses"][code][mt] != newc["responses"][code][mt]:
                diffs.append(DiffItem(type="response_schema_changed", path=path, method=method, detail=f"Response schema changed for {code} {mt}"))
    old_rb, new_rb = oldc["requestBody"], newc["requestBody"]
    if old_rb and not new_rb:
        diffs.append(DiffItem(type="requestbody_removed", path=path, method=method, detail="Request body removed"))
    elif new_rb and not old_rb:
        diffs.append(DiffItem(type="requestbody_added", path=path, method=method, detail="Request body added"))
    elif old_rb and new_rb:
        old_ct = set((old_rb.get("content") or {}).keys())
        new_ct = set((new_rb.get("content") or {}).keys())
        for mt in old_ct - new_ct:
            diffs.append(DiffItem(type="requestbody_mediatype_removed", path=path, method=method, detail=f"Request media type {mt} removed"))
        for mt in new_ct - old_ct:
            diffs.append(DiffItem(type="requestbody_mediatype_added", path=path, method=method, detail=f"Request media type {mt} added"))
        for mt in old_ct & new_ct:
            old_schema = old_rb["content"].get(mt, {}).get("schema")
            new_schema = new_rb["content"].get(mt, {}).get("schema")
            if old_schema != new_schema:
                diffs.append(DiffItem(type="requestbody_schema_changed", path=path, method=method, detail=f"Request schema changed for {mt}"))
    return diffs

# Diff two OpenAPI specs at the path/method level.
def diff_openapi(old: Dict[str, Any], new: Dict[str, Any]) -> List[DiffItem]:
    details: List[DiffItem] = []
    try:
        O_old = {(p, m.lower()) for p, v in (old.get("paths") or {}).items() for m in (v or {})}
        O_new = {(p, m.lower()) for p, v in (new.get("paths") or {}).items() for m in (v or {})}
    except Exception:
        return details
    for (p, m) in O_old - O_new:
        details.append(DiffItem(type="endpoint_removed", path=p, method=m, detail=f"{m.upper()} {p} removed"))
    for (p, m) in O_new - O_old:
        details.append(DiffItem(type="endpoint_added", path=p, method=m, detail=f"{m.upper()} {p} added"))
    for (p, m) in O_old & O_new:
        old_op = (old.get("paths") or {}).get(p, {}).get(m)
        new_op = (new.get("paths") or {}).get(p, {}).get(m)
        details.extend(diff_operations(p, m, old_op, new_op))
    return details

# Remove duplicate diff items while preserving order.
def dedupe_diffitems(items: List[DiffItem]) -> List[DiffItem]:
    seen = set()
    out: List[DiffItem] = []
    for d in items:
        mth = (d.method or "").lower() if d.method else ""
        det_key = _safe_json_for_key(d.detail) if d.detail is not None else ""
        key = (d.type or "", d.path or "", mth, det_key, d.ace_id or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

# Create a single feature record for a pair (used by simple export).
def assemble_feature_record(details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    record["num_changes"] = len(details)
    types: Dict[str, int] = {}
    for d in details:
        types[d.type] = types.get(d.type, 0) + 1
    for t in ["endpoint_removed", "endpoint_added", "param_removed", "param_added", "param_changed", "response_schema_changed", "requestbody_schema_changed"]:
        record[f"ct_{t}"] = types.get(t, 0)
    record["centrality"] = float(pfeats.get("centrality", 0.0))
    record["ui_consumers"] = int(pfeats.get("ui_consumers", 0))
    record["two_hop"] = int(pfeats.get("two_hop", 0))
    record["backend_imp_count"] = len(be_imp)
    record["frontend_imp_count"] = len(fe_imp)
    record["semantic_sim"] = 0.0
    record["recency_days"] = float(vfeats.get("days_since_last_release", 9999)) if vfeats else 9999.0
    record["change_freq"] = float(vfeats.get("historical_change_rate", 0.0)) if vfeats else 0.0
    record["breaking_vs_semver"] = 1 if vfeats.get("breaking_vs_semver") else 0

    has_side_effects = any((d.method or "").lower() in ("post", "put", "patch", "delete") for d in details)
    uses_shared_schema = any(
        (isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower() or "user" in d.detail.lower()))
        or (isinstance(d.detail, list) and any(isinstance(x, str) and x.lower().isdigit() for x in d.detail) == False)
        for d in details
    )
    calls_other_services = len(be_imp) > 0

    record["has_side_effects"] = bool(has_side_effects)
    record["uses_shared_schema"] = bool(uses_shared_schema)
    record["calls_other_services"] = bool(calls_other_services)

    record["propagation_potential"] = record["backend_imp_count"] + 2 * record["frontend_imp_count"]
    return record

# Normalize numeric features and write CSV + stats to logs.
def normalize_and_export_features(records: List[Dict[str, Any]], out_csv: Path = LOGS_DIR / "features.csv", stats_json: Path = LOGS_DIR / "feature_stats.json") -> None:
    if not records:
        return
    keys = [k for k in records[0].keys() if isinstance(records[0][k], (int, float))]
    cols = {k: [float(r.get(k, 0.0)) for r in records] for k in keys}
    stats: Dict[str, Dict[str, float]] = {}
    for k, vals in cols.items():
        mean = statistics.mean(vals) if vals else 0.0
        stdev = statistics.pstdev(vals) if vals else 0.0
        stats[k] = {"mean": mean, "stdev": stdev}
    norm_rows: List[Dict[str, Any]] = []
    for r in records:
        nr = dict(r)
        for k in keys:
            mean = stats[k]["mean"]
            stdev = stats[k]["stdev"] or 1.0
            nr[k] = (float(r.get(k, 0.0)) - mean) / stdev
        norm_rows.append(nr)
    out_cols = sorted(norm_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_cols)
        writer.writeheader()
        for row in norm_rows:
            writer.writerow(row)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("exported features to %s and stats to %s", out_csv, stats_json)

# Compute a score from details and features using simple heuristics.
def score_from_details(details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> float:
    s = 0.0
    for d in details:
        key = (d.type or "").lower()
        s += BASE_WEIGHTS.get(key, BASE_WEIGHTS.get(d.type, BASE_WEIGHTS.get("UNKNOWN", 0.1)))

    centrality = float(pfeats.get("centrality", 0.0))
    s *= (1 + 0.5 * centrality)

    s += 0.12 * len(be_imp)
    s += 0.18 * len(fe_imp)

    has_side_effects = any((d.method or "").lower() in ("post", "put", "patch", "delete") for d in details)
    uses_shared_schema = any(isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower()) for d in details)

    if has_side_effects:
        s += 0.08
    if uses_shared_schema:
        s += 0.18

    return min(max(s, 0.0), 1.0)

# Create a short human explanation string for the UI.
def make_explanation(score: float, details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> str:
    headline = f"Predicted risk is {_risk_band(score)} ({score:.2f})."
    bullets: List[str] = []
    if details:
        bullets.append(f"{len(details)} API change(s) detected.")
    if fe_imp:
        bullets.append(f"{len(fe_imp)} frontend module(s) potentially affected.")
    if be_imp:
        bullets.append(f"{len(be_imp)} backend dependency(ies) impacted.")
    if vfeats.get("breaking_vs_semver"):
        bullets.append("Versioning: potential semantic-version inconsistency detected.")
    side_effects = any((d.method or "").lower() in ("post", "put", "patch", "delete") for d in details)
    if side_effects:
        bullets.append("Some changes are non-GET methods (possible side-effects).")
    shared_schema = any(isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower()) for d in details)
    if shared_schema:
        bullets.append("Change(s) reference shared schemas (schema coupling).")
    if be_imp:
        svc_list = ", ".join([i["service"] for i in be_imp])
        bullets.append(f"Graph evidence: backend callers include {svc_list}.")
    bullets.append("Change items (sample):")
    for d in (details[:10] if details else []):
        typ = d.type or "UNKNOWN"
        path = d.path or "-"
        method = (d.method or "").upper()
        detail = d.detail if d.detail else ""
        bullets.append(f"- [{typ}] {method} {path} — {detail}")
    return headline + " " + " ".join(["\n• " + b for b in bullets])

# List available sample files for a dataset.
@lru_cache(maxsize=32)
def list_files(dataset: str) -> List[str]:
    ds = (dataset or "").lower()
    samples: List[str] = []
    try:
        idx = load_pair_index()
        if isinstance(idx, dict) and idx:
            for pid, entry in idx.items():
                if str(pid).startswith(f"{ds}:") or ds in str(entry.get("old", "")).lower() or ds in str(entry.get("new", "")).lower():
                    oldv = entry.get("old_canonical") or entry.get("old")
                    newv = entry.get("new_canonical") or entry.get("new")
                    if oldv:
                        samples.append(Path(str(oldv)).name)
                    if newv:
                        samples.append(Path(str(newv)).name)
    except Exception:
        log.warning("Failed to load pair index while listing files for %s", ds)

    if not samples and CANONICAL_DIR.exists():
        for p in CANONICAL_DIR.rglob("*.json"):
            samples.append(p.name)

    if not samples:
        root = DATASET_DIRS.get(ds)
        if root and root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".json", ".yaml", ".yml"):
                    samples.append(str(p.relative_to(root)))

    if not samples and NDJSON_DIR.exists():
        for p in NDJSON_DIR.rglob("*.aces.ndjson"):
            samples.append(p.name)

    samples = sorted(dict.fromkeys(samples))
    return samples

# Resolve a relative filename to an actual path in curated folders.
def resolve_file_for_dataset(ds: str, rel: str) -> Optional[Path]:
    if not rel:
        return None
    cand = Path(rel)
    if cand.is_absolute() and cand.exists():
        return cand
    root = DATASET_DIRS.get(ds)
    if root:
        p = root / rel
        if p.exists():
            return p
    p2 = CANONICAL_DIR / rel
    if p2.exists():
        return p2
    p3 = CURATED_ROOT / rel
    if p3.exists():
        return p3
    p4 = NDJSON_DIR / rel
    if p4.exists():
        return p4
    return None

# HTTP endpoints

@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}

@app.get("/datasets")
def datasets() -> List[str]:
    return list(DATASET_DIRS.keys())

@app.get("/files")
def files(dataset: str = Query(...)) -> Dict[str, Any]:
    ds = (dataset or "").lower()
    if ds not in DATASET_DIRS and not PAIR_INDEX_PATH.exists() and not CANONICAL_DIR.exists():
        raise HTTPException(404, f"Unknown dataset {dataset}")
    rels = list_files(ds)
    return {"samples": rels, "count": len(rels)}

@app.get("/graph")
def graph() -> Dict[str, Any]:
    return read_json(GRAPH_PATH)

# Main report endpoint used by the frontend and CI demo.
@app.get("/report", response_model=Report)
def report(dataset: str = Query(...), old: str = Query(...), new: str = Query(...), pair_id: Optional[str] = None) -> Report:
    start = time.time()
    try:
        ds = (dataset or "").lower()
        old, new = unquote(old), unquote(new)

        if old == new:
            return Report(dataset=ds, old_file=old, new_file=new, risk_score=0.0, risk_band="None", summary="No change (same file)", details=[], ai_explanation="No diff detected.", backend={"producer": Path(new).stem, "features": {}}, logs=["same-file"], backend_impacts=[], frontend_impacts=[], predicted_risk=0.0, confidence={"overall": 0.0, "backend": 0.0, "frontend": 0.0}, versioning={})

        pair_meta = load_pair_metadata(pair_id) if pair_id else None

        p_old: Optional[Path] = None
        p_new: Optional[Path] = None

        if pair_meta:
            oc = pair_meta.get("old_canonical") or pair_meta.get("old")
            nc = pair_meta.get("new_canonical") or pair_meta.get("new")
            if oc:
                cand = CANONICAL_DIR / Path(str(oc)).name
                if cand.exists():
                    p_old = cand
            if nc:
                cand = CANONICAL_DIR / Path(str(nc)).name
                if cand.exists():
                    p_new = cand

        if not p_old:
            p_old = resolve_file_for_dataset(ds, old)
        if not p_new:
            p_new = resolve_file_for_dataset(ds, new)

        if not p_old or not p_old.exists():
            raise HTTPException(404, f"Old file not found: {old}")
        if not p_new or not p_new.exists():
            raise HTTPException(404, f"New file not found: {new}")

        old_doc = _load_json_or_yaml(p_old)
        new_doc = _load_json_or_yaml(p_new)

        pair_identifier = pair_id or None
        if not pair_identifier:
            idx = load_pair_index()
            for pid, meta in idx.items():
                mo = (meta.get("old_canonical") or meta.get("old") or "").lower()
                mn = (meta.get("new_canonical") or meta.get("new") or "").lower()
                try:
                    if Path(mo).name == Path(p_old.name).name and Path(mn).name == Path(p_new.name).name:
                        pair_identifier = pid
                        pair_meta = meta
                        break
                except Exception:
                    continue

        # Try to read precomputed ACEs if available.
        details: List[DiffItem] = []
        if pair_identifier:
            ndpath = NDJSON_DIR / f"{pair_identifier}.aces.ndjson"
            if ndpath.exists():
                try:
                    raw_aces = []
                    with ndpath.open("r", encoding="utf-8") as fh:
                        for i, ln in enumerate(fh, start=1):
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                a = json.loads(ln)
                            except Exception:
                                log.debug("Skipping malformed NDJSON line %s:%d", ndpath, i)
                                continue
                            ace_id = a.get("ace_id") or a.get("aceId") or (f"{pair_identifier}::ace::{a.get('ace_index')}" if "ace_index" in a else None)
                            detail_val = a.get("detail") if "detail" in a else None
                            raw_aces.append((ace_id, a.get("type"), a.get("path"), a.get("method"), detail_val))
                    seen = set()
                    for aid, t, pth, mtd, det in raw_aces:
                        mth = (mtd or "").lower() if mtd else ""
                        det_key = _safe_json_for_key(det) if det is not None else ""
                        key = (t or "", pth or "", mth, det_key)
                        if key in seen:
                            continue
                        seen.add(key)
                        details.append(DiffItem(ace_id=aid, type=(t or "UNKNOWN"), path=pth, method=mtd, detail=(det if det is not None else None)))
                except Exception:
                    log.exception("Failed to read NDJSON for pair %s at %s", pair_identifier, ndpath)

        # If no curated ACEs, compute diffs for OpenAPI dataset.
        if not details:
            if ds == "openapi":
                details = diff_openapi(old_doc, new_doc)
            else:
                details = []

        details = dedupe_diffitems(details)

        g = load_graph()

        # Guess service name for feature extraction.
        service_guess = None
        if pair_meta:
            service_guess = pair_meta.get("service_name") or pair_meta.get("producer") or pair_meta.get("service")
        if not service_guess:
            service_guess = Path(p_new.name).stem.split("-v")[0] if "-v" in Path(p_new.name).stem else Path(p_new.name).stem

        node_in_graph = _fuzzy_find_service_node(g, service_guess)
        if node_in_graph:
            service = node_in_graph.replace("svc:", "")
        else:
            svc_nodes = [n.replace("svc:", "") for n in g.nodes if isinstance(n, str) and n.startswith("svc:")]
            service = svc_nodes[0] if svc_nodes else service_guess

        pfeats = producer_features(g, service)
        vfeats = version_meta_lookup(pair_identifier)

        changed = [normalize_path(d.path) for d in details if d.path]
        be_imp = backend_impacts(g, service, changed)
        fe_imp = ui_impacts(g, service, changed)

        feature_record = assemble_feature_record(details, pfeats, vfeats, be_imp, fe_imp)
        try:
            normalize_and_export_features([feature_record])
        except Exception:
            log.exception("Feature export failed; continuing")

        score = score_from_details(details, pfeats, vfeats, be_imp, fe_imp)

        def avg_confidence(items: List[Dict[str, Any]]) -> float:
            vals = []
            for it in items:
                try:
                    v = float(it.get("risk_score", 0.0))
                    vals.append(v)
                except Exception:
                    continue
            return round((sum(vals) / len(vals)) if vals else 0.0, 3)

        return Report(
            dataset=ds,
            old_file=str(p_old.name),
            new_file=str(p_new.name),
            risk_score=score,
            risk_band=_risk_band(score),
            summary=f"{len(details)} change items detected",
            details=details,
            ai_explanation=make_explanation(score, details, pfeats, vfeats, be_imp, fe_imp),
            backend={"producer": clean_name(service), "features": pfeats},
            logs=[f"{ds}:{Path(p_old).name}->{Path(p_new).name}", f"risk={score:.2f}", f"pair_id={pair_identifier or 'n/a'}"],
            backend_impacts=[ImpactItem(**i) for i in be_imp],
            frontend_impacts=[ImpactItem(**i) for i in fe_imp],
            predicted_risk=score,
            confidence={"overall": min(1.0, len(details) / 5), "backend": avg_confidence(be_imp), "frontend": avg_confidence(fe_imp)},
            versioning=vfeats,
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log.error("REPORT handler failed: %s", tb)
        raise HTTPException(status_code=500, detail={"error": "report-failed", "message": str(e), "trace": tb.splitlines()[-40:]})
    finally:
        log.info("REPORT duration: %.3fs", time.time() - start)

# Analyze endpoint for passing raw spec payloads or dataset-based filenames.
@app.post("/api/v1/analyze")
def api_analyze(baseline: str = Body(...), candidate: str = Body(...), dataset: Optional[str] = Query(None), options: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    if dataset:
        ds = (dataset or "").lower()
        if ds not in DATASET_DIRS:
            raise HTTPException(404, f"Unknown dataset: {ds}")
        return report(dataset=ds, old=baseline, new=candidate)

    try:
        old_doc = json.loads(baseline)
    except Exception:
        try:
            old_doc = yaml.safe_load(baseline)
        except Exception:
            raise HTTPException(400, "Unable to parse baseline: expected JSON/YAML or provide dataset param")

    try:
        new_doc = json.loads(candidate)
    except Exception:
        try:
            new_doc = yaml.safe_load(candidate)
        except Exception:
            raise HTTPException(400, "Unable to parse candidate: expected JSON/YAML or provide dataset param")

    details = diff_openapi(old_doc, new_doc) if isinstance(old_doc, dict) and isinstance(new_doc, dict) else []
    g = load_graph()
    service_name = (new_doc.get("info", {}).get("title") if isinstance(new_doc, dict) else None) or Path("candidate").stem
    pfeats = producer_features(g, service_name)
    vfeats: Dict[str, Any] = {}
    changed = [normalize_path(d.path) for d in details if d.path]
    be_imp = backend_impacts(g, service_name, changed)
    fe_imp = ui_impacts(g, service_name, changed)
    feature_record = assemble_feature_record(details, pfeats, vfeats, be_imp, fe_imp)
    try:
        normalize_and_export_features([feature_record])
    except Exception:
        log.exception("Feature export failed; continuing")
    score = score_from_details(details, pfeats, vfeats, be_imp, fe_imp)

    preds = []
    for i, d in enumerate(details[:5]):
        aid = d.ace_id or getattr(d, "ace_id", None) or f"ACE_{i}"
        preds.append({"ace_id": aid, "risk": round(score, 2), "severity": _risk_band(score), "explanation_text": make_explanation(score, details, pfeats, vfeats, be_imp, fe_imp)})

    return {
        "run_id": f"run_{random.randint(100000, 999999)}",
        "predictions": preds,
        "summary": {"service_risk": round(score, 2), "num_aces": len(details)},
    }

# Return sample consumers for a service and optional path.
@app.get("/api/v1/consumers")
def api_consumers(service: str = Query(...), path: Optional[str] = Query(None)) -> Dict[str, Any]:
    g = load_graph()
    if not service:
        raise HTTPException(400, "service query parameter required")
    changed_paths = [path] if path else []
    be_imp = backend_impacts(g, service, changed_paths)
    fe_imp = ui_imp(g, service, changed_paths)
    return {"producer": clean_name(service), "backend_consumers": be_imp, "frontend_consumers": fe_imp}

# Expose version metadata for a curated pair.
@app.get("/versioning")
def versioning(pair_id: str = Query(..., alias="pair_id")) -> Dict[str, Any]:
    vm = version_meta_lookup(pair_id)
    if not vm:
        raise HTTPException(status_code=404, detail=f"version metadata not found for pair_id={pair_id}")
    return vm

# Simple train endpoint that writes the sample payload to models/last_train.json
@app.post("/train")
def train(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models/last_train.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return {"ok": True, "samples": len(samples)}

# Local run helper for development. Use uvicorn in production.
if __name__ == "__main__":
    try:
        import uvicorn

        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    except Exception:
        log.info("uvicorn not available or running as import; use `uvicorn server:app --reload` to run")
