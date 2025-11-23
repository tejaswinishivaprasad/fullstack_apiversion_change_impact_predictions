#!/usr/bin/env python3
"""
server.py

Impact AI core HTTP server (updated for dataset-specific curated layout).

This version expects curated data layout:

datasets/curated/
  openapi/
    canonical/
    ndjson/
    metadata/
  petclinic/
    canonical/
    ndjson/
    metadata/
  openrewrite/
    canonical/
    ndjson/
    metadata/
  index.json
  graph.json
  version_meta.json
  dataset_oindex.json
  version_pairs.csv

It is resilient: it falls back to legacy single-folder layout when needed,
and gracefully handles missing folders.
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

# logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("impact-ai-core")

# curated root and global files
CURATED_ROOT = Path(os.getenv("CURATED_ROOT", "datasets/curated"))
INDEX_PATH = CURATED_ROOT / "index.json"
GRAPH_PATH = CURATED_ROOT / "graph.json"
VERSION_META_PATH = CURATED_ROOT / "version_meta.json"
DATASET_OINDEX_PATH = CURATED_ROOT / "dataset_oindex.json"
VERSION_PAIRS_CSV = CURATED_ROOT / "version_pairs.csv"

# Ensure logs folder exists.
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Known dataset names (keeps backward compatibility: add more if you create them)
KNOWN_DATASETS = ["openapi", "petclinic", "openrewrite"]

# Helper to return dataset-specific curated paths
def dataset_base(ds: str) -> Path:
    return CURATED_ROOT / ds

def dataset_paths(ds: str) -> Dict[str, Path]:
    base = dataset_base(ds)
    return {
        "base": base,
        "canonical": base / "canonical",
        "ndjson": base / "ndjson",
        "metadata": base / "metadata",
    }

# Legacy single-folder locations for backwards compatibility
LEGACY_CANONICAL = CURATED_ROOT / "canonical"
LEGACY_NDJSON = CURATED_ROOT / "ndjson"
LEGACY_METADATA = CURATED_ROOT / "metadata"

# Data models
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

# FastAPI app
app = FastAPI(title="Impact AI Core", version="2.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- file helpers ----------
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

def read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to parse JSON at %s", p)
        return {}

def _safe_json_for_key(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=lambda o: str(o))
    except Exception:
        try:
            return str(obj)
        except Exception:
            return ""

# ---------- risk utilities (kept mostly as you had them) ----------
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

def _risk_band(s: float) -> str:
    return "High" if s >= 0.7 else "Medium" if s >= 0.4 else "Low" if s > 0 else "None"

def normalize_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return re.sub(r"\{[^}]+\}", "{*}", p.strip())

# ---------- index loader (dataset-aware) ----------
@lru_cache(maxsize=1)
def load_pair_index() -> Dict[str, Any]:
    """
    Prefer the new global index.json (CURATED_ROOT/index.json).
    If missing, try to assemble an index by scanning each dataset's metadata folder.
    Returns dict: pair_id -> metadata dict
    """
    if INDEX_PATH.exists():
        try:
            data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
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
            log.warning("Failed to parse index.json at %s", INDEX_PATH)

    # fallback: scan per-dataset metadata folders
    pairs: Dict[str, Any] = {}
    for ds in KNOWN_DATASETS:
        md = dataset_paths(ds)["metadata"]
        if md.exists():
            for p in md.glob("*.meta.json"):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    pid = d.get("pair_id") or p.stem
                    # include dataset hint
                    d.setdefault("dataset", ds)
                    pairs[pid] = d
                except Exception:
                    log.warning("Failed to parse meta file %s", p)
    # legacy metadata folder
    if LEGACY_METADATA.exists():
        for p in LEGACY_METADATA.glob("*.meta.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                pid = d.get("pair_id") or p.stem
                pairs.setdefault(pid, d)
            except Exception:
                log.warning("Failed to parse legacy meta file %s", p)
    return pairs

def load_pair_metadata(pair_id: Optional[str]) -> Dict[str, Any]:
    if not pair_id:
        return {}
    idx = load_pair_index()
    if pair_id in idx:
        return idx[pair_id] or {}
    # fallback: try to find metadata files across datasets
    for ds in KNOWN_DATASETS:
        mpath = dataset_paths(ds)["metadata"] / f"{pair_id}.meta.json"
        if mpath.exists():
            try:
                d = json.loads(mpath.read_text(encoding="utf-8"))
                d.setdefault("dataset", ds)
                return d
            except Exception:
                log.warning("Failed to parse meta %s", mpath)
    # legacy metadata
    legacy = LEGACY_METADATA / f"{pair_id}.meta.json"
    if legacy.exists():
        try:
            return json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to parse legacy meta %s", legacy)
    return {}

# ---------- graph loader ----------
def load_graph() -> nx.DiGraph:
    data = read_json(GRAPH_PATH)
    g = nx.DiGraph()
    for e in data.get("edges", []):
        src = e.get("src"); dst = e.get("dst")
        if src and dst:
            g.add_edge(src, dst, path=e.get("path"), evidence=e.get("evidence"), confidence=e.get("confidence", 0.5), provenance=e.get("provenance", {}))
    return g

# ---------- simple graph helpers (kept) ----------
def producer_node(service: str) -> str:
    if service.startswith("svc:") or service.startswith("ui:"):
        return service
    return f"svc:{service}"

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

def clean_name(name: str) -> str:
    n = name.split(":")[-1]
    n = n.replace("_", "-").replace(".", "-")
    return n.title()

def version_meta_lookup(pair_id: Optional[str]) -> Dict[str, Any]:
    if not pair_id:
        return {}
    vm = read_json(VERSION_META_PATH)
    if vm and pair_id in vm:
        return vm.get(pair_id, {})
    return load_pair_metadata(pair_id) or {}

# backend/ui impact computations (kept)
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
# Insert this in server.py (place before normalize_and_export_features or before report)
def assemble_feature_record(details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a single feature record for a pair (used by simple export).
    Kept small and defensive so it won't crash on odd inputs.
    """
    record: Dict[str, Any] = {}
    record["num_changes"] = len(details) if details is not None else 0

    types: Dict[str, int] = {}
    for d in details or []:
        t = (d.type or "UNKNOWN")
        types[t] = types.get(t, 0) + 1

    # populate common counters (keep keys stable)
    for t in ["endpoint_removed", "endpoint_added", "param_removed", "param_added", "param_changed", "response_schema_changed", "requestbody_schema_changed"]:
        record[f"ct_{t}"] = types.get(t, 0)

    # graph features
    record["centrality"] = float(pfeats.get("centrality", 0.0)) if pfeats else 0.0
    record["ui_consumers"] = int(pfeats.get("ui_consumers", 0)) if pfeats else 0
    record["two_hop"] = int(pfeats.get("two_hop", 0)) if pfeats else 0

    # impact counts
    record["backend_imp_count"] = len(be_imp) if be_imp is not None else 0
    record["frontend_imp_count"] = len(fe_imp) if fe_imp is not None else 0

    # placeholders for semantic/temporal features (safe defaults)
    record["semantic_sim"] = float(vfeats.get("semantic_sim", 0.0)) if vfeats else 0.0
    record["recency_days"] = float(vfeats.get("days_since_last_release", 9999)) if vfeats else 9999.0
    record["change_freq"] = float(vfeats.get("historical_change_rate", 0.0)) if vfeats else 0.0
    record["breaking_vs_semver"] = 1 if vfeats and vfeats.get("breaking_vs_semver") else 0

    # boolean features inferred from details
    has_side_effects = any(((d.method or "").lower() in ("post", "put", "patch", "delete")) for d in (details or []))
    uses_shared_schema = any(
        (isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower()))
        or (isinstance(d.detail, list) and any(isinstance(x, str) for x in d.detail))
        for d in (details or [])
    )
    calls_other_services = (len(be_imp) > 0) if be_imp is not None else False

    record["has_side_effects"] = bool(has_side_effects)
    record["uses_shared_schema"] = bool(uses_shared_schema)
    record["calls_other_services"] = bool(calls_other_services)

    # propagation heuristic
    record["propagation_potential"] = record["backend_imp_count"] + 2 * record["frontend_imp_count"]

    return record


# ---------- canonicalize operation + diff functions (copied from your server with minimal changes) ----------
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

# ---------- dataset-aware file listing and resolution ----------
@lru_cache(maxsize=32)
def list_files(dataset: str) -> List[str]:
    ds = (dataset or "").lower()
    samples: List[str] = []
    # prefer index.json if present
    idx = load_pair_index()
    if idx:
        for pid, entry in idx.items():
            entry_ds = (entry.get("dataset") or "").lower()
            if ds and entry_ds != ds:
                continue
            oldv = entry.get("old_canonical") or entry.get("old")
            newv = entry.get("new_canonical") or entry.get("new")
            if oldv:
                samples.append(Path(str(oldv)).name)
            if newv:
                samples.append(Path(str(newv)).name)
    else:
        # fallback: gather from dataset canonical folder
        dp = dataset_paths(ds)
        can = dp["canonical"]
        if can.exists():
            for p in can.rglob("*.json"):
                samples.append(p.name)
        nd = dp["ndjson"]
        if nd.exists():
            for p in nd.rglob("*.aces.ndjson"):
                samples.append(p.name)
        # legacy places
        if not samples and LEGACY_CANONICAL.exists():
            for p in LEGACY_CANONICAL.rglob("*.json"):
                samples.append(p.name)
        if not samples and LEGACY_NDJSON.exists():
            for p in LEGACY_NDJSON.rglob("*.aces.ndjson"):
                samples.append(p.name)

    samples = sorted(dict.fromkeys(samples))
    return samples

def resolve_file_for_dataset(ds: str, rel: str) -> Optional[Path]:
    """
    Resolve a relative filename (e.g., "openapi--svc--v1.canonical.json")
    to a real Path within the curated structure for dataset `ds`.
    """
    if not rel:
        return None
    cand = Path(rel)
    if cand.is_absolute() and cand.exists():
        return cand

    ds = (ds or "").lower()
    # check dataset-specific canonical (preferred)
    dp = dataset_paths(ds)
    for folder in (dp["canonical"], dp["ndjson"], dp["metadata"]):
        p = folder / rel
        if p.exists():
            return p
    # try within the dataset base (some entries may store relative paths)
    p2 = dp["base"] / rel
    if p2.exists():
        return p2
    # check global index entry for direct path match
    idx = load_pair_index()
    for pid, meta in idx.items():
        oc = meta.get("old_canonical") or meta.get("old")
        nc = meta.get("new_canonical") or meta.get("new")
        if oc and Path(oc).name == rel:
            ds_meta = meta.get("dataset")
            candp = dataset_paths(ds_meta)["canonical"] / rel
            if candp.exists():
                return candp
        if nc and Path(nc).name == rel:
            ds_meta = meta.get("dataset")
            candp = dataset_paths(ds_meta)["canonical"] / rel
            if candp.exists():
                return candp

    # legacy fallbacks
    lc = LEGACY_CANONICAL / rel
    if lc.exists():
        return lc
    ln = LEGACY_NDJSON / rel
    if ln.exists():
        return ln
    lp = CURATED_ROOT / rel
    if lp.exists():
        return lp
    return None

# ---------- HTTP endpoints ----------
@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}

@app.get("/datasets")
def datasets() -> List[str]:
    # list known dataset dirs that exist (show all known names even if empty)
    return KNOWN_DATASETS

@app.get("/files")
def files(dataset: str = Query(...)) -> Dict[str, Any]:
    ds = (dataset or "").lower()
    if ds not in KNOWN_DATASETS:
        raise HTTPException(404, f"Unknown dataset {dataset}")
    rels = list_files(ds)
    return {"samples": rels, "count": len(rels)}

@app.get("/graph")
def graph() -> Dict[str, Any]:
    return read_json(GRAPH_PATH)


# --- Feature export & scoring helpers (paste before report()) ---

def normalize_and_export_features(records: List[Dict[str, Any]], out_csv: Path = LOGS_DIR / "features.csv", stats_json: Path = LOGS_DIR / "feature_stats.json") -> None:
    """
    Normalize numeric features across the given records, write CSV and stats JSON.
    Defensive: if records is empty or malformed, it returns quietly.
    """
    try:
        if not records:
            log.info("normalize_and_export_features: no records to export")
            return
        # pick numeric keys from first record
        keys = [k for k, v in records[0].items() if isinstance(v, (int, float))]
        if not keys:
            log.info("normalize_and_export_features: no numeric keys found")
            return

        cols = {k: [float(r.get(k, 0.0) or 0.0) for r in records] for k in keys}
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
                nr[k] = (float(r.get(k, 0.0) or 0.0) - mean) / stdev
            norm_rows.append(nr)

        out_cols = sorted(norm_rows[0].keys())
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_cols)
            writer.writeheader()
            for row in norm_rows:
                writer.writerow(row)

        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("exported features to %s and stats to %s", out_csv, stats_json)
    except Exception:
        log.exception("Feature export failed; continuing")


def score_from_details(details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> float:
    """
    Heuristic scoring function that converts detected diffs + features into a 0..1 risk.
    """
    try:
        s = 0.0
        for d in (details or []):
            key = (d.type or "").lower()
            # try to map common variants to base keys
            mapped = {
                "endpoint_added": "endpoint_added",
                "endpoint_removed": "endpoint_removed",
                "param_added": "param_added",
                "param_removed": "param_removed",
                "param_changed": "param_changed",
                "response_schema_changed": "response_schema_changed",
                "requestbody_schema_changed": "requestbody_schema_changed",
                "param_required_added": "param_required_added",
            }.get(key, key)
            s += BASE_WEIGHTS.get(mapped, BASE_WEIGHTS.get("UNKNOWN", 0.1))

        centrality = float(pfeats.get("centrality", 0.0)) if pfeats else 0.0
        s *= (1 + 0.5 * centrality)

        s += 0.12 * (len(be_imp) if be_imp is not None else 0)
        s += 0.18 * (len(fe_imp) if fe_imp is not None else 0)

        has_side_effects = any(((d.method or "").lower() in ("post", "put", "patch", "delete")) for d in (details or []))
        uses_shared_schema = any(isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower()) for d in (details or []))

        if has_side_effects:
            s += 0.08
        if uses_shared_schema:
            s += 0.18

        s = max(0.0, min(1.0, s))
        return round(s, 3)
    except Exception:
        log.exception("score_from_details failed; returning 0.0")
        return 0.0


def make_explanation(score: float, details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> str:
    """
    Produce a compact, grouped human explanation.
    """
    try:
        # header
        headline = f"Predicted risk is {_risk_band(score)} ({score:.2f})."
        # group by type
        type_counts: Dict[str, int] = {}
        for d in (details or []):
            typ = (d.type or "UNKNOWN").upper()
            type_counts[typ] = type_counts.get(typ, 0) + 1

        bullets: List[str] = []
        if type_counts:
            top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:6]
            bullets.append("Change types: " + ", ".join([f"{t}={c}" for t, c in top_types]))

        if fe_imp:
            bullets.append(f"{len(fe_imp)} frontend module(s) possibly affected")
        if be_imp:
            bullets.append(f"{len(be_imp)} backend dependency(ies) impacted")

        if vfeats and vfeats.get("breaking_vs_semver"):
            bullets.append("Versioning: potential semantic-version inconsistency detected.")

        # side-effect / shared schema hints
        side_effects = any((d.method or "").lower() in ("post", "put", "patch", "delete") for d in (details or []))
        if side_effects:
            bullets.append("Contains non-GET changes (possible side-effects).")

        shared_schema = any(isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower()) for d in (details or []))
        if shared_schema:
            bullets.append("Some changes reference shared schemas (schema coupling).")

        # representative ACEs (dedupe by type+path)
        seen = set()
        rep_aces = []
        for d in (details or []):
            k = ((d.type or ""), (d.path or ""), (d.method or ""))
            if k in seen:
                continue
            seen.add(k)
            rep_aces.append(d)
            if len(rep_aces) >= 5:
                break

        if rep_aces:
            bullets.append("Sample changes (up to 5):")
            for d in rep_aces:
                tid = d.ace_id or "-"
                typ = d.type or "UNKNOWN"
                path = d.path or "-"
                method = (d.method or "").upper() or "-"
                # try to surface confidence if present in d.detail or detail structure
                conf = None
                try:
                    # some ACEs store confidence in detail dict or as an attribute
                    if isinstance(d.detail, dict) and "confidence" in d.detail:
                        conf = d.detail.get("confidence")
                except Exception:
                    conf = None
                conf_str = f" conf={conf:.2f}" if isinstance(conf, (float, int)) else ""
                bullets.append(f"- [{tid}] {typ} {method} {path}{conf_str}")

        # produce final text
        text = headline + "\n" + "\n".join(["• " + b for b in bullets])
        return text
    except Exception:
        log.exception("make_explanation failed; returning short message")
        return f"Predicted risk: {_risk_band(score)} ({score:.2f})."
    
# ---------- main report endpoint ----------
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

        # If we have pair_meta from index, prefer dataset-aware canonical paths
        if pair_meta:
            ds_meta = (pair_meta.get("dataset") or "").lower()
            oc = pair_meta.get("old_canonical") or pair_meta.get("old")
            nc = pair_meta.get("new_canonical") or pair_meta.get("new")
            if oc:
                cand = dataset_paths(ds_meta)["canonical"] / Path(str(oc)).name
                if cand.exists():
                    p_old = cand
            if nc:
                cand = dataset_paths(ds_meta)["canonical"] / Path(str(nc)).name
                if cand.exists():
                    p_new = cand

        # resolve using dataset-specific lookup
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
                try:
                    mo = (meta.get("old_canonical") or meta.get("old") or "").lower()
                    mn = (meta.get("new_canonical") or meta.get("new") or "").lower()
                    if Path(mo).name == Path(p_old.name).name and Path(mn).name == Path(p_new.name).name:
                        pair_identifier = pid
                        pair_meta = meta
                        break
                except Exception:
                    continue

        # Try to read precomputed ACEs if available (dataset-aware)
        details: List[DiffItem] = []
        if pair_identifier:
            # locate ndjson using either dataset in pair_meta or by checking each dataset
            ndpath_candidates: List[Path] = []
            if pair_meta and pair_meta.get("dataset"):
                ndpath_candidates.append(dataset_paths(pair_meta.get("dataset"))["ndjson"] / f"{pair_identifier}.aces.ndjson")
            # also check all known datasets for that file
            for ds_check in KNOWN_DATASETS:
                ndpath_candidates.append(dataset_paths(ds_check)["ndjson"] / f"{pair_identifier}.aces.ndjson")
            # legacy location
            ndpath_candidates.append(LEGACY_NDJSON / f"{pair_identifier}.aces.ndjson")
            # root fallback
            ndpath_candidates.append(CURATED_ROOT / f"{pair_identifier}.aces.ndjson")

            found_nd = None
            for p in ndpath_candidates:
                if p and p.exists():
                    found_nd = p
                    break

            if found_nd:
                try:
                    raw_aces = []
                    with found_nd.open("r", encoding="utf-8") as fh:
                        for i, ln in enumerate(fh, start=1):
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                a = json.loads(ln)
                            except Exception:
                                log.debug("Skipping malformed NDJSON line %s:%d", found_nd, i)
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
                    log.exception("Failed to read NDJSON for pair %s at %s", pair_identifier, found_nd)

        # if no precomputed ACEs, compute diffs (openapi only)
        if not details:
            if ds == "openapi":
                details = diff_openapi(old_doc, new_doc)
            else:
                details = []

        details = dedupe_diffitems(details)

        g = load_graph()

        # guess service name
        service_guess = None
        if pair_meta:
            service_guess = pair_meta.get("service_name") or pair_meta.get("producer") or pair_meta.get("service")
        if not service_guess:
            service_guess = Path(p_new.name).stem.split("-v")[0] if "-v" in Path(p_new.name).stem else Path(p_new.name).stem

        node_in_graph = _fuzzy_find_service_node(g, service_guess)
        service = node_in_graph.replace("svc:", "") if node_in_graph else (list(g.nodes)[0].replace("svc:", "") if any(isinstance(n, str) and n.startswith("svc:") for n in g.nodes) else service_guess)

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


# ---------- analyze endpoint ----------
@app.post("/api/v1/analyze")
def api_analyze(baseline: str = Body(...), candidate: str = Body(...), dataset: Optional[str] = Query(None), options: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    if dataset:
        ds = (dataset or "").lower()
        if ds not in KNOWN_DATASETS:
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

# ---------- consumers endpoint ----------
@app.get("/api/v1/consumers")
def api_consumers(service: str = Query(...), path: Optional[str] = Query(None)) -> Dict[str, Any]:
    g = load_graph()
    if not service:
        raise HTTPException(400, "service query parameter required")
    changed_paths = [path] if path else []
    be_imp = backend_impacts(g, service, changed_paths)
    fe_imp = ui_imp(g, service, changed_paths)
    return {"producer": clean_name(service), "backend_consumers": be_imp, "frontend_consumers": fe_imp}

# ---------- versioning endpoint ----------
@app.get("/versioning")
def versioning(pair_id: str = Query(..., alias="pair_id")) -> Dict[str, Any]:
    vm = version_meta_lookup(pair_id)
    if not vm:
        raise HTTPException(status_code=404, detail=f"version metadata not found for pair_id={pair_id}")
    return vm

# ---------- train endpoint ----------
@app.post("/train")
def train(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models/last_train.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return {"ok": True, "samples": len(samples)}

# ---------- run (development) ----------
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    except Exception:
        log.info("uvicorn not available or running as import; use `uvicorn server:app --reload` to run")
