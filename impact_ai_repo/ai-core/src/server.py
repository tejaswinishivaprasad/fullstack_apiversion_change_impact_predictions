#!/usr/bin/env python3
"""
server.py

Impact AI core HTTP server (updated for dataset-specific curated layout).
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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import networkx as nx
import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import JSONResponse

# logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("impact-ai-core")

# ------------ curated root / variant resolution ------------
# CURATED_ROOT may be a container (default "datasets") or a legacy curated folder.
CURATED_ROOT_RAW = Path(os.getenv("CURATED_ROOT", "datasets"))
# explicit override for selecting a variant (optional)
CURATION_VARIANT = os.getenv("CURATION_VARIANT", None)

# base dataset names
BASE_DATASETS = ["openapi", "petclinic", "openrewrite"]

# variant folder names mapping
VARIANT_MAP = {
    "clean": "curated_clean",
    "noisy_light": "curated_noisy_light",
    "noisy_heavy": "curated_noisy_heavy",
}

def _discover_container(root: Path) -> Path:
    # return container dir that holds curated_* folders; fallback to parent
    try:
        if root.exists():
            if (root / "index.json").exists() or (root / "canonical").exists() or root.name.startswith("curated"):
                return root.parent
            if any(p.is_dir() and p.name.startswith("curated") for p in root.iterdir()):
                return root
        return root.parent
    except Exception:
        return root.parent

CURATED_CONTAINER = _discover_container(CURATED_ROOT_RAW)

def _variant_dir_for(variant_folder: str) -> Path:
    # return the directory for a variant name, try multiple sensible locations
    cand = CURATED_CONTAINER / variant_folder
    if cand.exists():
        return cand
    alt = CURATED_ROOT_RAW / variant_folder
    if alt.exists():
        return alt
    # if CURATED_ROOT_RAW itself is a curated folder, use it
    if CURATED_ROOT_RAW.exists() and (CURATED_ROOT_RAW / "index.json").exists():
        return CURATED_ROOT_RAW
    return cand  # may not exist, caller will handle

# effective default curated root uses curated_clean if present else CURATED_ROOT_RAW
DEFAULT_VARIANT = VARIANT_MAP["clean"]
EFFECTIVE_CURATED_ROOT = _variant_dir_for(DEFAULT_VARIANT) if _variant_dir_for(DEFAULT_VARIANT).exists() else CURATED_ROOT_RAW

# global file paths bound to effective curated root (clean by default)
INDEX_PATH = EFFECTIVE_CURATED_ROOT / "index.json"
GRAPH_PATH = EFFECTIVE_CURATED_ROOT / "graph.json"
VERSION_META_PATH = EFFECTIVE_CURATED_ROOT / "version_meta.json"
DATASET_OINDEX_PATH = EFFECTIVE_CURATED_ROOT / "dataset_oindex.json"
VERSION_PAIRS_CSV = EFFECTIVE_CURATED_ROOT / "version_pairs.csv"

log.info("Curated container=%s effective_curated=%s variant_override=%s", CURATED_CONTAINER, EFFECTIVE_CURATED_ROOT, CURATION_VARIANT)

# Ensure logs folder exists.
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Known dataset names (base list)
KNOWN_DATASETS = list(BASE_DATASETS)

# ---------- helpers to parse dataset keys ----------
def _parse_dataset_key(ds_key: str) -> Tuple[str, str]:
    # parse dataset key like "openapi_noisy_light" -> ("openapi","curated_noisy_light")
    if not ds_key:
        return ds_key, VARIANT_MAP["clean"]
    ds_key = ds_key.lower()
    if ds_key.endswith("_noisy_light"):
        return ds_key[: -len("_noisy_light")], VARIANT_MAP["noisy_light"]
    if ds_key.endswith("_noisy_heavy"):
        return ds_key[: -len("_noisy_heavy")], VARIANT_MAP["noisy_heavy"]
    # plain name -> clean variant
    return ds_key, VARIANT_MAP["clean"]

def all_dataset_keys() -> List[str]:
    # returns the 9 dataset keys (3 base × 3 variants)
    keys: List[str] = []
    for b in BASE_DATASETS:
        keys.append(b)
        keys.append(f"{b}_noisy_light")
        keys.append(f"{b}_noisy_heavy")
    return keys

def is_valid_dataset_key(ds_key: str) -> bool:
    # valid if base is known
    base, _ = _parse_dataset_key((ds_key or "").lower())
    return base in BASE_DATASETS

# ---------- dataset path resolution ----------
def dataset_base(ds_key: str) -> Path:
    # return the base folder for a dataset key
    base_name, variant_folder = _parse_dataset_key(ds_key)
    if CURATION_VARIANT:
        variant_folder = CURATION_VARIANT
    variant_dir = _variant_dir_for(variant_folder)
    return variant_dir / base_name

def dataset_paths(ds_key: str) -> Dict[str, Path]:
    # return canonical/ndjson/metadata paths for a dataset key
    base = dataset_base(ds_key)
    return {
        "base": base,
        "canonical": base / "canonical",
        "ndjson": base / "ndjson",
        "metadata": base / "metadata",
    }

# legacy single-folder locations for backwards compatibility
LEGACY_CANONICAL = EFFECTIVE_CURATED_ROOT / "canonical"
LEGACY_NDJSON = EFFECTIVE_CURATED_ROOT / "ndjson"
LEGACY_METADATA = EFFECTIVE_CURATED_ROOT / "metadata"

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

# ---------- risk utilities ----------
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

# ---------- ACE index (lazy conservative build) ----------
ACE_INDEX: Dict[str, Path] = {}

def build_ace_index(limit_per_file: int = 10, max_files: Optional[int] = None):
    # build a conservative ace_id -> ndjson path map
    try:
        idx = load_pair_index()
        files_scanned = 0
        for pid, meta in idx.items():
            if max_files and files_scanned >= max_files:
                break
            nd = None
            ds_hint = meta.get("dataset")
            if ds_hint:
                nd = dataset_paths(ds_hint)["ndjson"] / f"{pid}.aces.ndjson"
            if not nd or not nd.exists():
                # try each base dataset and variant keys
                for key in all_dataset_keys():
                    cand = dataset_paths(key)["ndjson"] / f"{pid}.aces.ndjson"
                    if cand.exists():
                        nd = cand
                        break
            if not nd or not nd.exists():
                for cand in [LEGACY_NDJSON / f"{pid}.aces.ndjson", EFFECTIVE_CURATED_ROOT / f"{pid}.aces.ndjson"]:
                    if cand.exists():
                        nd = cand
                        break
            if nd and nd.exists():
                files_scanned += 1
                try:
                    with nd.open("r", encoding="utf-8") as fh:
                        for i, ln in enumerate(fh):
                            if not ln.strip():
                                continue
                            try:
                                o = json.loads(ln)
                            except Exception:
                                continue
                            aid = o.get("ace_id") or o.get("aceId")
                            if aid:
                                ACE_INDEX[str(aid)] = nd
                                ACE_INDEX[unquote(str(aid))] = nd
                            if limit_per_file and i+1 >= limit_per_file:
                                break
                except Exception:
                    log.exception("build_ace_index: failed reading %s", nd)
        log.info("ACE_INDEX built files=%d entries=%d", files_scanned, len(ACE_INDEX))
    except Exception:
        log.exception("build_ace_index failed")

# do not eagerly build full index by default; keep ACE_INDEX empty until first use
# call build_ace_index(...) manually or set up a background job if you want full index

# ---------- index loader (dataset-aware) ----------
@lru_cache(maxsize=4)
def load_pair_index(dataset_hint: Optional[str] = None) -> Dict[str, Any]:
    # Variant-aware index loader. If dataset_hint provided, prefer that dataset's variant index.
    # Otherwise, try to load from all variant index.json files and fall back to EFFECTIVE_CURATED_ROOT.
    out: Dict[str, Any] = {}

    # helper to try reading an index file path and merge into out
    def try_load_index(p: Path):
        try:
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out.update(data)
            elif isinstance(data, list):
                for e in data:
                    pid = e.get("pair_id") or e.get("pair") or None
                    if pid:
                        out[pid] = e
        except Exception:
            log.exception("load_pair_index: failed to read/parse %s", p)

    # 1) If caller provided a dataset_hint, try that dataset's variant index first
    if dataset_hint:
        base, variant_folder = _parse_dataset_key((dataset_hint or "").lower())
        # try variant's root index.json
        vroot = _variant_dir_for(variant_folder)
        try_load_index(vroot / "index.json")
        # try dataset-level index under variant dir (e.g., curated_noisy_light/openapi/index.json)
        try:
            ds_index = dataset_paths(dataset_hint)["base"] / "index.json"
            try_load_index(ds_index)
        except Exception:
            pass

    # 2) try all variant roots (curated_clean, curated_noisy_light, curated_noisy_heavy)
    for vf in VARIANT_MAP.values():
        vroot = _variant_dir_for(vf)
        try_load_index(vroot / "index.json")

    # 3) final fallback: EFFECTIVE_CURATED_ROOT/index.json (keeps legacy behavior)
    try_load_index(EFFECTIVE_CURATED_ROOT / "index.json")

    # 4) If still empty, scan per-dataset metadata folders as before
    if not out:
        for base in BASE_DATASETS:
            md = dataset_paths(base)["metadata"]
            if md.exists():
                for p in md.glob("*.meta.json"):
                    try:
                        d = json.loads(p.read_text(encoding="utf-8"))
                        pid = d.get("pair_id") or p.stem
                        d.setdefault("dataset", base)
                        out[pid] = d
                    except Exception:
                        log.warning("Failed to parse meta file %s", p)
        # legacy metadata fallback
        if LEGACY_METADATA.exists():
            for p in LEGACY_METADATA.glob("*.meta.json"):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    pid = d.get("pair_id") or p.stem
                    out.setdefault(pid, d)
                except Exception:
                    log.warning("Failed to parse legacy meta file %s", p)

    return out


def load_pair_metadata(pair_id: Optional[str], dataset_hint: Optional[str] = None) -> Dict[str, Any]:
    # return metadata object for a pair_id
    if not pair_id:
        return {}
    idx = load_pair_index(dataset_hint)
    if pair_id in idx:
        return idx[pair_id] or {}
    for base in BASE_DATASETS:
        mpath = dataset_paths(base)["metadata"] / f"{pair_id}.meta.json"
        if mpath.exists():
            try:
                d = json.loads(mpath.read_text(encoding="utf-8"))
                d.setdefault("dataset", base)
                return d
            except Exception:
                log.warning("Failed to parse meta %s", mpath)
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

# ---------- simple graph helpers ----------
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
    # Variant-aware lookup for version metadata: check each curated_* variant first,
    # then the effective curated root, then per-pair metadata files.
    if not pair_id:
        return {}

    # 1) check variant roots (curated_clean, curated_noisy_light, curated_noisy_heavy)
    try:
        for variant_folder in VARIANT_MAP.values():
            vdir = _variant_dir_for(variant_folder)
            vm_path = vdir / "version_meta.json"
            if vm_path.exists():
                try:
                    jm = read_json(vm_path)
                    if isinstance(jm, dict) and pair_id in jm:
                        log.info("version_meta_lookup: found %s in %s", pair_id, vm_path)
                        return jm.get(pair_id, {}) or {}
                except Exception:
                    log.exception("version_meta_lookup: failed to read %s", vm_path)
    except Exception:
        log.exception("version_meta_lookup: variant scan failed")

    # 2) check the effective curated root's version_meta.json as previously
    try:
        if VERSION_META_PATH.exists():
            jm = read_json(VERSION_META_PATH)
            if isinstance(jm, dict) and pair_id in jm:
                log.info("version_meta_lookup: found %s in effective %s", pair_id, VERSION_META_PATH)
                return jm.get(pair_id, {}) or {}
    except Exception:
        log.exception("version_meta_lookup: failed reading effective VERSION_META_PATH")

    # 3) fall back to pair-specific metadata files (dataset-specific metadata folders)
    try:
        meta = load_pair_metadata(pair_id)
        if meta:
            log.info("version_meta_lookup: found %s via load_pair_metadata()", pair_id)
            return meta
    except Exception:
        log.exception("version_meta_lookup: load_pair_metadata failed for %s", pair_id)

    # not found
    log.info("version_meta_lookup: no version metadata for %s", pair_id)
    return {}


# backend/ui impact computations
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

def assemble_feature_record(details: List[DiffItem], pfeats: Dict[str, Any], vfeats: Dict[str, Any], be_imp: List[Dict[str, Any]], fe_imp: List[Dict[str, Any]]) -> Dict[str, Any]:
    # assemble a compact feature record
    record: Dict[str, Any] = {}
    record["num_changes"] = len(details) if details is not None else 0
    types: Dict[str, int] = {}
    for d in details or []:
        t = (d.type or "UNKNOWN")
        types[t] = types.get(t, 0) + 1
    for t in ["endpoint_removed", "endpoint_added", "param_removed", "param_added", "param_changed", "response_schema_changed", "requestbody_schema_changed"]:
        record[f"ct_{t}"] = types.get(t, 0)
    record["centrality"] = float(pfeats.get("centrality", 0.0)) if pfeats else 0.0
    record["ui_consumers"] = int(pfeats.get("ui_consumers", 0)) if pfeats else 0
    record["two_hop"] = int(pfeats.get("two_hop", 0)) if pfeats else 0
    record["backend_imp_count"] = len(be_imp) if be_imp is not None else 0
    record["frontend_imp_count"] = len(fe_imp) if fe_imp is not None else 0
    record["semantic_sim"] = float(vfeats.get("semantic_sim", 0.0)) if vfeats else 0.0
    record["recency_days"] = float(vfeats.get("days_since_last_release", 9999)) if vfeats else 9999.0
    record["change_freq"] = float(vfeats.get("historical_change_rate", 0.0)) if vfeats else 0.0
    record["breaking_vs_semver"] = 1 if vfeats and vfeats.get("breaking_vs_semver") else 0
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
    record["propagation_potential"] = record["backend_imp_count"] + 2 * record["frontend_imp_count"]
    return record

# ---------- canonicalize operation + diff functions ----------
def canonicalize_operation(op: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # normalize operation representation
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
    # compute diffs between two operations
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
    # compute diffs between two openapi documents
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
    # dedupe list of DiffItem
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
@lru_cache(maxsize=128)
def list_files(dataset: str) -> List[str]:
    """
    Return sample filenames for a dataset key.
    Prefer scanning the dataset-specific curated folders (variant-aware).
    Fall back to global index.json / legacy locations if dataset folders are absent.
    """
    ds = (dataset or "").lower()
    samples: List[str] = []

    # 1) Try dataset-specific curated folders first (variant-aware).
    dp = dataset_paths(ds)
    can = dp["canonical"]
    nd = dp["ndjson"]
    md = dp["metadata"]

    # scan canonical jsons if present
    if can.exists():
        for p in can.rglob("*.json"):
            samples.append(p.name)

    # scan ndjson (aces) if present
    if nd.exists():
        for p in nd.rglob("*.aces.ndjson"):
            samples.append(p.name)

    # scan metadata files if present (pair ids may be stored here)
    if md.exists():
        for p in md.glob("*.meta.json"):
            samples.append(p.name.replace(".meta.json", ""))

    # If we found anything in dataset-specific places, return them (deduped & sorted)
    if samples:
        samples = sorted(dict.fromkeys(samples))
        return samples

    # 2) Dataset-specific folders empty/missing -> fallback to global index.json (legacy behavior)
    idx = load_pair_index(ds)
    if idx:
        for pid, entry in idx.items():
            # match by base dataset name (ignore variant suffixes in requested ds)
            entry_ds = (entry.get("dataset") or "").lower()
            # allow match if entry dataset matches base of requested ds
            if ds and _parse_dataset_key(entry_ds)[0] != _parse_dataset_key(ds)[0]:
                continue
            oldv = entry.get("old_canonical") or entry.get("old")
            newv = entry.get("new_canonical") or entry.get("new")
            if oldv:
                samples.append(Path(str(oldv)).name)
            if newv:
                samples.append(Path(str(newv)).name)

    # 3) still empty -> try global canonical/ndjson legacy locations
    if not samples:
        # try effective curated root canonical
        if LEGACY_CANONICAL.exists():
            for p in LEGACY_CANONICAL.rglob("*.json"):
                samples.append(p.name)
        if LEGACY_NDJSON.exists():
            for p in LEGACY_NDJSON.rglob("*.aces.ndjson"):
                samples.append(p.name)

    samples = sorted(dict.fromkeys(samples))
    return samples

# resolve a filename relative to a dataset key into an existing Path (variant-aware)
def resolve_file_for_dataset(ds_key: str, rel: str) -> Optional[Path]:
    # quick guards
    if not rel:
        return None
    cand = Path(rel)
    if cand.is_absolute() and cand.exists():
        return cand

    ds = (ds_key or "").lower()
    # prefer dataset-specific locations (variant-aware)
    dp = dataset_paths(ds)
    for folder in (dp["canonical"], dp["ndjson"], dp["metadata"]):
        p = folder / rel
        if p.exists():
            return p

    # some metadata/index entries may store paths relative to dataset base
    p2 = dp["base"] / rel
    if p2.exists():
        return p2

    # use global index to try to locate canonical filenames across variants
    idx = load_pair_index(ds)
    for pid, meta in idx.items():
        try:
            oc = meta.get("old_canonical") or meta.get("old")
            nc = meta.get("new_canonical") or meta.get("new")
            if oc and Path(oc).name == rel:
                ds_meta = meta.get("dataset")
                if ds_meta:
                    candp = dataset_paths(ds_meta)["canonical"] / rel
                    if candp.exists():
                        return candp
            if nc and Path(nc).name == rel:
                ds_meta = meta.get("dataset")
                if ds_meta:
                    candp = dataset_paths(ds_meta)["canonical"] / rel
                    if candp.exists():
                        return candp
        except Exception:
            continue

    # legacy fallbacks
    lc = LEGACY_CANONICAL / rel
    if lc.exists():
        return lc
    ln = LEGACY_NDJSON / rel
    if ln.exists():
        return ln

    # check effective curated root for raw files named directly under it
    lp = EFFECTIVE_CURATED_ROOT / rel
    if lp.exists():
        return lp

    # nothing found
    return None



# ---------- HTTP endpoints ----------
@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}

@app.get("/datasets")
def datasets() -> List[str]:
    # return the 9 dataset keys for UI selection
    return all_dataset_keys()

@app.get("/files")
def files(dataset: str = Query(...)) -> Dict[str, Any]:
    """
    Return samples for a dataset key. Accepts variant keys like 'petclinic_noisy_heavy'.
    """
    ds = (dataset or "").lower()
    if not is_valid_dataset_key(ds):
        raise HTTPException(404, f"Unknown dataset {dataset}")
    rels = list_files(ds)
    return {"samples": rels, "count": len(rels)}

@app.get("/graph")
def graph() -> Dict[str, Any]:
    return read_json(GRAPH_PATH)

# --- Feature export & scoring helpers ---
def normalize_and_export_features(records: List[Dict[str, Any]], out_csv: Path = LOGS_DIR / "features.csv", stats_json: Path = LOGS_DIR / "feature_stats.json") -> None:
    # normalize numeric features and write CSV + stats
    try:
        if not records:
            log.info("normalize_and_export_features: no records to export")
            return
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
    # heuristic risk scoring
    try:
        s = 0.0
        for d in (details or []):
            key = (d.type or "").lower()
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
    # produce human readable explanation
    try:
        headline = f"Predicted risk is {_risk_band(score)} ({score:.2f})."
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
        side_effects = any((d.method or "").lower() in ("post", "put", "patch", "delete") for d in (details or []))
        if side_effects:
            bullets.append("Contains non-GET changes (possible side-effects).")
        shared_schema = any(isinstance(d.detail, str) and ("schema" in d.detail.lower() or "shared" in d.detail.lower()) for d in (details or []))
        if shared_schema:
            bullets.append("Some changes reference shared schemas (schema coupling).")
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
                conf = None
                try:
                    if isinstance(d.detail, dict) and "confidence" in d.detail:
                        conf = d.detail.get("confidence")
                except Exception:
                    conf = None
                conf_str = f" conf={conf:.2f}" if isinstance(conf, (float, int)) else ""
                bullets.append(f"- [{tid}] {typ} {method} {path}{conf_str}")
        text = headline + "\n" + "\n".join(["• " + b for b in bullets])
        return text
    except Exception:
        log.exception("make_explanation failed; returning short message")
        return f"Predicted risk: {_risk_band(score)} ({score:.2f})."

# ---------- main report endpoint ----------
@app.get("/report", response_model=Report)
def report(dataset: str = Query(...), old: str = Query(...), new: str = Query(...), pair_id: Optional[str] = None) -> Report:
    # main report handler
    start = time.time()
    try:
        ds = (dataset or "").lower()
        old, new = unquote(old), unquote(new)
        if old == new:
            return Report(dataset=ds, old_file=old, new_file=new, risk_score=0.0, risk_band="None", summary="No change (same file)", details=[], ai_explanation="No diff detected.", backend={"producer": Path(new).stem, "features": {}}, logs=["same-file"], backend_impacts=[], frontend_impacts=[], predicted_risk=0.0, confidence={"overall": 0.0, "backend": 0.0, "frontend": 0.0}, versioning={})
        if not is_valid_dataset_key(ds):
            raise HTTPException(404, f"Unknown dataset {dataset}")
        pair_meta = load_pair_metadata(pair_id,ds) if pair_id else None
        p_old: Optional[Path] = None
        p_new: Optional[Path] = None
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
            idx = load_pair_index(ds)
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
        details: List[DiffItem] = []
        if pair_identifier:
            ndpath_candidates: List[Path] = []
            if pair_meta and pair_meta.get("dataset"):
                ndpath_candidates.append(dataset_paths(pair_meta.get("dataset"))["ndjson"] / f"{pair_identifier}.aces.ndjson")
            for key in all_dataset_keys():
                ndpath_candidates.append(dataset_paths(key)["ndjson"] / f"{pair_identifier}.aces.ndjson")
            ndpath_candidates.append(LEGACY_NDJSON / f"{pair_identifier}.aces.ndjson")
            ndpath_candidates.append(EFFECTIVE_CURATED_ROOT / f"{pair_identifier}.aces.ndjson")
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
        if not details:
            base_ds = _parse_dataset_key(ds)[0]
            if base_ds == "openapi":
                details = diff_openapi(old_doc, new_doc)
            else:
                details = []
        details = dedupe_diffitems(details)
        g = load_graph()
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
    # analyze either by dataset key or by raw payloads
    if dataset:
        ds = (dataset or "").lower()
        if not is_valid_dataset_key(ds):
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
    # return backend/frontend consumers for a service
    g = load_graph()
    if not service:
        raise HTTPException(400, "service query parameter required")
    changed_paths = [path] if path else []
    be_imp = backend_impacts(g, service, changed_paths)
    fe_imp = ui_impacts(g, service, changed_paths)
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

def _ndjson_candidates_for_pair(pair_id: Optional[str]) -> List[Path]:
    # return candidate ndjson files for a pair_id
    candidates: List[Path] = []
    if not pair_id:
        return candidates
    try:
        meta = load_pair_metadata(pair_id)
        if meta and meta.get("dataset"):
            candidates.append(dataset_paths(meta.get("dataset"))["ndjson"] / f"{pair_id}.aces.ndjson")
        for key in all_dataset_keys():
            candidates.append(dataset_paths(key)["ndjson"] / f"{pair_id}.aces.ndjson")
        candidates.append(LEGACY_NDJSON / f"{pair_id}.aces.ndjson")
        candidates.append(EFFECTIVE_CURATED_ROOT / f"{pair_id}.aces.ndjson")
    except Exception:
        log.exception("Failed to assemble ndjson candidates for %s", pair_id)
    return candidates

def _find_ace_object(pair_id: Optional[str], ace_id: str) -> Optional[Dict[str, Any]]:
    # find an ace object by id using ACE_INDEX then scanning candidates
    if not ace_id:
        return None
    if ACE_INDEX:
        path = ACE_INDEX.get(str(ace_id)) or ACE_INDEX.get(unquote(str(ace_id)))
        if path and path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for ln in fh:
                        if not ln.strip():
                            continue
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        aid = obj.get("ace_id") or obj.get("aceId") or None
                        if aid and (str(aid) == str(ace_id) or str(aid) == unquote(str(ace_id)) or unquote(str(aid)) == str(ace_id)):
                            return obj
            except Exception:
                log.exception("Failed reading indexed NDJSON %s for ace_id=%s", path, ace_id)
    candidates = _ndjson_candidates_for_pair(pair_id) if pair_id else []
    for p in candidates:
        if not p or not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    aid = obj.get("ace_id") or obj.get("aceId") or None
                    if aid:
                        if str(aid) == str(ace_id) or str(aid) == unquote(str(ace_id)) or unquote(str(aid)) == str(ace_id):
                            return obj
        except Exception:
            log.exception("Failed reading NDJSON %s while searching for ace_id=%s", p, ace_id)
    return None

@app.get("/ace")
def ace(request: Request, ace_id: str = Query(..., alias="ace_id"), pair_id: Optional[str] = Query(None, alias="pair_id")) -> Dict[str, Any]:
    # retrieve ace object endpoint
    rid = request.headers.get("X-Request-ID") or f"rid-{int(time.time()*1000)}"
    try:
        if not ace_id or not ace_id.strip():
            raise HTTPException(status_code=400, detail="ace_id required")
        try:
            raw = ace_id
            ace_id = unquote(ace_id)
            ace_id = unquote(ace_id)
        except Exception:
            ace_id = raw
        if pair_id:
            try:
                p_raw = pair_id
                pair_id = unquote(pair_id)
                pair_id = unquote(pair_id)
            except Exception:
                pair_id = p_raw
        log.info("ACE request %s ace_id=%s pair_id=%s", rid, ace_id, pair_id)
        obj = _find_ace_object(pair_id, ace_id)
        if not obj:
            tried = [str(p) for p in (_ndjson_candidates_for_pair(pair_id) if pair_id else [])][:6]
            log.info("ACE not found %s ace_id=%s pair_id=%s tried=%s", rid, ace_id, pair_id, tried)
            raise HTTPException(status_code=404, detail={"error": "ace-not-found", "ace_id": ace_id, "pair_id": pair_id, "tried_files": tried})
        return JSONResponse(content=obj, headers={"X-Request-ID": rid})
    except HTTPException:
        raise
    except Exception as e:
        log.exception("ace handler failed (%s): %s", rid, e)
        raise HTTPException(status_code=500, detail={"error": "ace-failed", "message": str(e), "request_id": rid})

# ---------- run (development) ----------
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    except Exception:
        log.info("uvicorn not available or running as import; use `uvicorn server:app --reload` to run")
