#!/usr/bin/env python3
"""
curated_datasets.py - end-to-end curated dataset generator.

Purpose:
 - Produce canonical/, ndjson/, metadata/, index.json, version_meta.json, version_pairs.csv
 - Uses robust canonicalization and safe deterministic mapping from raw OpenAPI files
 - Guarantee safe pair_id generation (pair-<hex>) to avoid URL encoding/lookup issues
 - Provide dry-run mode for inspection before writing to disk


Sample run command:
  python curated_datasets.py --out datasets/curated --max 400 --seed 42
"""
from __future__ import annotations
import argparse
import csv
import datetime
import decimal
import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing as mp
import hashlib as _hashlib
import yaml

# ---------- Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- Defaults (override via CLI)
RAW_ROOT = Path("datasets/raw")
CURATED_ROOT = Path("datasets/curated")

OPENAPI_REPO = "https://github.com/APIs-guru/openapi-directory.git"
PETCLINIC_REPO = "https://github.com/spring-petclinic/spring-petclinic-microservices.git"
OPENREWRITE_REPO = "https://github.com/openrewrite/rewrite.git"

_MAX_CANONICAL_DEPTH = 2000

# ---------- Module-level caches and pool
_resolved_external_cache: Dict[str, Any] = {}
_canon_cache: Dict[str, Tuple[Optional[Dict[str, Any]], List[str]]] = {}
_CANON_POOL: Optional[ProcessPoolExecutor] = None

# ---------- Small deterministic pools for realistic naming
REAL_SERVICE_POOL = [
    "payment-service", "order-service", "inventory-service", "transaction-service",
    "user-service", "billing-service", "auth-service", "notifications-service",
    "catalog-service", "shipping-service", "analytics-service", "search-service"
]

UI_SERVICE_POOL = ["admin-ui", "storefront-ui", "dashboard-ui", "mobile-ui"]

RESOURCE_BASES = [
    "payments", "orders", "inventory", "transactions", "users",
    "billing", "shipments", "products", "carts", "reviews", "profiles"
]

# ---------- Utilities

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

def _default_json_serializer(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, datetime.datetime):
        dt = obj
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).replace(microsecond=0).isoformat()
    if isinstance(obj, (datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return list(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)

def sha_for_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]

# safe JSON writes
def _write_json(p: Path, obj):
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default_json_serializer), encoding="utf-8")

def _write_json_sorted(p: Path, obj):
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=_default_json_serializer), encoding="utf-8")

# ---------- Deterministic 'realistic' helpers (kept from your version)
def _deterministic_choice_from_str(seed_str: str, pool: List[str]) -> str:
    if not pool:
        return seed_str[:24]
    try:
        h = int(_hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:16], 16)
        return pool[h % len(pool)]
    except Exception:
        return pool[0]

def pick_real_service(seed_str: str) -> str:
    return _deterministic_choice_from_str(seed_str, REAL_SERVICE_POOL)

def pick_ui_service(seed_str: str) -> str:
    return _deterministic_choice_from_str(seed_str, UI_SERVICE_POOL)

def _normalize_token(token: str) -> str:
    t = re.sub(r"auto|autogen|autox|synth|synthetic|svc|ui", "", token, flags=re.I)
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    if not t:
        return token.lower()
    return t

def make_path_realistic(path: str, seed_str: str = "") -> str:
    if not path or not isinstance(path, str):
        return path
    p = path.strip()
    p = re.sub(r"//+", "/", p)
    def norm_param(m):
        inner = m.group(1)
        inner2 = re.sub(r"[^a-zA-Z0-9_]+", "_", inner).strip("_")
        if not inner2:
            inner2 = "id"
        base = _deterministic_choice_from_str(seed_str + inner2, ["id", "paymentId", "orderId", "sku", "userId"])
        return "{" + base + "}"
    p = re.sub(r"\{([^}]+)\}", norm_param, p)
    if re.search(r"/auto|autogen|autox|synth|synthetic", p, flags=re.I) or re.search(r"/\d{2,}", p):
        base = _deterministic_choice_from_str(seed_str + p, RESOURCE_BASES)
        return f"/{base}/{{id}}"
    if re.search(r"/\d+$", p):
        p = re.sub(r"/\d+$", "/{id}", p)
        return p
    tokens = [t for t in p.split("/") if t]
    if len(tokens) == 1:
        base = _deterministic_choice_from_str(seed_str + tokens[0], RESOURCE_BASES)
        return f"/{base}/{{id}}"
    new_tokens = []
    for t in tokens:
        cleaned = _normalize_token(t)
        if re.search(r"\d{2,}", t):
            new_tokens.append("{id}")
        else:
            new_tokens.append(cleaned or t)
    return "/" + "/".join(new_tokens)

def realisticize_canonical(canon: Dict[str, Any], seed_str: str):
    if not isinstance(canon, dict):
        return canon, []
    paths = canon.get("paths") or {}
    new_paths = {}
    op_samples = []
    for p in sorted(list(paths.keys())):
        safe_p = str(p)
        new_p = make_path_realistic(safe_p, seed_str + safe_p)
        methods = paths.get(p) or {}
        new_methods = {}
        for method_name, op in (methods.items() if isinstance(methods, dict) else []):
            if not isinstance(op, dict):
                new_methods[method_name] = op
                continue
            new_op = deepcopy(op)
            opid = new_op.get("operationId") or new_op.get("summary") or f"{method_name}_{re.sub(r'[^a-z0-9]+','_', new_p)}"
            opid_seed = seed_str + str(opid)
            new_op["operationId"] = _normalize_token(_deterministic_choice_from_str(opid_seed, [opid, opid + "_v2"])) or opid
            params = new_op.get("parameters") or []
            for param in (params if isinstance(params, list) else []):
                if isinstance(param, dict) and param.get("name"):
                    n = param["name"]
                    if re.search(r"^auto|^param|^\d", n, flags=re.I) or len(n) > 24:
                        param["name"] = _deterministic_choice_from_str(seed_str + n, ["id", "status", "limit", "offset", "q", "sku", "email"])
            new_methods[method_name] = new_op
            op_samples.append(new_p)
        new_paths[new_p] = new_methods
    canon["paths"] = new_paths
    return canon, sorted(list(set(op_samples)))[:10]

# ---------- Robust file reads with caching
@lru_cache(maxsize=4096)
def _read_file_cached_text(path_str: str) -> Optional[str]:
    try:
        txt = Path(path_str).read_text(encoding="utf-8")
        if not txt.strip():
            return None
        return txt
    except Exception as e:
        logging.debug(f"_read_file_cached_text failed {path_str}: {e}")
        return None

def _read_file(p: Path) -> Optional[Any]:
    """
    Robust read: try JSON, then YAML multi-doc (safe_load_all).
    Returns the first meaningful dict/list document, or None.
    """
    txt = _read_file_cached_text(str(p))
    if not txt:
        return None

    try:
        return json.loads(txt)
    except Exception:
        pass

    try:
        docs = list(yaml.safe_load_all(txt))
    except Exception as e:
        logging.debug(f"yaml.safe_load_all failed for {p}: {e}")
        # fallback: try single-doc YAML load
        try:
            return yaml.safe_load(txt)
        except Exception as e2:
            logging.debug(f"yaml.safe_load fallback also failed for {p}: {e2}")
            return None

    # Filter out empty docs
    docs = [d for d in docs if d not in (None, {}, [], "")]
    if not docs:
        return None

    # Prefer first dict/list doc
    for d in docs:
        if isinstance(d, (dict, list)):
            return d

    # Fallback: return first doc
    return docs[0]


_BAD_DIR = Path("datasets/raw/openrewrite_repo_malformed")

def _save_bad_file(path: Path, reason: str):
    """
    Save a copy of a problematic raw file (and a reason.txt) to a debug folder.
    Non-fatal; used only to aid manual inspection.
    """
    try:
        _ensure_dir(_BAD_DIR)
        safe_name = sha_for_text(str(path)) + "-" + path.name
        dst = _BAD_DIR / safe_name
        try:
            shutil.copy2(path, dst)
        except Exception:
            try:
                dst.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            except Exception:
                pass
        try:
            (dst.with_suffix(".reason.txt")).write_text(reason or "unknown", encoding="utf-8")
        except Exception:
            pass
    except Exception:
        logging.debug(f"_save_bad_file failed for {path}")

def is_noise_path(p: Path) -> bool:
    """
    Heuristic to skip non-API/noisy files (workflows, examples, test resources, meta files).
    Return True if file should be skipped early.
    """
    s = str(p.as_posix()).lower()
    #  directories and file stems that are not API specs
    skip_keywords = [
        "/.github/", "/.github/workflows/", "/meta-inf/", "/test/", "/tests/", "/examples/",
        "/.circleci/", "/.travis/", "/ci/", "/jenkins/", "readme", "changelog", "change-log",
        "license", "pom.xml", "gradle", ".gitignore"
    ]
    for kw in skip_keywords:
        if kw in s:
            return True
    # skip tiny or massive files (because they may not likely API specs)
    try:
        st = p.stat()
        if st.st_size < 120:   # too small
            return True
        if st.st_size > 2_000_000:  # too big
            return True
    except Exception:
        pass
    # extension-based filter (non-json/yaml already filtered upstream)
    if p.suffix.lower() not in (".yml", ".yaml", ".json"):
        return True
    return False

# ---------- Canonicalizer (keeps  robust resolver and caches)
def resolve_local_pointer(root: Any, pointer: str):
    if pointer == "#" or pointer == "#/":
        return deepcopy(root)
    parts = pointer.lstrip("#/").split("/")
    cur = root
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"Pointer part '{p}' missing")
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(f"Cannot navigate pointer part '{p}'")
    return deepcopy(cur)

def canonicalize_spec(spec: Any, base_dir: Path):
    warnings: List[str] = []
    def _find_dict_candidate(obj, depth=0):
        if depth > 8:
            return None
        if isinstance(obj, dict):
            if obj.get("paths") or obj.get("openapi") or obj.get("swagger"):
                return obj
            return obj
        if isinstance(obj, list):
            for it in obj:
                cand = _find_dict_candidate(it, depth + 1)
                if cand is not None:
                    return cand
        return None
    if not isinstance(spec, dict):
        if isinstance(spec, list):
            cand = _find_dict_candidate(spec)
            if cand:
                spec = cand
            else:
                warnings.append("spec parsed as list; fallback to empty paths")
                return {"paths": {}}, warnings
        else:
            warnings.append("spec not dict; fallback to empty paths")
            return {"paths": {}}, warnings

    depth = 0
    def _rec(node, cur_base: Path, stack: Tuple[str, ...] = ()):
        nonlocal depth
        depth += 1
        if depth > _MAX_CANONICAL_DEPTH:
            warnings.append("canonicalization truncated due to depth")
            depth -= 1
            return {"__truncated__": True}
        try:
            if isinstance(node, dict):
                if "$ref" in node and isinstance(node["$ref"], str):
                    ref = node["$ref"]
                    if ref.startswith("#"):
                        try:
                            key = ("local", ref)
                            if key in stack:
                                warnings.append(f"cycle local {ref}")
                                depth -= 1
                                return {"__ref_cycle__": ref}
                            resolved = resolve_local_pointer(spec, ref)
                            result = _rec(resolved, cur_base, stack + (key,))
                            depth -= 1
                            return result
                        except Exception as e:
                            warnings.append(f"local ref {ref} failed: {e}")
                            depth -= 1
                            return {"__unresolved_ref__": ref}
                    else:
                        parts = ref.split("#", 1)
                        file_part = parts[0]
                        ptr = "#" + parts[1] if len(parts) > 1 else "#"
                        target_file = (cur_base / file_part).resolve()
                        key = (str(target_file), ptr)
                        if key in stack:
                            warnings.append(f"cycle external {ref}")
                            depth -= 1
                            return {"__ref_cycle__": ref}
                        if not target_file.exists():
                            warnings.append(f"external ref file missing: {target_file}")
                            depth -= 1
                            return {"__unresolved_ref__": ref}
                        try:
                            cache_key = f"{str(target_file.resolve())}::{ptr}"
                            if cache_key in _resolved_external_cache:
                                loaded = _resolved_external_cache[cache_key]
                            else:
                                loaded = _read_file(target_file)
                                if loaded is not None:
                                    _resolved_external_cache[cache_key] = loaded
                            if loaded is None:
                                warnings.append(f"external ref empty file: {target_file}")
                                depth -= 1
                                return {"__unresolved_ref__": ref}
                            resolved = resolve_local_pointer(loaded, ptr) if ptr != "#" else loaded
                            result = _rec(resolved, target_file.parent, stack + (key,))
                            depth -= 1
                            return result
                        except Exception as e:
                            warnings.append(f"external ref load failed {ref}: {e}")
                            depth -= 1
                            return {"__unresolved_ref__": ref}
                else:
                    out = {}
                    for k, v in node.items():
                        try:
                            out[k] = _rec(v, cur_base, stack)
                        except Exception as e:
                            warnings.append(f"child canonicalize error for key {k}: {e}")
                            out[k] = {"__canonicalize_error__": str(e)}
                    if any(k in out for k in ("oneOf", "anyOf", "allOf")):
                        out["_polymorphic_marker"] = {k: True for k in ("oneOf", "anyOf", "allOf") if k in out}
                    depth -= 1
                    return out
            elif isinstance(node, list):
                res = []
                for i in node:
                    try:
                        res.append(_rec(i, cur_base, stack))
                    except Exception as e:
                        warnings.append(f"list item canonicalize error: {e}")
                        res.append({"__canonicalize_error__": str(e)})
                depth -= 1
                return res
            else:
                depth -= 1
                return deepcopy(node)
        except RecursionError as re:
            warnings.append(f"recursion error: {re}")
            depth -= 1
            return {"__recursion_error__": True}
        except Exception as e:
            warnings.append(f"canonicalize node error: {e}")
            depth -= 1
            return {"__canonicalize_error__": str(e)}
    try:
        canon = _rec(spec, base_dir)
    except Exception as e:
        warnings.append(f"canonicalization failed top-level: {e}")
        canon = deepcopy(spec)
    if not isinstance(canon, dict) or "paths" not in canon:
        canon = {"paths": {}}
    return canon, warnings

# ---------- Parallel canonicalization pool
def _start_canon_pool(max_workers: int = max(1, (mp.cpu_count() or 2) // 2)):
    global _CANON_POOL
    if _CANON_POOL is None:
        _CANON_POOL = ProcessPoolExecutor(max_workers=max_workers)
    return _CANON_POOL

def _canon_worker_serialized(spec_text: str, base_dir_str: str):
    import json as _json, yaml as _yaml
    from pathlib import Path as _Path
    try:
        spec = _json.loads(spec_text)
    except Exception:
        spec = _yaml.safe_load(spec_text)
    canon, warns = canonicalize_spec(spec, _Path(base_dir_str))
    return {"canon": canon, "warns": warns}

def canonicalize_with_timeout(spec, base_dir, timeout_sec: int = 8):
    try:
        spec_blob = json.dumps(spec, sort_keys=True, default=_default_json_serializer)
    except Exception:
        spec_blob = yaml.safe_dump(spec)
    key = sha_for_text(spec_blob)
    if key in _canon_cache:
        return _canon_cache[key]
    pool = _start_canon_pool()
    try:
        spec_text = json.dumps(spec, default=_default_json_serializer)
    except Exception:
        spec_text = yaml.safe_dump(spec)
    fut = pool.submit(_canon_worker_serialized, spec_text, str(base_dir))
    try:
        res = fut.result(timeout=timeout_sec)
    except TimeoutError:
        try:
            fut.cancel()
        except Exception:
            pass
        _canon_cache[key] = (None, [f"canonicalize timed out after {timeout_sec}s"])
        return None, [f"canonicalize timed out after {timeout_sec}s"]
    except Exception as e:
        _canon_cache[key] = (None, [f"canonicalize worker exception: {e}"])
        return None, [f"canonicalize worker exception: {e}"]
    canon = res.get("canon")
    warns = res.get("warns", [])
    _canon_cache[key] = (canon, warns)
    return canon, warns

# ---------- Pair helpers (centralized, safe)
def generate_pair_id(service_name: str, old_sha: str, new_sha: str) -> str:
    """
    Always produce a safe pair id of the form: pair-<hex>
    Deterministic for given inputs.
    """
    key = f"{service_name}::{old_sha}::{new_sha}"
    return f"pair-{sha_for_text(key)}"

def make_pair_token(service_name: str, old_sha: str, new_sha: str) -> str:
    """
    Create a stable shared filename token for both versions of a pair.
    Deterministic: same old_sha/new_sha always result in same token.
    Short token (8 chars) to keep filenames readable.
    """
    key = f"{service_name}::{old_sha}::{new_sha}"
    return sha_for_text(key)[:8]

def make_canonical_filename(dataset: str, service_name: str, pair_token: str, version_tag: str) -> str:
    """
    Unified canonical filename formatting:
      <dataset>--<service>--<pairtoken>--v1.canonical.json
    """
    safe_ds = re.sub(r"[^0-9a-zA-Z]+", "-", dataset).strip("-").lower()
    safe_svc = re.sub(r"[^0-9a-zA-Z\-_]+", "-", service_name).strip("-").lower()
    v = version_tag.lower()
    return f"{safe_ds}--{safe_svc}--{pair_token}--{v}.canonical.json"

# ---------- Small helper code to normalize/repair loaded index entries
def _normalize_pair_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure every entry has a safe pair_id and expected fields.
    If legacy or broken pair_id exists (e.g. 'n/a'), repair by regenerating when possible.
    """
    pid = entry.get("pair_id") or entry.get("pair") or None
    svc = entry.get("service_name") or entry.get("service") or "unknown"
    old_sha = entry.get("old_sha") or entry.get("oldSha") or ""
    new_sha = entry.get("new_sha") or entry.get("newSha") or ""
    if not pid or not isinstance(pid, str) or pid.strip().lower() in ("n", "n/a", "na", "none", ""):
        # regenerate
        pid = generate_pair_id(svc, old_sha or "old", new_sha or "new")
        entry["pair_id"] = pid
    # ensure canonical filenames are strings
    for k in ("old_canonical", "new_canonical", "old", "new"):
        if k in entry and entry[k] is None:
            entry[k] = ""
    return entry

# ---------- Diff detection and ACE logic 
def inject_openapi_diffs(doc: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    r = random.Random(seed)
    new_doc = deepcopy(doc)
    paths = new_doc.setdefault("paths", {})
    candidates = list(paths.keys())
    r.shuffle(candidates)
    for p in candidates[: max(0, min(4, len(candidates)))]:
        methods = paths.get(p) or {}
        for m in list(methods.keys())[:1]:
            if isinstance(methods[m], dict):
                methods[m]["description"] = f"modified {m} {p}"
                if r.random() < 0.4:
                    params = methods[m].setdefault("parameters", [])
                    if r.random() < 0.6:
                        params.append({
                            "name": f"added_{r.randint(1,99)}",
                            "in": "query",
                            "schema": {"type": "string"},
                            "required": False
                        })
                    else:
                        if params:
                            idx = r.randrange(len(params))
                            if isinstance(params[idx], dict):
                                params[idx]["required"] = True
    for _ in range(r.randint(1, 4)):
        new_path = f"/auto{r.randint(100,9999)}"
        if r.random() < 0.5:
            paths[new_path] = {
                "get": {
                    "summary": "synthetic diff",
                    "responses": {"200": {"description": "ok"}}
                }
            }
        else:
            paths[new_path] = {
                "post": {
                    "summary": "synthetic write",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {
                                            "type": "string",
                                            "enum": ["on", "off", "maybe"]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "created"},
                        "400": {"description": "bad"}
                    }
                }
            }
    if candidates and r.random() < 0.2:
        try:
            del paths[candidates[0]]
        except Exception:
            pass
    for p, methods in list(paths.items()):
        for m, op in (methods or {}).items():
            if not isinstance(op, dict):
                continue
            if r.random() < 0.15:
                ps = op.setdefault("parameters", [])
                if ps and isinstance(ps, list):
                    for param in ps[:]:
                        if isinstance(param, dict):
                            sch = param.setdefault("schema", {})
                            if "enum" not in sch and r.random() < 0.2:
                                sch["enum"] = ["a", "b", "c"]
                            elif "enum" in sch and r.random() < 0.5:
                                if isinstance(sch["enum"], list) and len(sch["enum"]) > 1:
                                    sch["enum"] = sch["enum"][:-1]
            if "responses" in op and r.random() < 0.12:
                resp_keys = list(op["responses"].keys())
                if len(resp_keys) > 1 and r.random() < 0.7:
                    rm = r.choice(resp_keys)
                    try:
                        del op["responses"][rm]
                    except Exception:
                        pass
            params = op.get("parameters") or []
            for param in params:
                if isinstance(param, dict) and r.random() < 0.08:
                    sch = param.setdefault("schema", {})
                    if "type" in sch:
                        sch["type"] = "integer" if sch["type"] == "string" else "string"
    return new_doc

def compare_operations(old_op, new_op, path, method):
    def _norm_op(op):
        if isinstance(op, dict):
            return op
        if isinstance(op, list):
            for item in op:
                if isinstance(item, dict):
                    return item
            return {}
        return {}
    old_op = _norm_op(old_op)
    new_op = _norm_op(new_op)
    aces = []
    if not old_op and new_op:
        aces.append({"type": "ENDPOINT_ADDED", "path": path, "method": method})
        return aces
    if old_op and not new_op:
        aces.append({"type": "ENDPOINT_REMOVED", "path": path, "method": method})
        return aces
    old_params_list = (old_op.get("parameters") or []) if isinstance(old_op, dict) else []
    new_params_list = (new_op.get("parameters") or []) if isinstance(new_op, dict) else []
    old_params = {p.get("name"): p for p in old_params_list if isinstance(p, dict) and p.get("name")}
    new_params = {p.get("name"): p for p in new_params_list if isinstance(p, dict) and p.get("name")}
    for name in set(new_params.keys()) - set(old_params.keys()):
        aces.append({"type": "PARAM_ADDED", "path": path, "method": method, "detail": name})
    for name in set(old_params.keys()) - set(new_params.keys()):
        aces.append({"type": "PARAM_REMOVED", "path": path, "method": method, "detail": name})
    for name, new_p in new_params.items():
        if not isinstance(new_p, dict):
            continue
        old_p = old_params.get(name)
        if new_p.get("required") and (old_p is None or not old_p.get("required")):
            aces.append({"type":"PARAM_REQUIRED_ADDED","path":path,"method":method,"detail":name})
    for name, new_p in new_params.items():
        old_p = old_params.get(name)
        if isinstance(new_p, dict) and isinstance(old_p, dict):
            old_type = (old_p.get("schema") or {}).get("type")
            new_type = (new_p.get("schema") or {}).get("type")
            if old_type and new_type and old_type != new_type:
                aces.append({"type":"PARAM_TYPE_CHANGED","path":path,"method":method,"detail":name})
    def extract_enums(op):
        enums = {}
        if not isinstance(op, dict):
            return enums
        for p in (op.get("parameters") or []):
            if not isinstance(p, dict):
                continue
            s = p.get("schema", {}) if isinstance(p, dict) else {}
            if isinstance(s, dict) and "enum" in s:
                enums[f"param:{p.get('name')}"] = list(s["enum"])
        rb = (op.get("requestBody") or {}) if isinstance(op, dict) else {}
        for ct, body in (rb.get("content") or {}).items():
            s = body.get("schema", {}) if isinstance(body, dict) else {}
            if isinstance(s, dict) and "enum" in s:
                enums[f"requestBody:{ct}"] = list(s["enum"])
        return enums
    old_enums = extract_enums(old_op or {})
    new_enums = extract_enums(new_op or {})
    for k, new_list in new_enums.items():
        old_list = old_enums.get(k)
        if old_list and set(new_list) < set(old_list):
            aces.append({"type":"ENUM_NARROWED","path":path,"method":method,"detail":k})
    old_rc = set(str(k) for k in ((old_op or {}).get("responses") or {}).keys()) if isinstance(old_op, dict) else set()
    new_rc = set(str(k) for k in ((new_op or {}).get("responses") or {}).keys()) if isinstance(new_op, dict) else set()
    removed = old_rc - new_rc
    if removed:
        aces.append({"type":"RESPONSE_CODE_REMOVED","path":path,"method":method,"detail":list(removed)})
    old_rb = (old_op or {}).get("requestBody") or {} if isinstance(old_op, dict) else {}
    new_rb = (new_op or {}).get("requestBody") or {} if isinstance(new_op, dict) else {}
    if bool(old_rb) != bool(new_rb):
        aces.append({"type":"REQUESTBODY_CHANGED","path":path,"method":method,"detail":None})
    return aces

def compute_confidence(ace, canonical_old, canonical_new):
    base = 0.65
    typ = ace.get("type","")
    if "ENUM" in typ:
        base -= 0.08
    if "PARAM" in typ:
        base += 0.04
    if ("_polymorphic_marker" in canonical_old) or ("_polymorphic_marker" in canonical_new):
        base -= 0.06
    return round(max(0.0, min(1.0, base)), 3)

# ---------- dataset path helpers
def dataset_paths(root: Path, dataset: str):
    base = Path(root) / dataset
    return {
        "base": base,
        "canonical": base / "canonical",
        "ndjson": base / "ndjson",
        "metadata": base / "metadata",
    }

# ---------- Git-based historical pair extractor 
def _git_file_commit_pairs(repo_dir: Path, file_path: Path, max_pairs: int = 3):
    try:
        rel = file_path.relative_to(repo_dir)
    except Exception:
        return []
    cmd = ["git", "-C", str(repo_dir), "log", "--follow", "--pretty=format:%H", "--", str(rel)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, encoding="utf-8")
        commits = [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []
    pairs = []
    for i in range(min(len(commits)-1, max_pairs)):
        old_c = commits[i+1]
        new_c = commits[i]
        try:
            old_blob = subprocess.check_output(["git", "-C", str(repo_dir), "show", f"{old_c}:{str(rel)}"], stderr=subprocess.DEVNULL, encoding="utf-8")
        except Exception:
            old_blob = None
        try:
            new_blob = subprocess.check_output(["git", "-C", str(repo_dir), "show", f"{new_c}:{str(rel)}"], stderr=subprocess.DEVNULL, encoding="utf-8")
        except Exception:
            new_blob = None
        if old_blob is None or new_blob is None:
            continue
        pairs.append((old_c, new_c, old_blob, new_blob))
    return pairs

# ---------- Curators (OpenAPI, PetClinic, OpenRewrite)
# Each curator writes canonical JSONs, ndjson ACE files and metadata JSON.
# Important: always use generate_pair_id() so pair ids are stable and safe.

def curate_openapi(
    pair_index: Dict,
    producers: Dict,
    version_meta: Dict,
    budget: Dict,
    seed: int = 42,
    dry_run: bool = False,
    raw_root: Path = RAW_ROOT,
    curated_root: Path = CURATED_ROOT,
    skip_clone: bool = False,
    openapi_timeout: int = 8,
    prefer_git_pairs: bool = True
):
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "openapi")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "openapi_repo"
    local_dir = raw_root / "openapi"

    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                logging.info(f"[curate_openapi] cloning openapi repo into {repo_dir}")
                subprocess.run(["git","clone","--depth","1", OPENAPI_REPO, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        except Exception as e:
            logging.warning(f"[curate_openapi] clone failed: {e}")

    scan_base = (
        local_dir if (local_dir.exists() and any(local_dir.rglob("*")))
        else repo_dir if (repo_dir.exists() and any(repo_dir.rglob("*")))
        else None
    )

    if not scan_base:
        logging.warning("[curate_openapi] No OpenAPI source files found; will produce 0 openapi pairs")
        return

    curated = 0
    processed_files = 0
    skipped_files = 0
    LOG_EVERY_FILES = 10

    for file in sorted(scan_base.rglob("*")):
        if budget.get("remaining", 0) <= 0:
            break
        processed_files += 1
        try:
            if not file.is_file():
                skipped_files += 1
                continue
            if file.suffix.lower() not in (".yaml", ".yml", ".json"):
                skipped_files += 1
                continue
            try:
                st_size = file.stat().st_size
            except Exception:
                st_size = 0
            if st_size == 0 or st_size > 2_000_000:
                skipped_files += 1
                continue
            doc = _read_file(file)
            if doc is None or not isinstance(doc, (dict, list)):
                skipped_files += 1
                continue

            pair_seed = (int(sha_for_text(str(file)), 16) ^ seed) & 0x7fffffff
            processed_real_pair = False

            # Try git-history pairs
            git_pairs = []
            try:
                if prefer_git_pairs and repo_dir.exists() and (repo_dir / ".git").exists():
                    git_pairs = _git_file_commit_pairs(repo_dir, file, max_pairs=2)
            except Exception:
                git_pairs = []

            if git_pairs:
                for old_c, new_c, old_blob, new_blob in git_pairs:
                    if budget.get("remaining", 0) <= 0:
                        break
                    try:
                        try:
                            old_doc = json.loads(old_blob)
                        except Exception:
                            old_doc = yaml.safe_load(old_blob)
                    except Exception:
                        continue
                    try:
                        try:
                            new_doc = json.loads(new_blob)
                        except Exception:
                            new_doc = yaml.safe_load(new_blob)
                    except Exception:
                        continue

                    old_can, warns_old = canonicalize_with_timeout(old_doc, file.parent, timeout_sec=openapi_timeout)
                    new_can, warns_new = canonicalize_with_timeout(new_doc, file.parent, timeout_sec=openapi_timeout)
                    if old_can is None or new_can is None:
                        continue

                    svc_base = re.sub(r"[^0-9a-zA-Z\-_]", "-", file.with_suffix("").as_posix()).lower()
                    svc_name = pick_real_service(svc_base)

                    old_can, old_sample = realisticize_canonical(old_can, svc_base + old_c[:8])
                    new_can, new_sample = realisticize_canonical(new_can, svc_base + new_c[:8])

                    old_sha = sha_for_text(json.dumps(old_can, sort_keys=True, default=_default_json_serializer))
                    new_sha = sha_for_text(json.dumps(new_can, sort_keys=True, default=_default_json_serializer))
                    pair_id = generate_pair_id(svc_name, old_sha, new_sha)
                    pair_token = make_pair_token(svc_name, old_sha, new_sha)

                    # filenames: unified pair token, v1/v2
                    old_can_path = paths["canonical"] / make_canonical_filename("openapi", svc_name, pair_token, "v1")
                    new_can_path = paths["canonical"] / make_canonical_filename("openapi", svc_name, pair_token, "v2")

                    if not dry_run:
                        _write_json(old_can_path, old_can)
                        _write_json(new_can_path, new_can)

                    pair_entry = {
                        "pair_id": pair_id,
                        "service_name": svc_name,
                        "dataset": "openapi",
                        "old_canonical": old_can_path.name,
                        "new_canonical": new_can_path.name,
                        "old": str(old_can_path),
                        "new": str(new_can_path),
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                        "warnings": (warns_old or []) + (warns_new or []),
                        "seed": pair_seed,
                        "generated_at": _now_iso(),
                    }
                    # attach producers sample (normalized paths) for graph building
                    eps = list((old_can.get("paths") or {}).keys())
                    eps_norm = [re.sub(r"\{[^}]+\}", "{*}", p) for p in eps]
                    producers.setdefault(svc_name, []).extend(eps_norm[:5])

                    # compute ACEs
                    aces = []
                    def ops_map(spec):
                        out = {}
                        for p, methods in (spec.get("paths") or {}).items():
                            if not isinstance(methods, dict):
                                continue
                            out[p] = {m.lower(): methods[m] for m in methods.keys()}
                        return out
                    old_ops = ops_map(old_can)
                    new_ops = ops_map(new_can)
                    all_paths = sorted(set(old_ops.keys()) | set(new_ops.keys()))
                    for p in all_paths:
                        methods = sorted(set(old_ops.get(p, {}).keys()) | set(new_ops.get(p, {}).keys()))
                        for m in methods:
                            old_op = old_ops.get(p, {}).get(m)
                            new_op = new_ops.get(p, {}).get(m)
                            try:
                                detected = compare_operations(old_op, new_op, p, m)
                            except Exception:
                                continue
                            for d in detected:
                                try:
                                    conf = compute_confidence(d, old_can, new_can)
                                except Exception:
                                    conf = 0.5
                                side_effect = m in ("post", "put", "patch", "delete")
                                callers = list(producers.keys())
                                calls = [rng.choice(callers)] if callers and rng.random() < 0.35 else []
                                shared = ["SharedSchema"] if isinstance(d.get("detail"), str) and ("schema" in d["detail"].lower() or "enum" in d["detail"].lower()) else []
                                ace = {
                                    "pair_id": pair_id,
                                    "ace_index": len(aces),
                                    "ace_id": f"{pair_id}::ace::{len(aces)}",
                                    "type": d.get("type"),
                                    "path": d.get("path"),
                                    "method": d.get("method"),
                                    "detail": d.get("detail"),
                                    "confidence": conf,
                                    "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                                    "side_effect": side_effect,
                                    "calls_services": calls,
                                    "shared_schemas": shared,
                                }
                                aces.append(ace)
                    if not aces:
                        aces.append({
                            "pair_id": pair_id,
                            "ace_index": 0,
                            "ace_id": f"{pair_id}::ace::0",
                            "type": "NO_IMPACT",
                            "path": None,
                            "method": None,
                            "detail": None,
                            "confidence": 0.0,
                            "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                            "side_effect": False,
                            "calls_services": [],
                            "shared_schemas": [],
                        })

                    ndjson_path = paths["ndjson"] / f"{pair_id}.aces.ndjson"
                    if not dry_run:
                        _ensure_dir(ndjson_path.parent)
                        with ndjson_path.open("w", encoding="utf-8") as fh:
                            for a in aces:
                                fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")
                        _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)

                    version_meta[pair_id] = {
                        "pair_id": pair_id,
                        "dataset": "openapi",
                        "service_name": svc_name,
                        "semver_old": "1.0.0",
                        "semver_new": "1.1.0",
                        "semver_delta": rng.choice(["patch", "minor", "major"]),
                        "breaking_vs_semver": rng.random() < 0.15,
                        "generated_at": pair_entry["generated_at"],
                    }

                    pair_index[pair_id] = pair_entry
                    curated += 1
                    budget["remaining"] = max(0, budget["remaining"] - 1)
                    processed_real_pair = True
                    logging.info(f"[curate_openapi][OK] git-pair {file} -> curated {curated} remaining={budget.get('remaining',0)}")
                    if budget.get("remaining", 0) <= 0:
                        break

            if budget.get("remaining", 0) <= 0:
                break
            if processed_real_pair:
                continue

            # FALLBACK: canonicalize current file and inject diffs
            canon, warns = canonicalize_with_timeout(doc, file.parent, timeout_sec=openapi_timeout)
            if canon is None:
                skipped_files += 1
                continue
            path_count = len((canon.get("paths") or {}).keys()) if isinstance(canon, dict) else 0
            if path_count == 0:
                skipped_files += 1
                continue

            v2doc = inject_openapi_diffs(canon, seed=pair_seed)

            svc_name = pick_real_service(str(file.with_suffix("").as_posix()))
            old_can_obj, _ = realisticize_canonical(canon, str(file) + "-v1")
            new_can_obj, _ = realisticize_canonical(v2doc, str(file) + "-v2")

            old_sha = sha_for_text(json.dumps(old_can_obj, sort_keys=True, default=_default_json_serializer))
            new_sha = sha_for_text(json.dumps(new_can_obj, sort_keys=True, default=_default_json_serializer))
            pair_id = generate_pair_id(svc_name, old_sha, new_sha)
            pair_token = make_pair_token(svc_name, old_sha, new_sha)

            old_can = paths["canonical"] / make_canonical_filename("openapi", svc_name, pair_token, "v1")
            new_can = paths["canonical"] / make_canonical_filename("openapi", svc_name, pair_token, "v2")
            if not dry_run:
                _write_json(old_can, old_can_obj)
                _write_json(new_can, new_can_obj)

            pair_entry = {
                "pair_id": pair_id,
                "service_name": svc_name,
                "dataset": "openapi",
                "old_canonical": old_can.name,
                "new_canonical": new_can.name,
                "old": str(old_can),
                "new": str(new_can),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "warnings": warns,
                "seed": pair_seed,
                "generated_at": _now_iso(),
            }
            pair_index[pair_id] = pair_entry

            eps = list((old_can_obj.get("paths") or {}).keys())
            eps_norm = [re.sub(r"\{[^}]+\}", "{*}", p) for p in eps]
            producers.setdefault(svc_name, []).extend(eps_norm[:5])

            aces = []
            def ops_map(spec):
                out = {}
                for p, m in (spec.get("paths") or {}).items():
                    if isinstance(m, dict):
                        out[p] = {mm.lower(): m[mm] for mm in m.keys()}
                return out

            old_ops = ops_map(old_can_obj)
            new_ops = ops_map(new_can_obj)
            all_paths = sorted(set(old_ops.keys()) | set(new_ops.keys()))
            for p in all_paths:
                methods = sorted(set(old_ops.get(p, {}).keys()) | set(new_ops.get(p, {}).keys()))
                for m in methods:
                    old_op = old_ops.get(p, {}).get(m)
                    new_op = new_ops.get(p, {}).get(m)
                    try:
                        detected = compare_operations(old_op, new_op, p, m)
                    except Exception:
                        continue
                    for d in detected:
                        try:
                            conf = compute_confidence(d, old_can_obj, new_can_obj)
                        except Exception:
                            conf = 0.5
                        side_effect = m in ("post", "put", "patch", "delete")
                        callers = list(producers.keys())
                        calls = [rng.choice(callers)] if callers and rng.random() < 0.35 else []
                        shared = ["SharedSchema"] if isinstance(d.get("detail"), str) and ("schema" in d["detail"].lower() or "enum" in d["detail"].lower()) else []
                        ace = {
                            "pair_id": pair_id,
                            "ace_index": len(aces),
                            "ace_id": f"{pair_id}::ace::{len(aces)}",
                            "type": d.get("type"),
                            "path": d.get("path"),
                            "method": d.get("method"),
                            "detail": d.get("detail"),
                            "confidence": conf,
                            "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                            "side_effect": side_effect,
                            "calls_services": calls,
                            "shared_schemas": shared,
                        }
                        aces.append(ace)

            if not aces:
                aces.append({
                    "pair_id": pair_id,
                    "ace_index": 0,
                    "ace_id": f"{pair_id}::ace::0",
                    "type": "NO_IMPACT",
                    "path": None,
                    "method": None,
                    "detail": None,
                    "confidence": 0.0,
                    "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                    "side_effect": False,
                    "calls_services": [],
                    "shared_schemas": [],
                })

            ndjson_path = paths["ndjson"] / f"{pair_id}.aces.ndjson"
            if not dry_run:
                _ensure_dir(ndjson_path.parent)
                with ndjson_path.open("w", encoding="utf-8") as fh:
                    for a in aces:
                        fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")
                _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)

            version_meta[pair_id] = {
                "pair_id": pair_id,
                "dataset": "openapi",
                "service_name": svc_name,
                "semver_old": "1.0.0",
                "semver_new": "1.1.0",
                "semver_delta": rng.choice(["patch", "minor", "major"]),
                "breaking_vs_semver": rng.random() < 0.15,
                "generated_at": pair_entry["generated_at"],
            }

            curated += 1
            budget["remaining"] = max(0, budget["remaining"] - 1)
            logging.info(f"[curate_openapi][OK] fallback {file} -> curated {curated} remaining={budget.get('remaining',0)}")

        except Exception as e:
            skipped_files += 1
            logging.debug(f"[curate_openapi] SKIP/EXC {file}: {e}", exc_info=True)
            continue

        if processed_files % LOG_EVERY_FILES == 0:
            logging.info(f"[curate_openapi] progress files={processed_files} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped_files}")

    logging.info(f"[curate_openapi] Finished: curated {curated} OpenAPI pairs into {paths['canonical']}")
    return

# ---------- Unified noisy-raw helper (paste here) ----------
import tempfile
from typing import Union

def generate_noisy_raw_copy(raw_root: Path, target_root: Path, mode: str = "light",
                            seed: int = 2025, counts: Optional[Dict[str,int]] = None):
    """
    Create a noisy copy of raw files into target_root.
    mode: "light" or "heavy"
    counts: optional dict like {"openapi":500,"openrewrite":300,"petclinic":30,"synth_openapi":50}
    Returns target_root Path. Writes noise_manifest.json to target_root.
    """
    rnd = random.Random(seed)
    counts = counts or {"openapi": 500, "openrewrite": 300, "petclinic": 30}
    datasets = ["openapi", "openrewrite", "petclinic"]
    # include any synth folders found under raw_root
    for synth in ["synth_openapi", "synth_openrewrite", "synth_petclinic"]:
        if (raw_root / synth).exists():
            datasets.append(synth)
            counts.setdefault(synth, 50)

    manifest = {"seed": seed, "mode": mode, "entries": []}
    _ensure_dir(target_root)

    for ds in datasets:
        src_ds = Path(raw_root) / ds
        tgt_ds = Path(target_root) / ds
        if not src_ds.exists():
            manifest["entries"].append({"dataset": ds, "status": "missing_source"})
            continue
        _ensure_dir(tgt_ds)

        # gather candidate files
        cand = [p for p in src_ds.rglob("*") if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".json", ".log", ".txt", ".md")]
        if not cand:
            # fallback: copy everything
            for f in src_ds.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(src_ds)
                    dst = tgt_ds / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(f, dst)
                    except Exception:
                        try:
                            dst.write_text(f.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                        except Exception:
                            pass
            manifest["entries"].append({"dataset": ds, "status": "copied_all_fallback", "count": 0})
            continue

        target_count = min(len(cand), counts.get(ds, len(cand)))
        sampled = cand if len(cand) <= target_count else rnd.sample(cand, target_count)

        corrupted_count = 0
        for src in sampled:
            rel = src.relative_to(src_ds)
            dst = tgt_ds / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw_text = src.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    shutil.copy2(src, dst)
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "binary_copied"})
                    continue
                except Exception:
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "read_error"})
                    continue

            # corruption probability depends on mode
            corr_prob = 0.25 if mode == "light" else 0.45
            if rnd.random() < corr_prob:
                if mode == "light":
                    corrupted_text, changes = apply_noisy_light_text(raw_text, rnd)
                else:
                    corrupted_text, changes = apply_noisy_heavy_text(raw_text, rnd)
                try:
                    dst.write_text(corrupted_text, encoding="utf-8")
                except Exception:
                    dst.write_bytes(corrupted_text.encode("utf-8", errors="replace"))
                manifest["entries"].append({"dataset": ds, "file": str(rel), "corrupted": True, "changes": changes})
                corrupted_count += 1
            else:
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    try:
                        dst.write_text(raw_text, encoding="utf-8")
                    except Exception:
                        pass
                manifest["entries"].append({"dataset": ds, "file": str(rel), "corrupted": False})

        # copy remaining non-sampled files so structure remains complete
        remaining = set(cand) - set(sampled)
        for src in remaining:
            rel = src.relative_to(src_ds)
            dst = tgt_ds / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception:
                try:
                    dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                except Exception:
                    pass

        manifest["entries"].append({"dataset": ds, "sampled": len(sampled), "corrupted_count": corrupted_count})

    # write manifest
    manifest_path = Path(target_root) / "noise_manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        logging.warning("Failed to write noise manifest")

    return Path(target_root)


def get_processing_root(raw_root: Path, curated_root: Path, mode: str, seed: int = 2025,
                        counts: Optional[Dict[str,int]] = None) -> Path:
    """
    Return a raw folder to use for processing. If mode == 'clean' return raw_root.
    Otherwise generate a noisy copy under parent of curated_root and return it.
    This is intended to be called once by the main driver before calling all curators.
    """
    mode = (mode or "clean")
    if mode == "clean":
        return raw_root
    # create a deterministic target folder next to curated root
    tgt_parent = Path(curated_root).parent
    out_name = f"raw_{mode}_{seed}"
    target_root = tgt_parent / out_name
    if target_root.exists():
        # assume reproducible; reuse existing copy to save time
        return target_root
    return generate_noisy_raw_copy(raw_root, target_root, mode="light" if mode == "noisy_light" else "heavy",
                                   seed=seed, counts=counts)
# ---------- end noisy helper ----------


# Petclinic and OpenRewrite curators: 

def synth_curate_petclinic(pair_index: Dict, version_meta: Dict, budget: Dict, seed: int = 43, dry_run: bool = False,
                     raw_root: Path = RAW_ROOT, curated_root: Path = CURATED_ROOT, skip_clone: bool = False):
    """
    Curate PetClinic pairs.
    - Use existing raw repo YAMLs when present.
    - For each YAML create multiple deterministic realistic variants (mutations) to reach budget.
    - Fall back to fully synthetic generation if no YAMLs found.
    - Preserves deterministic seed, budget handling, dry_run, and existing file layout.
    """
    import yaml

    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "petclinic")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "petclinic_repo"
    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                subprocess.run(["git", "clone", "--depth", "1", PETCLINIC_REPO, str(repo_dir)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        except Exception:
            logging.warning("[petclinic] git clone failed or skipped; falling back to local/synth")

    curated = 0
    sources = []
    if repo_dir.exists() and any(repo_dir.rglob("*.yml")):
        sources = list(repo_dir.rglob("*.yml"))
    else:
        local_pc = raw_root / "petclinic"
        if local_pc.exists() and any(local_pc.rglob("*.yml")):
            sources = list(local_pc.rglob("*.yml"))

    # --- Case A: No YAML sources -> generate fully synthetic pairs 
    if not sources:
        times = budget.get("remaining", 0)
        for i in range(times):
            if budget.get("remaining", 0) <= 0:
                break
            try:
                service = pick_real_service(f"petclinic-synth-{i}")
                base_paths = [f"/pets", f"/owners", f"/appointments", f"/{service}/reports"]
                doc = {"paths": {}}
                for p in base_paths[:3]:
                    doc["paths"][p] = {
                        "get": {"responses": {"200": {"description": "ok"}}},
                        "post": {"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                                 "responses": {"201": {"description": "created"}}}
                    }
                doc["paths"]["/pets"]["get"].setdefault("parameters", []).append({
                    "name": "status", "in": "query", "schema": {"type": "string", "enum": ["available", "adopted", "foster"]},
                    "required": False
                })
                v2 = deepcopy(doc)
                new_ep = f"/reports"
                v2["paths"][new_ep] = {"get": {"responses": {"200": {"description": "ok"}}}}
                old_sha = sha_for_text(json.dumps(doc, sort_keys=True, default=_default_json_serializer))
                new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                pair_id = generate_pair_id(service, old_sha, new_sha)
                pair_token = make_pair_token(service, old_sha, new_sha)
                old_can = paths["canonical"] / make_canonical_filename("petclinic", service, pair_token, "v1")
                new_can = paths["canonical"] / make_canonical_filename("petclinic", service, pair_token, "v2")
                if not dry_run:
                    _write_json(old_can, doc)
                    _write_json(new_can, v2)
                generated_at = _now_iso()
                pair_entry = {
                    "pair_id": pair_id,
                    "service_name": service,
                    "dataset": "petclinic",
                    "old_canonical": old_can.name,
                    "new_canonical": new_can.name,
                    "old": str(old_can),
                    "new": str(new_can),
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "generated_at": generated_at
                }
                pair_index[pair_id] = pair_entry
                aces = []
                ai = 0
                aces.append({
                    "pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}",
                    "type": "ENDPOINT_ADDED", "path": new_ep, "method": "get", "detail": new_ep, "confidence": 0.8,
                    "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [],
                    "shared_schemas": []
                }); ai += 1
                aces.append({
                    "pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}",
                    "type": "PARAM_ADDED", "path": "/pets", "method": "get", "detail": "status", "confidence": 0.7,
                    "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [],
                    "shared_schemas": []
                }); ai += 1
                if rng.random() < 0.3:
                    aces.append({
                        "pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}",
                        "type": "RESPONSE_CODE_REMOVED", "path": "/owners", "method": "post", "detail": ["201"], "confidence": 0.75,
                        "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [],
                        "shared_schemas": []
                    }); ai += 1
                nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
                if not dry_run:
                    _ensure_dir(nd.parent)
                    with nd.open("w", encoding="utf-8") as fh:
                        for a in aces:
                            fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")
                    _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)
                version_meta[pair_id] = {
                    "pair_id": pair_id,
                    "dataset": "petclinic",
                    "service_name": service,
                    "semver_old": "1.0.0",
                    "semver_new": "1.0.1",
                    "semver_delta": "patch",
                    "breaking_vs_semver": False,
                    "generated_at": generated_at,
                    "producers_sample": base_paths + [new_ep]
                }
                curated += 1
                budget["remaining"] = max(0, budget.get("remaining", 0) - 1)
                if dry_run and curated >= 5:
                    break
            except Exception as e:
                logging.warning(f"[petclinic synth] skip {i}: {e}")

    # --- Case B: YAML sources exist -> load each YAML and create multiple realistic deterministic variants 
    else:
        
        variants_per_file = 8
        for yml in sources:
            if budget.get("remaining", 0) <= 0:
                break
            try:
                service = yml.stem.lower()
                svc_name = pick_real_service(service)

                # --- Load YAML and normalize to a minimal base_doc with 'paths' ---
                try:
                    yaml_text = yml.read_text(encoding="utf-8")
                    loaded = yaml.safe_load(yaml_text)
                    if isinstance(loaded, dict) and "paths" in loaded and isinstance(loaded["paths"], dict):
                       
                        base_doc = {"paths": {}}
                        for p, node in loaded["paths"].items():
                            base_doc["paths"][p] = deepcopy(node)
                    else:
                        # fallback: attempt to infer a top-level service path or use svc_name
                        base_doc = {"paths": {f"/{svc_name}": {"get": {"description": "orig", "responses": {"200": {"description": "ok"}}}}}}
                except Exception as ex:
                    logging.warning(f"[petclinic] failed to load YAML {yml}: {ex}; using fallback base_doc")
                    base_doc = {"paths": {f"/{svc_name}": {"get": {"description": "orig", "responses": {"200": {"description": "ok"}}}}}}

                # list of existing paths to prefer for mutations
                existing_paths = list(base_doc.get("paths", {}).keys()) or [f"/{svc_name}"]

                for variant in range(variants_per_file):
                    if budget.get("remaining", 0) <= 0:
                        break

                    v2 = deepcopy(base_doc)
                    m = variant % 5  # rotate mutation types but applied to real paths when possible

                    # choose a target path deterministically from existing_paths
                    target_path = existing_paths[rng.randrange(len(existing_paths))] if existing_paths else f"/{svc_name}"

                    if m == 0:
                        # add new sibling endpoint under same prefix if possible, else generic new path
                        if target_path.startswith("/"):
                            prefix = target_path.rstrip("/").split("/")[1] if len(target_path.split("/")) > 1 else svc_name
                            new_path = f"/{prefix}/new{variant}"
                        else:
                            new_path = f"/{svc_name}/new{variant}"
                        v2["paths"][new_path] = {"get": {"description": "added endpoint", "responses": {"200": {"description": "ok"}}}}
                        ace_type = "ENDPOINT_ADDED"
                        ace_path = new_path
                        ace_detail = new_path
                    elif m == 1:
                        # add a query parameter to the chosen target_path
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"description": "ok"}}}})
                        v2["paths"][ppath].setdefault("get", {}).setdefault("parameters", []).append(
                            {"name": f"q{variant}", "in": "query", "schema": {"type": "string"}, "required": False}
                        )
                        ace_type = "PARAM_ADDED"
                        ace_path = ppath
                        ace_detail = f"q{variant}"
                    elif m == 2:
                        # simulate response code removal on target_path
                        ppath = target_path
                        # ensure responses exist in v2 for shape consistency
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"description": "ok"}}}})
                        ace_type = "RESPONSE_CODE_REMOVED"
                        ace_path = ppath
                        ace_detail = ["201"]
                    elif m == 3:
                        # change/add a schema property type in the first JSON response found on target_path
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object", "properties": {}}}}}}}})
                        # ensure nested keys exist and then add the property
                        try:
                            schema_props = v2["paths"][ppath]["get"]["responses"]["200"]["content"]["application/json"]["schema"].setdefault("properties", {})
                            schema_props[f"p{variant}"] = {"type": "integer"}
                        except Exception:
                            # fallback: create the whole structure
                            v2["paths"][ppath]["get"]["responses"]["200"]["content"] = {"application/json": {"schema": {"type": "object", "properties": {f"p{variant}": {"type": "integer"}}}}}
                        ace_type = "SCHEMA_PROPERTY_CHANGED"
                        ace_path = ppath
                        ace_detail = f"p{variant}"
                    else:
                        # rename parameter - simulated by adding a required param with a 'renamed_' name
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"description": "ok"}}}})
                        v2["paths"][ppath].setdefault("get", {}).setdefault("parameters", []).append(
                            {"name": f"renamed_{variant}", "in": "query", "schema": {"type": "string"}, "required": True}
                        )
                        ace_type = "PARAM_RENAMED"
                        ace_path = ppath
                        ace_detail = f"renamed_{variant}"

                    old_sha = sha_for_text(json.dumps(base_doc, sort_keys=True, default=_default_json_serializer))
                    new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                    pair_id = generate_pair_id(svc_name, old_sha, new_sha)
                    pair_token = make_pair_token(svc_name, old_sha, new_sha)
                    old_can = paths["canonical"] / make_canonical_filename("petclinic", f"petclinic-{svc_name}", pair_token, "v1")
                    new_can = paths["canonical"] / make_canonical_filename("petclinic", f"petclinic-{svc_name}", pair_token, "v2")
                    if not dry_run:
                        _write_json(old_can, base_doc)
                        _write_json(new_can, v2)
                    generated_at = _now_iso()
                    pair_entry = {
                        "pair_id": pair_id,
                        "service_name": f"petclinic-{svc_name}",
                        "dataset": "petclinic",
                        "old_canonical": old_can.name,
                        "new_canonical": new_can.name,
                        "old": str(old_can),
                        "new": str(new_can),
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                        "generated_at": generated_at
                    }
                    pair_index[pair_id] = pair_entry

                    aces = []
                    aces.append({
                        "pair_id": pair_id,
                        "ace_index": 0,
                        "ace_id": f"{pair_id}::ace::0",
                        "type": ace_type,
                        "path": ace_path,
                        "method": "get",
                        "detail": ace_detail,
                        "confidence": 0.8,
                        "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                        "side_effect": False,
                        "calls_services": [],
                        "shared_schemas": []
                    })

                    nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
                    if not dry_run:
                        _ensure_dir(nd.parent)
                        with nd.open("w", encoding="utf-8") as fh:
                            for a in aces:
                                fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")
                        _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)

                    version_meta[pair_id] = {
                        "pair_id": pair_id,
                        "dataset": "petclinic",
                        "service_name": pair_entry["service_name"],
                        "semver_old": "1.0.0",
                        "semver_new": "1.0.1",
                        "semver_delta": "patch",
                        "breaking_vs_semver": ace_type in ("RESPONSE_CODE_REMOVED", "PARAM_RENAMED"),
                        "generated_at": generated_at,
                        "producers_sample": [ace_path] if isinstance(ace_path, str) else [f"/{svc_name}"]
                    }
                    curated += 1
                    budget["remaining"] = max(0, budget.get("remaining", 0) - 1)
                    if dry_run and curated >= 5:
                        break
            except Exception as e:
                logging.warning(f"[petclinic] skip {yml}: {e}")

    logging.info(f"[petclinic] Curated %d pairs", curated)
    return

def curate_petclinic(
    pair_index: Dict,
    version_meta: Dict,
    budget: Dict,
    seed: int = 43,
    dry_run: bool = False,
    raw_root: Path = RAW_ROOT,
    curated_root: Path = CURATED_ROOT,
    skip_clone: bool = False,
):
    """
    Original PetClinic curator (formatted).

    Notes:
    - This function attempts to clone the petclinic repo (unless skip_clone=True).
    - If YAMLs are found in the repo (or local folder), it creates one pair per YAML.
    - If no YAMLs are found, it falls back to fully synthetic generation using `budget`.
    - Preserves deterministic behavior via `seed`.
    - Writes canonical JSON files, ACE NDJSON, and metadata entries.
    - No logic changes were made — only formatting and inline comments for readability.
    """
    rng = random.Random(seed)

    # Prepare curated dataset paths and ensure directories exist.
    paths = dataset_paths(curated_root, "petclinic")
    _ensure_dir(paths["canonical"])
    _ensure_dir(paths["ndjson"])
    _ensure_dir(paths["metadata"])

    # Ensure raw repo exists (clone if allowed and missing)
    repo_dir = raw_root / "petclinic_repo"
    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                subprocess.run(
                    ["git", "clone", "--depth", "1", PETCLINIC_REPO, str(repo_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
        except Exception:
            # swallow clone failures; fallback logic will handle missing repo
            pass

    curated = 0
    sources = []

    # Prefer YAML files from cloned repo; fallback to local/raw/petclinic
    if repo_dir.exists() and any(repo_dir.rglob("*.yml")):
        sources = list(repo_dir.rglob("*.yml"))
    else:
        local_pc = raw_root / "petclinic"
        if local_pc.exists() and any(local_pc.rglob("*.yml")):
            sources = list(local_pc.rglob("*.yml"))

 
    if not sources:
        times = budget.get("remaining", 0)
        for i in range(times):
            if budget.get("remaining", 0) <= 0:
                break
            try:
                service = pick_real_service(f"petclinic-synth-{i}")
                base_paths = [f"/pets", f"/owners", f"/appointments", f"/{service}/reports"]

                # Build minimal OpenAPI-like doc for this synthetic service
                doc = {"paths": {}}
                for p in base_paths[:3]:
                    doc["paths"][p] = {
                        "get": {"responses": {"200": {"description": "ok"}}},
                        "post": {
                            "requestBody": {
                                "content": {"application/json": {"schema": {"type": "object"}}}}
                            ,
                            "responses": {"201": {"description": "created"}},
                        },
                    }

                # Add a realistic query parameter on /pets GET
                doc["paths"]["/pets"]["get"].setdefault("parameters", []).append(
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["available", "adopted", "foster"]},
                        "required": False,
                    }
                )

                # Create v2 by adding a reports endpoint (simple synthetic change)
                v2 = deepcopy(doc)
                new_ep = f"/reports"
                v2["paths"][new_ep] = {"get": {"responses": {"200": {"description": "ok"}}}}

                # Deterministic SHAs and tokens
                old_sha = sha_for_text(json.dumps(doc, sort_keys=True, default=_default_json_serializer))
                new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                pair_id = generate_pair_id(service, old_sha, new_sha)
                pair_token = make_pair_token(service, old_sha, new_sha)

                # Canonical file paths
                old_can = paths["canonical"] / make_canonical_filename("petclinic", service, pair_token, "v1")
                new_can = paths["canonical"] / make_canonical_filename("petclinic", service, pair_token, "v2")

                # Write canonical JSON files (unless dry_run)
                if not dry_run:
                    _write_json(old_can, doc)
                    _write_json(new_can, v2)

                generated_at = _now_iso()
                pair_entry = {
                    "pair_id": pair_id,
                    "service_name": service,
                    "dataset": "petclinic",
                    "old_canonical": old_can.name,
                    "new_canonical": new_can.name,
                    "old": str(old_can),
                    "new": str(new_can),
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "generated_at": generated_at,
                }
                pair_index[pair_id] = pair_entry

                # Build ACEs for this synthetic pair
                aces = []
                ai = 0
                aces.append(
                    {
                        "pair_id": pair_id,
                        "ace_index": ai,
                        "ace_id": f"{pair_id}::ace::{ai}",
                        "type": "ENDPOINT_ADDED",
                        "path": new_ep,
                        "method": "get",
                        "detail": new_ep,
                        "confidence": 0.8,
                        "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                        "side_effect": False,
                        "calls_services": [],
                        "shared_schemas": [],
                    }
                )
                ai += 1

                aces.append(
                    {
                        "pair_id": pair_id,
                        "ace_index": ai,
                        "ace_id": f"{pair_id}::ace::{ai}",
                        "type": "PARAM_ADDED",
                        "path": "/pets",
                        "method": "get",
                        "detail": "status",
                        "confidence": 0.7,
                        "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                        "side_effect": False,
                        "calls_services": [],
                        "shared_schemas": [],
                    }
                )
                ai += 1

                # Occasionally add a response-code removal ACE for variety
                if rng.random() < 0.3:
                    aces.append(
                        {
                            "pair_id": pair_id,
                            "ace_index": ai,
                            "ace_id": f"{pair_id}::ace::{ai}",
                            "type": "RESPONSE_CODE_REMOVED",
                            "path": "/owners",
                            "method": "post",
                            "detail": ["201"],
                            "confidence": 0.75,
                            "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                            "side_effect": False,
                            "calls_services": [],
                            "shared_schemas": [],
                        }
                    )
                    ai += 1

                # Write ACE NDJSON and metadata
                nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
                if not dry_run:
                    _ensure_dir(nd.parent)
                    with nd.open("w", encoding="utf-8") as fh:
                        for a in aces:
                            fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")
                    _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)

                # Version metadata for tracking
                version_meta[pair_id] = {
                    "pair_id": pair_id,
                    "dataset": "petclinic",
                    "service_name": service,
                    "semver_old": "1.0.0",
                    "semver_new": "1.0.1",
                    "semver_delta": "patch",
                    "breaking_vs_semver": False,
                    "generated_at": generated_at,
                    "producers_sample": base_paths + [new_ep],
                }

                curated += 1
                budget["remaining"] = max(0, budget.get("remaining", 0) - 1)

                # If dry_run, stop early for quick checks (keeps original behavior)
                if dry_run and curated >= 5:
                    break

            except Exception as e:
                logging.warning(f"[petclinic synth] skip {i}: {e}")


    else:
        for yml in sources:
            if budget.get("remaining", 0) <= 0:
                break
            try:
                service = yml.stem.lower()
                svc_name = pick_real_service(service)

                # compute shas for deterministic pair token
                doc = {"paths": {f"/{svc_name}": {"get": {"description": "orig"}}}}
                v2 = deepcopy(doc)
                v2["paths"][f"/{svc_name}/new"] = {"get": {"description": "added endpoint"}}

                old_sha = sha_for_text(json.dumps(doc, sort_keys=True, default=_default_json_serializer))
                new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                pair_id = generate_pair_id(svc_name, old_sha, new_sha)
                pair_token = make_pair_token(svc_name, old_sha, new_sha)

                old_can = paths["canonical"] / make_canonical_filename(
                    "petclinic", f"petclinic-{svc_name}", pair_token, "v1"
                )
                new_can = paths["canonical"] / make_canonical_filename(
                    "petclinic", f"petclinic-{svc_name}", pair_token, "v2"
                )

                if not dry_run:
                    _write_json(old_can, doc)
                    _write_json(new_can, v2)

                generated_at = _now_iso()
                pair_entry = {
                    "pair_id": pair_id,
                    "service_name": f"petclinic-{svc_name}",
                    "dataset": "petclinic",
                    "old_canonical": old_can.name,
                    "new_canonical": new_can.name,
                    "old": str(old_can),
                    "new": str(new_can),
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "generated_at": generated_at,
                }
                pair_index[pair_id] = pair_entry

                # Build ACE list for this pair (single ENDPOINT_ADDED)
                aces = []
                aces.append(
                    {
                        "pair_id": pair_id,
                        "ace_index": 0,
                        "ace_id": f"{pair_id}::ace::0",
                        "type": "ENDPOINT_ADDED",
                        "path": f"/{svc_name}/new",
                        "method": "get",
                        "detail": f"/{svc_name}/new",
                        "confidence": 0.8,
                        "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                        "side_effect": False,
                        "calls_services": [],
                        "shared_schemas": [],
                    }
                )

                nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
                if not dry_run:
                    _ensure_dir(nd.parent)
                    with nd.open("w", encoding="utf-8") as fh:
                        for a in aces:
                            fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")
                    _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)

                version_meta[pair_id] = {
                    "pair_id": pair_id,
                    "dataset": "petclinic",
                    "service_name": pair_entry["service_name"],
                    "semver_old": "1.0.0",
                    "semver_new": "1.0.1",
                    "semver_delta": "patch",
                    "breaking_vs_semver": False,
                    "generated_at": generated_at,
                    "producers_sample": [f"/{svc_name}", f"/{svc_name}/new"],
                }

                curated += 1
                budget["remaining"] = max(0, budget.get("remaining", 0) - 1)

                if dry_run and curated >= 5:
                    break

            except Exception as e:
                logging.warning(f"[petclinic] skip {yml}: {e}")

    logging.info(f"[petclinic] Curated %d pairs", curated)
    return


def curate_openrewrite(
    pair_index: Dict,
    version_meta: Dict,
    budget: Dict,
    seed:int=44,
    dry_run:bool=False,
    raw_root: Path = RAW_ROOT,
    curated_root: Path = CURATED_ROOT,
    skip_clone: bool = False,
    prefer_git_pairs: bool = True,
    openapi_timeout: int = 8,
    max_git_pairs_per_file: int = 3
):
    """
    Robust OpenRewrite curator modeled after curate_openapi:
      - Prefer git-history pairs when available (real pairs)
      - Fallback to per-file multi-doc parsing and deterministic micro-changes
      - Filter noisy files (workflows, examples, meta)
      - Write canonical JSONs, ndjson ACEs, metadata and version_meta
    """
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "openrewrite")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "openrewrite_repo"

    # Clone if  requested and repo not present
    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                logging.info(f"[curate_openrewrite] cloning rewrite repo into {repo_dir}")
                subprocess.run(["git","clone", OPENREWRITE_REPO, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except Exception as e:
            logging.warning(f"[curate_openrewrite] clone failed: {e}")

    # gather candidate files
    files = []
    if repo_dir.exists():
        files = list(repo_dir.rglob("*.yml")) + list(repo_dir.rglob("*.yaml")) + list(repo_dir.rglob("*.json"))
    else:
        logging.warning("[openrewrite] repo_dir not found, nothing to curate")
        budget["remaining"] = 0
        return

    if not files:
        logging.warning("[openrewrite] no candidate files found; skipping")
        budget["remaining"] = 0
        return

    curated = 0
    LOG_EVERY = 25
    processed_files = 0
    skipped_files = 0

    def _is_candidate_file(f: Path) -> bool:
        # reuse earlier heuristic -- small, huge, or known noise -> skip
        if is_noise_path(f):
            return False
        try:
            sz = f.stat().st_size
            if sz < 120 or sz > 2_000_000:
                return False
        except Exception:
            pass
        return True

    # Helper to create pair from two spec dictionaries (v1/v2) — writes canon, aces, meta, version_meta and registers pair_index
    def _emit_pair_from_docs(dataset: str, svc_name: str, old_doc: dict, new_doc: dict, provenance_extra: dict = None):
        nonlocal curated
        try:
            old_can, warns_old = canonicalize_with_timeout(old_doc, repo_dir, timeout_sec=openapi_timeout)
            new_can, warns_new = canonicalize_with_timeout(new_doc, repo_dir, timeout_sec=openapi_timeout)
            if old_can is None or new_can is None:
                return False

            # realisticize as OpenAPI curator does (use svc_name seed)
            old_can_r, old_sample = realisticize_canonical(old_can, svc_name + "-old")
            new_can_r, new_sample = realisticize_canonical(new_can, svc_name + "-new")

            old_sha = sha_for_text(json.dumps(old_can_r, sort_keys=True, default=_default_json_serializer))
            new_sha = sha_for_text(json.dumps(new_can_r, sort_keys=True, default=_default_json_serializer))
            pair_id = generate_pair_id(svc_name, old_sha, new_sha)
            pair_token = make_pair_token(svc_name, old_sha, new_sha)

            old_path = paths["canonical"] / make_canonical_filename("openrewrite", svc_name, pair_token, "v1")
            new_path = paths["canonical"] / make_canonical_filename("openrewrite", svc_name, pair_token, "v2")

            if not dry_run:
                _write_json(old_path, old_can_r)
                _write_json(new_path, new_can_r)

            # compute ACEs by diffing operations where possible, for OpenRewrite we normally emit NON_API_CHANGE
            aces = []
            # if both have "paths" treat them like pseudo-OpenAPI and diff; otherwise emit NO_IMPACT or NON_API_CHANGE
            if isinstance(old_can_r, dict) and isinstance(new_can_r, dict) and (old_can_r.get("paths") or new_can_r.get("paths")):
                def ops_map(spec):
                    out = {}
                    for p, methods in (spec.get("paths") or {}).items():
                        if isinstance(methods, dict):
                            out[p] = {m.lower(): methods[m] for m in methods.keys()}
                    return out
                old_ops = ops_map(old_can_r)
                new_ops = ops_map(new_can_r)
                all_paths = sorted(set(old_ops.keys()) | set(new_ops.keys()))
                for p in all_paths:
                    methods = sorted(set(old_ops.get(p, {}).keys()) | set(new_ops.get(p, {}).keys()))
                    for m in methods:
                        old_op = old_ops.get(p, {}).get(m)
                        new_op = new_ops.get(p, {}).get(m)
                        try:
                            detected = compare_operations(old_op, new_op, p, m)
                        except Exception:
                            detected = []
                        for d in detected:
                            try:
                                conf = compute_confidence(d, old_can_r, new_can_r)
                            except Exception:
                                conf = 0.5
                            ace = {
                                "pair_id": pair_id,
                                "ace_index": len(aces),
                                "ace_id": f"{pair_id}::ace::{len(aces)}",
                                "type": d.get("type"),
                                "path": d.get("path"),
                                "method": d.get("method"),
                                "detail": d.get("detail"),
                                "confidence": conf,
                                "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                                "side_effect": (d.get("method") or "").lower() in ("post","put","patch","delete"),
                                "calls_services": [],
                                "shared_schemas": []
                            }
                            aces.append(ace)

            if not aces:
                # give a safe NON_API_CHANGE 
                aces.append({
                    "pair_id": pair_id,
                    "ace_index": 0,
                    "ace_id": f"{pair_id}::ace::0",
                    "type": "NON_API_CHANGE",
                    "path": None,
                    "method": None,
                    "detail": provenance_extra.get("detail") if provenance_extra else None,
                    "confidence": 0.1,
                    "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                    "side_effect": False,
                    "calls_services": [],
                    "shared_schemas": []
                })

            ndjson_path = paths["ndjson"] / f"{pair_id}.aces.ndjson"
            if not dry_run:
                _ensure_dir(ndjson_path.parent)
                with ndjson_path.open("w", encoding="utf-8") as fh:
                    for a in aces:
                        fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")

            meta = {
                "pair_id": pair_id,
                "dataset": dataset,
                "service_name": svc_name,
                "old_canonical": old_path.name,
                "new_canonical": new_path.name,
                "old": str(old_path),
                "new": str(new_path),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "generated_at": _now_iso(),
                "warnings": (warns_old or []) + (warns_new or []) if 'warns_old' in locals() else []
            }
            if not dry_run:
                _write_json(paths["metadata"] / f"{pair_id}.meta.json", meta)

            pair_index[pair_id] = meta
            version_meta[pair_id] = {
                "pair_id": pair_id,
                "dataset": dataset,
                "service_name": svc_name,
                "semver_old": "0.0.0",
                "semver_new": "0.0.1",
                "semver_delta": "patch",
                "breaking_vs_semver": False,
                "generated_at": meta["generated_at"],
                "producers_sample": []
            }

            curated += 1
            budget["remaining"] = max(0, budget.get("remaining", 0) - 1)
            return True

        except Exception as e:
            logging.debug(f"[openrewrite] emit_pair failed: {e}", exc_info=True)
            return False

    # Primary strategy: prefer git-history pairs (real commit pairs per file)
    if prefer_git_pairs and repo_dir.exists() and (repo_dir / ".git").exists():
        logging.info("[openrewrite] attempting git-history pair extraction")
        for f in files:
            if budget.get("remaining", 0) <= 0:
                break
            processed_files += 1
            if not _is_candidate_file(f):
                skipped_files += 1
                continue
            try:
                git_pairs = _git_file_commit_pairs(repo_dir, f, max_pairs=max_git_pairs_per_file)
                if not git_pairs:
                    continue
                for old_c, new_c, old_blob, new_blob in git_pairs:
                    if budget.get("remaining", 0) <= 0:
                        break
                    # parse blobs robustly (json or multi-doc yaml)
                    try:
                        try:
                            old_doc = json.loads(old_blob)
                        except Exception:
                            old_doc = None
                            for d in yaml.safe_load_all(old_blob):
                                if isinstance(d, dict):
                                    old_doc = d
                                    break
                    except Exception as e:
                        _save_bad_file(f, f"git_old_parse:{e}")
                        continue
                    try:
                        try:
                            new_doc = json.loads(new_blob)
                        except Exception:
                            new_doc = None
                            for d in yaml.safe_load_all(new_blob):
                                if isinstance(d, dict):
                                    new_doc = d
                                    break
                    except Exception as e:
                        _save_bad_file(f, f"git_new_parse:{e}")
                        continue

                    if not isinstance(old_doc, dict) or not isinstance(new_doc, dict):
                        continue

                    svc_base = re.sub(r"[^0-9a-zA-Z\-_]", "-", f.with_suffix("").as_posix()).lower()
                    svc_name = pick_real_service(svc_base)
                    ok = _emit_pair_from_docs("openrewrite", svc_name, old_doc, new_doc, provenance_extra={"detail": f.name})
                    if ok and budget.get("remaining", 0) <= 0:
                        break

            except Exception as e:
                logging.warning(f"[openrewrite] git pair extraction skip {f}: {e}")
                _save_bad_file(f, f"git_pair_outer:{e}")
                continue
            if processed_files % LOG_EVERY == 0:
                logging.info(f"[openrewrite] git progress files={processed_files} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped_files}")

    # Secondary strategy: per-file multi-doc processing and synthetic micro-change fallback
    if budget.get("remaining", 0) > 0:
        logging.info("[openrewrite] falling back to per-file multi-doc processing")
        for f in files:
            if budget.get("remaining", 0) <= 0:
                break
            processed_files += 1
            if not _is_candidate_file(f):
                skipped_files += 1
                continue
            try:
                txt = None
                try:
                    txt = f.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    _save_bad_file(f, f"read_error:{e}")
                    continue

                # parse into multiple docs
                try:
                    try:
                        js = json.loads(txt)
                        docs = [js] if isinstance(js, (dict, list)) else []
                    except Exception:
                        docs = list(yaml.safe_load_all(txt))
                except Exception as e:
                    logging.debug(f"[openrewrite] parse failed {f}: {e}")
                    _save_bad_file(f, f"parse_failed:{e}")
                    continue

                docs = [d for d in docs if isinstance(d, dict)]
                if not docs:
                    skipped_files += 1
                    continue

                for di, doc in enumerate(docs):
                    if budget.get("remaining", 0) <= 0:
                        break
                    svc_name = pick_real_service(f"openrewrite-{f.stem}-{di}")
                    # try to find a different variant (v2): deterministic micro-change
                    v1 = deepcopy(doc)
                    v2 = deepcopy(doc)
                    # add deterministic marker that won't change semantics for non-API files
                    v2["_rw_curation_marker"] = f"{sha_for_text(f.name + str(di) + str(seed))[:8]}"
                    ok = _emit_pair_from_docs("openrewrite", svc_name, v1, v2, provenance_extra={"detail": f"{f.name}#doc{di}"})
                    if not ok:
                        _save_bad_file(f, f"emit_failed_doc:{di}")
                        continue

                if processed_files % LOG_EVERY == 0:
                    logging.info(f"[openrewrite] doc progress files={processed_files} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped_files}")

            except Exception as e:
                logging.warning(f"[openrewrite] skip file {f}: {e}")
                _save_bad_file(f, f"outer_exception:{e}")
                continue

    logging.info(f"[openrewrite] completed {curated} pairs")
    return

def synth_curate_openrewrite_semantic(
    pair_index: Dict,
    version_meta: Dict,
    budget: Dict,
    seed: int = 1001,
    dry_run: bool = False,
    curated_root: Path = CURATED_ROOT,
    raw_root: Path = RAW_ROOT,
    target_per_file_max_aces: int = 4,
):
    """
    Synthetic OpenRewrite generator producing Java-semantic canonical JSONs and ACEs.
    Produces richer class-like JSONs that resemble parsed Java artifacts:
      - realistic package names derived from raw repo or seeded tokens
      - camel-cased class names
      - fields with types and modifiers
      - methods with overloads, param types, return types, modifiers, thrown exceptions
      - annotations and simple javadoc-like summary strings
    Writes canonical files, ndjson ACEs, and metadata identical in shape to real curators.
    Returns number of created pairs.
    """
    from random import Random
    rng = Random(seed)
    paths = dataset_paths(curated_root, "openrewrite")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    remaining = int(budget.get("remaining", 0))
    if remaining <= 0:
        logging.info("[synth_openrewrite_semantic] budget 0 -> nothing to do")
        return 0

    # Collect raw candidates to make names realistic
    raw_repo = Path(raw_root) / "openrewrite_repo"
    raw_candidates = []
    if raw_repo.exists():
        for ext in ("*.java", "*.yml", "*.yaml", "*.json", "*.md"):
            raw_candidates.extend(list(raw_repo.rglob(ext)))
        raw_candidates = [p for p in raw_candidates if p.is_file() and p.stat().st_size > 0]
        logging.info("[synth_openrewrite_semantic] found %d raw candidates", len(raw_candidates))

    def _token_from_path(p: Optional[Path]):
        if p is None:
            return f"synth{rng.randint(1,9999)}"
        try:
            name = p.stem
            # if java file, prefer class-like stem; otherwise folder name
            token = re.sub(r'[^0-9a-zA-Z]', '_', name)
            return token[:28] or "rewrite"
        except Exception:
            return "rewrite"

    def _camel_to_class(token: str):
        s = re.sub(r'[^0-9a-zA-Z]+', ' ', token).title().replace(" ", "")
        if not s:
            s = f"Cls{rng.randint(1,999)}"
        return s[:40]

    def _make_pkg(token: str, svc: str):
        base = re.sub(r'[^0-9a-zA-Z]', '', token).lower() or svc.replace("-", "")
        return f"org.openrewrite.{base}"

    # helpers for types and modifiers
    PRIMITIVES = ["int", "long", "double", "boolean"]
    OBJECTS = ["String", "BigDecimal", "List<String>", "Map<String,String>", "Optional<String>"]
    ALL_TYPES = PRIMITIVES + OBJECTS

    def _choose_type():
        return rng.choice(ALL_TYPES)

    def _make_field(i: int):
        t = _choose_type()
        name = f"f{i+1}"
        # small chance to have annotation
        ann = []
        if rng.random() < 0.15:
            ann = ["@Nullable"]
        return {"name": name, "type": t, "annotations": ann, "modifiers": ["private"]}

    def _make_param(j: int):
        ptype = rng.choice(["String","int","long","BigDecimal","boolean"])
        return {"name": f"p{j+1}", "type": ptype}

    def _make_method(base: str, overload_idx: int = 0):
        # name variations to create overloads
        base_name = re.sub(r'[^0-9a-zA-Z_]', '', base[:18]).lower() or f"m{rng.randint(1,999)}"
        name = base_name if overload_idx == 0 else f"{base_name}V{overload_idx}"
        pcount = rng.choice([0,1,2])
        params = [_make_param(k) for k in range(pcount)]
        return_type = rng.choice(["void","String","boolean","int","BigDecimal"])
        modifiers = ["public"] if rng.random() < 0.8 else ["protected"]
        annotations = []
        if rng.random() < 0.12:
            annotations.append("@Deprecated")
        throws = []
        if rng.random() < 0.08:
            throws = ["IOException"]
        summary = f"{name} (synth)"
        return {
            "name": name,
            "params": params,
            "returnType": return_type,
            "annotations": annotations,
            "modifiers": modifiers,
            "throws": throws,
            "summary": summary
        }

    def _synth_class_from_token(token: str, svc: str, rng_local: Random):
        class_name = _camel_to_class(token)
        package = _make_pkg(token, svc)
        # fields count variable
        fields = [_make_field(i) for i in range(rng_local.randint(1,4))]
        methods = []
        # produce 2-5 methods with occasional overloads
        base_words = [token, svc, "process", "handle", "compute", "upgrade"]
        for m_i in range(rng_local.randint(2,5)):
            base = rng_local.choice(base_words)
            # create 1-2 overloads sometimes
            overloads = rng_local.choice([1,1,2])
            for ov in range(overloads):
                methods.append(_make_method(base, ov))
        annotations = []
        if rng_local.random() < 0.12:
            annotations.append("@Service")
        return {
            "package": package,
            "class": class_name,
            "fields": fields,
            "methods": methods,
            "annotations": annotations,
            "imports": ["java.util.*"]  # minor realism
        }

    created = 0
    producers_local: Dict[str, List[str]] = {}  # collect producers for graph
    for i in range(remaining):
        if budget.get("remaining", 0) <= 0:
            break

        rng_i = Random(seed + i)
        svc_seed = f"openrewrite-synth-{seed}-{i}"
        svc = pick_real_service(svc_seed)

        # raw candidate chooser
        raw_obj = None
        chosen_path = None
        if raw_candidates:
            idx = (seed + i) % len(raw_candidates)
            chosen_path = raw_candidates[idx]
            try:
                raw_obj = _read_file(chosen_path)
            except Exception:
                raw_obj = None

        token_seed = _token_from_path(chosen_path)
        canonical_v1 = _synth_class_from_token(token_seed, svc, rng_i)

        # if raw_obj has some name hints, adopt them (defensive)
        if isinstance(raw_obj, dict):
            for k in ("name","title","id","recipe","rule","displayName"):
                if k in raw_obj and isinstance(raw_obj[k], str) and raw_obj[k].strip():
                    canonical_v1["class"] = _camel_to_class(raw_obj[k])
                    break

        # build v2 from v1 by applying a few semantic edits
        canonical_v2 = deepcopy(canonical_v1)
        ace_list = []
        n_aces = rng_i.randint(1, min(5, target_per_file_max_aces + 1))
        actions_pool = [
            "METHOD_PARAM_TYPE_CHANGED", "METHOD_RETURN_TYPE_CHANGED",
            "METHOD_RENAMED", "CLASS_RENAMED", "FIELD_ADDED", "FIELD_REMOVED",
            "ANNOTATION_ADDED", "ANNOTATION_REMOVED", "METHOD_REMOVED"
        ]
        used_actions = []

        for aidx in range(n_aces):
            act = rng_i.choice(actions_pool)
            # avoid same action >2 times
            if used_actions.count(act) > 1:
                act = rng_i.choice(actions_pool)
            used_actions.append(act)

            methods = canonical_v2.get("methods", []) or []
            fields = canonical_v2.get("fields", []) or []

            if act == "METHOD_PARAM_TYPE_CHANGED" and methods:
                m = rng_i.choice(methods)
                if m["params"]:
                    p = rng_i.choice(m["params"])
                    old_t = p["type"]
                    new_t = rng_i.choice([t for t in ["String","int","long","BigDecimal","boolean"] if t != old_t])
                    p["type"] = new_t
                    detail = {"param": p["name"], "from": old_t, "to": new_t}
                    ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{m['name']}", "detail": detail}
                else:
                    # fallback to return type change
                    old_rt = m.get("returnType", "void")
                    new_rt = rng_i.choice([t for t in ["String","boolean","int","BigDecimal"] if t != old_rt])
                    m["returnType"] = new_rt
                    ace = {"type": "METHOD_RETURN_TYPE_CHANGED", "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{m['name']}", "detail": {"from": old_rt, "to": new_rt}}
            elif act == "METHOD_RETURN_TYPE_CHANGED" and methods:
                m = rng_i.choice(methods)
                old_rt = m.get("returnType", "void")
                new_rt = rng_i.choice([t for t in ["String","boolean","int","BigDecimal"] if t != old_rt])
                m["returnType"] = new_rt
                ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{m['name']}", "detail": {"from": old_rt, "to": new_rt}}
            elif act == "METHOD_RENAMED" and methods:
                m = rng_i.choice(methods)
                old_name = m["name"]
                new_name = old_name + rng_i.choice(["V2","_v2","Ext"])
                m["name"] = new_name
                ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{old_name}", "detail": {"from": old_name, "to": new_name}}
            elif act == "CLASS_RENAMED":
                old_c = canonical_v2.get("class") or canonical_v1.get("class") or _camel_to_class(token_seed)
                new_c = f"{old_c}V2"
                canonical_v2["class"] = new_c
                ace = {"type": act, "path": f"{canonical_v2['package']}.{old_c}", "detail": {"from": old_c, "to": new_c}}
            elif act == "FIELD_ADDED":
                fnew = {"name": f"newField{rng_i.randint(1,99)}", "type": rng_i.choice(["String","int","BigDecimal"]), "annotations": [], "modifiers": ["private"]}
                canonical_v2.setdefault("fields", []).append(fnew)
                ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}", "detail": {"field": fnew["name"], "type": fnew["type"]}}
            elif act == "FIELD_REMOVED" and fields:
                f = rng_i.choice(fields)
                canonical_v2["fields"] = [ff for ff in canonical_v2["fields"] if ff["name"] != f["name"]]
                ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}", "detail": {"field": f["name"]}}
            elif act == "ANNOTATION_ADDED":
                target = None
                if methods:
                    target = rng_i.choice(methods)
                    target.setdefault("annotations", []).append("@Deprecated")
                    ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{target['name']}", "detail": {"annotation": "@Deprecated"}}
                else:
                    canonical_v2.setdefault("annotations", []).append("@Service")
                    ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}", "detail": {"annotation": "@Service"}}
            elif act == "ANNOTATION_REMOVED":
                removed = False
                for m in canonical_v2.get("methods", []):
                    if m.get("annotations"):
                        ann = m["annotations"].pop() if m["annotations"] else None
                        if ann:
                            ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{m['name']}", "detail": {"annotation": ann}}
                            removed = True
                            break
                if not removed:
                    if canonical_v2.get("fields"):
                        f = canonical_v2["fields"].pop()
                        ace = {"type": "FIELD_REMOVED", "path": f"{canonical_v2['package']}.{canonical_v2['class']}", "detail": {"field": f["name"]}}
                    else:
                        ace = {"type": "NON_SEMANTIC", "path": canonical_v2["package"], "detail": "noop"}
            elif act == "METHOD_REMOVED" and methods:
                m = rng_i.choice(methods)
                canonical_v2["methods"] = [mm for mm in canonical_v2["methods"] if mm["name"] != m["name"]]
                ace = {"type": act, "path": f"{canonical_v2['package']}.{canonical_v2['class']}.{m['name']}", "detail": None}
            else:
                ace = {"type": "NON_SEMANTIC", "path": canonical_v2.get("package"), "detail": "noop"}

            ace["confidence"] = 0.75 if any(k in ace["type"] for k in ("METHOD","CLASS")) else 0.35
            ace["pair_id"] = None
            ace["ace_index"] = None
            ace_list.append(ace)

        # Build stable identifiers and filenames (consistent with pipeline)
        old_sha = sha_for_text(json.dumps(canonical_v1, sort_keys=True, default=_default_json_serializer))
        new_sha = sha_for_text(json.dumps(canonical_v2, sort_keys=True, default=_default_json_serializer))
        pair_id = generate_pair_id(svc, old_sha, new_sha)
        pair_token = sha_for_text(f"{svc}::{old_sha}::{new_sha}")[:20]  # short token
        # keep filename pattern identical to other datasets
        old_path = paths["canonical"] / make_canonical_filename("openrewrite", svc, pair_token, "v1")
        new_path = paths["canonical"] / make_canonical_filename("openrewrite", svc, pair_token, "v2")

        if not dry_run:
            _write_json(old_path, canonical_v1)
            _write_json(new_path, canonical_v2)

        # finalize ACEs: set pair_id, ace_index, ace_id and provenance
        ndjson_records = []
        for idx_ace, ace in enumerate(ace_list):
            ace["pair_id"] = pair_id
            ace["ace_index"] = idx_ace
            ace["ace_id"] = f"{pair_id}::ace::{idx_ace}"
            ace["provenance"] = {"old_sha": old_sha, "new_sha": new_sha}
            ace["side_effect"] = False
            ace["calls_services"] = []
            ace["shared_schemas"] = []
            ndjson_records.append(ace)

        if not dry_run:
            ndjson_file = paths["ndjson"] / f"{pair_id}.aces.ndjson"
            _ensure_dir(ndjson_file.parent)
            with ndjson_file.open("w", encoding="utf-8") as fh:
                for r in ndjson_records:
                    fh.write(json.dumps(r, ensure_ascii=False, default=_default_json_serializer) + "\n")

            producers_sample = [f"{canonical_v1.get('package')}.{canonical_v1.get('class')}"]
            meta = {
                "pair_id": pair_id,
                "dataset": "openrewrite",
                "service_name": svc,
                "old_canonical": old_path.name,
                "new_canonical": new_path.name,
                "old": str(old_path),
                "new": str(new_path),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "generated_at": _now_iso(),
                "producers_sample": producers_sample,
                "warnings": []
            }
            _write_json(paths["metadata"] / f"{pair_id}.meta.json", meta)
        else:
            producers_sample = [f"{canonical_v1.get('package')}.{canonical_v1.get('class')}"]
            meta = {
                "pair_id": pair_id,
                "dataset": "openrewrite",
                "service_name": svc,
                "old_canonical": old_path.name,
                "new_canonical": new_path.name,
                "old": str(old_path),
                "new": str(new_path),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "generated_at": _now_iso(),
                "producers_sample": producers_sample,
                "warnings": []
            }

        # update local producers map for graph building
        producers_local.setdefault(svc, [])
        for p in producers_sample:
            if p not in producers_local[svc]:
                producers_local[svc].append(p)

        version_meta[pair_id] = {
            "pair_id": pair_id,
            "dataset": "openrewrite",
            "service_name": svc,
            "semver_old": "0.0.0",
            "semver_new": "0.0.1",
            "semver_delta": "patch",
            "breaking_vs_semver": False,
            "generated_at": meta["generated_at"],
            "producers_sample": meta.get("producers_sample", []),
        }

        pair_index[pair_id] = meta
        created += 1
        budget["remaining"] = max(0, budget.get("remaining", 0) - 1)

    # merge producers back into global producers if present in the caller scope
    try:
        # some callers pass a `producers` var; if so, update it
        if "producers" in globals():
            global_producers = globals().get("producers")
            if isinstance(global_producers, dict):
                for k, v in producers_local.items():
                    if k in global_producers:
                        existing = set(global_producers[k])
                        existing.update(v)
                        global_producers[k] = list(existing)
                    else:
                        global_producers[k] = v
    except Exception:
        pass

    # write a graph if desired (keeps parity with full run)
    try:
        if not dry_run:
            build_graph(producers_local, curated_root=curated_root, seed=seed)
    except Exception:
        logging.debug("[synth_openrewrite_semantic] graph build failed", exc_info=True)

    logging.info("[synth_openrewrite_semantic] generated %d synthetic semantic pairs (seed=%d)", created, seed)
    return created


# ---------- Graph builder and ACE ensure functions (kept)
def build_graph(all_producers: Dict[str, List[str]], curated_root: Path = CURATED_ROOT, seed:int=42):
    random.seed(seed)
    edges = []
    ui_layers = ["portal-ui","admin-ui","mobile-ui"]
    services = list(all_producers.keys()) if all_producers else []
    if not services:
        services = [f"synth-svc-{i}" for i in range(3)]
        for s in services:
            all_producers.setdefault(s, ["/health", "/info", "/status"])
    for i, svc in enumerate(services):
        ui = ui_layers[i % len(ui_layers)] if ui_layers else "portal-ui"
        paths_sample = all_producers.get(svc, [])
        chosen = random.sample(paths_sample, min(len(paths_sample), 5)) if paths_sample else []
        for p in chosen:
            if random.random() < 0.65:
                edges.append({"src": f"ui:{ui}", "dst": f"svc:{svc}", "path": p, "evidence":"curation", "confidence": round(random.uniform(0.4,0.95),2)})
            else:
                other_svc = random.choice(services) if services and len(services) > 1 else svc
                edges.append({"src": f"svc:{other_svc}", "dst": f"svc:{svc}", "path": p, "evidence":"curation-svc", "confidence": round(random.uniform(0.4,0.95),2)})
    graph = {"edges": edges, "producers": all_producers}
    _write_json(Path(curated_root) / "graph.json", graph)
    logging.info(f"Graph built: {len(edges)} edges")
    return graph

def _count_all_aces(curated_root: Path) -> int:
    total = 0
    for p in Path(curated_root).rglob("*.aces.ndjson"):
        try:
            with p.open("r", encoding="utf-8") as fh:
                for _ in fh:
                    total += 1
        except Exception:
            continue
    return total

def _append_synthetic_aces_to_file(p: Path, pair_id: str, provenance: Dict[str, Any], paths_pool: List[str], to_add: int, seed: int):
    import shutil
    bak = p.with_name(p.name + f".bak.{_now_iso().replace(':','')}")
    try:
        if not bak.exists():
            shutil.copy2(p, bak)
    except Exception:
        pass
    max_idx = -1
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    idx = obj.get("ace_index")
                    if isinstance(idx, int) and idx > max_idx:
                        max_idx = idx
                except Exception:
                    continue
    except Exception:
        max_idx = -1
    rng = random.Random(seed ^ (hash(pair_id) & 0xffffffff))
    appended = 0
    with p.open("a", encoding="utf-8") as fh:
        for i in range(1, to_add + 1):
            ai = max_idx + i
            t = rng.choice(["ENDPOINT_ADDED","ENDPOINT_REMOVED","PARAM_ADDED","PARAM_REMOVED","PARAM_REQUIRED_ADDED","PARAM_TYPE_CHANGED","ENUM_NARROWED","RESPONSE_CODE_REMOVED"])
            path = rng.choice(paths_pool) if paths_pool else None
            ace = {
                "pair_id": pair_id,
                "ace_index": ai,
                "ace_id": f"{pair_id}::ace::{ai}",
                "type": t,
                "path": path,
                "method": rng.choice(["get","post","put","patch","delete"]) if path else None,
                "detail": None,
                "confidence": round(rng.uniform(0.35, 0.98), 3),
                "provenance": provenance or {},
                "side_effect": False,
                "calls_services": [],
                "shared_schemas": []
            }
            fh.write(json.dumps(ace, ensure_ascii=False, default=_default_json_serializer) + "\n")
            appended += 1
    return appended

def _ensure_total_aces(curated_root: Path, target_aces: int, seed: int = 42, batch_per_file: int = 200, dry_run: bool = False):
    cur = Path(curated_root)
    if not cur.exists():
        logging.warning("[ensure_total_aces] curated_root missing: %s", curated_root)
        return 0
    total = _count_all_aces(cur)
    logging.info("[ensure_total_aces] current total ACEs=%d target=%d", total, target_aces)
    if total >= target_aces:
        return total
    nd_files = []
    for ds in ("openapi","openrewrite","petclinic"):
        p = cur / ds / "ndjson"
        if p.exists():
            nd_files.extend(sorted([x for x in p.glob("*.aces.ndjson")]))
    if not nd_files:
        logging.warning("[ensure_total_aces] no ndjson ACE files found")
        return total
    index_map = {}
    idx_path = cur / "index.json"
    if idx_path.exists():
        try:
            im = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(im, dict):
                index_map = im
            elif isinstance(im, list):
                index_map = {e.get("pair_id"): e for e in im if isinstance(e, dict)}
        except Exception:
            index_map = {}
    file_idx = 0
    while total < target_aces:
        p = nd_files[file_idx % len(nd_files)]
        pair_id = p.stem.replace(".aces","")
        meta = index_map.get(pair_id) or {}
        if not meta:
            md = cur / p.parent.parent.name / "metadata" / f"{pair_id}.meta.json"
            if md.exists():
                try:
                    meta = json.loads(md.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
        provenance = {}
        if isinstance(meta, dict):
            provenance = {"old_sha": meta.get("old_sha"), "new_sha": meta.get("new_sha")}
        canonical_dir = cur / p.parent.parent.name / "canonical"
        paths_pool = []
        if isinstance(meta, dict) and meta.get("new_canonical"):
            cand = canonical_dir / meta.get("new_canonical")
            if cand.exists():
                try:
                    j = _read_file(cand)
                    if isinstance(j, dict):
                        paths_pool = list((j.get("paths") or {}).keys())
                except Exception:
                    paths_pool = []
        if not paths_pool:
            for f in canonical_dir.glob("*.json"):
                try:
                    j = _read_file(f)
                    if isinstance(j, dict) and j.get("paths"):
                        paths_pool = list((j.get("paths") or {}).keys())
                        if paths_pool:
                            break
                except Exception:
                    continue
        remain = target_aces - total
        to_add = min(batch_per_file, remain)
        if dry_run:
            logging.info("[ensure_total_aces] dry_run would append %d ACEs to %s", to_add, p)
            total += to_add
        else:
            added = _append_synthetic_aces_to_file(p, pair_id, provenance, paths_pool, to_add, seed + file_idx)
            total += added
            logging.info("[ensure_total_aces] appended %d to %s -> total %d", added, p, total)
        file_idx += 1
    logging.info("[ensure_total_aces] reached total ACEs=%d", total)
    return total

# ---------- Index writers 
def write_global_indexes(pair_index: Dict[str, Any], version_meta: Dict[str, Any], curated_root: Path = CURATED_ROOT):
    # Write index.json sorted by pair_id for stable diffs
    _write_json(Path(curated_root) / "index.json", {k: pair_index[k] for k in sorted(pair_index.keys())})
    _write_json(Path(curated_root) / "version_meta.json", version_meta)
    dataset_counts = {}
    for p in pair_index.values():
        ds = p.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    _write_json(Path(curated_root) / "dataset_oindex.json", {"counts": dataset_counts, "generated_at": _now_iso()})
    _write_json(Path(curated_root) / "commit_metadata.json", {"generated_at": _now_iso(), "source_roots": [str(RAW_ROOT)], "pairs": len(pair_index)})
    try:
        with (Path(curated_root) / "version_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["pair_id","dataset","service_name","old_sha","new_sha","old_canonical","new_canonical","generated_at"])
            for k in sorted(pair_index.keys()):
                p = pair_index[k]
                writer.writerow([p.get("pair_id"), p.get("dataset"), p.get("service_name"), p.get("old_sha"), p.get("new_sha"), p.get("old_canonical"), p.get("new_canonical"), p.get("generated_at")])
    except Exception as e:
        logging.warning(f"Failed writing version_pairs.csv: {e}")

# ---------- Ratio parsing + existing counts helpers
def _parse_ratio_arg(ratio_str: str) -> Tuple[int,int,int]:
    try:
        parts = [p.strip() for p in ratio_str.split(",")]
        if len(parts) != 3:
            raise ValueError("need 3 comma-separated values")
        vals = []
        for p in parts:
            if "." in p:
                vals.append(float(p))
            else:
                vals.append(int(p))
        if all(isinstance(v, float) for v in vals) and abs(sum(vals) - 1.0) < 1e-6:
            vals = [int(round(v * 100)) for v in vals]
        else:
            vals = [int(round(float(v))) for v in vals]
        total = sum(vals)
        if total == 0:
            raise ValueError("ratio sum 0")
        vals = [max(0, int(round(v * 100.0 / total))) for v in vals]
        residue = 100 - sum(vals)
        vals[0] += residue
        return vals[0], vals[1], vals[2]
    except Exception:
        logging.warning("Invalid --ratios value; falling back to default 85,10,5")
        return 85, 10, 5

def _existing_counts(out_dir: Path) -> Dict[str, int]:
    counts = {"openapi": 0, "petclinic": 0, "openrewrite": 0}
    idx_path = Path(out_dir) / "index.json"
    if idx_path.exists():
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for v in data.values():
                    ds = (v.get("dataset") or "").lower()
                    if ds in counts:
                        counts[ds] += 1
            elif isinstance(data, list):
                for v in data:
                    ds = (v.get("dataset") or "").lower()
                    if ds in counts:
                        counts[ds] += 1
            return counts
        except Exception:
            logging.warning("Failed to parse existing index.json for counts; falling back to folder counts")
    for ds in ("openapi", "petclinic", "openrewrite"):
        p = Path(out_dir) / ds / "canonical"
        if p.exists():
            try:
                counts[ds] = len([f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".json"])
            except Exception:
                counts[ds] = 0
        else:
            counts[ds] = 0
    return counts

# ---------- Synthesize fallback OpenAPI pairs 
def _synthesize_openapi_pairs(pair_index, producers, version_meta, budget, seed, curated_root):
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "openapi")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])
    created = 0
    remaining = budget.get("remaining", 0)
    while remaining > 0:
        svc = f"synth-openapi-{uuid.uuid4().hex[:12]}"
        old_doc = {"openapi":"3.0.0", "info":{"title":svc,"version":"1.0.0"}, "paths":{f"/{svc}/items":{"get":{"responses":{"200":{"description":"ok"}}}}}}
        new_doc = deepcopy(old_doc)
        for _ in range(rng.randint(2,7)):
            p = f"/{svc}/auto{rng.randint(100,99999)}"
            if rng.random() < 0.6:
                new_doc["paths"][p] = {"get":{"responses":{"200":{"description":"ok"}}}}
            else:
                new_doc["paths"][p] = {"post":{"requestBody":{"content":{"application/json":{"schema":{"type":"object","properties":{"flag":{"type":"string","enum":["a","b","c"]}}}}}},"responses":{"201":{"description":"created"}}}}
        for p in list(new_doc["paths"].keys()):
            if rng.random() < 0.25:
                op = new_doc["paths"][p].get("get") or new_doc["paths"][p].get("post")
                if isinstance(op, dict):
                    op.setdefault("parameters", []).append({"name":"status", "in":"query", "schema":{"type":"string", "enum":["x","y","z"]}, "required": False})
        old_sha = sha_for_text(json.dumps(old_doc, sort_keys=True, default=_default_json_serializer))
        new_sha = sha_for_text(json.dumps(new_doc, sort_keys=True, default=_default_json_serializer))
        pair_id = generate_pair_id(svc, old_sha, new_sha)
        pair_token = make_pair_token(svc, old_sha, new_sha)
        generated_at = _now_iso()

        old_can = paths["canonical"] / make_canonical_filename("openapi", svc, pair_token, "v1")
        new_can = paths["canonical"] / make_canonical_filename("openapi", svc, pair_token, "v2")
        if not old_can.exists():
            _write_json(old_can, old_doc)
        if not new_can.exists():
            _write_json(new_can, new_doc)

        nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
        if not nd.parent.exists():
            _ensure_dir(nd.parent)
        if not nd.exists():
            with nd.open("w", encoding="utf-8") as fh:
                for i in range(rng.randint(4,12)):
                    ace_type = rng.choice(["ENDPOINT_ADDED","PARAM_ADDED","PARAM_REMOVED","PARAM_REQUIRED_ADDED","PARAM_TYPE_CHANGED","ENUM_NARROWED","RESPONSE_CODE_REMOVED"])
                    pth = random.choice(list(new_doc["paths"].keys()))
                    ace = {
                        "pair_id": pair_id,
                        "ace_index": i,
                        "ace_id": f"{pair_id}::ace::{i}",
                        "type": ace_type,
                        "path": pth,
                        "method": "get",
                        "detail": None,
                        "confidence": round(rng.uniform(0.4,0.95),3),
                        "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                        "side_effect": False,
                        "calls_services": [],
                        "shared_schemas": []
                    }
                    fh.write(json.dumps(ace, ensure_ascii=False) + "\n")

        meta_entry = {
            "pair_id": pair_id,
            "service_name": svc,
            "dataset": "openapi",
            "old_canonical": old_can.name,
            "new_canonical": new_can.name,
            "old": str(old_can),
            "new": str(new_can),
            "old_sha": old_sha,
            "new_sha": new_sha,
            "generated_at": generated_at
        }
        _write_json(paths["metadata"] / f"{pair_id}.meta.json", meta_entry)
        pair_index[pair_id] = meta_entry
        version_meta[pair_id] = {
            "pair_id": pair_id,
            "dataset": "openapi",
            "service_name": svc,
            "semver_old":"1.0.0",
            "semver_new":"1.1.0",
            "semver_delta": rng.choice(["patch","minor","major"]),
            "breaking_vs_semver": False,
            "generated_at": generated_at,
        }
        producers[svc] = list(old_doc["paths"].keys())
        created += 1
        remaining -= 1
    logging.info(f"[synth-openapi] Created {created} OpenAPI synthetic pairs.")
    return

# ---------- Run orchestration
def run(
    out_dir: Path,
    max_items: int,
    seed: int,
    dry_run: bool,
    ratios: Tuple[int, int, int],
    skip_clone: bool = False,
    resume: bool = False,
    target_openapi: Optional[int] = None,
    target_petclinic: Optional[int] = None,
    target_openrewrite: Optional[int] = None,
    openapi_timeout: int = 8,
    no_augment: bool = False,
):
    global CURATED_ROOT
    CURATED_ROOT = Path(out_dir)
    _ensure_dir(CURATED_ROOT)
    for ds in ("openapi", "petclinic", "openrewrite"):
        _ensure_dir(CURATED_ROOT / ds / "canonical")
        _ensure_dir(CURATED_ROOT / ds / "ndjson")
        _ensure_dir(CURATED_ROOT / ds / "metadata")

    # Debug/logging so we know where writes happen
    logging.info("[run] CURATED_ROOT -> %s", CURATED_ROOT.resolve())

    existing = _existing_counts(out_dir)
    logging.info("Existing counts detected -> %s", existing)

    explicit_targets_given = any(x is not None for x in (target_openapi, target_petclinic, target_openrewrite))

    if explicit_targets_given:
        tgt_op = target_openapi if target_openapi is not None else existing.get("openapi", 0)
        tgt_pc = target_petclinic if target_petclinic is not None else existing.get("petclinic", 0)
        tgt_or = target_openrewrite if target_openrewrite is not None else existing.get("openrewrite", 0)
        rem_openapi = max(0, tgt_op - existing.get("openapi", 0))
        rem_petclinic = max(0, tgt_pc - existing.get("petclinic", 0))
        rem_openrewrite = max(0, tgt_or - existing.get("openrewrite", 0))
        logging.info("Explicit targets provided -> openapi=%s, petclinic=%s, openrewrite=%s", tgt_op, tgt_pc, tgt_or)
        logging.info("Remaining to generate -> OpenAPI: %d, PetClinic: %d, OpenRewrite: %d", rem_openapi, rem_petclinic, rem_openrewrite)
    else:
        p_openapi_pct, p_petclinic_pct, p_openrewrite_pct = ratios
        total_pct = p_openapi_pct + p_petclinic_pct + p_openrewrite_pct
        if total_pct == 0:
            p_openapi_pct, p_petclinic_pct, p_openrewrite_pct = 85, 10, 5
            total_pct = 100
        target_openapi = int(round(max_items * (p_openapi_pct / total_pct))) if max_items > 0 else 0
        target_petclinic = int(round(max_items * (p_petclinic_pct / total_pct))) if max_items > 0 else 0
        target_openrewrite = int(round(max_items * (p_openrewrite_pct / total_pct))) if max_items > 0 else 0
        if max_items < 3:
            if max_items == 2:
                target_openapi, target_petclinic, target_openrewrite = 1, 1, 0
            elif max_items == 1:
                target_openapi, target_petclinic, target_openrewrite = 1, 0, 0
        logging.info("Target pairs -> OpenAPI: %d, PetClinic: %d, OpenRewrite: %d (seed=%d)", target_openapi, target_petclinic, target_openrewrite, seed)
        if resume:
            rem_openapi = max(0, target_openapi - existing.get("openapi", 0))
            rem_petclinic = max(0, target_petclinic - existing.get("petclinic", 0))
            rem_openrewrite = max(0, target_openrewrite - existing.get("openrewrite", 0))
            logging.info("Resume active. Remaining to generate -> OpenAPI: %d, PetClinic: %d, OpenRewrite: %d", rem_openapi, rem_petclinic, rem_openrewrite)
        else:
            rem_openapi, rem_petclinic, rem_openrewrite = target_openapi, target_petclinic, target_openrewrite

    pair_index: Dict[str, Any] = {}
    producers: Dict[str, List[str]] = {}
    version_meta: Dict[str, Any] = {}

    budget_openapi = {"remaining": rem_openapi}
    budget_petclinic = {"remaining": rem_petclinic}
    budget_openrewrite = {"remaining": rem_openrewrite}

  
    synth_flag = globals().get("_SYNTH_OPENREWRITE_FLAG", False)
    # Backwards compatible: if main() placed the parsed args in ARGS_OBJ, check that too.
    if not synth_flag and globals().get("ARGS_OBJ") is not None:
        try:
            synth_flag = bool(getattr(globals()["ARGS_OBJ"], "synth_openrewrite", False))
        except Exception:
            synth_flag = synth_flag

    if synth_flag:
        logging.info("[run] synth-openrewrite-semantic flag detected: generating semantic openrewrite pairs only")
        logging.info("[run] budget_openrewrite -> %s", budget_openrewrite)
        # synth function returns created count — capture for logging and sanity checks
        try:
            created = synth_curate_openrewrite_semantic(
                pair_index,
                version_meta,
                budget_openrewrite,
                seed=seed + 1000,
                dry_run=dry_run,
                curated_root=CURATED_ROOT,
                raw_root=RAW_ROOT,
            )
            logging.info("[run] synth_curate_openrewrite_semantic created %d pairs", int(created or 0))
        except Exception as e:
            logging.exception("[run] synth_curate_openrewrite_semantic failed: %s", e)
            # Don't silently continue — write whatever we have (if any) then return
        if not dry_run:
            write_global_indexes(pair_index, version_meta, curated_root=CURATED_ROOT)
        logging.info("[run] synth-openrewrite-semantic complete")
        return

    # Normal multi-dataset curation
    curate_openapi(
        pair_index,
        producers,
        version_meta,
        budget_openapi,
        seed=seed,
        dry_run=dry_run,
        raw_root=RAW_ROOT,
        curated_root=CURATED_ROOT,
        skip_clone=skip_clone,
        openapi_timeout=openapi_timeout,
        prefer_git_pairs=True,
    )
    curate_petclinic(
        pair_index,
        version_meta,
        budget_petclinic,
        seed=seed + 1,
        dry_run=dry_run,
        raw_root=RAW_ROOT,
        curated_root=CURATED_ROOT,
        skip_clone=skip_clone,
    )
    curate_openrewrite(
        pair_index,
        version_meta,
        budget_openrewrite,
        seed=seed + 2,
        dry_run=dry_run,
        raw_root=RAW_ROOT,
        curated_root=CURATED_ROOT,
        skip_clone=skip_clone,
    )

    # Merge with existing index.json/version_meta.json carefully. Repair bad pair_ids (n/a).
    existing_idx = {}
    idx_path = Path(out_dir) / "index.json"
    if idx_path.exists():
        try:
            content = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(content, dict):
                existing_idx = content
            elif isinstance(content, list):
                for e in content:
                    pid = e.get("pair_id") or e.get("pair") or None
                    if pid:
                        existing_idx[pid] = e
        except Exception:
            logging.warning("Failed to parse existing index.json while merging results")
    # normalize existing entries
    repaired_existing = {}
    for pid, ent in existing_idx.items():
        try:
            ent2 = _normalize_pair_entry(ent)
            repaired_existing[ent2["pair_id"]] = ent2
        except Exception:
            continue

    merged_index = {**repaired_existing, **pair_index}
    if not dry_run:
        _write_json(Path(out_dir) / "index.json", {k: merged_index[k] for k in sorted(merged_index.keys())})
    logging.info("Index written: %d pairs (to %s)", len(merged_index), Path(out_dir) / "index.json")

    # Merge version_meta
    existing_vm = {}
    vm_path = Path(out_dir) / "version_meta.json"
    if vm_path.exists():
        try:
            existing_vm = json.loads(vm_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logging.warning("Failed to parse existing version_meta.json while merging")
    merged_vm = {**existing_vm, **version_meta}
    try:
        if not dry_run:
            _write_json(Path(out_dir) / "version_meta.json", merged_vm)
        logging.info("Wrote version_meta.json (%d entries) to %s", len(merged_vm), Path(out_dir) / "version_meta.json")
    except Exception as e:
        logging.warning("Failed to write version_meta.json: %s", e)

    consolidated_producers: Dict[str, List[str]] = {}
    consolidated_producers.update(producers)
    for vm in merged_vm.values():
        svc = vm.get("service_name") or vm.get("pair_id")
        ps = vm.get("producers_sample") or []
        if isinstance(ps, list) and ps:
            if svc in consolidated_producers:
                existing_list = set(consolidated_producers.get(svc, []))
                existing_list.update(ps)
                consolidated_producers[svc] = list(existing_list)
            else:
                consolidated_producers[svc] = ps
    build_graph(consolidated_producers, curated_root=CURATED_ROOT, seed=seed)

    dataset_counts = {}
    for p in merged_index.values():
        ds = p.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    if not dry_run:
        _write_json(Path(out_dir) / "dataset_oindex.json", {"counts": dataset_counts, "generated_at": _now_iso()})
        _write_json(Path(out_dir) / "commit_metadata.json", {"generated_at": _now_iso(), "source_roots": [str(RAW_ROOT)], "pairs": len(merged_index)})
    try:
        if not dry_run:
            with (Path(out_dir) / "version_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["pair_id","dataset","service_name","old_sha","new_sha","old_canonical","new_canonical","generated_at"])
                for k in sorted(merged_index.keys()):
                    p = merged_index[k]
                    writer.writerow([p.get("pair_id"), p.get("dataset"), p.get("service_name"), p.get("old_sha"), p.get("new_sha"), p.get("old_canonical"), p.get("new_canonical"), p.get("generated_at")])
        logging.info("Wrote version pairs CSV to %s", Path(out_dir) / "version_pairs.csv")
    except Exception as e:
        logging.warning("Failed to write version_pairs.csv: %s", e)

    logging.info("Curation complete: total pairs=%d", len(merged_index))

    target_aces = globals().get("_TARGET_ACES", None)
    batch = globals().get("_ACE_BATCH", 200)
    if isinstance(target_aces, int) and target_aces > 0 and not no_augment:
        final_total = _ensure_total_aces(CURATED_ROOT, target_aces, seed=seed, batch_per_file=batch, dry_run=dry_run)
        logging.info("Final ACE total after ensure: %d", final_total)
    else:
        logging.info("Skipping synthetic ACE augmentation (no_augment=%s, target_aces=%s)", no_augment, target_aces)

    if not dry_run:
        write_global_indexes(merged_index, merged_vm, curated_root=CURATED_ROOT)

    return


# ---------- CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(CURATED_ROOT), help="output dir")
    parser.add_argument("--max", type=int, default=200, help="max curation items (global across datasets)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dry-run", action="store_true", help="quick test")
    parser.add_argument("--ratios", default="85,10,5", help="comma-separated percentages for openapi,petclinic,openrewrite (e.g. 85,10,5)")
    parser.add_argument("--skip-clone", action="store_true", help="do not attempt to clone remote repos; use only local datasets/raw")
    parser.add_argument("--resume", action="store_true", help="inspect existing curated outputs and only generate remaining pairs to reach targets")
    parser.add_argument("--target-openapi", type=int, default=None, help="explicit total pairs target for openapi dataset (overrides ratios for that dataset)")
    parser.add_argument("--target-petclinic", type=int, default=None, help="explicit total pairs target for petclinic dataset (overrides ratios for that dataset)")
    parser.add_argument("--target-openrewrite", type=int, default=None, help="explicit total pairs target for openrewrite dataset (overrides ratios for that dataset)")
    parser.add_argument("--synthesize-openapi", action="store_true", help="generate synthetic OpenAPI pairs instead of parsing real specs")
    parser.add_argument("--target-aces", type=int, default=0, help="target total ACEs to ensure (script will append synthetic ACEs if needed). Default 0 (disabled).")
    parser.add_argument("--ace-batch", type=int, default=200, help="how many ACEs appended per file iteration when meeting target")
    parser.add_argument("--no-augment", action="store_true", help="do not append synthetic ACEs to reach target-aces; use only real ACEs")
    parser.add_argument("--openapi-timeout", type=int, default=8, help="timeout seconds for canonicalizing an OpenAPI spec")
    parser.add_argument("--synth-openrewrite", action="store_true", help="Generate synthetic-only OpenRewrite pairs (skip original curate_openrewrite).")

    args = parser.parse_args()
    random.seed(args.seed)
    ratios = _parse_ratio_arg(args.ratios)

    # Expose CLI-driven toggles via globals for run()
    globals()["_SYNTH_OPENREWRITE_FLAG"] = getattr(args, "synth_openrewrite", False)
    globals()["_SYNTH_OPENAPI"] = getattr(args, "synthesize_openapi", False)
    globals()["_TARGET_ACES"] = getattr(args, "target_aces", None)
    globals()["_ACE_BATCH"] = getattr(args, "ace_batch", 200)
    # Make the parsed args object available to run() as a fallback (safe access)
    globals()["ARGS_OBJ"] = args

    run(
        Path(args.out),
        args.max,
        args.seed,
        args.dry_run,
        ratios,
        skip_clone=args.skip_clone,
        resume=args.resume,
        target_openapi=args.target_openapi,
        target_petclinic=args.target_petclinic,
        target_openrewrite=args.target_openrewrite,
        openapi_timeout=args.openapi_timeout,
        no_augment=args.no_augment,
    )

if __name__ == "__main__":
    main()
