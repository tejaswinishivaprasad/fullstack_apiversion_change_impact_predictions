#!/usr/bin/env python3
"""
curated_datasets.py - end-to-end curated dataset generator (patched + optimized)

Main runtime improvements (non-breaking):
 - _read_file uses an LRU cache to avoid re-parsing the same files repeatedly.
 - external $ref resolution is cached to avoid repeated file loads for the same referenced document.
 - canonicalize_with_timeout uses a persistent ProcessPoolExecutor (reuses worker processes).
 - normal JSON writes avoid unnecessary sort_keys; stable SHA blobs still use sorted serialization.
 - optional CLI flags (--skip-clone, --no-augment, --synthesize-openapi) unchanged and respected.
 - preserves folder structure and output filenames exactly.

Usage example:
  python curated_datasets.py --out datasets/curated --seed 42 --max 200 \
    --target-openapi 1000 --target-petclinic 200 --target-openrewrite 200 \
    --no-augment --openapi-timeout 10
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

import yaml

# ---------- Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- Defaults (can be overridden via CLI)
RAW_ROOT = Path("datasets/raw")
CURATED_ROOT = Path("datasets/curated")

OPENAPI_REPO = "https://github.com/APIs-guru/openapi-directory.git"
PETCLINIC_REPO = "https://github.com/spring-petclinic/spring-petclinic-microservices.git"
OPENREWRITE_REPO = "https://github.com/openrewrite/rewrite.git"

_MAX_CANONICAL_DEPTH = 2000

# ---------- Small shared caches (module-level)
_resolved_external_cache: Dict[str, Any] = {}
# cache for canonicalization results keyed by content hash (to avoid re-canonicalizing same doc)
_canon_cache: Dict[str, Tuple[Optional[Dict[str, Any]], List[str]]] = {}
# persistent process pool (lazy init)
_CANON_POOL: Optional[ProcessPoolExecutor] = None

# ---------- Helpers
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

# Use cached file reads to avoid repetitive disk parsing
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
    txt = _read_file_cached_text(str(p))
    if not txt:
        return None
    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            return yaml.safe_load(txt)
        except Exception as e:
            logging.debug(f"yaml.safe_load failed for {p}: {e}")
            try:
                # last resort
                return json.loads(txt)
            except Exception:
                return None
    try:
        return json.loads(txt)
    except Exception:
        try:
            return yaml.safe_load(txt)
        except Exception as e:
            logging.debug(f"_read_file failed to parse {p}: {e}")
            return None

# Non-sorted JSON write for speed; keep deterministic SHA helper for when sort_keys is required.
def _write_json(p: Path, obj):
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default_json_serializer), encoding="utf-8")

def _write_json_sorted(p: Path, obj):
    """Write JSON with sorted keys (only use when you need stable ordering)."""
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=_default_json_serializer), encoding="utf-8")

# ---------- Clone helper (unchanged behavior)
def clone_if_needed(repo_url: str, dest: Path, timeout_sec: int = 120):
    try:
        if dest.exists() and any(dest.rglob("*")):
            logging.info(f"Repo already present: {dest}")
            return
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Cloning {repo_url} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
    except Exception as e:
        logging.warning(f"git clone failed for {repo_url}: {e}")
        if dest.exists() and not any(dest.rglob("*")):
            try:
                shutil.rmtree(dest)
            except Exception:
                pass

# ---------- Canonicalize & $ref resolver (robust, with external ref caching)
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
    """
    Robust canonicalizer:
    - If `spec` is a list, recursively search for the first dict-like candidate
      that looks like an OpenAPI/Swagger document. Handles nested lists.
    - Defensive when encountering unexpected types (lists, strings) and records
      warnings instead of crashing. Returns (canon_dict, warnings_list).
    """
    warnings: List[str] = []

    # Helper: recursively find a dict candidate inside lists (limited depth)
    def _find_dict_candidate(obj, depth=0):
        if depth > 8:
            return None
        if isinstance(obj, dict):
            if obj.get("paths") or obj.get("openapi") or obj.get("swagger"):
                return obj
            # fallback: return dict even if it doesn't look like OpenAPI
            return obj
        if isinstance(obj, list):
            for it in obj:
                cand = _find_dict_candidate(it, depth + 1)
                if cand is not None:
                    return cand
        return None

    # Normalize top-level spec if it's a list or weird wrapper
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
                            # Use module-level cache for external refs
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

    # Ensure canonical shape expected by downstream code
    if not isinstance(canon, dict) or "paths" not in canon:
        canon = {"paths": {}}

    return canon, warnings

# ---------- Persistent ProcessPoolExecutor based canonicalize_with_timeout
def _start_canon_pool(max_workers: int = max(1, (mp.cpu_count() or 2) // 2)):
    global _CANON_POOL
    if _CANON_POOL is None:
        # use a small pool; too many processes harm IO-bound workloads
        _CANON_POOL = ProcessPoolExecutor(max_workers=max_workers)
    return _CANON_POOL

def _canon_worker_serialized(spec_text: str, base_dir_str: str):
    """
    Executed in worker process. Re-implements minimal local parsing and then calls
    canonicalize_spec defined in this module (worker will have module code).
    """
    import json as _json, yaml as _yaml
    from pathlib import Path as _Path
    try:
        spec = _json.loads(spec_text)
    except Exception:
        spec = _yaml.safe_load(spec_text)
    canon, warns = canonicalize_spec(spec, _Path(base_dir_str))
    return {"canon": canon, "warns": warns}

def canonicalize_with_timeout(spec, base_dir, timeout_sec: int = 8):
    """
    Use a persistent ProcessPoolExecutor to avoid per-call process spawn overhead.
    Returns (canon, warns) or (None, [error]).
    """
    # Cache by spec content hash to avoid repeated canonicalization on identical content
    try:
        spec_blob = json.dumps(spec, sort_keys=True, default=_default_json_serializer)
    except Exception:
        spec_blob = yaml.safe_dump(spec)
    key = sha_for_text(spec_blob)
    if key in _canon_cache:
        return _canon_cache[key]

    pool = _start_canon_pool()
    # Serialize spec to text for safer pickling
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

# ---------- Diff injection and ACE logic (unchanged behavior, small style improvements)
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
                rb = op.get("requestBody", {})
                for ct, body in (rb.get("content") or {}).items():
                    s = body.get("schema", {})
                    if isinstance(s, dict) and "enum" in s:
                        if isinstance(s["enum"], list) and len(s["enum"]) > 1 and r.random() < 0.6:
                            s["enum"] = s["enum"][:-1]

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

# ---------- Per-dataset folder helpers
def dataset_paths(root: Path, dataset: str):
    base = Path(root) / dataset
    return {
        "base": base,
        "canonical": base / "canonical",
        "ndjson": base / "ndjson",
        "metadata": base / "metadata",
    }

# ---------- Git history helper to extract real pairs (unchanged behavior)
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

# ---------- Curators (cleaned, same outputs)
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
    _ensure_dir(paths["canonical"])
    _ensure_dir(paths["ndjson"])
    _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "openapi_repo"
    local_dir = raw_root / "openapi"

    if not skip_clone:
        clone_if_needed(OPENAPI_REPO, repo_dir, timeout_sec=120)

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
                logging.debug(f"[curate_openapi][SKIP] {file} -> not_file")
                continue

            if file.suffix.lower() not in (".yaml", ".yml", ".json"):
                skipped_files += 1
                logging.debug(f"[curate_openapi][SKIP] {file} -> bad_suffix")
                continue

            try:
                st_size = file.stat().st_size
            except Exception:
                st_size = 0
            if st_size == 0:
                skipped_files += 1
                logging.info(f"[curate_openapi][SKIP] {file} -> empty")
                continue
            if st_size > 2_000_000:
                skipped_files += 1
                logging.info(f"[curate_openapi][SKIP] {file} -> too large {st_size}")
                continue

            doc = _read_file(file)
            if doc is None:
                skipped_files += 1
                logging.info(f"[curate_openapi][SKIP] {file} -> parsed None/empty")
                continue

            if not isinstance(doc, (dict, list)):
                skipped_files += 1
                logging.info(f"[curate_openapi][SKIP] {file} -> not dict/list ({type(doc).__name__})")
                continue

            if isinstance(doc, list):
                cand = next((s for s in doc if isinstance(s, dict) and (s.get("paths") or s.get("openapi") or s.get("swagger"))), None)
                if cand is None:
                    skipped_files += 1
                    logging.info(f"[curate_openapi][SKIP] {file} -> list wrapper with no spec candidate")
                    continue

            pair_seed = (int(sha_for_text(str(file)), 16) ^ seed) & 0x7fffffff

            processed_real_pair = False
            try:
                if prefer_git_pairs and repo_dir.exists() and (repo_dir / ".git").exists():
                    git_pairs = _git_file_commit_pairs(repo_dir, file, max_pairs=2)
                else:
                    git_pairs = []
            except Exception as e:
                git_pairs = []
                logging.debug(f"[curate_openapi] git_pairs failed for {file}: {e}")

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
                        logging.debug(f"[curate_openapi] git old_blob parse failed {file} {old_c[:8]}")
                        continue
                    try:
                        try:
                            new_doc = json.loads(new_blob)
                        except Exception:
                            new_doc = yaml.safe_load(new_blob)
                    except Exception:
                        logging.debug(f"[curate_openapi] git new_blob parse failed {file} {new_c[:8]}")
                        continue

                    old_can, warns_old = canonicalize_with_timeout(old_doc, file.parent, timeout_sec=openapi_timeout)
                    new_can, warns_new = canonicalize_with_timeout(new_doc, file.parent, timeout_sec=openapi_timeout)
                    if old_can is None or new_can is None:
                        logging.debug(f"[curate_openapi] git pair canonicalize failed/timeout for {file} {old_c[:8]}->{new_c[:8]} warns={warns_old or warns_new}")
                        continue

                    svc_base = re.sub(r"[^0-9a-zA-Z\-_]", "-", file.with_suffix("").as_posix()).lower()
                    old_can_path = paths["canonical"] / f"openapi--{svc_base}--{old_c[:8]}.canonical.json"
                    new_can_path = paths["canonical"] / f"openapi--{svc_base}--{new_c[:8]}.canonical.json"

                    if not dry_run:
                        _write_json(old_can_path, old_can)
                        _write_json(new_can_path, new_can)

                    # stable SHA uses sorted serialization for determinism
                    old_sha = sha_for_text(json.dumps(old_can, sort_keys=True, default=_default_json_serializer))
                    new_sha = sha_for_text(json.dumps(new_can, sort_keys=True, default=_default_json_serializer))
                    pair_fingerprint = sha_for_text(svc_base + old_sha + new_sha)
                    pair_id = f"pair-{pair_fingerprint}"
                    generated_at = _now_iso()

                    pair_entry = {
                        "pair_id": pair_id,
                        "service_name": svc_base,
                        "dataset": "openapi",
                        "old_canonical": old_can_path.name,
                        "new_canonical": new_can_path.name,
                        "old": str(old_can_path),
                        "new": str(new_can_path),
                        "old_sha": old_sha,
                        "new_sha": new_sha,
                        "git_old_commit": old_c,
                        "git_new_commit": new_c,
                        "warnings": (warns_old or []) + (warns_new or []),
                        "seed": pair_seed,
                        "generated_at": generated_at,
                    }
                    pair_index[pair_id] = pair_entry

                    eps = list((old_can.get("paths") or {}).keys())
                    eps_norm = [re.sub(r"\{[^}]+\}", "{*}", p) for p in eps]
                    producers[svc_base] = (rng.sample(eps_norm, 5) if len(eps_norm) > 5 else eps_norm)

                    # compute ACEs defensively
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
                            except Exception as op_e:
                                logging.warning(f"[curate_openapi] compare_operations ERROR {file} {p} {m}: {op_e}")
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

                    meta_path = paths["metadata"] / f"{pair_id}.meta.json"
                    if not dry_run:
                        _write_json(meta_path, pair_entry)

                    version_meta[pair_id] = {
                        "pair_id": pair_id,
                        "dataset": "openapi",
                        "service_name": svc_base,
                        "semver_old": "1.0.0",
                        "semver_new": "1.1.0",
                        "semver_delta": rng.choice(["patch", "minor", "major"]),
                        "breaking_vs_semver": rng.random() < 0.15,
                        "generated_at": generated_at,
                    }

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
                logging.info(f"[curate_openapi][SKIP] {file} -> canonicalize_none|{';'.join(warns or [])}")
                continue

            path_count = len((canon.get("paths") or {}).keys()) if isinstance(canon, dict) else 0
            if path_count == 0:
                skipped_files += 1
                logging.info(f"[curate_openapi][SKIP] {file} -> canonical has zero paths")
                continue

            v2doc = inject_openapi_diffs(canon, seed=pair_seed)
            svc_name = re.sub(r"[^0-9a-zA-Z\-_]", "-", file.with_suffix("").as_posix()).lower()
            old_can = paths["canonical"] / f"openapi--{svc_name}--v1.canonical.json"
            new_can = paths["canonical"] / f"openapi--{svc_name}--v2.canonical.json"
            if not dry_run:
                _write_json(old_can, canon)
                _write_json(new_can, v2doc)

            old_sha = sha_for_text(json.dumps(canon, sort_keys=True, default=_default_json_serializer))
            new_sha = sha_for_text(json.dumps(v2doc, sort_keys=True, default=_default_json_serializer))
            pair_id = f"pair-{sha_for_text(svc_name + old_sha + new_sha)}"
            generated_at = _now_iso()
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
                "generated_at": generated_at,
            }
            pair_index[pair_id] = pair_entry

            eps = list((canon.get("paths") or {}).keys())
            eps_norm = [re.sub(r"\{[^}]+\}", "{*}", p) for p in eps]
            producers[svc_name] = rng.sample(eps_norm, min(5, len(eps_norm))) if eps_norm else []

            aces = []
            def ops_map(spec):
                out = {}
                for p, m in (spec.get("paths") or {}).items():
                    if isinstance(m, dict):
                        out[p] = {mm.lower(): m[mm] for mm in m.keys()}
                return out

            old_ops = ops_map(canon)
            new_ops = ops_map(v2doc)
            all_paths = sorted(set(old_ops.keys()) | set(new_ops.keys()))
            for p in all_paths:
                methods = sorted(set(old_ops.get(p, {}).keys()) | set(new_ops.get(p, {}).keys()))
                for m in methods:
                    old_op = old_ops.get(p, {}).get(m)
                    new_op = new_ops.get(p, {}).get(m)
                    try:
                        detected = compare_operations(old_op, new_op, p, m)
                    except Exception as op_e:
                        logging.warning(f"[curate_openapi] compare_operations ERROR {file} {p} {m}: {op_e}")
                        continue
                    for d in detected:
                        try:
                            conf = compute_confidence(d, canon, v2doc)
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
                "generated_at": generated_at,
            }

            curated += 1
            budget["remaining"] = max(0, budget["remaining"] - 1)
            logging.info(f"[curate_openapi][OK] fallback {file} -> curated {curated} remaining={budget.get('remaining',0)}")
        except Exception as e:
            skipped_files += 1
            logging.warning(f"[curate_openapi][EXC] {file} -> {e}")
            logging.debug("traceback:", exc_info=True)
            continue

        if processed_files % LOG_EVERY_FILES == 0:
            logging.info(f"[curate_openapi] progress files={processed_files} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped_files}")

    logging.info(f"[curate_openapi] Finished: curated {curated} OpenAPI pairs into {paths['canonical']}")
    return

# ---------- Petclinic + OpenRewrite curators unchanged except for using new _write_json
def curate_petclinic(pair_index: Dict, version_meta: Dict, budget: Dict, seed:int=43, dry_run:bool=False, raw_root: Path = RAW_ROOT, curated_root: Path = CURATED_ROOT, skip_clone: bool = False):
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "petclinic")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "petclinic_repo"
    if not skip_clone:
        clone_if_needed(PETCLINIC_REPO, repo_dir, timeout_sec=60)

    sources = []
    if repo_dir.exists() and any(repo_dir.rglob("*.yml")):
        sources = list(repo_dir.rglob("*.yml"))
    else:
        local_pc = raw_root / "petclinic"
        if local_pc.exists() and any(local_pc.rglob("*.yml")):
            sources = list(local_pc.rglob("*.yml"))

    curated = 0
    if not sources:
        times = budget.get("remaining", 0)
        for i in range(times):
            if budget.get("remaining", 0) <= 0:
                break
            try:
                service = f"petclinic-synth-{i}"
                doc = {"paths": {}}
                base_paths = [f"/{service}/pets", f"/{service}/owners", f"/{service}/appointments"]
                for p in base_paths:
                    doc["paths"][p] = {
                        "get": {"responses": {"200": {"description": "ok"}}},
                        "post": {"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}}, "responses": {"201": {"description": "created"}}}
                    }
                v2 = deepcopy(doc)
                new_ep = f"/{service}/reports"
                v2["paths"][new_ep] = {"get": {"responses": {"200": {"description": "ok"}}}}
                pmod = base_paths[0]
                if "get" in v2["paths"][pmod]:
                    v2["paths"][pmod]["get"].setdefault("parameters", []).append({"name": "status", "in": "query", "schema": {"type": "string", "enum": ["active","inactive"]}, "required": False})
                if rng.random() < 0.3:
                    try:
                        del v2["paths"][base_paths[1]]["post"]["responses"]["201"]
                    except Exception:
                        pass

                old_can = paths["canonical"] / f"{service}-v1.canonical.json"
                new_can = paths["canonical"] / f"{service}-v2.canonical.json"
                if not dry_run:
                    _write_json(old_can, doc)
                    _write_json(new_can, v2)

                old_sha = sha_for_text(json.dumps(doc, sort_keys=True, default=_default_json_serializer))
                new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                pair_id = f"pair-{sha_for_text(service + old_sha + new_sha)}"
                generated_at = _now_iso()
                pair_entry = {
                    "pair_id": pair_id,
                    "service_name": service,
                    "dataset":"petclinic",
                    "old_canonical": str(old_can.name),
                    "new_canonical": str(new_can.name),
                    "old": str(old_can),
                    "new": str(new_can),
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "generated_at": generated_at
                }
                pair_index[pair_id] = pair_entry

                aces = []
                ai = 0
                aces.append({"pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}", "type": "ENDPOINT_ADDED", "path": new_ep, "method": "get", "detail": new_ep, "confidence": 0.8, "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [], "shared_schemas": []}); ai += 1
                aces.append({"pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}", "type": "PARAM_ADDED", "path": pmod, "method": "get", "detail": "status", "confidence": 0.7, "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [], "shared_schemas": []}); ai += 1
                if rng.random() < 0.3:
                    aces.append({"pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}", "type": "RESPONSE_CODE_REMOVED", "path": base_paths[1], "method": "post", "detail": ["201"], "confidence": 0.75, "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [], "shared_schemas": []}); ai += 1
                if rng.random() < 0.2:
                    aces.append({"pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}", "type": "PARAM_TYPE_CHANGED", "path": base_paths[0], "method": "get", "detail": "count", "confidence": 0.65, "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [], "shared_schemas": []}); ai += 1
                if rng.random() < 0.2:
                    aces.append({"pair_id": pair_id, "ace_index": ai, "ace_id": f"{pair_id}::ace::{ai}", "type": "NO_IMPACT", "path": None, "method": None, "detail": None, "confidence": 0.0, "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [], "shared_schemas": []}); ai += 1

                nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
                if not dry_run:
                    _ensure_dir(nd.parent)
                    with nd.open("w", encoding="utf-8") as fh:
                        for a in aces:
                            fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")

                if not dry_run:
                    _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)

                version_meta[pair_id] = {
                    "pair_id": pair_id,
                    "dataset": "petclinic",
                    "service_name": pair_entry["service_name"],
                    "semver_old":"1.0.0",
                    "semver_new":"1.0.1",
                    "semver_delta":"patch",
                    "breaking_vs_semver": False,
                    "generated_at": generated_at,
                    "producers_sample": base_paths + [new_ep]
                }

                curated += 1
                budget["remaining"] = max(0, budget["remaining"] - 1)
                if dry_run and curated >= 5:
                    break
            except Exception as e:
                logging.warning(f"[petclinic synth] skip {i}: {e}")
    else:
        for yml in sources:
            if budget.get("remaining",0) <= 0:
                break
            try:
                service = yml.stem.lower()
                old_can = paths["canonical"] / f"petclinic-{service}-v1.canonical.json"
                new_can = paths["canonical"] / f"petclinic-{service}-v2.canonical.json"
                doc = {"paths": {"/"+service: {"get": {"description":"orig"}}}}
                v2 = deepcopy(doc)
                v2["paths"][f"/{service}/new"] = {"get": {"description":"added endpoint"}}
                if not dry_run:
                    _write_json(old_can, doc)
                    _write_json(new_can, v2)
                old_sha = sha_for_text(json.dumps(doc, sort_keys=True, default=_default_json_serializer))
                new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                pair_id = f"pair-{sha_for_text(service + old_sha + new_sha)}"
                generated_at = _now_iso()
                pair_entry = {
                    "pair_id": pair_id,
                    "service_name": f"petclinic-{service}",
                    "dataset":"petclinic",
                    "old_canonical": str(old_can.name),
                    "new_canonical": str(new_can.name),
                    "old": str(old_can),
                    "new": str(new_can),
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "generated_at": generated_at
                }
                pair_index[pair_id] = pair_entry
                aces = []
                aces.append({"pair_id": pair_id, "ace_index": 0, "ace_id": f"{pair_id}::ace::0", "type": "ENDPOINT_ADDED", "path": f"/{service}/new", "method":"get", "detail": f"/{service}/new", "confidence":0.8, "provenance": {"old_sha": old_sha, "new_sha": new_sha}, "side_effect": False, "calls_services": [], "shared_schemas": []})
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
                    "semver_old":"1.0.0",
                    "semver_new":"1.0.1",
                    "semver_delta":"patch",
                    "breaking_vs_semver": False,
                    "generated_at": generated_at,
                    "producers_sample": [f"/{service}", f"/{service}/new"]
                }
                curated += 1
                budget["remaining"] = max(0, budget["remaining"] - 1)
                if dry_run and curated >= 5:
                    break
            except Exception as e:
                logging.warning(f"[petclinic] skip {yml}: {e}")

    logging.info(f"[petclinic] Curated {curated} pairs")
    return

def curate_openrewrite(pair_index: Dict, version_meta: Dict, budget: Dict, seed:int=44, dry_run:bool=False, raw_root: Path = RAW_ROOT, curated_root: Path = CURATED_ROOT, skip_clone: bool = False):
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "openrewrite")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "openrewrite_repo"
    if not skip_clone:
        clone_if_needed(OPENREWRITE_REPO, repo_dir, timeout_sec=60)

    curated = 0
    i = 0
    while budget.get("remaining", 0) > 0:
        try:
            name = f"openrewrite-recipe-{i}"
            v1 = {"recipeList":[{"name":f"org.demo.Recipe{i}"}]}
            if rng.random() < 0.6:
                v2 = {"recipeList":[{"name":f"org.demo.Recipe{i}"},{"name":f"org.demo.Added{i}"}]}
                ace_type = "RECIPE_ADDED"
                ace_detail = f"Added{i}"
            else:
                v2 = {"recipeList":[{"name":f"org.demo.Recipe{i}","config": {"opt": rng.choice(["a","b","c"])}}]}
                ace_type = "RECIPE_MODIFIED"
                ace_detail = f"Modified{i}"

            old_can = paths["canonical"] / f"{name}-v1.canonical.json"
            new_can = paths["canonical"] / f"{name}-v2.canonical.json"
            if not dry_run:
                _write_json(old_can, v1); _write_json(new_can, v2)
            old_sha = sha_for_text(json.dumps(v1, sort_keys=True, default=_default_json_serializer))
            new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
            pair_id = f"pair-{sha_for_text(str(i)+old_sha+new_sha)}"
            pair_entry = {
                "pair_id": pair_id,
                "service_name": name,
                "dataset":"openrewrite",
                "old_canonical": str(old_can.name),
                "new_canonical": str(new_can.name),
                "old": str(old_can),
                "new": str(new_can),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "generated_at": _now_iso()
            }
            pair_index[pair_id] = pair_entry

            nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
            ace = {
                "pair_id": pair_id,
                "ace_index": 0,
                "ace_id": f"{pair_id}::ace::0",
                "type": ace_type,
                "detail": ace_detail,
                "confidence": 0.9,
                "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                "side_effect": False,
                "calls_services": [],
                "shared_schemas": []
            }
            if not dry_run:
                with nd.open("w", encoding="utf-8") as fh:
                    fh.write(json.dumps(ace, ensure_ascii=False, default=_default_json_serializer) + "\n")
                _write_json(paths["metadata"] / f"{pair_id}.meta.json", pair_entry)
            version_meta[pair_id] = {
                "pair_id": pair_id,
                "dataset": "openrewrite",
                "service_name": pair_entry["service_name"],
                "semver_old":"0.1.0",
                "semver_new":"0.2.0",
                "semver_delta":"minor",
                "breaking_vs_semver": False,
                "generated_at": _now_iso(),
                "producers_sample": [f"recipe://{pair_entry['service_name']}"]
            }
            curated += 1
            budget["remaining"] = max(0, budget["remaining"] - 1)
            i += 1
            if dry_run and curated >= 10:
                break
        except Exception as e:
            logging.warning(f"[openrewrite] skip synth {i}: {e}")
            i += 1
    logging.info(f"[openrewrite] Curated {curated} pairs")
    return

# ---------- Graph builder
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

# ---------- Ensure total ACEs utility (unchanged)
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

# ---------- Index and utility writers
def write_global_indexes(pair_index: Dict[str, Any], version_meta: Dict[str, Any], curated_root: Path = CURATED_ROOT):
    # write index.json unsorted content (fast) but preserve stable CSV using sorted keys
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

# ---------- Budget ratio parsing + helpers (unchanged)
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

        old_can = paths["canonical"] / f"openapi--{svc}--v1.canonical.json"
        new_can = paths["canonical"] / f"openapi--{svc}--v2.canonical.json"
        _write_json(old_can, old_doc)
        _write_json(new_can, new_doc)

        old_sha = sha_for_text(json.dumps(old_doc, sort_keys=True))
        new_sha = sha_for_text(json.dumps(new_doc, sort_keys=True))
        pair_id = f"pair-{sha_for_text(svc + old_sha + new_sha)}"
        generated_at = _now_iso()

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
            "old_canonical": str(old_can.name),
            "new_canonical": str(new_can.name),
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

    # Run curators
    curate_openapi(pair_index, producers, version_meta, budget_openapi, seed=seed, dry_run=dry_run, raw_root=RAW_ROOT, curated_root=CURATED_ROOT, skip_clone=skip_clone, openapi_timeout=openapi_timeout, prefer_git_pairs=True)
    curate_petclinic(pair_index, version_meta, budget_petclinic, seed=seed + 1, dry_run=dry_run, raw_root=RAW_ROOT, curated_root=CURATED_ROOT, skip_clone=skip_clone)
    curate_openrewrite(pair_index, version_meta, budget_openrewrite, seed=seed + 2, dry_run=dry_run, raw_root=RAW_ROOT, curated_root=CURATED_ROOT, skip_clone=skip_clone)

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

    merged_index = {**existing_idx, **pair_index}
    if not dry_run:
        _write_json(Path(out_dir) / "index.json", {k: merged_index[k] for k in sorted(merged_index.keys())})
    logging.info("Index written: %d pairs (to %s)", len(merged_index), Path(out_dir) / "index.json")

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
    parser.add_argument("--target-aces", type=int, default=18000, help="target total ACEs to ensure (script will append synthetic ACEs if needed)")
    parser.add_argument("--ace-batch", type=int, default=200, help="how many ACEs appended per file iteration when meeting target")
    parser.add_argument("--no-augment", action="store_true", help="do not append synthetic ACEs to reach target-aces; use only real ACEs")
    parser.add_argument("--openapi-timeout", type=int, default=8, help="timeout seconds for canonicalizing an OpenAPI spec")
    args = parser.parse_args()
    random.seed(args.seed)
    ratios = _parse_ratio_arg(args.ratios)

    globals()["_SYNTH_OPENAPI"] = getattr(args, "synthesize_openapi", False)
    globals()["_TARGET_ACES"] = getattr(args, "target_aces", None)
    globals()["_ACE_BATCH"] = getattr(args, "ace_batch", 200)

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
