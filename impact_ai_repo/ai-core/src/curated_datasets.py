
# curated_datasets.py
from __future__ import annotations
import argparse, csv, datetime, decimal, hashlib, json, logging, random, re, shutil, subprocess, uuid, os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing as mp
import yaml
import signal
import sys

# ---------- Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- Default paths and repos (to be tweaked to work in local environment)
RAW_ROOT = Path("datasets/raw")
CURATED_ROOT = Path("datasets/curated")
OPENAPI_REPO = "https://github.com/APIs-guru/openapi-directory.git"
PETCLINIC_REPO = "https://github.com/spring-petclinic/spring-petclinic-microservices.git"
OPENREWRITE_REPO = "https://github.com/openrewrite/rewrite.git"
_MAX_CANONICAL_DEPTH = 2000# ---------- file size limits (bytes)
# default: 2 MB
MAX_ALLOWED_FILE_SIZE = int(os.environ.get("MAX_ALLOWED_FILE_SIZE", 2_000_000))



# ---------- tiny real-name pools
REAL_SERVICE_POOL = [
    "payment-service", "order-service", "inventory-service", "transaction-service",
    "user-service", "billing-service", "auth-service", "notifications-service",
    "catalog-service", "shipping-service", "analytics-service", "search-service"
]
RESOURCE_BASES = ["payments","orders","inventory","transactions","users","billing","shipments","products","carts","reviews","profiles"]

# ---------- caches & pool
_resolved_external_cache: Dict[str, Any] = {}
_canon_cache: Dict[str, Tuple[Optional[Dict[str, Any]], List[str]]] = {}
_CANON_POOL: Optional[ProcessPoolExecutor] = None

# ---------- helpers
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

def _default_json_serializer(obj: Any):
    if obj is None: return None
    if isinstance(obj, datetime.datetime):
        dt = obj if obj.tzinfo else obj.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).replace(microsecond=0).isoformat()
    if isinstance(obj, (datetime.date, datetime.time)): return obj.isoformat()
    if isinstance(obj, decimal.Decimal): return float(obj)
    if isinstance(obj, (set, tuple)): return list(obj)
    if isinstance(obj, bytes):
        try: return obj.decode("utf-8")
        except Exception: return list(obj)
    if isinstance(obj, uuid.UUID): return str(obj)
    return str(obj)

def sha_for_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]

def stable_seed_from_str(s: str, salt: int = 0) -> int:
    try:
        h = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)
    except Exception:
        h = abs(hash(s)) & 0xffffffff
    return (h ^ (salt & 0xffffffff)) & 0x7fffffff

def make_canonical_filename(dataset: str, service_name: str, pair_token: str, version_tag: str) -> str:
    safe_ds = re.sub(r"[^0-9a-zA-Z]+", "-", dataset).strip("-").lower()
    safe_svc = re.sub(r"[^0-9a-zA-Z\-_]+", "-", service_name).strip("-").lower()
    v = version_tag.lower()
    return f"{safe_ds}--{safe_svc}--{pair_token}--{v}.canonical.json"

def generate_pair_id(service_name: str, old_sha: str, new_sha: str) -> str:
    key = f"{service_name}::{old_sha}::{new_sha}"
    return f"pair-{sha_for_text(key)}"

def make_pair_token(service_name: str, old_sha: str, new_sha: str) -> str:
    key = f"{service_name}::{old_sha}::{new_sha}"
    return sha_for_text(key)[:8]

def _write_json(p: Path, obj):
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default_json_serializer), encoding="utf-8")

def _write_json_sorted(p: Path, obj):
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=_default_json_serializer), encoding="utf-8")

# ---------- producers helper
_MAX_PRODUCERS_PER_SERVICE = 200
def producers_add(service: str, eps: List[str], producers_map: Dict[str, List[str]]):
    cur = producers_map.setdefault(service, [])
    cur.extend(eps)
    seen = set(); new=[]
    for e in cur:
        if e not in seen:
            seen.add(e); new.append(e)
    producers_map[service] = new[-_MAX_PRODUCERS_PER_SERVICE:]

# ---------- file readers/canonicalizer (robust)
@lru_cache(maxsize=4096)
def _read_file_cached_text(path_str: str) -> Optional[str]:
    try:
        txt = Path(path_str).read_text(encoding="utf-8")
        if not txt.strip(): return None
        return txt
    except Exception:
        return None

def _read_file(p: Path) -> Optional[Any]:
    txt = _read_file_cached_text(str(p))
    if not txt: return None
    try:
        return json.loads(txt)
    except Exception:
        pass
    try:
        docs = list(yaml.safe_load_all(txt))
    except Exception:
        try: return yaml.safe_load(txt)
        except Exception: return None
    docs = [d for d in docs if d not in (None, {}, [], "")]
    if not docs: return None
    for d in docs:
        if isinstance(d, (dict, list)): return d
    return docs[0]

_BAD_DIR = Path("datasets/raw/openrewrite_repo_malformed")
def _save_bad_file(path: Path, reason: str):
    try:
        _ensure_dir(_BAD_DIR)
        safe_name = sha_for_text(str(path)) + "-" + path.name
        dst = _BAD_DIR / safe_name
        try: shutil.copy2(path, dst)
        except Exception:
            try: dst.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            except Exception: pass
        try: (dst.with_suffix(".reason.txt")).write_text(reason or "unknown", encoding="utf-8")
        except Exception: pass
    except Exception:
        logging.debug(f"_save_bad_file failed for {path}")

def is_noise_path(p: Path) -> bool:
    s = str(p.as_posix()).lower()
    skip_keywords = [
        "/.github/", "/.github/workflows/", "/meta-inf/", "/test/", "/tests/", "/examples/",
        "/.circleci/", "/.travis/", "/ci/", "/jenkins/", "readme", "changelog", "change-log",
        "license", "pom.xml", "gradle", ".gitignore"
    ]
    for kw in skip_keywords:
        if kw in s: return True
    try:
        st = p.stat()
        if st.st_size < 120: 
            return True
        # use global threshold
        max_sz = globals().get("MAX_ALLOWED_FILE_SIZE", MAX_ALLOWED_FILE_SIZE)
        if st.st_size > max_sz:
            return True
    except Exception: pass
    if p.suffix.lower() not in (".yml", ".yaml", ".json"): return True
    return False


# ---------- canonicalization
def resolve_local_pointer(root: Any, pointer: str):
    if pointer == "#" or pointer == "#/": return deepcopy(root)
    parts = pointer.lstrip("#/").split("/")
    cur = root
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur: raise KeyError(f"Pointer part '{p}' missing")
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(f"Cannot navigate pointer part '{p}'")
    return deepcopy(cur)

def canonicalize_spec(spec: Any, base_dir: Path):
    warnings: List[str] = []
    def _find_dict_candidate(obj, depth=0):
        if depth > 8: return None
        if isinstance(obj, dict):
            if obj.get("paths") or obj.get("openapi") or obj.get("swagger"):
                return obj
            return obj
        if isinstance(obj, list):
            for it in obj:
                cand = _find_dict_candidate(it, depth + 1)
                if cand is not None: return cand
        return None
    if not isinstance(spec, dict):
        if isinstance(spec, list):
            cand = _find_dict_candidate(spec)
            if cand: spec = cand
            else: return {"paths": {}}, ["spec parsed as list; fallback to empty paths"]
        else:
            return {"paths": {}}, ["spec not dict; fallback to empty paths"]

    depth = 0
    def _rec(node, cur_base: Path, stack: Tuple[str,...]=()):
        nonlocal depth
        depth += 1
        if depth > _MAX_CANONICAL_DEPTH:
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
                                depth -= 1; return {"__ref_cycle__": ref}
                            resolved = resolve_local_pointer(spec, ref)
                            result = _rec(resolved, cur_base, stack + (key,))
                            depth -= 1; return result
                        except Exception as e:
                            warnings.append(f"local ref {ref} failed: {e}")
                            depth -= 1; return {"__unresolved_ref__": ref}
                    else:
                        parts = ref.split("#",1)
                        file_part = parts[0]; ptr = "#" + parts[1] if len(parts)>1 else "#"
                        target_file = (cur_base / file_part).resolve()
                        key = (str(target_file), ptr)
                        if key in stack:
                            depth -= 1; return {"__ref_cycle__": ref}
                        if not target_file.exists():
                            warnings.append(f"external ref file missing: {target_file}")
                            depth -= 1; return {"__unresolved_ref__": ref}
                        try:
                            cache_key = f"{str(target_file.resolve())}::{ptr}"
                            if cache_key in _resolved_external_cache:
                                loaded = _resolved_external_cache[cache_key]
                            else:
                                loaded = _read_file(target_file)
                                if loaded is not None: _resolved_external_cache[cache_key] = loaded
                            if loaded is None:
                                warnings.append(f"external ref empty file: {target_file}")
                                depth -= 1; return {"__unresolved_ref__": ref}
                            resolved = resolve_local_pointer(loaded, ptr) if ptr != "#" else loaded
                            result = _rec(resolved, target_file.parent, stack + (key,))
                            depth -= 1; return result
                        except Exception as e:
                            warnings.append(f"external ref load failed {ref}: {e}")
                            depth -= 1; return {"__unresolved_ref__": ref}
                else:
                    out = {}
                    for k, v in node.items():
                        try:
                            out[k] = _rec(v, cur_base, stack)
                        except Exception as e:
                            warnings.append(f"child canonicalize error for key {k}: {e}")
                            out[k] = {"__canonicalize_error__": str(e)}
                    if any(k in out for k in ("oneOf","anyOf","allOf")):
                        out["_polymorphic_marker"] = {k: True for k in ("oneOf","anyOf","allOf") if k in out}
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
        try: fut.cancel()
        except Exception: pass
        _canon_cache[key] = (None, [f"canonicalize timed out after {timeout_sec}s"])
        return None, [f"canonicalize timed out after {timeout_sec}s"]
    except Exception as e:
        _canon_cache[key] = (None, [f"canonicalize worker exception: {e}"])
        return None, [f"canonicalize worker exception: {e}"]
    canon = res.get("canon")
    warns = res.get("warns", [])
    _canon_cache[key] = (canon, warns)
    return canon, warns

def shutdown_canon_pool():
    global _CANON_POOL
    try:
        if _CANON_POOL is not None:
            _CANON_POOL.shutdown(wait=True)
            _CANON_POOL = None
            logging.info("[canon_pool] shutdown complete")
    except Exception as e:
        logging.debug(f"[canon_pool] shutdown error: {e}")

# ---------- ops_map helper
def ops_map_from_canonical(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not isinstance(spec, dict): return out
    for p, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict): continue
        out[p] = {m.lower(): methods[m] for m in methods.keys() if isinstance(m, str)}
    return out

# ---------- realistic path / service helpers
def _deterministic_choice_from_str(seed_str: str, pool: List[str]) -> str:
    if not pool: return seed_str[:24]
    try:
        h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:16], 16)
        return pool[h % len(pool)]
    except Exception:
        return pool[0]

def pick_real_service(seed_str: str) -> str:
    return _deterministic_choice_from_str(seed_str, REAL_SERVICE_POOL)

def _normalize_token(token: str) -> str:
    t = re.sub(r"auto|autogen|autox|synth|synthetic|svc|ui", "", token, flags=re.I)
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t or token.lower()

def make_path_realistic(path: str, seed_str: str = "") -> str:
    if not path or not isinstance(path, str): return path
    p = path.strip(); p = re.sub(r"//+", "/", p)
    def norm_param(m):
        inner = m.group(1)
        inner2 = re.sub(r"[^a-zA-Z0-9_]+", "_", inner).strip("_")
        if not inner2: inner2 = "id"
        base = _deterministic_choice_from_str(seed_str + inner2, ["id", "paymentId", "orderId", "sku", "userId"])
        return "{" + base + "}"
    p = re.sub(r"\{([^}]+)\}", norm_param, p)
    if re.search(r"/auto|autogen|autox|synth|synthetic", p, flags=re.I) or re.search(r"/\d{2,}", p):
        base = _deterministic_choice_from_str(seed_str + p, RESOURCE_BASES)
        return f"/{base}/{{id}}"
    if re.search(r"/\d+$", p):
        return re.sub(r"/\d+$", "/{id}", p)
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
    # non-mutating
    if not isinstance(canon, dict): return canon, []
    canon_copy = deepcopy(canon)
    paths = canon_copy.get("paths") or {}
    new_paths = {}
    op_samples = []
    for p in sorted(list(paths.keys())):
        safe_p = str(p)
        new_p = make_path_realistic(safe_p, seed_str + safe_p)
        methods = paths.get(p) or {}
        new_methods = {}
        for method_name, op in (methods.items() if isinstance(methods, dict) else []):
            if not isinstance(op, dict):
                new_methods[method_name] = op; continue
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
    canon_copy["paths"] = new_paths
    return canon_copy, sorted(list(set(op_samples)))[:10]

# ---------- noisy text appliers
def apply_noisy_light_text(s: str, rnd: random.Random) -> Tuple[str, List[str]]:
    ops = []
    t = s
    if rnd.random() < 0.3:
        t = re.sub(r"(?m)^\s*#.*$\n?", "", t)
        ops.append("strip_comments")
    if rnd.random() < 0.4:
        t = re.sub(r'("description"\s*:\s*")([^"]+)(")', lambda m: m.group(1) + m.group(2)[:max(3,len(m.group(2))//2)] + m.group(3), t)
        ops.append("truncate_descs")
    if rnd.random() < 0.25:
        t = t.replace(':"', ': "').replace('",', '",\n')
        ops.append("whitespace_noise")
    return t, ops

def apply_noisy_heavy_text(s: str, rnd: random.Random) -> Tuple[str, List[str]]:
    """
    Aggressive corruption for 'noisy_heavy' mode.
    Structural changes, truncation, bracket/quote removal, random byte injections,
    key scrambling, and block deletions to simulate real-world breakage and garbage.
    Returns (corrupted_text, list_of_ops_applied).
    """
    ops = []
    t = s

    # 1) strip large blocks (simulate removed sections)
    if rnd.random() < 0.7:
        t = re.sub(r'(?s)\n\s*-\s*.*?(\n\s*-|\Z)', '\n', t)
        ops.append("drop_list_blocks")

    # 2) truncate long descriptions and long strings
    if rnd.random() < 0.8:
        def trunc(m):
            txt = m.group(2)
            keep = max(8, len(txt)//4)
            return m.group(1) + txt[:keep] + m.group(3)
        t = re.sub(r'("description"\s*:\s*")([^"]+)(")', trunc, t)
        ops.append("truncate_descriptions")

    # 3) remove/replace braces, brackets and quotes randomly (break JSON/YAML)
    if rnd.random() < 0.5:
        if rnd.random() < 0.5:
            t = t.replace('{', '').replace('}', '')
            ops.append("strip_braces")
        if rnd.random() < 0.4:
            t = t.replace('[', '').replace(']', '')
            ops.append("strip_brackets")
        if rnd.random() < 0.35:
            t = t.replace('"', '')
            ops.append("remove_quotes")

    # 4) scramble some key names to simulate schema drift
    if rnd.random() < 0.5:
        def scramble_key(m):
            k = m.group(1)
            if len(k) <= 3: return m.group(0)
            cut = max(1, len(k)//3)
            k2 = k[:cut] + ''.join(rnd.choice('xyz') for _ in range(cut)) + k[cut*2:]
            return f'"{k2}"{m.group(2)}'
        t, nsub = re.subn(r'"([a-zA-Z0-9_\-]{4,})"(\s*:\s*)', scramble_key, t)
        if nsub:
            ops.append(f"scramble_keys:{nsub}")

    # 5) colon-to-equals and equals-to-colon confusion
    if rnd.random() < 0.6:
        t = re.sub(r'(\s*:\s*)', ' = ', t)
        ops.append("colon_to_equals")

    # 6) random deletion of JSON/YAML lines (simulate truncation/corruption)
    if rnd.random() < 0.6:
        lines = t.splitlines()
        keep = []
        for L in lines:
            if rnd.random() < 0.85:
                keep.append(L)
        if len(keep) < len(lines):
            t = "\n".join(keep)
            ops.append("delete_some_lines")

    # 7) inject binary-like or random junk in a few places
    if rnd.random() < 0.5:
        junk = "".join(chr(rnd.randint(0x20, 0x7E)) for _ in range(rnd.randint(8,64)))
        insert_at = rnd.randint(0, max(0, len(t)-1))
        t = t[:insert_at] + ("\n/*BIN_JUNK:%s*/\n" % junk) + t[insert_at:]
        ops.append("inject_junk_block")

    # 8) occasionally make JSON keys repeat or be duplicated (bad merges)
    if rnd.random() < 0.3:
        t = re.sub(r'("([a-zA-Z0-9_\-]{2,})"\s*=\s*)', lambda m: m.group(0) + m.group(1), t)
        ops.append("duplicate_keys")

    # 9) final small edits: remove some commas, add stray equal signs
    if rnd.random() < 0.4:
        t = re.sub(r',\s*\n', '\n', t)
        ops.append("drop_commas")

    # 10) last-resort short truncation of file (simulate truncated upload)
    if rnd.random() < 0.25:
        cut = rnd.randint(max(100, len(t)//6), max(200, len(t)//3))
        t = t[:cut]
        ops.append("truncate_file_tail")

    return t, ops


# ---------- noisy raw copy generator (updated)
def generate_noisy_raw_copy(raw_root: Path, target_root: Path, mode: str = "light",
                            seed: int = 2025, counts: Optional[Dict[str,int]] = None):
    """
    Create a noisy copy of raw datasets under `target_root`.
    - mode: "light" / "noisy_light" -> light corruption (lower prob)
            "heavy" / "noisy_heavy" -> heavy corruption (higher prob, destructive)
    - respects global MAX_ALLOWED_FILE_SIZE (bytes) if set, otherwise uses 2_000_000
    - returns Path(target_root)
    """
    rnd = random.Random(seed)
    counts = counts or {"openapi": 500, "openrewrite": 300, "petclinic": 30}
    datasets = ["openapi", "openrewrite", "petclinic"]
    for synth in ("synth_openapi", "synth_openrewrite", "synth_petclinic"):
        if (raw_root / synth).exists():
            datasets.append(synth)
            counts.setdefault(synth, 50)

    manifest = {"seed": seed, "mode": mode, "entries": []}
    _ensure_dir(target_root)

    # centralized max file size (bytes) - prefer user-provided global, else fallback
    max_sz = int(globals().get("MAX_ALLOWED_FILE_SIZE", 2_000_000))

    for ds in datasets:
        src_ds = Path(raw_root) / ds
        tgt_ds = Path(target_root) / ds
        if not src_ds.exists():
            manifest["entries"].append({"dataset": ds, "status": "missing_source"})
            continue
        _ensure_dir(tgt_ds)

        # gather candidate files with sensible text suffixes
        all_candidates = [p for p in src_ds.rglob("*")
                          if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".json", ".log", ".txt", ".md")]

        # filter by centralized size threshold; record skipped-large files
        cand = []
        skipped_by_size = []
        for p in sorted(all_candidates):
            try:
                sz = p.stat().st_size
            except Exception:
                # if cannot stat, conservatively include
                cand.append(p)
                continue
            if sz <= max_sz:
                cand.append(p)
            else:
                skipped_by_size.append((p, sz))

        if skipped_by_size:
            logging.info(f"[generate_noisy_raw_copy] Skipped {len(skipped_by_size)} large files in {ds} (> {max_sz} bytes)")
            for p, sz in skipped_by_size:
                try:
                    rel = p.relative_to(src_ds)
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "skipped_large", "size": sz})
                except Exception:
                    manifest["entries"].append({"dataset": ds, "file": p.name, "status": "skipped_large", "size": sz})

        # fallback: if no suitable text candidates, copy other files (still skip large ones)
        if not cand:
            for f in sorted(src_ds.rglob("*")):
                if not f.is_file():
                    continue
                try:
                    sz = f.stat().st_size
                except Exception:
                    sz = 0
                if sz > max_sz:
                    try:
                        rel = f.relative_to(src_ds)
                        manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "skipped_large", "size": sz})
                    except Exception:
                        manifest["entries"].append({"dataset": ds, "file": f.name, "status": "skipped_large", "size": sz})
                    continue

                rel = f.relative_to(src_ds); dst = tgt_ds / rel; dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(f, dst)
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copied_all_fallback"})
                except Exception:
                    try:
                        dst.write_text(f.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                        manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copied_all_text"})
                    except Exception:
                        manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copy_failed"})
            continue

        # sample a manageable subset (or use all if smaller)
        target_count = min(len(cand), counts.get(ds, len(cand)))
        sampled = cand if len(cand) <= target_count else sorted(rnd.sample(cand, target_count))

        corrupted_count = 0
        for src in sampled:
            rel = src.relative_to(src_ds); dst = tgt_ds / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw_text = src.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # binary or unreadable: just copy if small enough
                try:
                    shutil.copy2(src, dst)
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "binary_copied"})
                    continue
                except Exception:
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "read_error"})
                    continue

            # corruption probability: light vs heavy
            if mode in ("light", "noisy_light"):
                corr_prob = 0.25
            elif mode in ("heavy", "noisy_heavy"):
                corr_prob = 0.65
            else:
                corr_prob = 0.25

            if rnd.random() < corr_prob:
                if mode in ("light", "noisy_light"):
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
                # mostly intact copy
                try:
                    shutil.copy2(src, dst)
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "corrupted": False})
                except Exception:
                    try:
                        dst.write_text(raw_text, encoding="utf-8")
                        manifest["entries"].append({"dataset": ds, "file": str(rel), "corrupted": False})
                    except Exception:
                        manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copy_failed"})

        # copy remaining (non-sampled) candidate files — they are within size threshold
        remaining = [r for r in cand if r not in sampled]
        for src in remaining:
            rel = src.relative_to(src_ds); dst = tgt_ds / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copied_remaining"})
            except Exception:
                try:
                    dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copied_remaining_text"})
                except Exception:
                    manifest["entries"].append({"dataset": ds, "file": str(rel), "status": "copy_failed"})

        manifest["entries"].append({
            "dataset": ds,
            "sampled": len(sampled),
            "corrupted_count": corrupted_count,
            "skipped_large_count": len(skipped_by_size)
        })

    manifest_path = Path(target_root) / "noise_manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        logging.warning("Failed to write noise manifest")
    return Path(target_root)



def get_processing_root(raw_root: Path, curated_root: Path, mode: str, seed: int = 2025, counts: Optional[Dict[str,int]] = None) -> Path:
    mode = (mode or "clean")
    if mode == "clean": return raw_root
    tgt_parent = Path(curated_root).parent
    out_name = f"raw_{mode}_{seed}"
    target_root = tgt_parent / out_name
    if target_root.exists(): return target_root
    return generate_noisy_raw_copy(raw_root, target_root, mode="light" if mode=="noisy_light" else "heavy", seed=seed, counts=counts)


# ---------- small pair entry normalizer
def _normalize_pair_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    pid = entry.get("pair_id") or entry.get("pair") or None
    svc = entry.get("service_name") or entry.get("service") or "unknown"
    old_sha = entry.get("old_sha") or entry.get("oldSha") or ""
    new_sha = entry.get("new_sha") or entry.get("newSha") or ""
    if not pid or not isinstance(pid, str) or pid.strip().lower() in ("n", "n/a", "na", "none", ""):
        pid = generate_pair_id(svc, old_sha or "old", new_sha or "new")
        entry["pair_id"] = pid
    for k in ("old_canonical", "new_canonical", "old", "new"):
        if k in entry and entry[k] is None: entry[k] = ""
    return entry

# ---------- synth OpenAPI pairs (used as fallback)
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
        if not old_can.exists(): _write_json(old_can, old_doc)
        if not new_can.exists(): _write_json(new_can, new_doc)

        nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
        _ensure_dir(nd.parent)
        with nd.open("w", encoding="utf-8", newline="\n") as fh:
            for i in range(rng.randint(4,12)):
                ace_type = rng.choice(["ENDPOINT_ADDED","PARAM_ADDED","PARAM_REMOVED","PARAM_REQUIRED_ADDED","PARAM_TYPE_CHANGED","ENUM_NARROWED","RESPONSE_CODE_REMOVED"])
                pth = rng.choice(list(new_doc["paths"].keys()))
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
            "semver_delta": random.choice(["patch","minor","major"]),
            "breaking_vs_semver": False,
            "generated_at": generated_at,
        }
        producers[svc] = list(old_doc["paths"].keys())
        created += 1
        remaining -= 1
    logging.info(f"[synth-openapi] Created {created} OpenAPI synthetic pairs.")
    return created

# ---------- ACE counting / appending helpers
def _count_all_aces(curated_root: Path) -> int:
    total = 0
    for p in Path(curated_root).rglob("*.aces.ndjson"):
        try:
            with p.open("r", encoding="utf-8") as fh:
                for _ in fh: total += 1
        except Exception: continue
    return total

def _append_synthetic_aces_to_file(p: Path, pair_id: str, provenance: Dict[str, Any], paths_pool: List[str], to_add: int, seed: int):
    bak = p.with_name(p.name + f".bak.{_now_iso().replace(':','')}")
    try:
        if not bak.exists(): shutil.copy2(p, bak)
    except Exception: pass
    max_idx = -1
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line); idx = obj.get("ace_index")
                    if isinstance(idx, int) and idx > max_idx: max_idx = idx
                except Exception: continue
    except Exception: max_idx = -1
    rng = random.Random(seed ^ (hash(pair_id) & 0xffffffff))
    appended = 0
    with p.open("a", encoding="utf-8", newline="\n") as fh:
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
        logging.warning("[ensure_total_aces] curated_root missing: %s", curated_root); return 0
    total = _count_all_aces(cur)
    logging.info("[ensure_total_aces] current total ACEs=%d target=%d", total, target_aces)
    if total >= target_aces: return total
    nd_files = []
    for ds in ("openapi","openrewrite","petclinic"):
        p = cur / ds / "ndjson"
        if p.exists(): nd_files.extend(sorted([x for x in p.glob("*.aces.ndjson")]))
    if not nd_files: logging.warning("[ensure_total_aces] no ndjson ACE files found"); return total
    index_map = {}
    idx_path = cur / "index.json"
    if idx_path.exists():
        try:
            im = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(im, dict): index_map = im
            elif isinstance(im, list): index_map = {e.get("pair_id"): e for e in im if isinstance(e, dict)}
        except Exception: index_map = {}
    file_idx = 0
    while total < target_aces:
        p = nd_files[file_idx % len(nd_files)]
        pair_id = p.stem.replace(".aces","")
        meta = index_map.get(pair_id) or {}
        if not meta:
            md = cur / p.parent.parent.name / "metadata" / f"{pair_id}.meta.json"
            if md.exists():
                try: meta = json.loads(md.read_text(encoding="utf-8"))
                except Exception: meta = {}
        provenance = {}
        if isinstance(meta, dict): provenance = {"old_sha": meta.get("old_sha"), "new_sha": meta.get("new_sha")}
        canonical_dir = cur / p.parent.parent.name / "canonical"
        paths_pool = []
        if isinstance(meta, dict) and meta.get("new_canonical"):
            cand = canonical_dir / meta.get("new_canonical")
            if cand.exists():
                try:
                    j = _read_file(cand)
                    if isinstance(j, dict): paths_pool = list((j.get("paths") or {}).keys())
                except Exception: paths_pool = []
        if not paths_pool:
            for f in canonical_dir.glob("*.json"):
                try:
                    j = _read_file(f)
                    if isinstance(j, dict) and j.get("paths"):
                        paths_pool = list((j.get("paths") or {}).keys()); break
                except Exception: continue
        remain = target_aces - total
        to_add = min(batch_per_file, remain)
        if dry_run:
            logging.info("[ensure_total_aces] dry_run would append %d ACEs to %s", to_add, p); total += to_add
        else:
            added = _append_synthetic_aces_to_file(p, pair_id, provenance, paths_pool, to_add, seed + file_idx)
            total += added
            logging.info("[ensure_total_aces] appended %d to %s -> total %d", added, p, total)
        file_idx += 1
    logging.info("[ensure_total_aces] reached total ACEs=%d", total)
    return total

# ---------- index writers
def write_global_indexes(pair_index: Dict[str, Any], version_meta: Dict[str, Any], curated_root: Path = CURATED_ROOT):
    _write_json(Path(curated_root) / "index.json", {k: pair_index[k] for k in sorted(pair_index.keys())})
    _write_json(Path(curated_root) / "version_meta.json", version_meta)
    dataset_counts = {}
    for p in pair_index.values():
        ds = p.get("dataset", "unknown"); dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
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

# ---------- graph builder
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

# ---------- dataset path helper
def dataset_paths(root: Path, dataset: str):
    base = Path(root) / dataset
    return {"base": base, "canonical": base / "canonical", "ndjson": base / "ndjson", "metadata": base / "metadata"}

# ---------- Curate OpenRewrite 
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
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "openrewrite")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "openrewrite_repo"

    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                logging.info(f"[curate_openrewrite] cloning rewrite repo into {repo_dir}")
                subprocess.run(["git","clone", OPENREWRITE_REPO, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except Exception as e:
            logging.warning(f"[curate_openrewrite] clone failed: {e}")

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
        # be permissive for openrewrite: allow recipes/rules and small YAML fragments,
        # but still skip known noise and very large files.
        s = str(f.as_posix()).lower()
        if is_noise_path(f):
            # allow rewrite/recipe files even if they match some noise keywords
            if any(k in s for k in ("recipe", "rewrite", "openrewrite", "rule")):
                pass
            else:
                return False
        try:
            sz = f.stat().st_size
            max_sz = globals().get("MAX_ALLOWED_FILE_SIZE", MAX_ALLOWED_FILE_SIZE)
            # reject empty files and oversized files, but accept small fragments
            if sz == 0 or sz > max_sz:
                return False
        except Exception:
            # if stat fails, keep it as candidate (best effort)
            pass
        # allow common textual/spec suffixes
        if f.suffix.lower() in (".yml", ".yaml", ".json", ".md", ".txt"):
            return True
        return False


    def _emit_pair_from_docs(dataset: str, svc_name: str, old_doc: dict, new_doc: dict, provenance_extra: dict = None):
        nonlocal curated
        try:
            old_can, warns_old = canonicalize_with_timeout(old_doc, repo_dir, timeout_sec=openapi_timeout)
            new_can, warns_new = canonicalize_with_timeout(new_doc, repo_dir, timeout_sec=openapi_timeout)
            if old_can is None or new_can is None:
                return False

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

            aces = []
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
                            detected = []
                            if old_op is None and new_op is not None:
                                detected.append({"type":"ENDPOINT_ADDED","path":p,"method":m,"detail":p})
                            elif old_op is not None and new_op is None:
                                detected.append({"type":"ENDPOINT_REMOVED","path":p,"method":m,"detail":p})
                            else:
                                old_params = {pp.get("name") for pp in (old_op or {}).get("parameters") or [] if isinstance(pp, dict)}
                                new_params = {pp.get("name") for pp in (new_op or {}).get("parameters") or [] if isinstance(pp, dict)}
                                added = (new_params - old_params) if new_params else set()
                                removed = (old_params - new_params) if old_params else set()
                                if added:
                                    for a in added:
                                        detected.append({"type":"PARAM_ADDED","path":p,"method":m,"detail":a})
                                if removed:
                                    for r in removed:
                                        detected.append({"type":"PARAM_REMOVED","path":p,"method":m,"detail":r})
                        except Exception:
                            detected = []
                        for d in detected:
                            try:
                                conf = 0.9 if d.get("type") in ("ENDPOINT_ADDED","ENDPOINT_REMOVED") else 0.6
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
                "dataset": "openrewrite",
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
                "dataset": "openrewrite",
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
                git_pairs = []
                try:
                    res = subprocess.run(
                        ["git", "-C", str(repo_dir), "log", "--pretty=format:%H", "--", str(f)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
                    )
                    if res.returncode != 0:
                        commits = []
                        logging.debug(f"[openrewrite] git log non-zero for {f}: {res.stderr.decode('utf-8', errors='replace')}")
                    else:
                        commits = [c.strip() for c in res.stdout.decode("utf-8", errors="replace").splitlines() if c.strip()]
                    # limit commits (we need at least two commits to form pairs)
                    commits = commits[: max(2, max_git_pairs_per_file + 1)]
                    if len(commits) >= 2:
                        for i in range(min(max_git_pairs_per_file, len(commits)-1)):
                            old_c = commits[i+1]
                            new_c = commits[i]
                            try:
                                old_blob_proc = subprocess.run(
                                    ["git","-C", str(repo_dir), "show", f"{old_c}:{str(f.relative_to(repo_dir))}"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
                                )
                                new_blob_proc = subprocess.run(
                                    ["git","-C", str(repo_dir), "show", f"{new_c}:{str(f.relative_to(repo_dir))}"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
                                )
                                if old_blob_proc.returncode != 0 or new_blob_proc.returncode != 0:
                                    logging.debug(f"[openrewrite] git show failed for {f} commits {old_c}/{new_c}")
                                    continue
                                old_blob = old_blob_proc.stdout.decode("utf-8", errors="replace")
                                new_blob = new_blob_proc.stdout.decode("utf-8", errors="replace")
                                git_pairs.append((old_c, new_c, old_blob, new_blob))
                            except Exception:
                                continue
                except Exception:
                    git_pairs = []

                if not git_pairs:
                    continue

                for old_c, new_c, old_blob, new_blob in git_pairs:
                    if budget.get("remaining", 0) <= 0: break
                    try:
                        try:
                            old_doc = json.loads(old_blob)
                        except Exception:
                            old_doc = None
                            for d in yaml.safe_load_all(old_blob):
                                if isinstance(d, dict):
                                    old_doc = d; break
                    except Exception as e:
                        _save_bad_file(f, f"git_old_parse:{e}"); continue
                    try:
                        try:
                            new_doc = json.loads(new_blob)
                        except Exception:
                            new_doc = None
                            for d in yaml.safe_load_all(new_blob):
                                if isinstance(d, dict):
                                    new_doc = d; break
                    except Exception as e:
                        _save_bad_file(f, f"git_new_parse:{e}"); continue

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

    # Secondary: per-file multi-doc parsing and deterministic multi-variant fallback
    if budget.get("remaining", 0) > 0:
        logging.info("[openrewrite] falling back to per-file multi-doc processing")
        for f in files:
            if budget.get("remaining", 0) <= 0: break
            processed_files += 1
            if not _is_candidate_file(f):
                skipped_files += 1; continue
            try:
                txt = None
                try:
                    txt = f.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    _save_bad_file(f, f"read_error:{e}"); continue

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
                    skipped_files += 1; continue

                variants_per_doc = 3
                for di, doc in enumerate(docs):
                    if budget.get("remaining", 0) <= 0: break
                    svc_name = pick_real_service(f"openrewrite-{f.stem}-{di}")
                    for variant_i in range(variants_per_doc):
                        if budget.get("remaining", 0) <= 0: break
                        v1 = deepcopy(doc)
                        v2 = deepcopy(doc)
                        marker = sha_for_text(f.name + str(di) + str(seed) + str(variant_i))[:6]
                        if variant_i % 3 == 0:
                            v2["_rw_curation_marker"] = f"{marker}"
                        elif variant_i % 3 == 1:
                            pnew = f"/{svc_name}/auto{marker}"
                            v2.setdefault("paths", {})[pnew] = {"get": {"responses": {"200": {"description": "ok"}}}}
                        else:
                            try:
                                first = next(iter(v2.get("paths", {}) or {}))
                                op = v2["paths"][first].get("get") or v2["paths"][first].get("post") or {}
                                op.setdefault("parameters", []).append({"name": f"q{marker}", "in":"query", "schema":{"type":"string"}, "required": False})
                                # ensure assigned back
                                if "get" in v2["paths"][first]:
                                    v2["paths"][first]["get"] = op
                                else:
                                    v2["paths"][first]["post"] = op
                            except StopIteration:
                                pnew = f"/{svc_name}/auto{marker}"
                                v2.setdefault("paths", {})[pnew] = {"get": {"responses": {"200": {"description": "ok"}}}}
                        ok = _emit_pair_from_docs("openrewrite", svc_name, v1, v2, provenance_extra={"detail": f"{f.name}#doc{di}#v{variant_i}"})
                        if not ok:
                            _save_bad_file(f, f"emit_failed_doc:{di}:v{variant_i}")
                if processed_files % LOG_EVERY == 0:
                    logging.info(f"[openrewrite] doc progress files={processed_files} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped_files}")

            except Exception as e:
                logging.warning(f"[openrewrite] skip file {f}: {e}")
                _save_bad_file(f, f"outer_exception:{e}")
                continue

    logging.info(f"[openrewrite] completed {curated} pairs")
    return

# ---------- Petclinic curator 
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
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "petclinic")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "petclinic_repo"
    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                subprocess.run(["git", "clone", "--depth", "1", PETCLINIC_REPO, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
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

    # Case A: No YAML sources -> generate fully synthetic pairs 
    if not sources:
        times = budget.get("remaining", 0)
        for i in range(times):
            if budget.get("remaining", 0) <= 0: break
            try:
                service = pick_real_service(f"petclinic-synth-{i}")
                base_paths = [f"/pets", f"/owners", f"/appointments", f"/{service}/reports"]
                doc = {"paths": {}}
                for p in base_paths[:3]:
                    doc["paths"][p] = {
                        "get": {"responses": {"200": {"description": "ok"}}},
                        "post": {"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}}, "responses": {"201": {"description": "created"}}}
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
                    _write_json(old_can, doc); _write_json(new_can, v2)
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
                if dry_run and curated >= 5: break
            except Exception as e:
                logging.warning(f"[petclinic synth] skip {i}: {e}")

    # Case B: YAML sources exist -> create multiple deterministic realistic variants
    else:
        variants_per_file = 8
        for yml in sources:
            if budget.get("remaining", 0) <= 0: break
            try:
                service = yml.stem.lower()
                svc_name = pick_real_service(service)
                try:
                    yaml_text = yml.read_text(encoding="utf-8")
                except Exception as ex:
                    logging.warning(f"[petclinic] failed reading YAML {yml}: {ex}; using fallback base_doc")
                    yaml_text = None

                base_doc = {"paths": {}}
                if yaml_text:
                    try:
                        # Support multi-document YAML (--- separators)
                        docs = list(yaml.safe_load_all(yaml_text))
                        # Keep only dict-like docs
                        docs = [d for d in docs if isinstance(d, dict)]
                        # If any document looks like an OpenAPI (has 'paths'), prefer that
                        picked = None
                        for d in docs:
                            if "paths" in d and isinstance(d["paths"], dict) and d["paths"]:
                                picked = d
                                break
                        # If none had 'paths' but there are several dict docs, merge 'paths' keys
                        if picked is None and docs:
                            merged = {"paths": {}}
                            for d in docs:
                                pmap = d.get("paths")
                                if isinstance(pmap, dict):
                                    for pk, pv in pmap.items():
                                        if pk not in merged["paths"]:
                                            merged["paths"][pk] = deepcopy(pv)
                            if merged["paths"]:
                                picked = merged
                        # If we found a useful doc, use it
                        if picked and isinstance(picked.get("paths"), dict) and picked["paths"]:
                            for p, node in picked["paths"].items():
                                base_doc["paths"][p] = deepcopy(node)
                        else:
                            # If file is an application.yml or other config with no paths, skip heavy processing
                            fname = yml.name.lower()
                            if "application" in fname or "config" in fname or not docs:
                                logging.debug(f"[petclinic] {yml} appears to be config/no-api; using fallback base_doc")
                                base_doc = {"paths": {f"/{svc_name}": {"get": {"description": "orig", "responses": {"200": {"description": "ok"}}}}}}
                            else:
                                # Fall back to first dict doc if nothing better found
                                cand = docs[0]
                                if isinstance(cand.get("paths"), dict):
                                    for p, node in cand["paths"].items():
                                        base_doc["paths"][p] = deepcopy(node)
                                else:
                                    base_doc = {"paths": {f"/{svc_name}": {"get": {"description": "orig", "responses": {"200": {"description": "ok"}}}}}}
                    except Exception as ex:
                        logging.warning(f"[petclinic] failed to parse YAML {yml}: {ex}; using fallback base_doc")
                        base_doc = {"paths": {f"/{svc_name}": {"get": {"description": "orig", "responses": {"200": {"description": "ok"}}}}}}


                existing_paths = list(base_doc.get("paths", {}).keys()) or [f"/{svc_name}"]

                for variant in range(variants_per_file):
                    if budget.get("remaining", 0) <= 0: break
                    v2 = deepcopy(base_doc); m = variant % 5
                    target_path = existing_paths[rng.randrange(len(existing_paths))] if existing_paths else f"/{svc_name}"
                    if m == 0:
                        prefix = target_path.rstrip("/").split("/")[1] if len(target_path.split("/")) > 1 else svc_name
                        new_path = f"/{prefix}/new{variant}"
                        v2["paths"][new_path] = {"get": {"description": "added endpoint", "responses": {"200": {"description": "ok"}}}}
                        ace_type = "ENDPOINT_ADDED"; ace_path = new_path; ace_detail = new_path
                    elif m == 1:
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"description": "ok"}}}})
                        v2["paths"][ppath].setdefault("get", {}).setdefault("parameters", []).append(
                            {"name": f"q{variant}", "in": "query", "schema": {"type": "string"}, "required": False}
                        )
                        ace_type = "PARAM_ADDED"; ace_path = ppath; ace_detail = f"q{variant}"
                    elif m == 2:
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"description": "ok"}}}})
                        ace_type = "RESPONSE_CODE_REMOVED"; ace_path = ppath; ace_detail = ["201"]
                    elif m == 3:
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"content": {"application/json": {"schema": {"type": "object", "properties": {}}}}}}}})
                        try:
                            schema_props = v2["paths"][ppath]["get"]["responses"]["200"]["content"]["application/json"]["schema"].setdefault("properties", {})
                            schema_props[f"p{variant}"] = {"type": "integer"}
                        except Exception:
                            v2["paths"][ppath]["get"]["responses"]["200"]["content"] = {"application/json": {"schema": {"type": "object", "properties": {f"p{variant}": {"type": "integer"}}}}}
                        ace_type = "SCHEMA_PROPERTY_CHANGED"; ace_path = ppath; ace_detail = f"p{variant}"
                    else:
                        ppath = target_path
                        v2["paths"].setdefault(ppath, {"get": {"responses": {"200": {"description": "ok"}}}})
                        v2["paths"][ppath].setdefault("get", {}).setdefault("parameters", []).append(
                            {"name": f"renamed_{variant}", "in": "query", "schema": {"type": "string"}, "required": True}
                        )
                        ace_type = "PARAM_RENAMED"; ace_path = ppath; ace_detail = f"renamed_{variant}"

                    old_sha = sha_for_text(json.dumps(base_doc, sort_keys=True, default=_default_json_serializer))
                    new_sha = sha_for_text(json.dumps(v2, sort_keys=True, default=_default_json_serializer))
                    pair_id = generate_pair_id(svc_name, old_sha, new_sha)
                    pair_token = make_pair_token(svc_name, old_sha, new_sha)
                    old_can = paths["canonical"] / make_canonical_filename("petclinic", f"petclinic-{svc_name}", pair_token, "v1")
                    new_can = paths["canonical"] / make_canonical_filename("petclinic", f"petclinic-{svc_name}", pair_token, "v2")
                    if not dry_run:
                        _write_json(old_can, base_doc); _write_json(new_can, v2)
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
                    if dry_run and curated >= 5: break
            except Exception as e:
                logging.warning(f"[petclinic] skip {yml}: {e}")

    logging.info(f"[petclinic] Curated %d pairs", curated)
    return


# ---------- Curate OpenAPI 
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
    prefer_git_pairs: bool = True,
):
    rng = random.Random(seed)
    paths = dataset_paths(curated_root, "openapi")
    _ensure_dir(paths["canonical"]); _ensure_dir(paths["ndjson"]); _ensure_dir(paths["metadata"])

    repo_dir = raw_root / "openapi_repo"

    # attempt clone if allowed and missing
    if not skip_clone:
        try:
            if not repo_dir.exists() or not any(repo_dir.rglob("*")):
                logging.info(f"[curate_openapi] cloning openapi repo into {repo_dir}")
                subprocess.run(["git","clone", OPENAPI_REPO, str(repo_dir)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except Exception as e:
            logging.warning(f"[curate_openapi] clone failed: {e}")

    # gather candidate spec files: local raw openapi folder or cloned repo
    files = []
    if repo_dir.exists():
        files = list(repo_dir.rglob("*.yaml")) + list(repo_dir.rglob("*.yml")) + list(repo_dir.rglob("*.json"))
    else:
        logging.info("[curate_openapi] repo not present; looking in raw/openapi")
        local = raw_root / "openapi"
        if local.exists():
            files = list(local.rglob("*.yaml")) + list(local.rglob("*.yml")) + list(local.rglob("*.json"))
    if not files:
        logging.info("[curate_openapi] no candidate files found; will synthesize if budget > 0")
        if budget.get("remaining",0) > 0:
            _synthesize_openapi_pairs(pair_index, producers, version_meta, budget, seed+999, curated_root)
        return

    LOG_EVERY = 50
    processed = 0
    curated = 0
    skipped = 0

    def _is_candidate(f: Path):
        if is_noise_path(f): return False
        try:
            s = f.stat().st_size
            max_sz = globals().get("MAX_ALLOWED_FILE_SIZE", MAX_ALLOWED_FILE_SIZE)
            if s < 200 or s > max_sz: 
                return False
        except Exception: pass
        return True


    def _emit_from_docs(dataset: str, svc_name: str, old_doc: dict, new_doc: dict, provenance_extra: dict = None):
        nonlocal curated
        try:
            old_can, warns_old = canonicalize_with_timeout(old_doc, repo_dir if repo_dir.exists() else raw_root, timeout_sec=openapi_timeout)
            new_can, warns_new = canonicalize_with_timeout(new_doc, repo_dir if repo_dir.exists() else raw_root, timeout_sec=openapi_timeout)
            if old_can is None or new_can is None:
                return False
            old_can_r, old_sample = realisticize_canonical(old_can, svc_name + "-old")
            new_can_r, new_sample = realisticize_canonical(new_can, svc_name + "-new")
            old_sha = sha_for_text(json.dumps(old_can_r, sort_keys=True, default=_default_json_serializer))
            new_sha = sha_for_text(json.dumps(new_can_r, sort_keys=True, default=_default_json_serializer))
            pair_id = generate_pair_id(svc_name, old_sha, new_sha)
            pair_token = make_pair_token(svc_name, old_sha, new_sha)
            old_path = paths["canonical"] / make_canonical_filename("openapi", svc_name, pair_token, "v1")
            new_path = paths["canonical"] / make_canonical_filename("openapi", svc_name, pair_token, "v2")
            if not dry_run:
                _write_json(old_path, old_can_r); _write_json(new_path, new_can_r)

            aces = []
            old_ops = ops_map_from_canonical(old_can_r)
            new_ops = ops_map_from_canonical(new_can_r)
            all_paths = sorted(set(old_ops.keys()) | set(new_ops.keys()))
            for p in all_paths:
                methods = sorted(set(old_ops.get(p, {}).keys()) | set(new_ops.get(p, {}).keys()))
                for m in methods:
                    old_op = old_ops.get(p, {}).get(m)
                    new_op = new_ops.get(p, {}).get(m)
                    try:
                        # basic comparisons
                        if old_op is None and new_op is not None:
                            aces.append({"type":"ENDPOINT_ADDED","path":p,"method":m,"detail":p})
                        elif old_op is not None and new_op is None:
                            aces.append({"type":"ENDPOINT_REMOVED","path":p,"method":m,"detail":p})
                        else:
                            old_params = {pp.get("name") for pp in (old_op or {}).get("parameters") or [] if isinstance(pp, dict)}
                            new_params = {pp.get("name") for pp in (new_op or {}).get("parameters") or [] if isinstance(pp, dict)}
                            added = (new_params - old_params) if new_params else set()
                            removed = (old_params - new_params) if old_params else set()
                            for a in added: aces.append({"type":"PARAM_ADDED","path":p,"method":m,"detail":a})
                            for r in removed: aces.append({"type":"PARAM_REMOVED","path":p,"method":m,"detail":r})
                            # response code diffs
                            old_codes = set((old_op or {}).get("responses",{}).keys()) if isinstance((old_op or {}).get("responses"), dict) else set()
                            new_codes = set((new_op or {}).get("responses",{}).keys()) if isinstance((new_op or {}).get("responses"), dict) else set()
                            for rc in (old_codes - new_codes): aces.append({"type":"RESPONSE_CODE_REMOVED","path":p,"method":m,"detail":[rc]})
                    except Exception:
                        continue

            if not aces:
                aces.append({"type":"NON_API_CHANGE","path":None,"method":None,"detail":provenance_extra.get("detail") if provenance_extra else None})

            # finalize ACEs
            nd = paths["ndjson"] / f"{pair_id}.aces.ndjson"
            if not dry_run:
                _ensure_dir(nd.parent)
                with nd.open("w", encoding="utf-8") as fh:
                    for idx, a in enumerate(aces):
                        conf = 0.9 if a["type"].startswith("ENDPOINT") else 0.6
                        ace = {
                            "pair_id": pair_id,
                            "ace_index": idx,
                            "ace_id": f"{pair_id}::ace::{idx}",
                            "type": a.get("type"),
                            "path": a.get("path"),
                            "method": a.get("method"),
                            "detail": a.get("detail"),
                            "confidence": conf,
                            "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                            "side_effect": (a.get("method") or "").lower() in ("post","put","patch","delete"),
                            "calls_services": [],
                            "shared_schemas": []
                        }
                        fh.write(json.dumps(ace, ensure_ascii=False, default=_default_json_serializer) + "\n")

            meta = {
                "pair_id": pair_id,
                "dataset": "openapi",
                "service_name": svc_name,
                "old_canonical": old_path.name,
                "new_canonical": new_path.name,
                "old": str(old_path),
                "new": str(new_path),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "generated_at": _now_iso(),
                "producers_sample": (old_sample or [])[:5] + (new_sample or [])[:5]
            }
            if not dry_run:
                _write_json(paths["metadata"] / f"{pair_id}.meta.json", meta)
            pair_index[pair_id] = meta
            version_meta[pair_id] = {
                "pair_id": pair_id,
                "dataset": "openapi",
                "service_name": svc_name,
                "semver_old": "1.0.0",
                "semver_new": "1.0.1",
                "semver_delta": "patch",
                "breaking_vs_semver": any(a["type"] in ("ENDPOINT_REMOVED","RESPONSE_CODE_REMOVED","PARAM_RENAMED") for a in aces),
                "generated_at": meta["generated_at"],
                "producers_sample": meta.get("producers_sample", [])
            }
            producers_add(svc_name, meta.get("producers_sample", []), producers)
            curated += 1
            budget["remaining"] = max(0, budget.get("remaining", 0) - 1)
            return True
        except Exception as e:
            logging.debug(f"[openapi] emit failed: {e}", exc_info=True)
            return False

    # Primary: try to extract real pairs from repo git history
    if prefer_git_pairs and repo_dir.exists() and (repo_dir / ".git").exists():
        logging.info("[openapi] attempting git-history pair extraction")
        for f in files:
            if budget.get("remaining", 0) <= 0: break
            processed += 1
            if not _is_candidate(f): skipped += 1; continue
            try:
                # Get last 2 commits touching file
                res = subprocess.run(["git","-C", str(repo_dir), "log", "--pretty=format:%H", "--", str(f)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                commits = res.stdout.decode("utf-8").strip().splitlines()
                commits = commits[:2]
                if not commits: continue
                # generate commit pairs
                for i in range(len(commits)-1):
                    old_c = commits[i+1]; new_c = commits[i]
                    try:
                        old_blob = subprocess.run(["git","-C", str(repo_dir), "show", f"{old_c}:{str(f.relative_to(repo_dir))}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20).stdout.decode("utf-8", errors="replace")
                        new_blob = subprocess.run(["git","-C", str(repo_dir), "show", f"{new_c}:{str(f.relative_to(repo_dir))}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20).stdout.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    try:
                        old_doc = json.loads(old_blob)
                    except Exception:
                        old_doc = None
                        for d in yaml.safe_load_all(old_blob):
                            if isinstance(d, dict): old_doc = d; break
                    try:
                        new_doc = json.loads(new_blob)
                    except Exception:
                        new_doc = None
                        for d in yaml.safe_load_all(new_blob):
                            if isinstance(d, dict): new_doc = d; break
                    if not isinstance(old_doc, dict) or not isinstance(new_doc, dict):
                        continue
                    svc_base = re.sub(r"[^0-9a-zA-Z\-_]", "-", f.with_suffix("").as_posix()).lower()
                    svc_name = pick_real_service(svc_base)
                    ok = _emit_from_docs("openapi", svc_name, old_doc, new_doc, provenance_extra={"detail": f.name})
                    if ok and budget.get("remaining", 0) <= 0: break
            except Exception as e:
                logging.debug(f"[openapi] git extraction problem for {f}: {e}")
                continue
            if processed % LOG_EVERY == 0:
                logging.info(f"[openapi] processed={processed} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped}")

    # Secondary: per-file parsing, multi-doc treatment, micro-change fallback
    if budget.get("remaining", 0) > 0:
        logging.info("[openapi] falling back to per-file processing")
        for f in files:
            if budget.get("remaining", 0) <= 0: break
            processed += 1
            if not _is_candidate(f): skipped += 1; continue
            try:
                txt = None
                try: txt = f.read_text(encoding="utf-8", errors="replace")
                except Exception as e: _save_bad_file(f, f"read_error:{e}"); continue
                try:
                    try:
                        js = json.loads(txt)
                        docs = [js] if isinstance(js, (dict, list)) else []
                    except Exception:
                        docs = list(yaml.safe_load_all(txt))
                except Exception as e:
                    logging.debug(f"[openapi] parse failed {f}: {e}")
                    _save_bad_file(f, f"parse_failed:{e}"); continue
                docs = [d for d in docs if isinstance(d, dict)]
                if not docs: skipped += 1; continue
                for di, doc in enumerate(docs):
                    if budget.get("remaining", 0) <= 0: break
                    svc_name = pick_real_service(f"openapi-{f.stem}-{di}")
                    v1 = deepcopy(doc); v2 = deepcopy(doc)
                    # deterministic micro-change: add marker or add endpoint/param
                    marker = sha_for_text(f.name + str(di) + str(seed))[:6]
                    if "paths" not in v2 or not isinstance(v2.get("paths"), dict) or len(v2.get("paths",{}))==0:
                        # inject a minimal path
                        pnew = f"/{svc_name}/auto{marker}"
                        v2.setdefault("paths", {})[pnew] = {"get":{"responses":{"200":{"description":"ok"}}}}
                    else:
                        # add param or endpoint deterministically
                        if di % 3 == 0:
                            # add param to first path
                            first = next(iter(v2["paths"].keys()))
                            op = v2["paths"][first].get("get") or v2["paths"][first].get("post") or {}
                            op.setdefault("parameters", []).append({"name": f"q{marker}", "in":"query", "schema":{"type":"string"}, "required": False})
                        else:
                            pnew = f"/{svc_name}/added{marker}"
                            v2.setdefault("paths", {})[pnew] = {"get":{"responses":{"200":{"description":"ok"}}}}

                    ok = _emit_from_docs("openapi", svc_name, v1, v2, provenance_extra={"detail": f"{f.name}#doc{di}"})
                    if not ok:
                        _save_bad_file(f, f"emit_failed_doc:{di}"); continue
                if processed % LOG_EVERY == 0:
                    logging.info(f"[openapi] doc progress processed={processed} curated={curated} remaining={budget.get('remaining',0)} skipped={skipped}")
            except Exception as e:
                logging.warning(f"[openapi] skip file {f}: {e}")
                _save_bad_file(f, f"outer_exception:{e}")
                continue

    # If still budget remains, synth
    if budget.get("remaining", 0) > 0:
        created = _synthesize_openapi_pairs(pair_index, producers, version_meta, budget, seed+9999, curated_root)
        logging.info("[openapi] synthesized fallback created %d pairs", created)

    logging.info("[openapi] complete")
    return

# ---------- Orchestration: run()
def _parse_ratio_arg(ratio_str: str) -> Tuple[int,int,int]:
    try:
        parts = [p.strip() for p in ratio_str.split(",")]
        if len(parts) != 3: raise ValueError("need 3")
        vals = [float(p) if "." in p else int(p) for p in parts]
        if all(isinstance(v, float) for v in vals) and abs(sum(vals)-1.0) < 1e-6:
            vals = [int(round(v * 100)) for v in vals]
        else:
            vals = [int(round(float(v))) for v in vals]
        total = sum(vals)
        if total == 0: raise ValueError("sum 0")
        vals = [max(0, int(round(v * 100.0 / total))) for v in vals]
        residue = 100 - sum(vals); vals[0] += residue
        return vals[0], vals[1], vals[2]
    except Exception:
        logging.warning("Invalid --ratios value; falling back to default 85,10,5"); return 85,10,5

def _existing_counts(out_dir: Path) -> Dict[str, int]:
    counts = {"openapi":0,"petclinic":0,"openrewrite":0}
    idx_path = Path(out_dir)/"index.json"
    if idx_path.exists():
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for v in data.values():
                    ds = (v.get("dataset") or "").lower()
                    if ds in counts: counts[ds] += 1
            elif isinstance(data, list):
                for v in data:
                    ds = (v.get("dataset") or "").lower()
                    if ds in counts: counts[ds] += 1
            return counts
        except Exception:
            logging.warning("Failed to parse existing index.json; using folder counts")
    for ds in ("openapi","petclinic","openrewrite"):
        p = Path(out_dir)/ds/"canonical"
        if p.exists():
            try: counts[ds] = len([f for f in p.iterdir() if f.is_file() and f.suffix.lower()==".json"])
            except Exception: counts[ds]=0
        else: counts[ds]=0
    return counts

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
    """
    Main orchestration for curation pipeline.
    Defensive: guarded synth invocation (only calls synth function if defined).
    Adds:
      - optional global _NO_GIT_PAIRS to skip slow git-history extraction
      - robust cleanup on KeyboardInterrupt / Exception (shutdown_canon_pool)
    """
    global CURATED_ROOT
    CURATED_ROOT = Path(out_dir)
    _ensure_dir(CURATED_ROOT)

    # --- diagnostic probe 
    try:
        probe_dir = Path(out_dir).resolve()
        _ensure_dir(probe_dir)
        probe_marker = probe_dir / f".curation_probe_seed{seed}.txt"
        probe_marker.write_text(f"probe: {seed} ts={_now_iso()}\nraw_root={RAW_ROOT}\n", encoding="utf-8")
        logging.info("[run-probe] wrote probe marker -> %s", probe_marker)
    except Exception as e:
        logging.warning("[run-probe] probe marker write failed: %s", e)

    # list raw root summary
    try:
        if Path(RAW_ROOT).exists():
            entries = list(Path(RAW_ROOT).iterdir())
            logging.info("[run-probe] RAW_ROOT exists (%s) entries=%d example=%s", RAW_ROOT, len(entries), entries[:5])
        else:
            logging.warning("[run-probe] RAW_ROOT does not exist: %s", RAW_ROOT)
    except Exception as e:
        logging.warning("[run-probe] RAW_ROOT listing failed: %s", e)
    # --- end probe ---

    # ensure dataset dirs exist
    for ds in ("openapi", "petclinic", "openrewrite"):
        _ensure_dir(CURATED_ROOT / ds / "canonical")
        _ensure_dir(CURATED_ROOT / ds / "ndjson")
        _ensure_dir(CURATED_ROOT / ds / "metadata")

    logging.info("[run] CURATED_ROOT -> %s", CURATED_ROOT.resolve())

    existing = _existing_counts(out_dir)
    logging.info("Existing counts detected -> %s", existing)

    # noise/processing mode resolution
    mode = "clean"
    if globals().get("ARGS_OBJ") is not None:
        try:
            mode = getattr(globals()["ARGS_OBJ"], "mode", mode) or mode
        except Exception:
            pass
    mode = os.environ.get("NOISE_MODE", mode)

    processing_root = get_processing_root(
        RAW_ROOT,
        CURATED_ROOT,
        mode=mode,
        seed=seed,
        counts={"openapi": 500, "openrewrite": 300, "petclinic": 30, "synth_openapi": 50, "synth_openrewrite": 50, "synth_petclinic": 10},
    )
    logging.info("[run] using processing_root -> %s (mode=%s)", str(processing_root), mode)

    # target calculation
    explicit_targets_given = any(x is not None for x in (target_openapi, target_petclinic, target_openrewrite))
    if explicit_targets_given:
        tgt_op = target_openapi if target_openapi is not None else existing.get("openapi", 0)
        tgt_pc = target_petclinic if target_petclinic is not None else existing.get("petclinic", 0)
        tgt_or = target_openrewrite if target_openrewrite is not None else existing.get("openrewrite", 0)
        rem_openapi = max(0, tgt_op - existing.get("openapi", 0))
        rem_petclinic = max(0, tgt_pc - existing.get("petclinic", 0))
        rem_openrewrite = max(0, tgt_or - existing.get("openrewrite", 0))
        target_openapi = tgt_op
        target_petclinic = tgt_pc
        target_openrewrite = tgt_or
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

        if resume:
            rem_openapi = max(0, target_openapi - existing.get("openapi", 0))
            rem_petclinic = max(0, target_petclinic - existing.get("petclinic", 0))
            rem_openrewrite = max(0, target_openrewrite - existing.get("openrewrite", 0))
        else:
            rem_openapi, rem_petclinic, rem_openrewrite = target_openapi, target_petclinic, target_openrewrite

    logging.info("Targets -> OpenAPI %d, PetClinic %d, OpenRewrite %d", target_openapi, target_petclinic, target_openrewrite)

    # initialize state
    pair_index: Dict[str, Any] = {}
    producers: Dict[str, List[str]] = {}
    version_meta: Dict[str, Any] = {}

    budget_openapi = {"remaining": rem_openapi}
    budget_petclinic = {"remaining": rem_petclinic}
    budget_openrewrite = {"remaining": rem_openrewrite}

    # synth-openrewrite flag resolution
    synth_flag = globals().get("_SYNTH_OPENREWRITE_FLAG", False)
    if not synth_flag and globals().get("ARGS_OBJ") is not None:
        try:
            synth_flag = bool(getattr(globals()["ARGS_OBJ"], "synth_openrewrite", False))
        except Exception:
            synth_flag = synth_flag

    # If synth requested, call only if function is present and callable
    if synth_flag:
        synth_func = globals().get("synth_curate_openrewrite_semantic")
        if callable(synth_func):
            logging.info("[run] synth-openrewrite-semantic flag detected; invoking synth_curate_openrewrite_semantic()")
            try:
                created = synth_func(
                    pair_index,
                    version_meta,
                    budget_openrewrite,
                    seed=seed + 1000,
                    dry_run=dry_run,
                    curated_root=CURATED_ROOT,
                    raw_root=RAW_ROOT,
                )
                logging.info("[run] synth created %d openrewrite pairs", int(created or 0))
            except Exception as e:
                logging.exception("[run] synth_curate_openrewrite_semantic failed: %s", e)
            if not dry_run:
                write_global_indexes(pair_index, version_meta, curated_root=CURATED_ROOT)
            logging.info("[run] synth-openrewrite complete")
            return
        else:
            logging.warning("[run] synth_openrewrite requested but synth_curate_openrewrite_semantic() is not defined. Skipping synth step.")

    # decide whether to prefer git-history pairs (can be toggled by global flag)
    prefer_git_pairs = True
    # Priority of configuration: explicit global _NO_GIT_PAIRS, ARGS_OBJ.no_git_pairs, env NO_GIT_PAIRS
    if globals().get("_NO_GIT_PAIRS", False):
        prefer_git_pairs = False
    elif globals().get("ARGS_OBJ") is not None:
        try:
            if getattr(globals()["ARGS_OBJ"], "no_git_pairs", False):
                prefer_git_pairs = False
        except Exception:
            pass
    if os.environ.get("NO_GIT_PAIRS", "0") in ("1", "true", "True", "yes", "YES"):
        prefer_git_pairs = False

    logging.info("[run] prefer_git_pairs=%s (set _NO_GIT_PAIRS / --no-git-pairs / NO_GIT_PAIRS to change)", prefer_git_pairs)

    # Call curators in sequence with robust error handling so pools get shut down
    try:
        curate_openapi(
            pair_index,
            producers,
            version_meta,
            budget_openapi,
            seed=seed,
            dry_run=dry_run,
            raw_root=processing_root,
            curated_root=CURATED_ROOT,
            skip_clone=skip_clone,
            openapi_timeout=openapi_timeout,
            prefer_git_pairs=prefer_git_pairs,
        )

        curate_petclinic(
            pair_index,
            version_meta,
            budget_petclinic,
            seed=seed + 1,
            dry_run=dry_run,
            raw_root=processing_root,
            curated_root=CURATED_ROOT,
            skip_clone=skip_clone,
        )

        curate_openrewrite(
            pair_index,
            version_meta,
            budget_openrewrite,
            seed=seed + 2,
            dry_run=dry_run,
            raw_root=processing_root,
            curated_root=CURATED_ROOT,
            skip_clone=skip_clone,
            prefer_git_pairs=prefer_git_pairs,
        )

    except KeyboardInterrupt:
        logging.info("[run] KeyboardInterrupt received — shutting down workers and exiting")
        try: shutdown_canon_pool()
        except Exception: pass
        raise
    except Exception as e:
        logging.exception("[run] unexpected error during curation: %s", e)
        try: shutdown_canon_pool()
        except Exception: pass
        raise

    # Merge and write indexes
    existing_idx: Dict[str, Any] = {}
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

    repaired_existing: Dict[str, Any] = {}
    for pid, ent in existing_idx.items():
        try:
            ent2 = _normalize_pair_entry(ent)
            repaired_existing[ent2["pair_id"]] = ent2
        except Exception:
            continue

    merged_index = {**repaired_existing, **pair_index}
    if not dry_run:
        _write_json(Path(out_dir) / "index.json", {k: merged_index[k] for k in sorted(merged_index.keys())})
    logging.info("Index written: %d pairs", len(merged_index))

    existing_vm: Dict[str, Any] = {}
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
    except Exception as e:
        logging.warning("Failed to write version_meta.json: %s", e)

    # Consolidate producers
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

    # dataset counts / commit metadata
    dataset_counts: Dict[str, int] = {}
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
                writer.writerow(["pair_id", "dataset", "service_name", "old_sha", "new_sha", "old_canonical", "new_canonical", "generated_at"])
                for k in sorted(merged_index.keys()):
                    p = merged_index[k]
                    writer.writerow([
                        p.get("pair_id"),
                        p.get("dataset"),
                        p.get("service_name"),
                        p.get("old_sha"),
                        p.get("new_sha"),
                        p.get("old_canonical"),
                        p.get("new_canonical"),
                        p.get("generated_at"),
                    ])
    except Exception as e:
        logging.warning("Failed to write version_pairs.csv: %s", e)

    logging.info("Curation complete: total pairs=%d", len(merged_index))

    # optionally ensure target ACE count
    target_aces = globals().get("_TARGET_ACES", None)
    batch = globals().get("_ACE_BATCH", 200)
    if isinstance(target_aces, int) and target_aces > 0 and not no_augment:
        final_total = _ensure_total_aces(CURATED_ROOT, target_aces, seed=seed, batch_per_file=batch, dry_run=dry_run)
        logging.info("Final ACE total after ensure: %d", final_total)
    else:
        logging.info("Skipping synthetic ACE augmentation (no_augment=%s, target_aces=%s)", no_augment, target_aces)

    if not dry_run:
        write_global_indexes(merged_index, merged_vm, curated_root=CURATED_ROOT)

    # cleanup
    shutdown_canon_pool()
    return



# ---------- CLI main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(CURATED_ROOT))
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ratios", default="85,10,5")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-openapi", type=int, default=None)
    parser.add_argument("--target-petclinic", type=int, default=None)
    parser.add_argument("--target-openrewrite", type=int, default=None)
    parser.add_argument("--synthesize-openapi", action="store_true")
    parser.add_argument("--target-aces", type=int, default=0)
    parser.add_argument("--ace-batch", type=int, default=200)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--openapi-timeout", type=int, default=8)
    parser.add_argument("--synth-openrewrite", action="store_true")
    parser.add_argument("--mode", choices=["clean","noisy_light","noisy_heavy"], default="clean")
    parser.add_argument("--max-file-size", type=int, default=None,
                        help="Max file size in bytes to consider during curation (default: env MAX_ALLOWED_FILE_SIZE or 2000000)")

    args = parser.parse_args()
    if args.max_file_size is not None:
        globals()["MAX_ALLOWED_FILE_SIZE"] = args.max_file_size
    else:
        # if env var specified earlier, ensure global reflects it
        globals()["MAX_ALLOWED_FILE_SIZE"] = globals().get("MAX_ALLOWED_FILE_SIZE", MAX_ALLOWED_FILE_SIZE)

    random.seed(args.seed)
    ratios = _parse_ratio_arg(args.ratios)
    globals()["_SYNTH_OPENREWRITE_FLAG"] = getattr(args, "synth_openrewrite", False)
    globals()["_SYNTH_OPENAPI"] = getattr(args, "synthesize_openapi", False)
    globals()["_TARGET_ACES"] = getattr(args, "target_aces", None)
    globals()["_ACE_BATCH"] = getattr(args, "ace_batch", 200)
    globals()["ARGS_OBJ"] = args
    run(Path(args.out), args.max, args.seed, args.dry_run, ratios, skip_clone=args.skip_clone, resume=args.resume, target_openapi=args.target_openapi, target_petclinic=args.target_petclinic, target_openrewrite=args.target_openrewrite, openapi_timeout=args.openapi_timeout, no_augment=args.no_augment)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user (KeyboardInterrupt) — shutting down")
    except Exception as e:
        logging.exception("Unhandled exception in main: %s", e)
    finally:
        try:
            shutdown_canon_pool()
        except Exception:
            logging.debug("shutdown_canon_pool() failed during final cleanup")
        logging.info("Exiting.")


def _graceful_exit(signum=None, frame=None):
    logging.info("[shutdown] received signal, shutting down pools...")
    try:
        shutdown_canon_pool()
    except Exception:
        pass
    # attempt to kill lingering child processes from subprocess calls
    try:
        # best-effort: send SIGTERM to process group
        os.killpg(0, signal.SIGTERM)
    except Exception:
        pass
    sys.exit(1)

# install handler for SIGINT/SIGTERM
signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)