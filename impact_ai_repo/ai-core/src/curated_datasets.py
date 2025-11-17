#!/usr/bin/env python3
"""
curated_datasets.py

Create curated pairs for OpenAPI, PetClinic and OpenRewrite.

This script creates small, deterministic examples used in the thesis.
It reads source files from datasets/raw when available or uses simple
synthetic examples. The output is written under datasets/curated by default.

Notes:
- Fixes pointer parsing issues and resolves local and external $ref.
- Adds extra ACE fields for some endpoint-added cases:
  side_effect, calls_services, shared_schemas
- Filenames and folder layout are preserved for reproducibility.
"""
from __future__ import annotations
import argparse
import logging
import json
import yaml
import hashlib
import random
import shutil
import subprocess
import time
import re
import datetime
import decimal
import uuid
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Defaults for input and output directories.
# These can be changed via the --out argument.
RAW_ROOT = Path("datasets/raw")
CURATED_ROOT = Path("datasets/curated")
CANONICAL_DIR = CURATED_ROOT / "canonical"
NDJSON_DIR = CURATED_ROOT / "ndjson"
META_DIR = CURATED_ROOT / "metadata"
GRAPH_PATH = CURATED_ROOT / "graph.json"
INDEX_PATH = CURATED_ROOT / "index.json"
VERSION_META_PATH = CURATED_ROOT / "version_meta.json"

# Remote repos used as optional sources. Cloning is attempted only if needed.
OPENAPI_REPO = "https://github.com/APIs-guru/openapi-directory.git"
PETCLINIC_REPO = "https://github.com/spring-petclinic/spring-petclinic-microservices.git"
OPENREWRITE_REPO = "https://github.com/openrewrite/rewrite.git"

# Small helper that returns a normalized copy for objects that json can't handle easily.
def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj

# Filesystem helpers
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_file(p: Path):
    txt = p.read_text(encoding="utf-8")
    try:
        if p.suffix in (".yml", ".yaml"):
            return yaml.safe_load(txt)
        return json.loads(txt)
    except Exception:
        return yaml.safe_load(txt)

# JSON serializer that can handle common non-serializable objects.
def _default_json_serializer(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
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

def _write_json(p: Path, obj):
    _ensure_dir(p.parent)
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=_default_json_serializer)
        p.write_text(text, encoding="utf-8")
    except TypeError:
        p.write_text(json.dumps(_normalize(obj), indent=2, ensure_ascii=False, default=_default_json_serializer), encoding="utf-8")

# Return a short sha for a text input.
def sha_for_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:20]


def clone_if_needed(repo_url: str, dest: Path):
    # Clone repo only if destination is missing or empty.
    if dest.exists() and any(dest.iterdir()):
        logging.info(f"Repo already present and non-empty: {dest}")
        return
    if dest.exists():
        logging.info(f"Removing existing empty directory: {dest}")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Cloning {repo_url} → {dest}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logging.warning(f"git clone failed for {repo_url}: {e}")


def curate_openapi(
    pair_index: Dict,
    producers: Dict,
    version_meta: Dict,
    max_items: int = 200,
    seed: int = 42,
    dry_run: bool = False,
):
    import time as _t
    random.seed(seed)

    _ensure_dir(CANONICAL_DIR)
    _ensure_dir(NDJSON_DIR)
    _ensure_dir(META_DIR)

    repo_dir = RAW_ROOT / "openapi_repo"
    local_dir = RAW_ROOT / "openapi"

    # Try to clone the openapi repo if no local copy exists.
    if not repo_dir.exists() or not any(repo_dir.iterdir()):
        try:
            clone_if_needed(OPENAPI_REPO, repo_dir)
        except Exception as e:
            logging.warning(f"[curate_openapi] clone_if_needed failed: {e} - will try local RAW_ROOT")

    scan_base = local_dir if local_dir.exists() and any(local_dir.rglob("*")) else repo_dir
    if not scan_base.exists() or not any(scan_base.rglob("*")):
        logging.warning(f"[curate_openapi] No source openapi files found under {scan_base}; skipping.")
        return

    curated = 0

    for file in scan_base.rglob("*"):
        if curated >= max_items:
            break

        try:
            if not file.is_file():
                continue
            if file.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            if file.stat().st_size > 2_000_000:
                logging.info(f"[curate_openapi] Skipping large file {file} (>2MB)")
                continue

            doc = _read_file(file)
            if not isinstance(doc, dict) or not doc.get("paths"):
                continue

            try:
                rel = file.relative_to(scan_base)
                rel_key = rel.with_suffix("").as_posix()
            except Exception:
                rel_key = f"{file.parent.name}/{file.stem}"
            service_name = re.sub(r"[^0-9a-zA-Z\-_]", "-", rel_key).replace("/", "-").replace(" ", "-").lower()

            canon, warns = canonicalize_spec(doc, file.parent)

            pair_seed = (int(sha_for_text(str(file)), 16) ^ seed) & 0x7fffffff

            v2doc = inject_openapi_diffs(canon, seed=pair_seed)

            old_can = CANONICAL_DIR / f"openapi--{service_name}--v1.canonical.json"
            new_can = CANONICAL_DIR / f"openapi--{service_name}--v2.canonical.json"

            _write_json(old_can, _normalize(canon))
            _write_json(new_can, _normalize(v2doc))

            old_sha = sha_for_text(json.dumps(canon, sort_keys=True))
            new_sha = sha_for_text(json.dumps(v2doc, sort_keys=True))
            pair_fingerprint = sha_for_text(service_name + old_sha + new_sha)
            pair_id = f"pair-{pair_fingerprint}"

            pair_entry = {
                "pair_id": pair_id,
                "service_name": service_name,
                "dataset": "openapi",
                "old_canonical": str(old_can.name),
                "new_canonical": str(new_can.name),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "warnings": warns,
                "seed": pair_seed,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            pair_index[pair_id] = {
                "pair_id": pair_id,
                "dataset": "openapi",
                "service_name": service_name,
                "old_canonical": str(old_can.name),
                "new_canonical": str(new_can.name),
                "old": str(old_can),
                "new": str(new_can),
                **pair_entry,
            }

            # pick a small sample of endpoints for the producers map
            eps = list((canon.get("paths") or {}).keys())
            eps_norm = [re.sub(r"\{[^}]+\}", "{*}", p) for p in eps]
            max_sample = 5
            if len(eps_norm) > max_sample:
                eps_norm = random.sample(eps_norm, max_sample)
            producers[service_name] = eps_norm

            # Build ACEs by comparing canonical operations
            aces = []
            def ops_map(spec):
                out = {}
                for p, methods in (spec.get("paths") or {}).items():
                    if not isinstance(methods, dict):
                        continue
                    out[p] = {m.lower(): methods[m] for m in methods.keys() if isinstance(m, str)}
                return out

            old_ops = ops_map(canon)
            new_ops = ops_map(v2doc)
            all_paths = sorted(set(list(old_ops.keys()) + list(new_ops.keys())))

            for p in all_paths:
                methods = sorted(set(list(old_ops.get(p, {}).keys()) + list(new_ops.get(p, {}).keys())))
                for m in methods:
                    old_op = old_ops.get(p, {}).get(m)
                    new_op = new_ops.get(p, {}).get(m)
                    detected = compare_operations(old_op, new_op, p, m)
                    for d in detected:
                        conf = compute_confidence(d, canon, v2doc)
                        # add a few heuristic metadata fields
                        side_effect = (m.lower() in ("post", "put", "patch", "delete"))
                        calls = []
                        if random.random() < 0.35:
                            callers = list(producers.keys())
                            if callers:
                                calls = [random.choice(callers)]
                        shared = []
                        if isinstance(d.get("detail"), str) and ("schema" in d.get("detail").lower() or "enum" in d.get("detail").lower()):
                            shared = ["SharedSchema"]
                        ace = {
                            "pair_id": pair_id,
                            "ace_index": len(aces),
                            "ace_id": f"{pair_id}::ace::{len(aces)}",
                            "type": d.get("type"),
                            "path": d.get("path"),
                            "method": d.get("method"),
                            "detail": d.get("detail") if "detail" in d else None,
                            "confidence": conf,
                            "provenance": {"old_sha": old_sha, "new_sha": new_sha},
                            "side_effect": bool(side_effect),
                            "calls_services": calls,
                            "shared_schemas": shared,
                        }
                        aces.append(ace)

            ndjson_path = NDJSON_DIR / f"{pair_id}.aces.ndjson"
            with ndjson_path.open("w", encoding="utf-8") as fh:
                for a in aces:
                    fh.write(json.dumps(a, ensure_ascii=False, default=_default_json_serializer) + "\n")

            meta_path = META_DIR / f"{pair_id}.meta.json"
            _write_json(meta_path, pair_entry)

            version_meta[pair_id] = {
                "pair_id": pair_id,
                "dataset": "openapi",
                "service_name": service_name,
                "semver_old": "1.0.0",
                "semver_new": "1.1.0",
                "semver_delta": random.choice(["patch", "minor", "major"]),
                "breaking_vs_semver": random.random() < 0.15,
                "deprecated": random.random() < 0.05,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            curated += 1
            if curated % 25 == 0:
                logging.info(f"[curate_openapi] Curated {curated} OpenAPI pairs so far")

            if dry_run and curated >= 10:
                break

        except Exception as e:
            logging.warning(f"[curate_openapi] Skipping {file}: {e}")

    logging.info(f"[curate_openapi] Finished: curated {curated} OpenAPI pairs into {CANONICAL_DIR}")
    return


def resolve_local_pointer(root: Any, pointer: str):
    # Resolve a JSON pointer inside the given root object.
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
    # Walk the spec and resolve local and external references.
    warnings = []
    def _rec(node, cur_base: Path, stack: Tuple[str, ...] = ()):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if ref.startswith("#"):
                    try:
                        resolved = resolve_local_pointer(spec, ref)
                        if ref in stack:
                            warnings.append(f"cycle local {ref}")
                            return {"__ref_cycle__": ref}
                        return _rec(resolved, cur_base, stack + (ref,))
                    except Exception as e:
                        warnings.append(f"local ref {ref} failed: {e}")
                        return {"__unresolved_ref__": ref}
                else:
                    parts = ref.split("#", 1)
                    file_part = parts[0]
                    ptr = "#" + parts[1] if len(parts) > 1 else "#"
                    target_file = (cur_base / file_part).resolve()
                    if not target_file.exists():
                        warnings.append(f"external ref file missing: {target_file}")
                        return {"__unresolved_ref__": ref}
                    try:
                        loaded = _read_file(target_file)
                        resolved = resolve_local_pointer(loaded, ptr) if ptr != "#" else loaded
                        key = f"{str(target_file)}::{ptr}"
                        if key in stack:
                            warnings.append(f"cycle external {ref}")
                            return {"__ref_cycle__": ref}
                        return _rec(resolved, target_file.parent, stack + (key,))
                    except Exception as e:
                        warnings.append(f"external ref load failed {ref}: {e}")
                        return {"__unresolved_ref__": ref}
            else:
                out = {}
                for k,v in node.items():
                    out[k] = _rec(v, cur_base, stack)
                if any(k in out for k in ("oneOf","anyOf","allOf")):
                    out["_polymorphic_marker"] = {k: True for k in ("oneOf","anyOf","allOf") if k in out}
                return out
        elif isinstance(node, list):
            return [_rec(i, cur_base, stack) for i in node]
        else:
            return deepcopy(node)
    try:
        canon = _rec(spec, base_dir)
    except Exception as e:
        warnings.append(f"canonicalization failed: {e}")
        canon = deepcopy(spec)
    return canon, warnings


def inject_openapi_diffs(doc: Dict[str,Any], seed:int=0) -> Dict[str,Any]:
    # Make small, deterministic changes to a canonical spec to produce a new version.
    r = random.Random(seed)
    new_doc = deepcopy(doc)
    paths = new_doc.setdefault("paths", {})
    candidates = list(paths.keys())
    r.shuffle(candidates)
    # update descriptions for a few endpoints
    for p in candidates[: max(0, min(3, len(candidates)))]:
        methods = paths.get(p) or {}
        for m in list(methods.keys())[:1]:
            if isinstance(methods[m], dict):
                methods[m]["description"] = f"modified {m} {p}"
    # add synthetic endpoints to simulate new features
    for _ in range(r.randint(1,3)):
        new_path = f"/auto{r.randint(100,999)}"
        if r.random() < 0.5:
            paths[new_path] = {
                "get": {"summary":"synthetic diff","responses":{"200":{"description":"ok"}}}
            }
        else:
            paths[new_path] = {
                "post": {"summary":"synthetic write","requestBody":{"content":{"application/json":{"schema":{"type":"object"}}}},"responses":{"201":{"description":"created"}}}
            }
    # sometimes remove an endpoint
    if candidates and r.random() < 0.25:
        try:
            del paths[candidates[0]]
        except KeyError:
            pass
    return new_doc


def compare_operations(old_op, new_op, path, method):
    # Detect simple change types between two operation objects.
    aces = []
    if old_op is None and new_op is not None:
        aces.append({"type": "ENDPOINT_ADDED", "path": path, "method": method})
        return aces
    if old_op is not None and new_op is None:
        aces.append({"type": "ENDPOINT_REMOVED", "path": path, "method": method})
        return aces
    old_params = {p.get("name"): p for p in (old_op.get("parameters") or [])} if old_op else {}
    new_params = {p.get("name"): p for p in (new_op.get("parameters") or [])} if new_op else {}
    for name, new_p in new_params.items():
        old_p = old_params.get(name)
        if new_p.get("required") and (old_p is None or not old_p.get("required")):
            aces.append({"type":"PARAM_REQUIRED_ADDED","path":path,"method":method,"detail":name})
    def extract_enums(op):
        enums = {}
        for p in (op.get("parameters") or []):
            s = p.get("schema", {})
            if isinstance(s, dict) and "enum" in s:
                enums[f"param:{p.get('name')}"] = list(s["enum"])
        rb = op.get("requestBody",{}) or {}
        for ct, body in (rb.get("content") or {}).items():
            s = body.get("schema",{})
            if isinstance(s, dict) and "enum" in s:
                enums[f"requestBody:{ct}"] = list(s["enum"])
        return enums
    old_enums = extract_enums(old_op or {})
    new_enums = extract_enums(new_op or {})
    for k, new_list in new_enums.items():
        old_list = old_enums.get(k)
        if old_list and set(new_list) < set(old_list):
            aces.append({"type":"ENUM_NARROWED","path":path,"method":method,"detail":k})
    old_rc = set(str(k) for k in ((old_op or {}).get("responses") or {}).keys())
    new_rc = set(str(k) for k in ((new_op or {}).get("responses") or {}).keys())
    removed = old_rc - new_rc
    if removed:
        aces.append({"type":"RESPONSE_CODE_REMOVED","path":path,"method":method,"detail":list(removed)})
    return aces

def compute_confidence(ace, canonical_old, canonical_new):
    # Simple heuristic to compute a confidence score for an ACE.
    base = 0.65
    typ = ace.get("type","")
    if "ENUM" in typ:
        base -= 0.08
    if "PARAM" in typ:
        base += 0.04
    if ("_polymorphic_marker" in canonical_old) or ("_polymorphic_marker" in canonical_new):
        base -= 0.06
    return round(max(0.0, min(1.0, base)), 3)


def curate_petclinic(pair_index: Dict, version_meta: Dict, dry_run:bool=False):
    # Create a few synthetic pairs based on petclinic files or the local folder.
    repo_dir = RAW_ROOT / "petclinic_repo"
    if not repo_dir.exists():
        try:
            clone_if_needed(PETCLINIC_REPO, repo_dir)
        except Exception as e:
            logging.warning(f"Could not clone petclinic: {e}")
            repo_dir = RAW_ROOT / "petclinic"
    _ensure_dir(CANONICAL_DIR)
    _ensure_dir(NDJSON_DIR)
    _ensure_dir(META_DIR)
    curated = 0
    source_iter = (repo_dir.rglob("*.yml") if repo_dir.exists() else (RAW_ROOT / "petclinic").rglob("*.yml"))
    for yml in source_iter:
        try:
            p = yml
            service = p.stem.lower()
            old_can = CANONICAL_DIR / f"petclinic-{service}-v1.canonical.json"
            new_can = CANONICAL_DIR / f"petclinic-{service}-v2.canonical.json"
            doc = {"paths": {"/"+service: {"get": {"description":"orig"}}}}
            v2 = deepcopy(doc)
            v2["paths"][f"/{service}/new"] = {"get": {"description":"added endpoint"}}
            _write_json(old_can, doc)
            _write_json(new_can, v2)
            old_sha = sha_for_text(json.dumps(doc, sort_keys=True))
            new_sha = sha_for_text(json.dumps(v2, sort_keys=True))
            pair_id = f"pair-{sha_for_text(service + old_sha + new_sha)}"
            pair_entry = {
                "pair_id": pair_id,
                "service_name": f"petclinic-{service}",
                "old_canonical": str(old_can.name),
                "new_canonical": str(new_can.name),
                "old": str(old_can),
                "new": str(new_can),
                "old_sha": old_sha,
                "new_sha": new_sha,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            pair_index[pair_id] = pair_entry
            nd = NDJSON_DIR / f"{pair_id}.aces.ndjson"
            ace = {"pair_id": pair_id, "ace_index": 0, "ace_id": f"{pair_id}::ace::0", "type": "ENDPOINT_ADDED", "path": f"/{service}/new", "method":"get", "detail": f"/{service}/new", "confidence":0.8, "side_effect": False, "calls_services": [], "shared_schemas": []}
            with nd.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(ace, ensure_ascii=False, default=_default_json_serializer) + "\n")
            _write_json(META_DIR / f"{pair_id}.meta.json", pair_entry)
            version_meta[pair_id] = {"pair_id": pair_id, "service_name": pair_entry["service_name"], "semver_old": "1.0.0", "semver_new": "1.0.1", "semver_delta": "patch", "breaking_vs_semver": False}
            curated += 1
            if dry_run and curated>=5:
                break
        except Exception as e:
            logging.warning(f"petclinic skip {yml}: {e}")
    logging.info(f"[PETCLINIC] Curated {curated} pairs")
    return


def curate_openrewrite(pair_index: Dict, version_meta: Dict, dry_run:bool=False):
    # Create small synthetic OpenRewrite recipe changes.
    _ensure_dir(CANONICAL_DIR)
    _ensure_dir(NDJSON_DIR)
    _ensure_dir(META_DIR)
    curated = 0
    for i in range(10):
        v1 = {"recipeList":[{"name":f"org.demo.Recipe{i}"}]}
        v2 = {"recipeList":[{"name":f"org.demo.Recipe{i}"},{"name":f"org.demo.Added{i}"}]}
        old_can = CANONICAL_DIR / f"openrewrite-recipe-{i}-v1.canonical.json"
        new_can = CANONICAL_DIR / f"openrewrite-recipe-{i}-v2.canonical.json"
        _write_json(old_can, v1)
        _write_json(new_can, v2)
        old_sha = sha_for_text(json.dumps(v1, sort_keys=True)); new_sha = sha_for_text(json.dumps(v2, sort_keys=True))
        pair_id = f"pair-{sha_for_text(str(i)+old_sha+new_sha)}"
        pair_entry = {
            "pair_id": pair_id,
            "service_name": f"openrewrite-recipe-{i}",
            "old_canonical": str(old_can.name),
            "new_canonical": str(new_can.name),
            "old": str(old_can),
            "new": str(new_can),
            "old_sha": old_sha,
            "new_sha": new_sha,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        pair_index[pair_id] = pair_entry
        nd = NDJSON_DIR / f"{pair_id}.aces.ndjson"
        ace = {"pair_id": pair_id, "ace_index": 0, "ace_id": f"{pair_id}::ace::0", "type": "RECIPE_ADDED", "detail": f"Added{i}", "confidence": 0.9, "side_effect": False, "calls_services": [], "shared_schemas": []}
        with nd.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(ace, ensure_ascii=False, default=_default_json_serializer) + "\n")
        _write_json(META_DIR / f"{pair_id}.meta.json", pair_entry)
        version_meta[pair_id] = {"pair_id": pair_id, "service_name": pair_entry["service_name"], "semver_old": "0.1.0", "semver_new": "0.2.0", "semver_delta": "minor", "breaking_vs_semver": False}
        curated += 1
    logging.info(f"[OPENREWRITE] Curated {curated} pairs")
    return


def build_graph(producers: Dict[str, List[str]], seed:int=42):
    # Build a small producer-consumer graph for the curated dataset.
    random.seed(seed)
    edges = []
    ui_layers = ["portal-ui","admin-ui","mobile-ui"]
    services = list(producers.keys())
    for i, svc in enumerate(services):
        ui = ui_layers[i % len(ui_layers)] if ui_layers else "portal-ui"
        paths = producers.get(svc, [])
        chosen = random.sample(paths, min(len(paths), 5)) if paths else []
        for p in chosen:
            if random.random() < 0.65:
                edges.append({"src": f"ui:{ui}", "dst": f"svc:{svc}", "path": p, "evidence":"curation", "confidence": round(random.uniform(0.4,0.95),2)})
            else:
                other_svc = random.choice(services) if services and len(services) > 1 else svc
                edges.append({"src": f"svc:{other_svc}", "dst": f"svc:{svc}", "path": p, "evidence":"curation-svc", "confidence": round(random.uniform(0.4,0.95),2)})
    graph = {"edges": edges, "producers": producers}
    _write_json(GRAPH_PATH, graph)
    logging.info(f"Graph built: {len(edges)} edges")
    return graph


def run(out_dir: Path, max_items:int, seed:int, dry_run:bool):
    # Main entry that runs all curators and writes index and version metadata.
    global CURATED_ROOT, CANONICAL_DIR, NDJSON_DIR, META_DIR, GRAPH_PATH, INDEX_PATH, VERSION_META_PATH
    CURATED_ROOT = Path(out_dir)
    CANONICAL_DIR = CURATED_ROOT / "canonical"
    NDJSON_DIR = CURATED_ROOT / "ndjson"
    META_DIR = CURATED_ROOT / "metadata"
    GRAPH_PATH = CURATED_ROOT / "graph.json"
    INDEX_PATH = CURATED_ROOT / "index.json"
    VERSION_META_PATH = CURATED_ROOT / "version_meta.json"

    _ensure_dir(CANONICAL_DIR)
    _ensure_dir(NDJSON_DIR)
    _ensure_dir(META_DIR)

    pair_index: Dict[str, Any] = {}
    producers: Dict[str, List[str]] = {}
    version_meta: Dict[str, Any] = {}

    curate_openapi(pair_index, producers, version_meta, max_items, seed, dry_run)
    curate_petclinic(pair_index, version_meta, dry_run)
    curate_openrewrite(pair_index, version_meta, dry_run)
    build_graph(producers, seed)

    _write_json(INDEX_PATH, {k: pair_index[k] for k in sorted(pair_index.keys())})
    logging.info(f"Index written: {len(pair_index)} pairs (to {INDEX_PATH})")

    try:
        _write_json(VERSION_META_PATH, version_meta)
        logging.info("Wrote version_meta.json (%d entries) to %s", len(version_meta), VERSION_META_PATH)
    except Exception as e:
        logging.warning("Failed to write version_meta.json: %s", e)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(CURATED_ROOT), help="output dir")
    parser.add_argument("--max", type=int, default=200, help="max curation items")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    run(Path(args.out), args.max, args.seed, args.dry_run)

if __name__ == "__main__":
    main()
