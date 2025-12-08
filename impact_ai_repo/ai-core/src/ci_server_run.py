#!/usr/bin/env python3
"""
ci_server_run.py

Lightweight CI wrapper that:
 - finds changed files in a PR (git)
 - maps canonical v2 -> v1 or fetches base from origin/main
 - lightly dereferences local '#/components/schemas/...' refs (shallow)
 - primarily uses server.api_analyze(...) plus graph enrichment
 - writes pr-impact-full.json (detailed) and pr-impact-report.json (compact)

This version:
 - locates curated datasets and index.json robustly
 - uses index.json only to infer pair_id / service_name from filenames
 - DOES NOT rely on model.joblib being present in CI
 - computes risk score via a heuristic over ACEs, with model score as a lower-bound
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve repo and ai-core paths (CI sets GITHUB_WORKSPACE and AI_CORE_DIR)
REPO_ROOT = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve()
AI_CORE_DIR_ENV = os.environ.get("AI_CORE_DIR", "impact_ai_repo/ai-core/src")
AI_CORE_SRC = (REPO_ROOT / AI_CORE_DIR_ENV).resolve()

# Make sure ai-core/src is first on sys.path so `import server` loads local server.py
HERE = AI_CORE_SRC if AI_CORE_SRC.exists() else Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return (0, out.decode())
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.output.decode() if e.output else "")
    except Exception as e:
        return (1, str(e))


# Import server module now
try:
    import server  # noqa: E402
    print(f"DEBUG: imported server from {getattr(server, '__file__', 'unknown')}", file=sys.stderr)
    try:
        print("DEBUG: dataset_paths(openapi) =", server.dataset_paths("openapi"), file=sys.stderr)
    except Exception:
        print("DEBUG: server.dataset_paths('openapi') failed", file=sys.stderr)
    print("DEBUG: CWD =", os.getcwd(), file=sys.stderr)
except Exception as e:
    print("ERROR: failed to import server.py:", e, file=sys.stderr)
    raise


# --- helpers to resolve candidate files / roots robustly ----
def _resolve_candidate_root(candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    p = Path(candidate)
    if p.is_absolute() and p.exists():
        return p
    # try relative to ai-core/src
    p2 = (HERE / candidate).resolve()
    if p2.exists():
        return p2
    # try relative to repo root
    p3 = (REPO_ROOT / candidate).resolve()
    if p3.exists():
        return p3
    # try inside HERE/datasets/<candidate>
    p4 = (HERE / "datasets" / candidate).resolve()
    if p4.exists():
        return p4
    # try sibling of HERE
    p5 = (HERE.parent / candidate).resolve()
    if p5.exists():
        return p5
    return None


def _find_index_json_under_datasets() -> Optional[Path]:
    """
    Search for an index.json (pair index) under plausible dataset roots.
    Returns path to the directory that contains index.json (curated variant root).
    """
    candidates: List[Path] = []

    # 1) server.EFFECTIVE_CURATED_ROOT if set
    try:
        eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
        if eff:
            resolved = _resolve_candidate_root(str(eff))
            if resolved:
                candidates.append(resolved)
    except Exception:
        pass

    # 2) HERE/datasets (ai-core/src/datasets)
    maybe = (HERE / "datasets")
    if maybe.exists():
        candidates.append(maybe.resolve())

    # 3) REPO_ROOT/AI_CORE_DIR_ENV/datasets
    maybe2 = (REPO_ROOT / AI_CORE_DIR_ENV / "datasets")
    if maybe2.exists():
        candidates.append(maybe2.resolve())

    # 4) dataset paths returned by server.dataset_paths() for dataset keys
    try:
        for key in server.all_dataset_keys():
            try:
                dp = server.dataset_paths(key)
                for root_candidate in (
                    dp.get("base"),
                    dp.get("canonical"),
                    dp.get("ndjson"),
                    dp.get("metadata"),
                ):
                    if root_candidate:
                        rc = _resolve_candidate_root(str(root_candidate))
                        if rc:
                            candidates.append(rc)
                            candidates.append(rc.parent)
            except Exception:
                continue
    except Exception:
        pass

    # 5) fallback: scan HERE/datasets immediate children for index.json
    try:
        dd = (HERE / "datasets")
        if dd.exists():
            for child in dd.iterdir():
                if child.is_dir():
                    if (child / "index.json").exists():
                        candidates.append(child.resolve())
                    if (child / "openapi" / "index.json").exists():
                        candidates.append(child.resolve())
    except Exception:
        pass

    # Deduplicate preserving order
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        try:
            rp = str(c.resolve())
        except Exception:
            rp = str(c)
        if rp not in seen:
            seen.add(rp)
            uniq.append(Path(rp))

    # Look for index.json in common curated roots
    alt_names = ["index.json", "dataset_index.json", "version_meta.json"]
    for root in uniq:
        for alt in alt_names:
            p_alt = root / alt
            if p_alt.exists():
                print(f"DEBUG: located index file {p_alt}", file=sys.stderr)
                return root
        # also check typical curated variants
        for sub in ("curated_clean", "curated_noisy_light", "curated_noisy_heavy"):
            psub = root / sub / "index.json"
            if psub.exists():
                print(f"DEBUG: located index.json at {psub}", file=sys.stderr)
                return (root / sub).resolve()

    # rglob fallback
    try:
        dd = (HERE / "datasets")
        if dd.exists():
            for sub in dd.rglob("index.json"):
                print(f"DEBUG: found index.json via rglob: {sub}", file=sys.stderr)
                return sub.parent.resolve()
    except Exception:
        pass

    return None


# --- Normalize EFFECTIVE_CURATED_ROOT only (do not override INDEX_PATH) ---
INDEX_ROOT: Optional[Path] = None
PAIR_INDEX: Optional[Dict[str, Any]] = None

try:
    eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
    if eff:
        res = _resolve_candidate_root(str(eff))
        if res:
            server.EFFECTIVE_CURATED_ROOT = res
            print(
                f"DEBUG: server.EFFECTIVE_CURATED_ROOT resolved -> {server.EFFECTIVE_CURATED_ROOT} "
                f"(type={type(server.EFFECTIVE_CURATED_ROOT)})",
                file=sys.stderr,
            )
        else:
            fallback = (HERE / "datasets")
            if fallback.exists():
                server.EFFECTIVE_CURATED_ROOT = fallback.resolve()
                print(
                    f"DEBUG: server.EFFECTIVE_CURATED_ROOT forced -> {server.EFFECTIVE_CURATED_ROOT} "
                    f"(type={type(server.EFFECTIVE_CURATED_ROOT)})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"DEBUG: server.EFFECTIVE_CURATED_ROOT ({eff}) could not be resolved",
                    file=sys.stderr,
                )
    else:
        fallback = (HERE / "datasets")
        if fallback.exists():
            server.EFFECTIVE_CURATED_ROOT = fallback.resolve()
            print(
                f"DEBUG: server.EFFECTIVE_CURATED_ROOT set -> {server.EFFECTIVE_CURATED_ROOT} "
                f"(type={type(server.EFFECTIVE_CURATED_ROOT)})",
                file=sys.stderr,
            )
except Exception as e:
    print("WARN: error normalizing EFFECTIVE_CURATED_ROOT:", repr(e), file=sys.stderr)

try:
    idx_root = _find_index_json_under_datasets()
    if idx_root:
        idx_root = Path(idx_root).resolve()
        INDEX_ROOT = idx_root
        print(
            f"DEBUG: index.json root candidate={idx_root}, existing EFFECTIVE_CURATED_ROOT={getattr(server, 'EFFECTIVE_CURATED_ROOT', None)}",
            file=sys.stderr,
        )

        # Load pair index directly from curated_clean/index.json (or similar)
        index_path = idx_root / "index.json"
        if index_path.exists():
            try:
                txt = index_path.read_text(encoding="utf-8")
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    PAIR_INDEX = obj
                    print(
                        f"DEBUG: loaded PAIR_INDEX from {index_path} with {len(PAIR_INDEX)} entries",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"WARN: index.json at {index_path} is not a dict; type={type(obj)}",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(
                    f"WARN: failed to load PAIR_INDEX from {index_path}: {repr(e)}",
                    file=sys.stderr,
                )
        else:
            print(
                f"WARN: index.json expected at {index_path} but file does not exist",
                file=sys.stderr,
            )
    else:
        print(
            "WARN: Could not locate any index.json under candidate dataset roots. "
            "PAIR_INDEX will remain None; pair_id lookup may not work.",
            file=sys.stderr,
        )
except Exception as e:
    print("WARN: error while attempting to find index root:", repr(e), file=sys.stderr)


# GRAPH_PATH normalization (lightweight)
try:
    gp = getattr(server, "GRAPH_PATH", None)
    if gp:
        res_gp = _resolve_candidate_root(str(gp))
        if res_gp and res_gp.exists():
            server.GRAPH_PATH = str(res_gp)
            print(f"DEBUG: server.GRAPH_PATH resolved -> {server.GRAPH_PATH}", file=sys.stderr)
        else:
            cand1 = (HERE / "datasets" / "graph.json")
            cand2 = (REPO_ROOT / AI_CORE_DIR_ENV / "datasets" / "graph.json")
            if cand1.exists():
                server.GRAPH_PATH = str(cand1.resolve())
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            elif cand2.exists():
                server.GRAPH_PATH = str(cand2.resolve())
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            else:
                print(f"DEBUG: server.GRAPH_PATH ({gp}) could not be resolved", file=sys.stderr)
    else:
        cand = (HERE / "datasets" / "graph.json")
        if cand.exists():
            server.GRAPH_PATH = str(cand.resolve())
            print(f"DEBUG: server.GRAPH_PATH set -> {server.GRAPH_PATH}", file=sys.stderr)
except Exception as e:
    print("WARN: error normalizing GRAPH_PATH:", repr(e), file=sys.stderr)


# ----------------- git helpers -----------------
def git_changed_files() -> List[str]:
    code, out = run_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    _ = run_cmd(["git", "fetch", "origin", "main", "--depth=1"])
    code, out = run_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    code, out = run_cmd(["git", "diff", "--name-only", "HEAD~1..HEAD"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    code, out = run_cmd(["git", "ls-files", "--modified"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    return []


def git_show_file(ref_path: str) -> str:
    code, out = run_cmd(["git", "show", ref_path])
    return out if code == 0 else ""


def find_counterpart_v1(v2path: Path) -> Path:
    name = v2path.name
    if "--v2." in name:
        alt = name.replace("--v2.", "--v1.")
    elif "-v2." in name:
        alt = name.replace("-v2.", "-v1.")
    else:
        alt = name.replace("v2", "v1", 1)
    cand = v2path.parent / alt
    return cand


def _variant_folders() -> List[str]:
    try:
        return list(server.VARIANT_MAP.values())
    except Exception:
        return ["curated_clean", "curated_noisy_light", "curated_noisy_heavy"]


def _all_dataset_keys() -> List[str]:
    try:
        return server.all_dataset_keys()
    except Exception:
        return [
            "openapi",
            "openapi_noisy_light",
            "openapi_noisy_heavy",
            "petclinic",
            "petclinic_noisy_light",
            "petclinic_noisy_heavy",
            "openrewrite",
            "openrewrite_noisy_light",
            "openrewrite_noisy_heavy",
        ]


def find_local_v1_across_variants(v2path: Path) -> Path:
    cand = find_counterpart_v1(v2path)
    if cand.exists():
        return cand
    name_alt = cand.name
    # try dataset_paths
    for key in _all_dataset_keys():
        try:
            dp = server.dataset_paths(key)
            can = dp.get("canonical")
            if can:
                croot = _resolve_candidate_root(str(can))
                if croot:
                    p = croot / name_alt
                    if p.exists():
                        print(f"DEBUG: found v1 in dataset_paths[{key}] -> {p}", file=sys.stderr)
                        return p
        except Exception:
            continue
    # try variant folders
    for vf in _variant_folders():
        try:
            vroot = server._variant_dir_for(vf)
            rv = _resolve_candidate_root(str(vroot))
            if rv:
                p = rv / name_alt
                if p.exists():
                    return p
            for base in getattr(server, "BASE_DATASETS", []):
                p2 = (rv / base / "canonical") if rv else None
                if p2 and p2.exists():
                    candp = p2 / name_alt
                    if candp.exists():
                        return candp
        except Exception:
            continue
    # fallback under EFFECTIVE_CURATED_ROOT
    try:
        eff = getattr(server, "EFFECTIVE_CURATED_ROOT", "")
        if eff:
            effp = _resolve_candidate_root(str(eff))
            if effp:
                p = effp / name_alt
                if p.exists():
                    return p
                for sub in ("canonical", "ndjson", "metadata"):
                    p2 = effp / sub / name_alt
                    if p2.exists():
                        return p2
    except Exception:
        pass
    return cand


def read_json_file_if_exists(p: Path) -> Dict[str, Any]:
    try:
        if p and p.exists():
            txt = p.read_text(encoding="utf-8")
            try:
                return json.loads(txt)
            except Exception:
                try:
                    return server._load_json_or_yaml(p)
                except Exception:
                    return {}
        return {}
    except Exception:
        return {}


def dereference_components(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return spec
    comp = spec.get("components") or {}
    schemas = comp.get("schemas") or {}

    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                ref = obj["$ref"]
                if ref.startswith("#/components/schemas/"):
                    key = ref.split("/")[-1]
                    target = schemas.get(key)
                    if isinstance(target, dict):
                        return resolve(dict(target))
                    return target
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(x) for x in obj]
        return obj

    out = dict(spec)
    try:
        if "paths" in spec and isinstance(spec["paths"], dict):
            new_paths = {}
            for p, methods in spec["paths"].items():
                if not isinstance(methods, dict):
                    new_paths[p] = methods
                    continue
                new_methods = {}
                for m, op in methods.items():
                    new_methods[m] = resolve(op)
                new_paths[p] = new_methods
            out["paths"] = new_paths
    except Exception:
        pass
    try:
        out["components"] = dict(out.get("components") or {})
    except Exception:
        pass
    return out


def load_json_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        try:
            return server._load_json_or_yaml(Path(text))
        except Exception:
            return {}


# -------- pair_id lookup helper (index-based, independent of server.report) --------
def _load_pair_index_any(dataset_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    In CI, prefer the locally loaded PAIR_INDEX (from curated index.json).
    Fall back to server.load_pair_index only if PAIR_INDEX is missing.
    """
    global PAIR_INDEX

    if isinstance(PAIR_INDEX, dict) and PAIR_INDEX:
        print(
            f"DEBUG: _load_pair_index_any using PAIR_INDEX with {len(PAIR_INDEX)} entries",
            file=sys.stderr,
        )
        return PAIR_INDEX

    # Fallback: try server.load_pair_index
    idx = None
    src = "none"
    try:
        try:
            idx = server.load_pair_index()
            src = "global"
        except TypeError:
            idx = None
    except Exception as e:
        print("DEBUG: global load_pair_index() raised in helper:", repr(e), file=sys.stderr)
        idx = None

    if (not isinstance(idx, dict)) or (not idx):
        if dataset_key:
            try:
                try:
                    idx = server.load_pair_index(dataset_key)
                    src = f"dataset={dataset_key}"
                except TypeError:
                    idx = None
            except Exception as e:
                print(
                    f"DEBUG: dataset load_pair_index({dataset_key}) raised in helper:",
                    repr(e),
                    file=sys.stderr,
                )
                idx = None

    if isinstance(idx, dict) and idx:
        print(
            f"DEBUG: _load_pair_index_any loaded {len(idx)} entries from {src}",
            file=sys.stderr,
        )
        return idx

    print("DEBUG: _load_pair_index_any found no usable index", file=sys.stderr)
    return None


def lookup_pair_meta_for_relpath(
    relname: str, dataset_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Best-effort: infer pair metadata (pair_id, service_name, etc.)
    from index.json based on filename only.
    """
    if not relname:
        return None

    name = Path(relname).name.lower()
    idx = _load_pair_index_any(dataset_key)

    if not isinstance(idx, dict) or not idx:
        print(
            f"DEBUG: lookup_pair_meta_for_relpath: no index loaded for dataset={dataset_key}",
            file=sys.stderr,
        )
        return None

    for pid, meta in idx.items():
        try:
            mo_raw = meta.get("old_canonical") or meta.get("old")
            mn_raw = meta.get("new_canonical") or meta.get("new")
            if not (mo_raw and mn_raw):
                continue

            mo_name = Path(str(mo_raw)).name.lower()
            mn_name = Path(str(mn_raw)).name.lower()

            if name in (mo_name, mn_name):
                out = dict(meta)
                out.setdefault("pair_id", pid)
                print(
                    f"DEBUG: lookup_pair_meta_for_relpath matched rel={name} to pid={pid}",
                    file=sys.stderr,
                )
                return out

        except Exception as e:
            print("DEBUG: error iterating index entry in lookup_pair_meta:", repr(e), file=sys.stderr)

    print(f"DEBUG: lookup_pair_meta_for_relpath no match for {name}", file=sys.stderr)
    return None


# -------- ACE-based heuristic risk (model-independent) --------
BREAKING_TYPES = {
    "ENDPOINT_REMOVED",
    "RESPONSE_SCHEMA_CHANGED",
    "REQUESTBODY_SCHEMA_CHANGED",
    "PARAM_CHANGED",
    "PARAM_REMOVED",
    "PARAMETER_REMOVED",
    "RESPONSE_REMOVED",
}


def heuristic_risk_from_diffs(diffs: List[Dict[str, Any]]) -> float:
    """
    Simple, deterministic risk heuristic based purely on ACEs.

    - Endpoint removals / schema changes drive score toward 1.0
    - Non-breaking but numerous changes still give medium-ish scores
    """
    if not diffs:
        return 0.0

    total = len(diffs)
    types_upper = [str(d.get("type", "")).upper() for d in diffs]
    breaking = sum(1 for t in types_upper if t in BREAKING_TYPES)
    removed = sum(1 for t in types_upper if t == "ENDPOINT_REMOVED")

    # No changes? No risk.
    if total == 0:
        return 0.0

    score = 0.0

    if breaking or removed:
        # Strongly breaking changes – go high.
        score = 0.7 + 0.01 * min(total, 30)
    else:
        # Mostly additive / minor tweaks.
        score = 0.3 + 0.01 * min(total, 20)

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    return round(score, 3)


# -------- core pair analysis --------
def analyze_pair_files(
    old_doc: Dict[str, Any],
    new_doc: Dict[str, Any],
    rel_path: Optional[str] = None,
    dataset_hint: Optional[str] = None,
) -> Dict[str, Any]:
    # Try to get pair metadata (pair_id, service_name) upfront from index.
    pair_id_hint: Optional[str] = None
    service_from_index: Optional[str] = None
    if rel_path:
        meta = lookup_pair_meta_for_relpath(rel_path, dataset_hint)
        if meta:
            pair_id_hint = meta.get("pair_id")
            service_from_index = meta.get("service_name")

    # Try loading metadata from curated folder the same way server.report does
    service_from_meta: Optional[str] = None
    if not service_from_index and pair_id_hint:
        try:
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            effp = eff if isinstance(eff, Path) else Path(str(eff)) if eff else None
            if effp:
                meta_path = effp / "openapi" / "metadata" / f"{pair_id_hint}.json"
                if meta_path.exists():
                    m = json.loads(meta_path.read_text(encoding="utf-8"))
                    service_from_meta = m.get("service_name")
                    print(
                        f"DEBUG: loaded service from metadata for {pair_id_hint}: {service_from_meta}",
                        file=sys.stderr,
                    )
        except Exception as e:
            print("DEBUG: metadata load failed:", e, file=sys.stderr)

    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN: server.diff_openapi failed:", e, file=sys.stderr)
        diffs = []

    diffs_serial: List[Dict[str, Any]] = []
    for d in diffs:
        try:
            if hasattr(d, "dict"):
                dd = d.dict()
            else:
                dd = {
                    "type": getattr(d, "type", None),
                    "path": getattr(d, "path", None),
                    "method": getattr(d, "method", None),
                    "detail": getattr(d, "detail", None),
                    "ace_id": getattr(d, "ace_id", None),
                }
        except Exception:
            dd = {"type": str(d)}
        if "type" in dd and isinstance(dd["type"], str):
            dd["type"] = dd["type"].upper()
        diffs_serial.append(dd)

    # Call api_analyze for features / explanation,
    # but DO NOT trust its risk blindly (model might be missing).
    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy_raw = server.api_analyze(
            baseline=baseline_str,
            candidate=candidate_str,
            dataset=None,   # keep None so this path works even without model.joblib
            options=None,
        )
    except Exception as e:
        print("WARN: server.api_analyze raised:", e, file=sys.stderr)
        analy_raw = {
            "run_id": None,
            "predictions": [],
            "summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)},
        }

    if not isinstance(analy_raw, dict):
        analy_raw = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # Risk from model (if any)
    try:
        model_score = float((analy_raw.get("summary") or {}).get("service_risk", 0.0))
    except Exception:
        model_score = 0.0

    # Risk from heuristic (fallback only)
    heuristic_score = heuristic_risk_from_diffs(diffs_serial)

    # If model is available, trust it. Otherwise, fall back to heuristic.
    if model_score > 0:
        final_score = model_score
        source = "model"
    else:
        final_score = heuristic_score
        source = "heuristic"

    print(
        f"DEBUG: risk model={model_score:.3f} heuristic={heuristic_score:.3f} "
        f"final={final_score:.3f} source={source}",
        file=sys.stderr,
    )


    # enrichment: try to load graph if available
    g = None
    pfeats: Dict[str, Any] = {}
    be_imp: List[Dict[str, Any]] = []
    fe_imp: List[Dict[str, Any]] = []

    # Determine service name
    service_guess: Optional[str] = service_from_index or service_from_meta

    if not service_guess and rel_path:
        try:
            stem = Path(rel_path).stem
            if "--v" in stem:
                service_guess = stem.split("--v")[0]
            elif "-v" in stem:
                service_guess = stem.split("-v")[0]
            else:
                service_guess = stem
        except Exception:
            service_guess = None

    if not service_guess:
        try:
            service_guess = (new_doc.get("info", {}) or {}).get("title")
        except Exception:
            service_guess = "unknown"

    try:
        try:
            g = server.load_graph()
            print("DEBUG: server.load_graph() returned:", type(g), file=sys.stderr)
        except Exception as e:
            print("DEBUG: server.load_graph() raised:", e, file=sys.stderr)
            g = None

        changed_paths = [
            server.normalize_path(d.get("path"))
            for d in diffs_serial
            if d.get("path")
        ]

        print(
            f"DEBUG: service_guess used for graph = {service_guess}",
            file=sys.stderr,
        )
        print(
            f"DEBUG: changed_paths for impacts = {changed_paths}",
            file=sys.stderr,
        )

        if g is not None:
            pfeats = server.producer_features(g, service_guess)
            # first attempt: with changed_paths as filter
            be_imp = server.backend_impacts(g, service_guess, changed_paths)
            fe_imp = server.ui_impacts(g, service_guess, changed_paths)
            print(
                f"DEBUG: initial enriched impacts -> backend:{len(be_imp)} frontend:{len(fe_imp)}",
                file=sys.stderr,
            )

            # if totally empty, retry without path filter (show all known dependents)
            if not be_imp and not fe_imp:
                print(
                    "DEBUG: impacts empty with path filter; retrying without path filter",
                    file=sys.stderr,
                )
                be_imp = server.backend_impacts(g, service_guess, None)
                fe_imp = server.ui_impacts(g, service_guess, None)
                print(
                    f"DEBUG: fallback impacts (no path filter) -> backend:{len(be_imp)} frontend:{len(fe_imp)}",
                    file=sys.stderr,
                )

            print(
                f"DEBUG: final enriched impacts -> backend:{len(be_imp)} frontend:{len(fe_imp)}",
                file=sys.stderr,
            )
        else:
            print("DEBUG: skipping backend/ui impact heuristics (no graph)", file=sys.stderr)
    except Exception as e:
        print("WARN: enrichment impact computation failed:", e, file=sys.stderr)
        be_imp = []
        fe_imp = []

    ai_expl = (
        analy_raw.get("ai_explanation")
        or analy_raw.get("explanation")
        or ""
    )
    if not ai_expl:
        try:
            ai_expl = server.make_explanation(final_score, diffs, pfeats, {}, be_imp, fe_imp)
        except Exception:
            ai_expl = ""

    # Ensure we propagate a pair_id if index knew it,
    # even though api_analyze itself has no concept of pair_id.
    # Ensure we propagate a pair_id if index knew it,
    # even though api_analyze itself has no concept of pair_id.
    final_pair_id = pair_id_hint or analy_raw.get("pair_id") or ""

    # Attach deterministic ace_id if missing, same style as UI:
    #   <pair_id>::ace::<index>
    if final_pair_id:
        for idx, dd in enumerate(diffs_serial):
            if not dd.get("ace_id"):
                dd["ace_id"] = f"{final_pair_id}::ace::{idx}"

    return {
        "diffs": diffs_serial,
        "analyze": {
            "summary": {"service_risk": final_score, "num_aces": len(diffs_serial)},
            "backend_impacts": be_imp,
            "frontend_impacts": fe_imp,
            "ai_explanation": ai_expl,
            "pair_id": final_pair_id,
        },
    

    }


# ----------------- main -----------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", "unknown"))
    parser.add_argument("--output-full", default="pr-impact-full.json")
    parser.add_argument("--output-summary", default="pr-impact-report.json")
    args = parser.parse_args()

    changed = git_changed_files()
    print("CI: changed files:", changed, file=sys.stderr)

    api_files = [f for f in changed if f.lower().endswith((".json", ".yaml", ".yml"))]

    curated_tokens = ["datasets/curated", "canonical"]
    try:
        curated_tokens += list(server.VARIANT_MAP.values())
    except Exception:
        curated_tokens += ["curated_clean", "curated_noisy_light", "curated_noisy_heavy"]

    api_files = [
        f for f in api_files if any(tok in f for tok in curated_tokens) or "canonical" in f
    ] or api_files

    results: List[Dict[str, Any]] = []
    files_processed: List[str] = []

    for rel in api_files:
        print("DEBUG: rel_path =", rel, file=sys.stderr)
        p = Path(rel)
        files_processed.append(rel)

        dataset_hint = None
        lrel = rel.lower()
        if "/openapi/" in lrel or "openapi--" in lrel or "/openapi_" in lrel:
            dataset_hint = "openapi"
        elif "/petclinic/" in lrel or "petclinic--" in lrel:
            dataset_hint = "petclinic"
        elif "/openrewrite/" in lrel or "openrewrite--" in lrel:
            dataset_hint = "openrewrite"

        try:
            if p.exists():
                new_doc = read_json_file_if_exists(p)
                v1cand = find_local_v1_across_variants(p)
                print(f"DEBUG: v1cand guessed as {v1cand}", file=sys.stderr)
                if v1cand.exists():
                    print(f"DEBUG: found local v1 counterpart at {v1cand}", file=sys.stderr)
                    old_doc = read_json_file_if_exists(v1cand)
                else:
                    blob = git_show_file(f"origin/main:{rel}")
                    if blob:
                        old_doc = load_json_text(blob)
                    else:
                        code, out = run_cmd(["git", "show", "HEAD~1:" + rel])
                        old_doc = load_json_text(out) if code == 0 else {}
            else:
                ref_branch = (
                    os.environ.get("GITHUB_REF_NAME", None)
                    or os.environ.get("BRANCH", None)
                    or "HEAD"
                )
                blob_new = (
                    git_show_file(f"origin/{ref_branch}:{rel}")
                    or git_show_file(f"HEAD:{rel}")
                    or git_show_file(f"origin/main:{rel}")
                )
                new_doc = load_json_text(blob_new)
                blob_old = git_show_file(f"origin/main:{rel}")
                old_doc = load_json_text(blob_old) if blob_old else {}
        except Exception as e:
            print(f"WARN: failed to load files for {rel}: {e}", file=sys.stderr)
            old_doc, new_doc = {}, {}

        try:
            pair_res = analyze_pair_files(
                old_doc,
                new_doc,
                rel_path=rel,
                dataset_hint=dataset_hint,
            )
        except Exception as e:
            print("WARN: analyze_pair_files exception:", e, file=sys.stderr)
            pair_res = {
                "diffs": [],
                "analyze": {"summary": {"service_risk": 0.0, "num_aces": 0}, "predictions": []},
            }

        results.append({"file": rel, "result": pair_res})

    full_out = {
        "status": "ok" if results else "partial",
        "pr": str(args.pr),
        "files_changed": files_processed,
        "files_analyzed": len(results),
        "entries": results,
    }
    Path(args.output_full).write_text(
        json.dumps(full_out, indent=2),
        encoding="utf-8",
    )
    print("Wrote full output to", args.output_full, file=sys.stderr)

    max_risk = 0.0
    atomic_aces: List[Dict[str, Any]] = []
    pair_id_top = None
    ai_expl_top = None
    for e in results:
        analy = e["result"].get("analyze", {}) or {}
        summary = analy.get("summary") or {}
        try:
            s_r = float(summary.get("service_risk", 0.0))
        except Exception:
            s_r = 0.0
        max_risk = max(max_risk, s_r)
        for d in e["result"].get("diffs", []):
            if isinstance(d.get("type"), str):
                d["type"] = d["type"].upper()
            atomic_aces.append(d)
        if not pair_id_top:
            pair_id_top = analy.get("pair_id")
        if not ai_expl_top:
            ai_expl_top = analy.get("ai_explanation") or analy.get("explanation")

    def _band_label(score: float) -> Tuple[str, str]:
        if score >= 0.7:
            return "High", "BLOCK"
        if score >= 0.4:
            return "Medium", "WARN"
        return "Low", "PASS"

    band, level = _band_label(max_risk)

    compact = {
        "status": full_out["status"],
        "pr": str(args.pr),
        "files_changed": files_processed,
        "api_files_changed": files_processed,
        "atomic_change_events": atomic_aces,
        "impact_assessment": {
            "score": round(float(max_risk), 3),
            "label": "high"
            if max_risk >= 0.6
            else "medium"
            if max_risk >= 0.25
            else "low",
            "breaking_count": sum(
                1
                for a in atomic_aces
                if a.get("type", "").upper()
                in (
                    "PARAM_CHANGED",
                    "RESPONSE_SCHEMA_CHANGED",
                    "REQUESTBODY_SCHEMA_CHANGED",
                    "ENDPOINT_REMOVED",
                )
            ),
            "total_aces": len(atomic_aces),
        },
        "risk_score": round(float(max_risk), 3),
        "risk_band": band,
        "risk_level": level,
        "ai_explanation": ai_expl_top or "",
        "pair_id": pair_id_top or "",
        "metadata": {"pair_id": pair_id_top or ""},
    }

    Path(args.output_summary).write_text(
        json.dumps(compact, indent=2),
        encoding="utf-8",
    )
    print("Wrote summary to", args.output_summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
