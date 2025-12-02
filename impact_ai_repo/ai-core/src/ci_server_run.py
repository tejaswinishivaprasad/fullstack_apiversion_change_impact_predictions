#!/usr/bin/env python3
"""
ci_server_run.py

Lightweight CI wrapper that:
 - finds changed files in a PR (git)
 - maps canonical v2 -> v1 or fetches base from origin/main
 - lightly dereferences local '#/components/schemas/...' refs (shallow)
 - invokes server.report(...) (preferred) or server.api_analyze(...) (fallback)
 - writes pr-impact-full.json (detailed) and pr-impact-report.json (compact)

This version includes robust Path coercion for server module variables
and extra DEBUG logging to explain why backend/frontend impacts may be empty.
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

# --- BEGIN: robust path coercion helpers (ensures server expects Path objects) ---
def _coerce_to_path(obj: Optional[Any]) -> Optional[Path]:
    if obj is None:
        return None
    if isinstance(obj, Path):
        return obj
    try:
        return Path(str(obj)).resolve()
    except Exception:
        return None

def _set_server_root_paths(root_path: Optional[Any]):
    """
    Ensure server.EFFECTIVE_CURATED_ROOT and module-level constants are Path objects.
    Also update INDEX_PATH / VERSION_META_PATH / DATASET_OINDEX_PATH / VERSION_PAIRS_CSV
    so server.load_pair_index() and friends use the discovered container immediately.
    """
    rp = _coerce_to_path(root_path)
    if not rp:
        return
    try:
        server.EFFECTIVE_CURATED_ROOT = rp
        print(f"DEBUG: _set_server_root_paths -> server.EFFECTIVE_CURATED_ROOT set to {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
    except Exception as e:
        print("WARN: failed to set server.EFFECTIVE_CURATED_ROOT via assignment:", e, file=sys.stderr)
        try:
            setattr(server, "EFFECTIVE_CURATED_ROOT", rp)
        except Exception:
            pass
    # update related module-level derived paths (best-effort)
    try:
        setattr(server, "INDEX_PATH", rp / "index.json")
        setattr(server, "VERSION_META_PATH", rp / "version_meta.json")
        setattr(server, "DATASET_OINDEX_PATH", rp / "dataset_oindex.json")
        setattr(server, "VERSION_PAIRS_CSV", rp / "version_pairs.csv")
        print(f"DEBUG: module-level INDEX_PATH set -> {getattr(server, 'INDEX_PATH', None)}", file=sys.stderr)
    except Exception as e:
        print("WARN: failed updating module-level derived paths:", e, file=sys.stderr)

def _set_server_graph_path(graph_file: Optional[Any]):
    """Ensure server.GRAPH_PATH is a Path pointing to a file (or a Path-like)."""
    gp = _coerce_to_path(graph_file)
    if not gp:
        return
    try:
        server.GRAPH_PATH = gp
        print(f"DEBUG: _set_server_graph_path -> server.GRAPH_PATH set to {server.GRAPH_PATH}", file=sys.stderr)
    except Exception as e:
        print("WARN: failed to set server.GRAPH_PATH via assignment:", e, file=sys.stderr)
        try:
            setattr(server, "GRAPH_PATH", gp)
        except Exception:
            pass
# --- END helpers ---

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
    Returns path to the directory that should be used as EFFECTIVE_CURATED_ROOT
    (i.e., directory that contains index.json).
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
                for root_candidate in (dp.get("base"), dp.get("canonical"), dp.get("ndjson"), dp.get("metadata")):
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
    uniq = []
    for c in candidates:
        try:
            rp = str(c.resolve())
        except Exception:
            rp = str(c)
        if rp not in seen:
            seen.add(rp)
            uniq.append(Path(rp))

    for root in uniq:
        idx = root / "index.json"
        if idx.exists():
            print(f"DEBUG: located index.json at {idx}", file=sys.stderr)
            return root
        # alternative index filenames
        alt_names = ["dataset_index.json", "index.json", "version_meta.json", "dataset_oindex.json", "dataset_oindex.json"]
        for alt in alt_names:
            p_alt = root / alt
            if p_alt.exists():
                print(f"DEBUG: located alternative index file {p_alt}", file=sys.stderr)
                return root
        for sub in ("curated_clean", "curated_noisy_light", "curated_noisy_heavy"):
            psub = root / sub / "index.json"
            if psub.exists():
                print(f"DEBUG: located index.json at {psub}", file=sys.stderr)
                return (root / sub).resolve()

    # As last attempt, search one level deep under HERE/datasets for index.json
    try:
        dd = (HERE / "datasets")
        if dd.exists():
            for sub in dd.rglob("index.json"):
                print(f"DEBUG: found index.json via rglob: {sub}", file=sys.stderr)
                return sub.parent.resolve()
    except Exception:
        pass

    return None

# --- Try to normalize EFFECTIVE_CURATED_ROOT and GRAPH_PATH using Path helpers ---
try:
    eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
    if eff:
        res = _resolve_candidate_root(str(eff))
        if res:
            _set_server_root_paths(res)
            print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT resolved -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
        else:
            # set to HERE/datasets if present (best-effort)
            fallback = (HERE / "datasets")
            if fallback.exists():
                _set_server_root_paths(fallback.resolve())
                print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT forced -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
            else:
                print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT ({eff}) could not be resolved", file=sys.stderr)
    else:
        fallback = (HERE / "datasets")
        if fallback.exists():
            _set_server_root_paths(fallback.resolve())
            print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT set -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
except Exception as e:
    print("WARN: error normalizing EFFECTIVE_CURATED_ROOT:", e, file=sys.stderr)

# locate index.json and set EFFECTIVE_CURATED_ROOT to container that has it (so load_pair_index() works)
try:
    idx_root = _find_index_json_under_datasets()
    if idx_root:
        _set_server_root_paths(idx_root)
        print(f"DEBUG: enforcing server.EFFECTIVE_CURATED_ROOT -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
        try:
            idx = server.load_pair_index()
            if isinstance(idx, dict):
                print(f"DEBUG: server.load_pair_index() returned {len(idx)} entries", file=sys.stderr)
            else:
                print(f"DEBUG: server.load_pair_index() returned type {type(idx)}", file=sys.stderr)
        except Exception as e:
            print("WARN: server.load_pair_index() raised after enforce:", e, file=sys.stderr)
    else:
        print("WARN: Could not locate any index.json under candidate dataset roots. server.report may fail.", file=sys.stderr)
except Exception as e:
    print("WARN: error while attempting to find/set index root:", e, file=sys.stderr)

# GRAPH_PATH normalization (same idea as earlier)
try:
    gp = getattr(server, "GRAPH_PATH", None)
    if gp:
        res_gp = _resolve_candidate_root(str(gp))
        if res_gp and res_gp.exists():
            _set_server_graph_path(res_gp)
            print(f"DEBUG: server.GRAPH_PATH resolved -> {server.GRAPH_PATH}", file=sys.stderr)
        else:
            cand1 = (HERE / "datasets" / "graph.json")
            cand2 = (REPO_ROOT / AI_CORE_DIR_ENV / "datasets" / "graph.json")
            # prefer any graph.json we can find
            if cand1.exists():
                _set_server_graph_path(cand1)
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            elif cand2.exists():
                _set_server_graph_path(cand2)
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            else:
                print(f"DEBUG: server.GRAPH_PATH ({gp}) could not be resolved", file=sys.stderr)
    else:
        cand = (HERE / "datasets" / "graph.json")
        if cand.exists():
            _set_server_graph_path(cand)
            print(f"DEBUG: server.GRAPH_PATH set -> {server.GRAPH_PATH}", file=sys.stderr)
except Exception as e:
    print("WARN: error normalizing GRAPH_PATH:", e, file=sys.stderr)

# Extra diagnostic: if server.GRAPH_PATH points to a file, try to parse and print a quick summary
try:
    gp = getattr(server, "GRAPH_PATH", None)
    if gp:
        gp_path = _coerce_to_path(gp)
        if gp_path and gp_path.exists():
            try:
                with gp_path.open("r", encoding="utf-8") as fh:
                    gjson = json.load(fh)
                n_edges = len(gjson.get("edges", [])) if isinstance(gjson, dict) else None
                print(f"DEBUG: graph.json loaded from {gp_path} — edges={n_edges}", file=sys.stderr)
            except Exception as e:
                print("WARN: failed to parse server.GRAPH_PATH graph.json:", e, file=sys.stderr)
    else:
        print("DEBUG: server.GRAPH_PATH not set (no graph.json available)", file=sys.stderr)
except Exception as e:
    print("WARN: graph.json diagnostic step failed:", e, file=sys.stderr)

# ----------------- rest of wrapper (unchanged except more logging) -----------------
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
        return ["openapi", "openapi_noisy_light", "openapi_noisy_heavy",
                "petclinic", "petclinic_noisy_light", "petclinic_noisy_heavy",
                "openrewrite", "openrewrite_noisy_light", "openrewrite_noisy_heavy"]

def find_local_v1_across_variants(v2path: Path) -> Path:
    cand = find_counterpart_v1(v2path)
    if cand.exists():
        return cand
    name_alt = cand.name
    # try dataset_paths canonical roots
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
    # variant dir helpers
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

def analyze_pair_files(old_doc: Dict[str, Any], new_doc: Dict[str, Any],
                       rel_path: Optional[str] = None, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
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

    diffs_serial = []
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

def try_call_report(dataset_key: str, relname: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Robust attempts to call server.report for a dataset file candidate.
    Tries:
      1) call server.report(dataset, old_name, new_name)
      2) call server.report with absolute canonical paths
      3) lookup pair_id in index and call server.report(..., pair_id=pid)

    Emits verbose DEBUG lines to stderr so CI logs make the failure mode obvious.
    Returns the same structure expected by the wrapper or None if all attempts fail.
    NOTE: relies on `diffs_serial` being available in the outer scope (same as original wrapper).
    """
    if not dataset_key:
        print("DEBUG: try_call_report called with empty dataset_key", file=sys.stderr)
        return None

    try:
        # dataset canonical/candidate roots
        dp = server.dataset_paths(dataset_key)
        canonical_root = dp.get("canonical")
        print(f"DEBUG: dataset_paths({dataset_key}) -> {dp}", file=sys.stderr)

        # resolve canonical root to an absolute Path where possible
        croot = None
        if canonical_root:
            croot = _resolve_candidate_root(str(canonical_root))
        if not croot:
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            if eff:
                attempt = _resolve_candidate_root(str(eff))
                if attempt and canonical_root:
                    try:
                        joined = (attempt / str(canonical_root)).resolve()
                        if joined.exists():
                            croot = joined
                    except Exception:
                        pass

        if croot:
            print(f"DEBUG: canonical_root resolved to: {croot}", file=sys.stderr)
        else:
            print(f"DEBUG: canonical_root could not be resolved for dataset {dataset_key} (candidate: {canonical_root})", file=sys.stderr)

        # build old/new names
        new_name = Path(relname).name if relname else None
        old_name = None
        if new_name:
            if "--v2." in new_name:
                old_name = new_name.replace("--v2.", "--v1.")
            elif "-v2." in new_name:
                old_name = new_name.replace("-v2.", "-v1.")
            else:
                old_name = new_name.replace("v2", "v1", 1)

        # candidate roots to check
        candidate_roots = []
        if croot:
            candidate_roots.append(croot)
        eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
        if eff:
            effp = _resolve_candidate_root(str(eff))
            if effp:
                candidate_roots.append(effp)
        ad = (HERE / "datasets")
        if ad.exists():
            candidate_roots.append(ad.resolve())
        ar = (REPO_ROOT / AI_CORE_DIR_ENV / "datasets")
        if ar.exists():
            candidate_roots.append(ar.resolve())
        # include canonical_root raw value if it's an existing path
        try:
            if canonical_root:
                cand_raw = Path(str(canonical_root))
                if cand_raw.exists():
                    candidate_roots.append(cand_raw.resolve())
        except Exception:
            pass

        # dedupe preserving order
        seen = set()
        candidate_roots_n = []
        for r in candidate_roots:
            try:
                rp = str(r.resolve())
            except Exception:
                rp = str(r)
            if rp not in seen:
                seen.add(rp)
                candidate_roots_n.append(Path(rp))
        candidate_roots = candidate_roots_n

        # Helper: attempt a server.report call and return dict or None
        def _call_report(old_arg, new_arg, pair_id_arg=None):
            try:
                print(f"DEBUG: attempting server.report(dataset={dataset_key}, old={old_arg}, new={new_arg}, pair_id={pair_id_arg})", file=sys.stderr)
                report_obj = server.report(dataset=dataset_key, old=old_arg, new=new_arg, pair_id=pair_id_arg)
                repd = report_obj.dict() if hasattr(report_obj, "dict") else dict(report_obj)
                print("DEBUG: server.report call succeeded", file=sys.stderr)
                return repd
            except Exception as e:
                # show short traceback-like message without flooding logs
                try:
                    msg = str(e)
                except Exception:
                    msg = "Exception (unstringifiable)"
                print(f"DEBUG: server.report(...) raised: {msg}", file=sys.stderr)
                return None

        # Try candidate roots for co-located canonical files
        for root in candidate_roots:
            new_path = (root / new_name) if new_name else None
            old_path = (root / old_name) if old_name else None
            new_exists = new_path.exists() if new_path else False
            old_exists = old_path.exists() if old_path else False
            print(f"DEBUG: checking root {root} -> new_exists: {new_exists} old_exists: {old_exists}", file=sys.stderr)

            if new_exists and old_exists:
                print(f"DEBUG: found both canonical files under {root}", file=sys.stderr)

                # Attempt 1: name-only (legacy)
                repd = None
                if old_name and new_name:
                    repd = _call_report(old_name, new_name, None)

                # Attempt 2: absolute/resolved paths (some server variants accept these)
                if repd is None and old_path and new_path:
                    try:
                        abs_old = str(old_path.resolve())
                        abs_new = str(new_path.resolve())
                        repd = _call_report(abs_old, abs_new, None)
                    except Exception as e:
                        print("DEBUG: failed to resolve absolute path for attempt:", e, file=sys.stderr)
                        repd = None

                # Attempt 3: use pair-index to call by pair_id
                if repd is None:
                    try:
                        idx = server.load_pair_index(dataset_key)
                        print(f"DEBUG: load_pair_index returned entries={len(idx) if isinstance(idx, dict) else 'unknown'}", file=sys.stderr)
                        found_pid = None
                        if isinstance(idx, dict):
                            for pid, meta in idx.items():
                                mo = str(meta.get("old_canonical") or meta.get("old") or "")
                                mn = str(meta.get("new_canonical") or meta.get("new") or "")
                                mo_name = Path(mo).name if mo else ""
                                mn_name = Path(mn).name if mn else ""
                                try:
                                    # prefer exact filename matches
                                    if mo_name and mn_name and old_name and new_name and mo_name.lower() == old_name.lower() and mn_name.lower() == new_name.lower():
                                        found_pid = pid
                                        break
                                except Exception:
                                    pass
                                # also try resolving meta paths into same root
                                try:
                                    if mo and mn and old_path and new_path:
                                        mo_path = _resolve_candidate_root(mo) or (Path(str(root)) / mo_name)
                                        mn_path = _resolve_candidate_root(mn) or (Path(str(root)) / mn_name)
                                        if mo_path.exists() and mn_path.exists():
                                            if mo_path.resolve() == old_path.resolve() and mn_path.resolve() == new_path.resolve():
                                                found_pid = pid
                                                break
                                except Exception:
                                    pass
                        if found_pid:
                            print(f"DEBUG: matched pair_id {found_pid} from pair-index; attempting server.report with pair_id", file=sys.stderr)
                            repd = _call_report(None, None, found_pid)
                        else:
                            print("DEBUG: no matching pair_id found in pair-index for these old/new names", file=sys.stderr)
                    except Exception as e:
                        print("DEBUG: load_pair_index() attempt failed:", e, file=sys.stderr)

                # If we got a report, normalize and return
                if repd is not None:
                    be_imp = repd.get("backend_impacts") or []
                    fe_imp = repd.get("frontend_impacts") or []
                    print(f"DEBUG: server.report returned - backend_impacts_count={len(be_imp)} frontend_impacts_count={len(fe_imp)}", file=sys.stderr)
                    try:
                        print("DEBUG: sample backend_impacts:", json.dumps(be_imp[:4], default=str), file=sys.stderr)
                    except Exception:
                        print("DEBUG: backend_impacts present but failed to serialize sample", file=sys.stderr)
                    try:
                        print("DEBUG: sample frontend_impacts:", json.dumps(fe_imp[:4], default=str), file=sys.stderr)
                    except Exception:
                        print("DEBUG: frontend_impacts present but failed to serialize sample", file=sys.stderr)

                    ai_expl = repd.get("ai_explanation") or repd.get("explanation") or ""
                    pair_id = repd.get("metadata", {}).get("pair_id") or repd.get("pair_id") or ""
                    summary = repd.get("summary") or {}
                    service_risk = 0.0
                    try:
                        if isinstance(summary, dict):
                            service_risk = float(summary.get("service_risk", repd.get("risk_score", 0.0)))
                        else:
                            service_risk = float(repd.get("risk_score", 0.0))
                    except Exception:
                        service_risk = float(repd.get("risk_score", 0.0) or 0.0)

                    return {
                        "diffs": diffs_serial,
                        "analyze": {
                            "summary": {"service_risk": service_risk, "num_aces": len(diffs_serial)},
                            "backend_impacts": be_imp,
                            "frontend_impacts": fe_imp,
                            "ai_explanation": ai_expl,
                            "pair_id": pair_id,
                            "versioning": repd.get("versioning") or {},
                            "metadata": repd.get("metadata") or {}
                        }
                    }
                else:
                    print("WARN: server.report attempts (name-only, abs-path, pair-id) all failed for this canonical root; trying next root", file=sys.stderr)

        # exhausted candidate roots
        print("DEBUG: exhausted candidate roots for try_call_report without success", file=sys.stderr)

    except Exception as e:
        print("DEBUG: try_call_report encountered unexpected exception:", e, file=sys.stderr)

    return None


def main():
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

    api_files = [f for f in api_files if any(tok in f for tok in curated_tokens) or "canonical" in f] or api_files

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
                ref_branch = os.environ.get("GITHUB_REF_NAME", None) or os.environ.get("BRANCH", None) or "HEAD"
                blob_new = git_show_file(f"origin/{ref_branch}:{rel}") or git_show_file(f"HEAD:{rel}") or git_show_file(f"origin/main:{rel}")
                new_doc = load_json_text(blob_new)
                blob_old = git_show_file(f"origin/main:{rel}")
                old_doc = load_json_text(blob_old) if blob_old else {}
        except Exception as e:
            print(f"WARN: failed to load files for {rel}: {e}", file=sys.stderr)
            old_doc, new_doc = {}, {}

        try:
            pair_res = analyze_pair_files(old_doc, new_doc, rel_path=rel, dataset_hint=dataset_hint)
        except Exception as e:
            print("WARN: analyze_pair_files exception:", e, file=sys.stderr)
            pair_res = {"diffs": [], "analyze": {"summary": {"service_risk": 0.0, "num_aces": 0}, "predictions": []}}

        results.append({"file": rel, "result": pair_res})

    full_out = {
        "status": "ok" if results else "partial",
        "pr": str(args.pr),
        "files_changed": files_processed,
        "files_analyzed": len(results),
        "entries": results
    }
    Path(args.output_full).write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print("Wrote full output to", args.output_full, file=sys.stderr)

    max_risk = 0.0
    atomic_aces = []
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
            pair_id_top = analy.get("pair_id") or (analy.get("versioning") or {}).get("pair_id") or (analy.get("metadata") or {}).get("pair_id")
        if not ai_expl_top:
            ai_expl_top = analy.get("ai_explanation") or analy.get("explanation")

    def _band_label(score: float):
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
            "label": "high" if max_risk >= 0.6 else "medium" if max_risk >= 0.25 else "low",
            "breaking_count": sum(1 for a in atomic_aces if a.get("type", "").lower() in ("param_changed","response_schema_changed","requestbody_schema_changed","endpoint_removed")),
            "total_aces": len(atomic_aces)
        },
        "risk_score": round(float(max_risk), 3),
        "risk_band": band,
        "risk_level": level,
        "ai_explanation": ai_expl_top or "",
        "pair_id": pair_id_top or "",
        "metadata": {"pair_id": pair_id_top or ""}
    }

    Path(args.output_summary).write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print("Wrote summary to", args.output_summary, file=sys.stderr)
    return 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
