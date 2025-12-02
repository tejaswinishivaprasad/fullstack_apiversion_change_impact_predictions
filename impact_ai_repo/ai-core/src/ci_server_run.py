#!/usr/bin/env python3
"""
ci_server_run.py

Lightweight CI wrapper that:
 - finds changed files in a PR (git)
 - maps canonical v2 -> v1 or fetches base from origin/main
 - lightly dereferences local '#/components/schemas/...' refs (shallow)
 - invokes server.report(...) (preferred) or server.api_analyze(...) (fallback)
 - writes pr-impact-full.json (detailed) and pr-impact-report.json (compact)
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Helpers to determine repo / ai-core locations early ---
REPO_ROOT = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve()
# AI_CORE_DIR env (as used by your workflow) is relative to repo root in CI
AI_CORE_DIR_ENV = os.environ.get("AI_CORE_DIR", "impact_ai_repo/ai-core/src")
AI_CORE_SRC = (REPO_ROOT / AI_CORE_DIR_ENV).resolve()

# Ensure we can import server.py from the same directory (ai-core/src)
HERE = AI_CORE_SRC if AI_CORE_SRC.exists() else Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import server  # noqa: E402
    # Minimal info printed early so logs show import path
    print(f"DEBUG: imported server from {getattr(server, '__file__', 'unknown')}", file=sys.stderr)
    # Show dataset_paths sample
    try:
        print("DEBUG: dataset_paths(openapi) =", server.dataset_paths("openapi"), file=sys.stderr)
    except Exception:
        print("DEBUG: server.dataset_paths('openapi') failed", file=sys.stderr)
    print("DEBUG: CWD =", os.getcwd(), file=sys.stderr)
except Exception as e:
    print("ERROR: failed to import server.py:", e, file=sys.stderr)
    raise

# --- Ensure server uses absolute dataset root / graph path if possible ---
def _resolve_candidate_root(candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    p = Path(candidate)
    # already absolute
    if p.is_absolute() and p.exists():
        return p
    # try relative to ai-core/src (HERE)
    p2 = (HERE / candidate).resolve()
    if p2.exists():
        return p2
    # try relative to repo root (maybe effect of running from repo root)
    p3 = (REPO_ROOT / candidate).resolve()
    if p3.exists():
        return p3
    # try if candidate is a simple folder under datasets in ai-core
    p4 = (HERE / "datasets" / candidate).resolve()
    if p4.exists():
        return p4
    # last ditch: sibling of HERE (in case script runs from ai-core/src)
    p5 = (HERE.parent / candidate).resolve()
    if p5.exists():
        return p5
    return None

# Try to normalize server.EFFECTIVE_CURATED_ROOT and server.GRAPH_PATH to absolute paths
try:
    eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
    if eff:
        resolved = _resolve_candidate_root(str(eff))
        if resolved:
            server.EFFECTIVE_CURATED_ROOT = str(resolved)
            print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT resolved -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
        else:
            # if not resolvable, try setting it to ai-core/src/datasets if that exists
            fallback = (HERE / "datasets")
            if fallback.exists():
                server.EFFECTIVE_CURATED_ROOT = str(fallback.resolve())
                print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT forced -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
            else:
                print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT ({eff}) could not be resolved", file=sys.stderr)
    else:
        # set to HERE/datasets if available
        fallback = (HERE / "datasets")
        if fallback.exists():
            server.EFFECTIVE_CURATED_ROOT = str(fallback.resolve())
            print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT set -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
except Exception as e:
    print("WARN: error while normalizing EFFECTIVE_CURATED_ROOT:", e, file=sys.stderr)

try:
    gp = getattr(server, "GRAPH_PATH", None)
    if gp:
        resolved_gp = _resolve_candidate_root(str(gp))
        if resolved_gp and resolved_gp.exists():
            server.GRAPH_PATH = str(resolved_gp)
            print(f"DEBUG: server.GRAPH_PATH resolved -> {server.GRAPH_PATH}", file=sys.stderr)
        else:
            # try ai-core/src/datasets/graph.json
            candidate = (HERE / "datasets" / "graph.json")
            if candidate.exists():
                server.GRAPH_PATH = str(candidate.resolve())
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            else:
                print(f"DEBUG: server.GRAPH_PATH ({gp}) could not be resolved", file=sys.stderr)
    else:
        candidate = (HERE / "datasets" / "graph.json")
        if candidate.exists():
            server.GRAPH_PATH = str(candidate.resolve())
            print(f"DEBUG: server.GRAPH_PATH set -> {server.GRAPH_PATH}", file=sys.stderr)
except Exception as e:
    print("WARN: error while normalizing GRAPH_PATH:", e, file=sys.stderr)

# --- existing functions (mostly unchanged) ---
def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return (0, out.decode())
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.output.decode() if e.output else "")
    except Exception as e:
        return (1, str(e))

def git_changed_files() -> List[str]:
    # Primary attempt
    code, out = run_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    # try fetching origin/main and retry (helps shallow checkouts)
    _ = run_cmd(["git", "fetch", "origin", "main", "--depth=1"])
    code, out = run_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    # fallback local
    code, out = run_cmd(["git", "diff", "--name-only", "HEAD~1..HEAD"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    # last resort
    code, out = run_cmd(["git", "ls-files", "--modified"])
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    return []

def git_show_file(ref_path: str) -> str:
    # ref_path example: origin/main:impact_ai_repo/...
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
    # Try dataset_paths returned roots; resolve them robustly
    for key in _all_dataset_keys():
        try:
            dp = server.dataset_paths(key)
            can = dp.get("canonical")
            if can:
                # try to resolve canonical relative/absolute using our helper
                croot = _resolve_candidate_root(str(can))
                if croot:
                    p = croot / name_alt
                    if p.exists():
                        print(f"DEBUG: found v1 in dataset_paths[{key}] -> {p}", file=sys.stderr)
                        return p
        except Exception:
            continue
    # try server helpers / variant folders
    for vf in _variant_folders():
        try:
            vroot = server._variant_dir_for(vf)
            rv = _resolve_candidate_root(str(vroot))
            if rv:
                p = rv / name_alt
                if p.exists():
                    return p
            # fallback check inside variant/base/dataset/canonical
            for base in getattr(server, "BASE_DATASETS", []):
                p2 = (rv / base / "canonical") if rv else None
                if p2 and p2.exists():
                    candp = p2 / name_alt
                    if candp.exists():
                        return candp
        except Exception:
            continue
    # check EFFECTIVE_CURATED_ROOT if provided
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
    return cand  # last best guess (may not exist)

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
            # rarely used fallback
            return server._load_json_or_yaml(Path(text))
        except Exception:
            return {}

# ------------- FIXED analyze_pair_files ----------------
def analyze_pair_files(
    old_doc: Dict[str, Any],
    new_doc: Dict[str, Any],
    rel_path: Optional[str] = None,
    dataset_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prefer calling server.report(dataset, old_name, new_name, pair_id) so backend/frontend
    impacts and version metadata are filled. If dataset_hint is provided, we force
    resolution using that dataset. Otherwise attempt automatic resolution; finally
    fallback to server.api_analyze.
    """
    # shallow deref for better diff detection
    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    # compute diffs
    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN: server.diff_openapi failed:", e, file=sys.stderr)
        diffs = []

    # serialize diffs
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

    # helper to try to find canonical filenames and call server.report
    def try_call_report(dataset_key: str, relname: Optional[str]) -> Optional[Dict[str, Any]]:
        # relname is the filename (e.g., openapi--catalog-service--...--v2.canonical.json)
        if not dataset_key:
            return None
        try:
            dp = server.dataset_paths(dataset_key)
            canonical_root = dp.get("canonical")
            print(f"DEBUG: dataset_paths({dataset_key}) -> {dp}", file=sys.stderr)
            # Resolve canonical_root robustly to an absolute Path
            croot = _resolve_candidate_root(str(canonical_root)) if canonical_root else None
            if not croot:
                # try server.EFFECTIVE_CURATED_ROOT + canonical_root (if canonical_root was relative)
                eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
                if eff:
                    attempt = _resolve_candidate_root(str(eff))
                    if attempt and canonical_root:
                        # if canonical_root was relative fragment, join
                        joined = (attempt / str(canonical_root)).resolve()
                        if joined.exists():
                            croot = joined
            if not croot:
                print(f"DEBUG: canonical_root could not be resolved for dataset {dataset_key} (candidate: {canonical_root})", file=sys.stderr)
                # continue — we'll attempt other heuristics below
            else:
                print(f"DEBUG: canonical_root resolved to: {croot}", file=sys.stderr)

            # try the filename directly at canonical_root
            new_name = Path(relname).name if relname else None
            old_name = None
            # attempt to get v1 candidate by replacing v2 -> v1 in the filename
            if new_name:
                if "--v2." in new_name:
                    old_name = new_name.replace("--v2.", "--v1.")
                elif "-v2." in new_name:
                    old_name = new_name.replace("-v2.", "-v1.")
                else:
                    # best-effort: replace last '-v2' occurrence
                    old_name = new_name.replace("v2", "v1", 1)

            # candidate checks across a few roots: canonical_root resolved, HERE/datasets, REPO_ROOT/AI_CORE_DIR/datasets, dp values
            candidate_roots = []
            if croot:
                candidate_roots.append(croot)
            # prefer server.EFFECTIVE_CURATED_ROOT if absolute
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            if eff:
                effp = _resolve_candidate_root(str(eff))
                if effp:
                    candidate_roots.append(effp)
            # ai-core/src/datasets (HERE/datasets)
            ad = (HERE / "datasets")
            if ad.exists():
                candidate_roots.append(ad.resolve())
            # repo-root + AI_CORE_DIR/datasets (if structure nested)
            ar = (REPO_ROOT / AI_CORE_DIR_ENV / "datasets")
            if ar.exists():
                candidate_roots.append(ar.resolve())
            # also attempt canonical_root as returned (if it's already absolute string)
            try:
                if canonical_root:
                    cand_raw = Path(str(canonical_root))
                    if cand_raw.exists():
                        candidate_roots.append(cand_raw.resolve())
            except Exception:
                pass

            # remove duplicates while preserving order
            seen = set()
            candidate_roots_n = []
            for r in candidate_roots:
                rp = str(r)
                if rp not in seen:
                    seen.add(rp)
                    candidate_roots_n.append(Path(rp))
            candidate_roots = candidate_roots_n

            # search for new and old path under candidate roots
            for root in candidate_roots:
                new_path = (root / new_name) if new_name else None
                old_path = (root / old_name) if old_name else None
                new_exists = new_path.exists() if new_path else False
                old_exists = old_path.exists() if old_path else False
                print(f"DEBUG: checking root {root} -> new_exists: {new_exists} old_exists: {old_exists}", file=sys.stderr)
                if new_exists and old_exists:
                    print(f"DEBUG: found both canonical files under {root}", file=sys.stderr)
                    try:
                        report_obj = server.report(dataset=dataset_key, old=old_name, new=new_name, pair_id=None)
                        repd = report_obj.dict() if hasattr(report_obj, "dict") else dict(report_obj)
                        # normalize structure expected by caller
                        be_imp = repd.get("backend_impacts") or []
                        fe_imp = repd.get("frontend_impacts") or []
                        ai_expl = repd.get("ai_explanation") or repd.get("explanation") or ""
                        pair_id = repd.get("metadata", {}).get("pair_id") or repd.get("pair_id") or ""
                        summary = repd.get("summary") or {}
                        service_risk = float(summary.get("service_risk", repd.get("risk_score", 0.0)))
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
                    except Exception as e:
                        print("WARN: server.report(dataset) call failed:", e, file=sys.stderr)
                        # continue trying other roots / index fallback

            # if file(s) not found, try scanning variant folders inside dataset_paths or server helpers
            try:
                idx = server.load_pair_index(dataset_key)
                print(f"DEBUG: loaded pair index for {dataset_key}, entries={len(idx) if isinstance(idx, dict) else 'unknown'}", file=sys.stderr)
                if isinstance(idx, dict):
                    # if relname matches either old/new name in index, pick that pair
                    for pid, meta in idx.items():
                        mo = Path(str(meta.get("old_canonical") or meta.get("old") or "")).name if meta.get("old") else None
                        mn = Path(str(meta.get("new_canonical") or meta.get("new") or "")).name if meta.get("new") else None
                        if mo and mn and new_name and (mn.lower() == new_name.lower() or mo.lower() == new_name.lower()):
                            try:
                                print(f"DEBUG: calling server.report using pair_id {pid} (index match)", file=sys.stderr)
                                report_obj = server.report(dataset=dataset_key, old=mo, new=mn, pair_id=pid)
                                repd = report_obj.dict() if hasattr(report_obj, "dict") else dict(report_obj)
                                be_imp = repd.get("backend_impacts") or []
                                fe_imp = repd.get("frontend_impacts") or []
                                ai_expl = repd.get("ai_explanation") or repd.get("explanation") or ""
                                pair_id = repd.get("metadata", {}).get("pair_id") or repd.get("pair_id") or pid
                                summary = repd.get("summary") or {}
                                service_risk = float(summary.get("service_risk", repd.get("risk_score", 0.0)))
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
                            except Exception as e:
                                print("WARN: server.report with pair-index failed:", e, file=sys.stderr)
                                continue
            except Exception as e:
                print("DEBUG: load_pair_index threw:", e, file=sys.stderr)
                pass
        except Exception as e:
            print("DEBUG: try_call_report encountered exception:", e, file=sys.stderr)
            pass
        return None

    # If dataset_hint provided, try the direct report resolution first
    if dataset_hint and rel_path:
        maybe = try_call_report(dataset_hint, Path(rel_path).name)
        if maybe:
            return maybe

    # If no dataset_hint or it failed, attempt auto-detection across dataset keys using rel_path.filename
    if rel_path:
        fn = Path(rel_path).name
        for key in _all_dataset_keys():
            maybe = try_call_report(key, fn)
            if maybe:
                return maybe

    # If report-based resolution failed, fallback to api_analyze + graph-based enrichment
    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy = server.api_analyze(baseline=baseline_str, candidate=candidate_str, dataset=None, options=None)
    except Exception as e:
        print("WARN: server.api_analyze raised:", e, file=sys.stderr)
        analy = {"run_id": None, "predictions": [], "summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    try:
        score = float((analy.get("summary") or {}).get("service_risk", 0.0))
    except Exception:
        score = 0.0

    # attempt to enrich with graph impacts (best-effort)
    try:
        g = None
        try:
            g = server.load_graph()
            print("DEBUG: server.load_graph() returned:", type(g), file=sys.stderr)
        except Exception as e:
            print("DEBUG: server.load_graph() raised:", e, file=sys.stderr)
            g = None

        # If server.load_graph returned None, try to force-load graph.json from candidate locations
        if g is None:
            gp_candidates = []
            # server.GRAPH_PATH if present
            gp = getattr(server, "GRAPH_PATH", None)
            if gp:
                gp_candidates.append(Path(str(gp)))
            # HERE/datasets/graph.json
            gp_candidates.append(HERE / "datasets" / "graph.json")
            # REPO_ROOT / AI_CORE_DIR_ENV / datasets / graph.json
            gp_candidates.append(REPO_ROOT / AI_CORE_DIR_ENV / "datasets" / "graph.json")
            # EFFECTIVE_CURATED_ROOT/graph.json
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            if eff:
                gp_candidates.append(Path(str(eff)) / "graph.json")
            # dedupe & test exists
            seen = set()
            for cand in gp_candidates:
                try:
                    candp = cand.resolve()
                except Exception:
                    candp = Path(cand)
                if str(candp) in seen:
                    continue
                seen.add(str(candp))
                if candp.exists():
                    print(f"DEBUG: found graph.json at {candp}", file=sys.stderr)
                    # update server.GRAPH_PATH so server helpers use the right path
                    try:
                        server.GRAPH_PATH = str(candp)
                        print(f"DEBUG: server.GRAPH_PATH forced to {server.GRAPH_PATH}", file=sys.stderr)
                        # try load_graph again
                        g = server.load_graph()
                        print("DEBUG: server.load_graph() after forcing GRAPH_PATH ->", type(g), file=sys.stderr)
                        break
                    except Exception as e:
                        print("WARN: server.load_graph() after forcing GRAPH_PATH failed:", e, file=sys.stderr)
                        g = None
            if g is None:
                print("DEBUG: no usable graph loaded after trying candidates", file=sys.stderr)
    except Exception as e:
        print("WARN: graph enrichment step failed:", e, file=sys.stderr)
        g = None

    service_guess = None
    if rel_path:
        try:
            # heuristic: filename stem before last --vN or before first '-v2'
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

    pfeats = {}
    be_imp = []
    fe_imp = []
    try:
        if g is not None:
            pfeats = server.producer_features(g, service_guess)
            changed_paths = [server.normalize_path(d.get("path")) for d in diffs_serial if d.get("path")]
            be_imp = server.backend_impacts(g, service_guess, changed_paths)
            fe_imp = server.ui_impacts(g, service_guess, changed_paths)
            print(f"DEBUG: enriched impacts -> backend:{len(be_imp)} frontend:{len(fe_imp)}", file=sys.stderr)
        else:
            print("DEBUG: skipping backend/ui impact heuristics (no graph)", file=sys.stderr)
    except Exception as e:
        # non-fatal
        print("WARN: enrichment impact computation failed:", e, file=sys.stderr)
        be_imp = []
        fe_imp = []

    ai_expl = analy.get("explanation") or analy.get("ai_explanation") or ""
    if not ai_expl:
        try:
            ai_expl = server.make_explanation(score, diffs, pfeats, {}, be_imp, fe_imp)
        except Exception:
            ai_expl = ""

    return {
        "diffs": diffs_serial,
        "analyze": {
            "summary": {"service_risk": score, "num_aces": len(diffs_serial)},
            "backend_impacts": be_imp,
            "frontend_impacts": fe_imp,
            "ai_explanation": ai_expl,
            "pair_id": analy.get("pair_id") or "",
        },
    }

# ------------------ main() ------------------
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

        # Dataset hint heuristic based on path (script runs from ai_core/src so paths may be long)
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
                # try to find local v1 counterpart first across curated variants
                v1cand = find_local_v1_across_variants(p)
                print(f"DEBUG: v1cand guessed as {v1cand}", file=sys.stderr)
                if v1cand.exists():
                    print(f"DEBUG: found local v1 counterpart at {v1cand}", file=sys.stderr)
                    old_doc = read_json_file_if_exists(v1cand)
                else:
                    # try to fetch from origin/main (path relative to repo root)
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
    total_aces = 0
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
        naces = int(summary.get("num_aces", 0) or 0)
        total_aces += naces
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
