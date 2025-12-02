#!/usr/bin/env python3
"""
ci_server_run.py

Lightweight CI wrapper that:
 - finds changed files in a PR (git)
 - maps canonical v2 -> v1 or fetches base from origin/main
 - lightly dereferences local '#/components/schemas/...' refs (shallow)
 - invokes server.report(...) (preferred) or server.api_analyze(...) (fallback)
 - writes pr-impact-full.json (detailed) and pr-impact-report.json (compact)
This version includes aggressive path resolution and an index.json discovery step
to ensure server.EFFECTIVE_CURATED_ROOT and server.GRAPH_PATH are correct in CI.
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
    try:
        if p.is_absolute() and p.exists():
            return p.resolve()
        p2 = (HERE / candidate)
        if p2.exists():
            return p2.resolve()
        p3 = (REPO_ROOT / candidate)
        if p3.exists():
            return p3.resolve()
        p4 = (HERE / "datasets" / candidate)
        if p4.exists():
            return p4.resolve()
        p5 = (HERE.parent / candidate)
        if p5.exists():
            return p5.resolve()
    except Exception:
        pass
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
        # try common alternative names
        alt_names = ["dataset_index.json", "index.json", "version_meta.json", "dataset_oindex.json"]
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

# --- Normalize EFFECTIVE_CURATED_ROOT and GRAPH_PATH early so server.report / load_pair_index behave ---
try:
    eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
    if eff:
        res = _resolve_candidate_root(str(eff))
        if res:
            server.EFFECTIVE_CURATED_ROOT = res
            print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT resolved -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
        else:
            fallback = (HERE / "datasets")
            if fallback.exists():
                server.EFFECTIVE_CURATED_ROOT = fallback.resolve()
                print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT forced -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
            else:
                print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT ({eff}) could not be resolved", file=sys.stderr)
    else:
        fallback = (HERE / "datasets")
        if fallback.exists():
            server.EFFECTIVE_CURATED_ROOT = fallback.resolve()
            print(f"DEBUG: server.EFFECTIVE_CURATED_ROOT set -> {server.EFFECTIVE_CURATED_ROOT}", file=sys.stderr)
except Exception as e:
    print("WARN: error normalizing EFFECTIVE_CURATED_ROOT:", e, file=sys.stderr)

# locate index.json and set EFFECTIVE_CURATED_ROOT to container that has it (so load_pair_index() works)
try:
    idx_root = _find_index_json_under_datasets()
    if idx_root:
        server.EFFECTIVE_CURATED_ROOT = idx_root.resolve()
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
            server.GRAPH_PATH = str(res_gp)
            print(f"DEBUG: server.GRAPH_PATH resolved -> {server.GRAPH_PATH}", file=sys.stderr)
        else:
            cand1 = (HERE / "datasets" / "graph.json")
            cand2 = (REPO_ROOT / AI_CORE_DIR_ENV / "datasets" / "graph.json")
            if cand1.exists():
                server.GRAPH_PATH = cand1.resolve()
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            elif cand2.exists():
                server.GRAPH_PATH = cand2.resolve()
                print(f"DEBUG: server.GRAPH_PATH forced -> {server.GRAPH_PATH}", file=sys.stderr)
            else:
                print(f"DEBUG: server.GRAPH_PATH ({gp}) could not be resolved", file=sys.stderr)
    else:
        cand = (HERE / "datasets" / "graph.json")
        if cand.exists():
            server.GRAPH_PATH = str(cand.resolve())
            print(f"DEBUG: server.GRAPH_PATH set -> {server.GRAPH_PATH}", file=sys.stderr)
except Exception as e:
    print("WARN: error normalizing GRAPH_PATH:", e, file=sys.stderr)

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
    Robust try-call to server.report with aggressive diagnostics and graph enrichment fallback.
    """
    if not dataset_key:
        print("DEBUG: try_call_report called with empty dataset_key", file=sys.stderr)
        return None

    def _diagnose_and_extract(report_obj):
        repd = report_obj.dict() if hasattr(report_obj, "dict") else dict(report_obj)
        try:
            print(f"DEBUG: report keys -> {list(repd.keys())}", file=sys.stderr)
        except Exception:
            pass
        be_imp = repd.get("backend_impacts") or []
        fe_imp = repd.get("frontend_impacts") or []
        ver = repd.get("versioning") or {}
        meta = repd.get("metadata") or {}
        print(f"DEBUG: report backend_impacts count={len(be_imp)} frontend_impacts count={len(fe_imp)}", file=sys.stderr)
        return repd, be_imp, fe_imp, ver, meta

    try:
        # dataset paths and canonical root resolution
        try:
            dp = server.dataset_paths(dataset_key)
        except Exception as e:
            print("DEBUG: server.dataset_paths() failed:", e, file=sys.stderr)
            dp = {}
        canonical_root = dp.get("canonical")
        print(f"DEBUG: dataset_paths({dataset_key}) -> {dp}", file=sys.stderr)
        croot = _resolve_candidate_root(str(canonical_root)) if canonical_root else None
        if not croot:
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            if eff:
                attempt = _resolve_candidate_root(str(eff))
                if attempt and canonical_root:
                    joined = (attempt / str(canonical_root)).resolve()
                    if joined.exists():
                        croot = joined
        if croot:
            print(f"DEBUG: canonical_root resolved to: {croot}", file=sys.stderr)
        else:
            print(f"DEBUG: canonical_root could not be resolved for dataset {dataset_key} (candidate: {canonical_root})", file=sys.stderr)

        new_name = Path(relname).name if relname else None
        old_name = None
        if new_name:
            if "--v2." in new_name:
                old_name = new_name.replace("--v2.", "--v1.")
            elif "-v2." in new_name:
                old_name = new_name.replace("-v2.", "-v1.")
            else:
                old_name = new_name.replace("v2", "v1", 1)

        # candidate roots to search
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
            rp = str(r)
            if rp not in seen:
                seen.add(rp)
                candidate_roots_n.append(Path(rp))
        candidate_roots = candidate_roots_n

        # 1) Try direct canonical file report() call
        tried_roots = []
        for root in candidate_roots:
            tried_roots.append(str(root))
            new_path = (root / new_name) if new_name else None
            old_path = (root / old_name) if old_name else None
            new_exists = new_path.exists() if new_path else False
            old_exists = old_path.exists() if old_path else False
            print(f"DEBUG: checking root {root} -> new_exists: {new_exists} old_exists: {old_exists}", file=sys.stderr)
            if new_exists and old_exists:
                print(f"DEBUG: found both canonical files under {root}", file=sys.stderr)
                try:
                    report_obj = server.report(dataset=dataset_key, old=old_name, new=new_name, pair_id=None)
                    repd, be_imp, fe_imp, ver, meta = _diagnose_and_extract(report_obj)
                    # gather service candidates
                    service_candidates = []
                    if isinstance(ver, dict):
                        ps = ver.get("producers_sample") or ver.get("producers") or []
                        if isinstance(ps, list):
                            service_candidates.extend([s for s in ps if isinstance(s, str)])
                    if isinstance(meta, dict):
                        svc = meta.get("service_name") or meta.get("producer") or meta.get("service")
                        if svc:
                            service_candidates.append(svc)
                    # filename guess
                    try:
                        stem = Path(relname).stem if relname else ""
                        if stem:
                            fg = stem.split("--v")[0] if "--v" in stem else (stem.split("-v")[0] if "-v" in stem else stem)
                            if fg:
                                service_candidates.append(fg)
                    except Exception:
                        pass
                    # normalize candidates
                    sc_norm = []
                    seen_s = set()
                    for s in service_candidates:
                        try:
                            s0 = s.strip()
                        except Exception:
                            s0 = str(s)
                        if s0 and s0 not in seen_s:
                            seen_s.add(s0)
                            sc_norm.append(s0)
                    print(f"DEBUG: service candidates from report/file -> {sc_norm}", file=sys.stderr)

                    # If impacts missing, try graph enrichment
                    if (not be_imp) or (not fe_imp):
                        try:
                            g = None
                            try:
                                g = server.load_graph()
                                print("DEBUG: server.load_graph() OK inside try_call_report", file=sys.stderr)
                            except Exception as e:
                                print("DEBUG: server.load_graph() inside try_call_report raised:", e, file=sys.stderr)
                                # attempt to force several GRAPH_PATH candidates
                                candidates = [
                                    getattr(server, "GRAPH_PATH", None),
                                    str(HERE / "datasets" / "graph.json"),
                                    str(REPO_ROOT / AI_CORE_DIR_ENV / "datasets" / "graph.json"),
                                ]
                                for p in candidates:
                                    if not p:
                                        continue
                                    try:
                                        pp = Path(p)
                                        if pp.exists():
                                            server.GRAPH_PATH = str(pp)
                                            g = server.load_graph()
                                            print("DEBUG: server.load_graph() succeeded after forcing GRAPH_PATH", file=sys.stderr)
                                            break
                                    except Exception as e2:
                                        print("DEBUG: forcing GRAPH_PATH candidate failed:", p, e2, file=sys.stderr)
                                        continue

                            if g is not None:
                                changed_paths = [server.normalize_path(d.get("path")) for d in diffs_serial if d.get("path")]
                                # try each candidate
                                for svc in sc_norm + [None]:
                                    try:
                                        svc_name = svc
                                        maybe_be = server.backend_impacts(g, svc_name, changed_paths)
                                        maybe_fe = server.ui_impacts(g, svc_name, changed_paths)
                                        if maybe_be:
                                            print(f"DEBUG: graph-based backend_impacts found for svc='{svc_name}': {maybe_be}", file=sys.stderr)
                                            be_imp = maybe_be
                                        if maybe_fe:
                                            print(f"DEBUG: graph-based frontend_impacts found for svc='{svc_name}': {maybe_fe}", file=sys.stderr)
                                            fe_imp = maybe_fe
                                        if be_imp:
                                            break
                                    except Exception as e:
                                        print("DEBUG: backend/ui impact attempt failed for svc:", svc, e, file=sys.stderr)
                                        continue
                            else:
                                print("DEBUG: graph not available for enrichment inside try_call_report", file=sys.stderr)
                        except Exception as e:
                            print("WARN: enrichment attempt inside try_call_report failed:", e, file=sys.stderr)

                    ai_expl = repd.get("ai_explanation") or repd.get("explanation") or ""
                    pair_id = repd.get("metadata", {}).get("pair_id") or repd.get("pair_id") or repd.get("versioning", {}).get("pair_id", "") or ""
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
                    print("WARN: server.report(dataset) call failed at canonical success branch:", e, file=sys.stderr)
                    # continue to index-based heuristics

        # 2) pair-index fallback
        try:
            idx = server.load_pair_index(dataset_key)
            print(f"DEBUG: loaded pair index for {dataset_key}, entries={len(idx) if isinstance(idx, dict) else 'unknown'}", file=sys.stderr)
            if isinstance(idx, dict):
                for pid, meta in idx.items():
                    try:
                        mo = Path(str(meta.get("old_canonical") or meta.get("old") or "")).name if meta.get("old") else None
                        mn = Path(str(meta.get("new_canonical") or meta.get("new") or "")).name if meta.get("new") else None
                        if mo and mn and new_name and (mn.lower() == new_name.lower() or mo.lower() == new_name.lower()):
                            print(f"DEBUG: index match -> pid={pid} mo={mo} mn={mn}", file=sys.stderr)
                            report_obj = server.report(dataset=dataset_key, old=mo, new=mn, pair_id=pid)
                            repd, be_imp, fe_imp, ver, meta = _diagnose_and_extract(report_obj)
                            # same enrichment logic as above
                            service_candidates = []
                            ps = ver.get("producers_sample") or ver.get("producers") or []
                            if isinstance(ps, list):
                                service_candidates.extend([s for s in ps if isinstance(s, str)])
                            svc = meta.get("service_name") or meta.get("producer") or meta.get("service")
                            if svc:
                                service_candidates.append(svc)
                            if new_name:
                                stem = Path(new_name).stem
                                fg = stem.split("--v")[0] if "--v" in stem else (stem.split("-v")[0] if "-v" in stem else stem)
                                service_candidates.append(fg)
                            sc_norm = []
                            seen_s = set()
                            for s in service_candidates:
                                s0 = s.strip() if isinstance(s, str) else str(s)
                                if s0 and s0 not in seen_s:
                                    seen_s.add(s0)
                                    sc_norm.append(s0)
                            if (not be_imp) or (not fe_imp):
                                try:
                                    g = server.load_graph()
                                    changed_paths = [server.normalize_path(d.get("path")) for d in diffs_serial if d.get("path")]
                                    for svc_name in sc_norm + [None]:
                                        try:
                                            maybe_be = server.backend_impacts(g, svc_name, changed_paths)
                                            maybe_fe = server.ui_impacts(g, svc_name, changed_paths)
                                            if maybe_be:
                                                be_imp = maybe_be
                                            if maybe_fe:
                                                fe_imp = maybe_fe
                                            if be_imp:
                                                break
                                        except Exception:
                                            continue
                                except Exception as e:
                                    print("DEBUG: graph enrichment after index-match failed:", e, file=sys.stderr)

                            ai_expl = repd.get("ai_explanation") or repd.get("explanation") or ""
                            pair_id = pid or repd.get("metadata", {}).get("pair_id") or ""
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
                        print("DEBUG: skipping index entry due to error:", e, file=sys.stderr)
                        continue
        except Exception as e:
            print("DEBUG: load_pair_index threw:", e, file=sys.stderr)
            pass

    except Exception as e:
        print("DEBUG: try_call_report encountered exception:", e, file=sys.stderr)
        pass

    # If everything failed, return None so caller falls back to api_analyze
    print(f"DEBUG: try_call_report exhausted candidates: roots_checked={tried_roots if 'tried_roots' in locals() else 'none'}", file=sys.stderr)
    return None

    # dataset_hint first
    if dataset_hint and rel_path:
        maybe = try_call_report(dataset_hint, Path(rel_path).name)
        if maybe:
            return maybe

    # try all keys
    if rel_path:
        fn = Path(rel_path).name
        for key in _all_dataset_keys():
            maybe = try_call_report(key, fn)
            if maybe:
                return maybe

    # fallback to api_analyze
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

    # enrichment: try to load graph if available
    try:
        g = None
        try:
            g = server.load_graph()
            print("DEBUG: server.load_graph() returned:", type(g), file=sys.stderr)
        except Exception as e:
            print("DEBUG: server.load_graph() raised:", e, file=sys.stderr)
            g = None

        if g is None:
            gp_candidates = []
            gp = getattr(server, "GRAPH_PATH", None)
            if gp:
                gp_candidates.append(Path(str(gp)))
            gp_candidates.append(HERE / "datasets" / "graph.json")
            gp_candidates.append(REPO_ROOT / AI_CORE_DIR_ENV / "datasets" / "graph.json")
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            if eff:
                gp_candidates.append(Path(str(eff)) / "graph.json")
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
                    try:
                        server.GRAPH_PATH = str(candp)
                        print(f"DEBUG: server.GRAPH_PATH forced to {server.GRAPH_PATH}", file=sys.stderr)
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

def ensure_graph_loaded(repo_root: Optional[str] = None, ai_core_dir_env: Optional[str] = None) -> None:
    """
    Best-effort: locate graph.json, set server.GRAPH_PATH to it, and call server.load_graph()
    so backend/frontend impact helpers have the graph available in CI.

    - Looks in several sensible places (server.EFFECTIVE_CURATED_ROOT, HERE/datasets,
      repo_root + ai_core_dir/datasets, datasets/curated_* etc).
    - Prints lots of DEBUG/WARN lines to stderr so CI logs show what happened.
    - Non-fatal: never raises (just logs), because missing graph should not crash CI.
    """
    try:
        candidates: List[Path] = []

        # 1) If server exposes EFFECTIVE_CURATED_ROOT, prefer graph under it or its parent
        try:
            eff = getattr(server, "EFFECTIVE_CURATED_ROOT", None)
            if eff:
                effp = Path(str(eff))
                candidates.append(effp / "graph.json")
                candidates.append(effp.parent / "graph.json")
                # also check curated variants under effective root
                candidates.append(effp / "curated_clean" / "graph.json")
                candidates.append(effp / "curated_noisy_light" / "graph.json")
                candidates.append(effp / "curated_noisy_heavy" / "graph.json")
        except Exception:
            print("DEBUG: could not read server.EFFECTIVE_CURATED_ROOT", file=sys.stderr)

        # 2) Common locations relative to this script
        candidates.append(HERE / "datasets" / "graph.json")
        candidates.append(HERE / "datasets" / "curated_clean" / "graph.json")
        candidates.append(HERE / "datasets" / "curated_noisy_light" / "graph.json")
        candidates.append(HERE / "datasets" / "curated_noisy_heavy" / "graph.json")

        # 3) If repo_root and AI_CORE_DIR provided, try those combos
        try:
            if repo_root and ai_core_dir_env:
                ai_core_dir = Path(repo_root) / ai_core_dir_env
                candidates.append(ai_core_dir / "datasets" / "graph.json")
                candidates.append(ai_core_dir / "datasets" / "curated_clean" / "graph.json")
                candidates.append(Path(repo_root) / "datasets" / "graph.json")
                candidates.append(Path(repo_root) / "datasets" / "curated_clean" / "graph.json")
        except Exception:
            pass

        # 4) fallback: relative to server module file location
        try:
            server_file = Path(getattr(server, "__file__", "")) if getattr(server, "__file__", None) else None
            if server_file:
                candidates.append(server_file.parent / "datasets" / "graph.json")
                candidates.append(server_file.parent.parent / "datasets" / "graph.json")
        except Exception:
            pass

        # normalize and dedupe candidates while logging attempts
        seen = set()
        resolved_candidates: List[Path] = []
        for c in candidates:
            try:
                rc = c.resolve()
            except Exception:
                rc = c
            if str(rc) not in seen:
                seen.add(str(rc))
                resolved_candidates.append(rc)
                print(f"DEBUG: graph candidate -> {rc}", file=sys.stderr)

        found: Optional[Path] = None
        for c in resolved_candidates:
            try:
                if c.exists():
                    found = c
                    break
            except Exception:
                continue

        if not found:
            print("WARN: no graph.json located in candidates; backend/frontend impacts may be skipped.", file=sys.stderr)
            return


        # force server.GRAPH_PATH (give server a Path object — load_graph expects Path-like)
        old_graph_path = getattr(server, "GRAPH_PATH", None)
        try:
            # prefer passing a Path object (server.load_graph uses .exists())
            server.GRAPH_PATH = found
            print(f"DEBUG: forced server.GRAPH_PATH -> {server.GRAPH_PATH}", file=sys.stderr)
        except Exception as e:
            # as a last resort also set the string form to be safe for code that expects str
            try:
                server.GRAPH_PATH = str(found)
                print(f"DEBUG: forced server.GRAPH_PATH (string fallback) -> {server.GRAPH_PATH}", file=sys.stderr)
            except Exception:
                print("WARN: failed to set server.GRAPH_PATH:", e, file=sys.stderr)


        # Attempt to load graph and print diagnostics
        try:
            g = server.load_graph()
            # print small diagnostics about loaded graph
            try:
                size = len(g) if hasattr(g, "__len__") else "n/a"
            except Exception:
                size = "n/a"
            print(f"DEBUG: server.load_graph() succeeded; type={type(g)} size={size}", file=sys.stderr)
            # If graph is a dict, print top-level keys sample
            try:
                if isinstance(g, dict):
                    keys_sample = list(g.keys())[:10]
                    print(f"DEBUG: graph top-level keys (sample 10): {keys_sample}", file=sys.stderr)
            except Exception:
                pass
        except Exception as e:
            print("WARN: server.load_graph() failed after forcing GRAPH_PATH:", e, file=sys.stderr)
            # try to emit a tiny snippet of the file so logs help debugging
            try:
                snippet = found.read_text(encoding="utf-8")[:800]
                print("DEBUG: graph.json snippet (first 800 chars):", file=sys.stderr)
                print(snippet, file=sys.stderr)
            except Exception:
                print("DEBUG: unable to read graph.json for snippet", file=sys.stderr)

        # sanity: print what server.GRAPH_PATH is now
        try:
            print("DEBUG: server.GRAPH_PATH final value ->", getattr(server, "GRAPH_PATH", None), file=sys.stderr)
        except Exception:
            pass

    except Exception as exc:
        print("WARN: unexpected error in ensure_graph_loaded():", exc, file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", "unknown"))
    parser.add_argument("--output-full", default="pr-impact-full.json")
    parser.add_argument("--output-summary", default="pr-impact-report.json")
    args = parser.parse_args()

    changed = git_changed_files()
    print("CI: changed files:", changed, file=sys.stderr)

    # Ensure graph is discoverable & loaded before any server.report()/api_analyze() calls
    ensure_graph_loaded(repo_root=str(REPO_ROOT), ai_core_dir_env=str(AI_CORE_DIR_ENV))

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
