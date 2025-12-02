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

# Ensure we can import server.py from the same directory
HERE = Path(__file__).resolve().parent
# If this script is inside ai_core/src, the server module should be importable from HERE
sys.path.insert(0, str(HERE))

try:
    import server  # noqa: E402
    print("DEBUG: dataset_paths(openapi) =", server.dataset_paths("openapi"), file=sys.stderr)
    print("DEBUG: CWD =", os.getcwd(), file=sys.stderr)

except Exception as e:
    print("ERROR: failed to import server.py:", e, file=sys.stderr)
    raise

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
    for key in _all_dataset_keys():
        try:
            dp = server.dataset_paths(key)
            can = dp.get("canonical")
            if can:
                p = Path(can) / name_alt
                if p.exists():
                    return p
        except Exception:
            continue
    for vf in _variant_folders():
        try:
            vroot = server._variant_dir_for(vf)
            p = Path(vroot) / name_alt
            if p.exists():
                return p
            for base in getattr(server, "BASE_DATASETS", []):
                p2 = Path(vroot) / base / "canonical" / name_alt
                if p2.exists():
                    return p2
        except Exception:
            continue
    try:
        eff = Path(getattr(server, "EFFECTIVE_CURATED_ROOT", ""))
        if eff:
            p = eff / name_alt
            if p.exists():
                return p
            for sub in ("canonical", "ndjson", "metadata"):
                p2 = eff / sub / name_alt
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
            # rarely used fallback
            return server._load_json_or_yaml(Path(text))
        except Exception:
            return {}
print("DEBUG: rel_path =", rel, file=sys.stderr)


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
            if not canonical_root:
                return None
            canonical_root = Path(canonical_root)
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
            # check existence
            new_path = canonical_root / new_name if new_name else None
            old_path = canonical_root / old_name if old_name else None
            if new_path and new_path.exists() and old_path and old_path.exists():
                # call report using base filenames
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
                    return None
            # if file(s) not found, try scanning variant folders inside dataset_paths or server helpers
            # fallback: probe server.load_pair_index to match names if possible
            try:
                idx = server.load_pair_index(dataset_key)
                if isinstance(idx, dict):
                    # if relname matches either old/new name in index, pick that pair
                    for pid, meta in idx.items():
                        mo = Path(str(meta.get("old_canonical") or meta.get("old") or "")).name if meta.get("old") else None
                        mn = Path(str(meta.get("new_canonical") or meta.get("new") or "")).name if meta.get("new") else None
                        if mo and mn and new_name and (mn.lower() == new_name.lower() or mo.lower() == new_name.lower()):
                            try:
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
                            except Exception:
                                continue
            except Exception:
                pass
        except Exception:
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
        g = server.load_graph()
    except Exception:
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
    except Exception:
        # non-fatal
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
                if v1cand.exists():
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
