#!/usr/bin/env python3
"""
ci_server_run.py

Lightweight CI wrapper that:
 - finds changed files in a PR (git)
 - maps canonical v2 -> v1 or fetches base from origin/main
 - lightly dereferences local '#/components/schemas/...' refs (shallow)
 - invokes server.api_analyze(...) (in-process) to reuse server logic
 - writes pr-impact-full.json (detailed) and pr-impact-report.json (compact)
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure we can import server.py from the same directory
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import server  # noqa: E402
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
    """
    Heuristic: replace '--v2.' with '--v1.' or '-v2.' with '-v1.' in filename.
    If file exists on filesystem, return Path. Otherwise return Path with same dir & name.
    """
    name = v2path.name
    if "--v2." in name:
        alt = name.replace("--v2.", "--v1.")
    elif "-v2." in name:
        alt = name.replace("-v2.", "-v1.")
    else:
        # fallback: try replacing first 'v2' segment (risky)
        alt = name.replace("v2", "v1", 1)
    cand = v2path.parent / alt
    return cand

# ---------- new helpers to support curated_* variants ----------
def _variant_folders() -> List[str]:
    # e.g., curated_clean, curated_noisy_light, curated_noisy_heavy
    try:
        return list(server.VARIANT_MAP.values())
    except Exception:
        return ["curated_clean", "curated_noisy_light", "curated_noisy_heavy"]

def _all_dataset_keys() -> List[str]:
    try:
        return server.all_dataset_keys()
    except Exception:
        # fallback
        return ["openapi", "openapi_noisy_light", "openapi_noisy_heavy",
                "petclinic", "petclinic_noisy_light", "petclinic_noisy_heavy",
                "openrewrite", "openrewrite_noisy_light", "openrewrite_noisy_heavy"]

def find_local_v1_across_variants(v2path: Path) -> Path:
    """
    Try to find the corresponding v1 file across the curated variant folders and legacy locations.
    Returns Path (possibly non-existing) as candidate; prefer existing file if found.
    """
    # first try the simple local heuristic
    cand = find_counterpart_v1(v2path)
    if cand.exists():
        return cand

    name_alt = cand.name
    # try scanning dataset variants: use all dataset keys and try their canonical folders
    for key in _all_dataset_keys():
        try:
            dp = server.dataset_paths(key)
            can_candidate = dp.get("canonical")
            if not can_candidate:
                continue
            can = Path(can_candidate)
            p = can / name_alt
            if p.exists():
                return p
        except Exception:
            continue

    # try variant-level index roots
    for vf in _variant_folders():
        # try both CURATED_CONTAINER/variant and CURATED_ROOT_RAW/variant behaviors via server._variant_dir_for if available
        try:
            vroot = server._variant_dir_for(vf)
            p = Path(vroot) / name_alt
            if p.exists():
                return p
            # also check canonical under each base dataset in that variant
            for base in getattr(server, "BASE_DATASETS", []):
                p2 = Path(vroot) / base / "canonical" / name_alt
                if p2.exists():
                    return p2
        except Exception:
            continue

    # legacy effective curated root - server.EFFECTIVE_CURATED_ROOT
    try:
        eff = Path(getattr(server, "EFFECTIVE_CURATED_ROOT", ""))
        if eff and eff.exists():
            p = eff / name_alt
            if p.exists():
                return p
            # also try canonical and ndjson/metadata places
            for sub in ("canonical", "ndjson", "metadata"):
                p2 = eff / sub / name_alt
                if p2.exists():
                    return p2
    except Exception:
        pass

    # finally, return the initial candidate (may not exist)
    return cand

def read_json_file_if_exists(p: Path) -> Dict[str, Any]:
    try:
        if p and p.exists():
            txt = p.read_text(encoding="utf-8")
            try:
                return json.loads(txt)
            except Exception:
                # try YAML via server helper if available
                try:
                    return server._load_json_or_yaml(p)
                except Exception:
                    return {}
        return {}
    except Exception:
        return {}

def dereference_components(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very small inliner for '#/components/schemas/X' refs used by responses/requestBody parameters.
    This is intentionally shallow and only resolves local components.schemas.* objects.
    It mutates a shallow copy of the spec to inline referenced schemas where practical.
    """
    if not isinstance(spec, dict):
        return spec
    comp = spec.get("components") or {}
    schemas = comp.get("schemas") or {}
    # simple resolver
    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                ref = obj["$ref"]
                if ref.startswith("#/components/schemas/"):
                    key = ref.split("/")[-1]
                    target = schemas.get(key)
                    if isinstance(target, dict):
                        return resolve(dict(target))  # inline and continue resolving
                    return target
            # recurse into dict
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(x) for x in obj]
        return obj
    out = dict(spec)
    # Only replace schemas and also go into paths responses and requestBody
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
    # shallow copy components too (keep original for diagnostics)
    try:
        out["components"] = dict(out.get("components") or {})
    except Exception:
        pass
    return out

def load_json_text(text: str) -> Dict[str, Any]:
    # If `text` appears to be file content, attempt json.loads; otherwise attempt server helper (if present)
    try:
        return json.loads(text)
    except Exception:
        try:
            # if provided a path-like string, try to read as file (but usually git_show returns content)
            return server._load_json_or_yaml(Path(text))
        except Exception:
            return {}

def analyze_pair_files(old_doc: Dict[str, Any], new_doc: Dict[str, Any], rel_path: str = None) -> Dict[str, Any]:
    """
    Robust dataset-aware analyze_pair_files.

    - Normalizes rel_path to just the filename (CI often reports deep paths).
    - Tries to resolve dataset + pair index and call server.report(...) first (preferred).
    - Falls back to server.api_analyze(...) and computes backend/frontend impacts
      using the graph helpers so the CI summary still contains impacts.
    - Defensive: tolerates missing server helpers without crashing.
    """
    # 1) shallow deref for diffs
    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    # 2) diffs
    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN: server.diff_openapi failed:", e, file=sys.stderr)
        diffs = []

    # serialize diffs to plain dicts
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

    # helper: try to find dataset + canonical names using just filenames (no full paths)
    def _find_dataset_and_names_from_filenames(old_name: str, new_name: str):
        try:
            for key in server.all_dataset_keys():
                try:
                    dp = server.dataset_paths(key)
                    can = dp.get("canonical") if isinstance(dp.get("canonical"), (str, Path)) else dp.get("canonical")
                    # if canonical is a path-like, compare names
                    if can:
                        # prefer matching by filename(s) (case-insensitive)
                        if old_name and (Path(can) / old_name).exists() and new_name and (Path(can) / new_name).exists():
                            return key, old_name, new_name, None
                    # try pair index lookup (if available)
                    try:
                        idx = server.load_pair_index(key)
                        if isinstance(idx, dict):
                            for pid, meta in idx.items():
                                mo = Path(str(meta.get("old_canonical") or meta.get("old") or "")).name if meta else None
                                mn = Path(str(meta.get("new_canonical") or meta.get("new") or "")).name if meta else None
                                if mo and mn and old_name and new_name and mo.lower() == old_name.lower() and mn.lower() == new_name.lower():
                                    return key, mo, mn, pid
                    except Exception:
                        # ignore index lookup failure for this dataset key
                        pass
                except Exception:
                    continue
        except Exception:
            pass
        return None, None, None, None

    # Normalize rel_path -> filename (CI gives deep paths; we only need the file name)
    filename = None
    if rel_path:
        try:
            filename = Path(rel_path).name
        except Exception:
            filename = rel_path.split("/")[-1] if "/" in rel_path else rel_path

    # attempt dataset resolution via filenames
    dataset_key = old_name = new_name = pair_hint = None
    try:
        # derive candidate filenames from docs if possible
        cand_old_name = None
        cand_new_name = None
        # if rel_path filename looks canonical-like, use it
        if filename:
            cand_old_name = cand_new_name = filename
        # else try to use info.title fallback to synthesize a name (last resort)
        if not cand_new_name:
            try:
                cand_new_name = (new_doc.get("info", {}) or {}).get("title")
            except Exception:
                cand_new_name = None
        dataset_key, old_name, new_name, pair_hint = _find_dataset_and_names_from_filenames(cand_old_name, cand_new_name)
    except Exception:
        dataset_key = old_name = new_name = pair_hint = None

    # If dataset resolved, call server.report(...) which should populate backend/frontend impacts and pair_id.
    if dataset_key and old_name and new_name:
        try:
            # call server.report(dataset, old, new, pair_id) — adapt if signature differs
            report_obj = server.report(dataset=dataset_key, old=old_name, new=new_name, pair_id=pair_hint)
            repd = report_obj.dict() if hasattr(report_obj, "dict") else dict(report_obj)
            be_imp = repd.get("backend_impacts") or repd.get("backend_impacts", []) or []
            fe_imp = repd.get("frontend_impacts") or repd.get("frontend_impacts", []) or []
            ai_expl = repd.get("ai_explanation") or repd.get("ai_explanation", "") or ""
            # prefer numeric summary service_risk from report, fall back to repd.risk_score
            summary = repd.get("summary") or {}
            try:
                service_risk = float(summary.get("service_risk", repd.get("risk_score", 0.0)))
            except Exception:
                service_risk = float(repd.get("risk_score", 0.0) or 0.0)
            pair_id = repd.get("metadata", {}).get("pair_id") or repd.get("pair_id") or pair_hint or ""
            return {
                "diffs": diffs_serial,
                "analyze": {
                    "summary": {"service_risk": service_risk, "num_aces": len(diffs_serial)},
                    "backend_impacts": be_imp,
                    "frontend_impacts": fe_imp,
                    "ai_explanation": ai_expl,
                    "pair_id": pair_id,
                },
            }
        except Exception as e:
            print("WARN: server.report(...) call failed or unavailable:", e, file=sys.stderr)
            # fall through to api_analyze path

    # FALLBACK: call server.api_analyze(...) and enrich with graph impacts
    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy = server.api_analyze(baseline=baseline_str, candidate=candidate_str, dataset=None, options=None)
    except Exception as e:
        print("WARN: server.api_analyze raised:", e, file=sys.stderr)
        analy = {"run_id": None, "predictions": [], "summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # extract score if present
    try:
        score = float((analy.get("summary") or {}).get("service_risk", 0.0))
    except Exception:
        try:
            score = float(analy.get("risk_score", 0.0))
        except Exception:
            score = 0.0

    # Try to compute backend/frontend impacts via graph helpers where available
    be_imp = []
    fe_imp = []
    try:
        # defensive: server.load_graph may raise if graph not built
        g = server.load_graph()
        # service guess: prefer pair index hint or filename-based name or info.title
        service_guess = None
        if pair_hint:
            service_guess = pair_hint
        elif filename:
            # attempt to simplify filename to service stem
            stem = Path(filename).stem
            # most canonical names are like openapi--<service>--<hash>--v2.canonical
            # try to extract <service> portion heuristically
            parts = stem.split("--")
            if len(parts) >= 3:
                # parts[1] is likely service name
                service_guess = parts[1]
            else:
                # fallback: take portion before '-v' or '--v'
                if "-v" in stem:
                    service_guess = stem.split("-v")[0]
                else:
                    service_guess = stem
        if not service_guess:
            try:
                service_guess = (new_doc.get("info", {}) or {}).get("title") or "unknown"
            except Exception:
                service_guess = "unknown"

        # producer features (defensive)
        try:
            pfeats = server.producer_features(g, service_guess)
        except Exception:
            pfeats = {}
        # normalized changed paths
        changed_paths = []
        for d in diffs_serial:
            pth = d.get("path")
            if pth and hasattr(server, "normalize_path"):
                try:
                    changed_paths.append(server.normalize_path(pth))
                except Exception:
                    changed_paths.append(pth)
            elif pth:
                changed_paths.append(pth)

        try:
            be_imp = server.backend_impacts(g, service_guess, changed_paths) or []
        except Exception:
            be_imp = []
        try:
            fe_imp = server.ui_impacts(g, service_guess, changed_paths) or []
        except Exception:
            fe_imp = []
    except Exception as e:
        # graph not available or helpers missing — keep impacts empty but don't fail
        print("INFO: unable to compute graph impacts:", e, file=sys.stderr)
        be_imp = be_imp or []
        fe_imp = fe_imp or []

    # Build ai_explanation if not provided by analy
    ai_expl = analy.get("explanation") or analy.get("ai_explanation") or ""
    if not ai_expl:
        try:
            # server.make_explanation may expect diffs or diffs_serial; pass diffs_serial
            ai_expl = server.make_explanation(score, diffs_serial, pfeats if 'pfeats' in locals() else {}, {}, be_imp, fe_imp)
        except Exception:
            try:
                # fallback to a short human-readable explanation
                ai_expl = f"Predicted risk: {score:.2f}. ACEs: {len(diffs_serial)}."
            except Exception:
                ai_expl = ""

    # assemble final shape
    pair_id_out = analy.get("pair_id") or analy.get("metadata", {}).get("pair_id") or pair_hint or ""
    return {
        "diffs": diffs_serial,
        "analyze": {
            "summary": {"service_risk": score, "num_aces": len(diffs_serial)},
            "backend_impacts": be_imp,
            "frontend_impacts": fe_imp,
            "ai_explanation": ai_expl,
            "pair_id": pair_id_out,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", "unknown"))
    parser.add_argument("--output-full", default="pr-impact-full.json")
    parser.add_argument("--output-summary", default="pr-impact-report.json")
    args = parser.parse_args()

    changed = git_changed_files()
    print("CI: changed files:", changed, file=sys.stderr)

    # filter for likely API/canonical changes
    api_files = [f for f in changed if f.lower().endswith(".json") or f.lower().endswith((".yaml", ".yml"))]

    # --- Updated: prefer curated_* variant folders (curated_clean, curated_noisy_light, curated_noisy_heavy)
    curated_tokens = ["datasets/curated", "canonical"]
    # also include curated_* tokens discovered from server.VARIANT_MAP if available
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
        # if v2 canonical file, try to find v1 on filesystem (including across curated variants)
        # otherwise, try to fetch origin/main copy
        try:
            if p.exists():
                # local workspace file present
                # read it (try JSON, fallback YAML via server helper)
                new_doc = read_json_file_if_exists(p)
                # find local v1 counterpart across variants first
                v1cand = find_local_v1_across_variants(p)
                if v1cand.exists():
                    old_doc = read_json_file_if_exists(v1cand)
                else:
                    # try to fetch from origin/main (path relative to repo root)
                    blob = git_show_file(f"origin/main:{rel}")
                    if blob:
                        old_doc = load_json_text(blob)
                    else:
                        # fallback: try HEAD~1 version
                        code, out = run_cmd(["git", "show", "HEAD~1:" + rel])
                        old_doc = load_json_text(out) if code == 0 else {}
            else:
                # file doesn't exist in workspace (maybe deleted) -> try git show for new and old
                # prefer the PR branch ref name if present, else HEAD
                ref_branch = os.environ.get("GITHUB_REF_NAME", None) or os.environ.get("BRANCH", None) or "HEAD"
                blob_new = git_show_file(f"origin/{ref_branch}:{rel}") or git_show_file(f"HEAD:{rel}") or git_show_file(f"origin/main:{rel}")
                new_doc = load_json_text(blob_new)
                blob_old = git_show_file(f"origin/main:{rel}")
                old_doc = load_json_text(blob_old) if blob_old else {}
        except Exception as e:
            print(f"WARN: failed to load files for {rel}: {e}", file=sys.stderr)
            old_doc, new_doc = {}, {}

        # skip trivial empty pairs
        try:
            pair_res = analyze_pair_files(old_doc, new_doc, rel_path=rel)
        except Exception as e:
            print("WARN: analyze_pair_files exception:", e, file=sys.stderr)
            pair_res = {"diffs": [], "analyze": {"summary": {"service_risk": 0.0, "num_aces": 0}, "predictions": []}}

        results.append({"file": rel, "result": pair_res})
    # Compose full output
    full_out = {
        "status": "ok" if results else "partial",
        "pr": str(args.pr),
        "files_changed": files_processed,
        "files_analyzed": len(results),
        "entries": results
    }
    Path(args.output_full).write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print("Wrote full output to", args.output_full, file=sys.stderr)

    # Compose compact summary used by workflow comment
    # We aggregate simple predicted risk as max of per-file service_risk or 0.0
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
        # collect diffs (atomic change events)
        for d in e["result"].get("diffs", []):
            # ensure type is uniform
            if isinstance(d.get("type"), str):
                d["type"] = d["type"].upper()
            atomic_aces.append(d)
        # pick the first non-empty pair_id and ai_explanation we find (authoritative)
        if not pair_id_top:
            pair_id_top = analy.get("pair_id") or (analy.get("versioning") or {}).get("pair_id") or (analy.get("metadata") or {}).get("pair_id")
        if not ai_expl_top:
            ai_expl_top = analy.get("ai_explanation") or analy.get("explanation")

    # Compute band / label consistent with other code paths
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
        # convenience fields expected by downstream scripts
        "risk_score": round(float(max_risk), 3),
        "risk_band": band,
        "risk_level": level,
        "ai_explanation": ai_expl_top or "",
        "pair_id": pair_id_top or "",
        "metadata": {
            "pair_id": pair_id_top or ""
        }
    }

    Path(args.output_summary).write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print("Wrote summary to", args.output_summary, file=sys.stderr)

    # exit 0 (CI can inspect outputs). If you want to fail on high risk, change policy here.
    return 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
