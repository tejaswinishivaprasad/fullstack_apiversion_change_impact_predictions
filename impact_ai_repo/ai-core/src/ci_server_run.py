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
from typing import Any, Dict, List, Tuple, Optional

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
            can = Path(dp.get("canonical") if isinstance(dp.get("canonical"), (str, Path)) else dp["canonical"])
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
            for base in server.BASE_DATASETS:
                p2 = Path(vroot) / base / "canonical" / name_alt
                if p2.exists():
                    return p2
        except Exception:
            continue

    # legacy effective curated root - server.EFFECTIVE_CURATED_ROOT
    try:
        eff = Path(server.EFFECTIVE_CURATED_ROOT)
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
    try:
        return json.loads(text)
    except Exception:
        # fallback to server helper if available
        try:
            return server._load_json_or_yaml(Path(text)) if False else {}
        except Exception:
            return {}

# ---- New: normalize repository-relative/absolute paths to workspace-relative dataset paths
def normalize_rel_path(rel: str) -> str:
    """
    Normalize a changed-file path (rel) to the repository-relative path used by server.dataset_paths.
    Handles:
      - absolute local paths containing 'ai-core/src' (your local machine)
      - github prefix like 'impact_ai_repo/ai-core/src/...'
      - direct paths already relative to ai-core/src (no-op)
      - paths already starting with 'datasets/' or 'curated_clean/...'
    Returns the normalized portion starting at the ai-core/src relative tree (i.e. the path under ai-core/src)
    or the original rel if no transformation applies.
    """
    if not rel:
        return rel
    # unify separators
    r = rel.replace("\\", "/")
    # common anchors we expect
    anchors = ["ai-core/src/", "ai-core/src", "ai-core/"]
    for a in anchors:
        if a in r:
            # strip everything up to and including a
            idx = r.index(a) + len(a)
            return r[idx:].lstrip("/")
    # sometimes repo contains a top-level folder like 'impact_ai_repo/ai-core/src/...'
    if "impact_ai_repo/" in r:
        # strip impact_ai_repo/
        try:
            return r.split("impact_ai_repo/", 1)[1].lstrip("/")
        except Exception:
            pass
    # if path already starts with datasets or curated_clean, return as-is
    tokens = ["datasets/", "curated_clean/", "curated_noisy_light/", "curated_noisy_heavy/"]
    for t in tokens:
        if t in r:
            idx = r.index(t)
            return r[idx:]
    # fallback: return original
    return r

def analyze_pair_files(old_doc: Dict[str, Any], new_doc: Dict[str, Any], rel_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Robust analyze_pair_files replacement.

    - Parse blobs robustly (JSON -> YAML fallback using server helper).
    - Try diffing both dereferenced and raw documents.
    - If diffs empty, attempt to resolve dataset/pair via pair index and call server.report.
    - Always attempt backend/frontend impact enrichment via server.backend_impacts/ui_impacts.
    - Emit DEBUG logs to stderr for easier troubleshooting.
    """
    # helper to parse text/blob into dict (tries json then server helper)
    def _parse_blob_or_doc(obj):
        if obj is None:
            return {}
        # if it's already a dict
        if isinstance(obj, dict):
            return obj
        # if it's a Path-like (rare here), try reading
        if isinstance(obj, Path) and obj.exists():
            try:
                txt = obj.read_text(encoding="utf-8")
            except Exception:
                txt = ""
        else:
            # assume it's a string blob
            txt = str(obj)
        if not txt:
            return {}
        # try json
        try:
            return json.loads(txt)
        except Exception:
            pass
        # try server yaml/json helper if available
        try:
            if hasattr(server, "_load_json_or_yaml"):
                # _load_json_or_yaml expects a Path or string path; some implementations also accept a text blob
                # We'll attempt to write to a tempfile if helper needs a Path.
                # First try passing a Path-like (create a temp file) to be safe.
                with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tf:
                    tf.write(txt)
                    tf.flush()
                    tfname = tf.name
                try:
                    return server._load_json_or_yaml(Path(tfname))
                except Exception:
                    # last resort: try server._load_json_or_yaml on the text - some forks accept text
                    try:
                        return server._load_json_or_yaml(txt)
                    except Exception:
                        return {}
                finally:
                    try:
                        os.unlink(tfname)
                    except Exception:
                        pass
        except Exception:
            pass
        return {}

    # parse inputs robustly
    o_doc = _parse_blob_or_doc(old_doc)
    n_doc = _parse_blob_or_doc(new_doc)

    print(f"DEBUG: analyze_pair_files: rel_path={rel_path} | old_doc keys={list(o_doc.keys())[:6]} | new_doc keys={list(n_doc.keys())[:6]}", file=sys.stderr)

    # try deref first (helps schema diffs)
    try:
        old2 = dereference_components(o_doc)
        new2 = dereference_components(n_doc)
    except Exception as e:
        print("DEBUG: dereference_components failed:", e, file=sys.stderr)
        old2, new2 = o_doc, n_doc

    # function to run diff attempts and return (diffs, used_variant) where used_variant notes which docs used
    def _attempt_diffs():
        attempts = [
            ("deref", old2, new2),
            ("raw", o_doc, n_doc),
        ]
        for tag, a, b in attempts:
            try:
                d = server.diff_openapi(a, b)
                # normalize to list
                if d is None:
                    d = []
                # convert pydantic models to dicts if needed
                ds = []
                for item in d:
                    try:
                        if hasattr(item, "dict"):
                            di = item.dict()
                        elif isinstance(item, dict):
                            di = dict(item)
                        else:
                            # fallback to attribute extraction
                            di = {
                                "type": getattr(item, "type", None),
                                "path": getattr(item, "path", None),
                                "method": getattr(item, "method", None),
                                "detail": getattr(item, "detail", None),
                                "ace_id": getattr(item, "ace_id", None),
                            }
                    except Exception:
                        di = {"type": str(item)}
                    if "type" in di and isinstance(di["type"], str):
                        di["type"] = di["type"].upper()
                    ds.append(di)
                if ds:
                    return ds, tag
            except Exception as e:
                print(f"DEBUG: diff_openapi attempt ({tag}) failed: {e}", file=sys.stderr)
                continue
        # nothing found
        return [], None

    diffs_serial, used_tag = _attempt_diffs()

    print(f"DEBUG: diff attempts finished: used_tag={used_tag} | diffs_count={len(diffs_serial)}", file=sys.stderr)

    # If no diffs, try to find pair_id via pair index and call server.report (authoritative)
    if not diffs_serial:
        # try to derive canonical filenames from rel_path
        if rel_path:
            try:
                rel_norm = normalize_rel_path(rel_path)
                fname = Path(rel_norm).name
                # map --v2 to --v1
                v1fname = fname.replace("--v2.", "--v1.") if "--v2." in fname else (fname.replace("-v2.", "-v1.") if "-v2." in fname else fname.replace("v2", "v1", 1))
                print(f"DEBUG: trying report lookup via dataset indices for fname={fname} v1fname={v1fname}", file=sys.stderr)
                # try every dataset index
                for key in server.all_dataset_keys():
                    try:
                        idx = server.load_pair_index(key)
                        for pid, meta in (idx or {}).items():
                            mo = Path(str(meta.get("old_canonical") or meta.get("old") or "")).name if meta.get("old") else None
                            mn = Path(str(meta.get("new_canonical") or meta.get("new") or "")).name if meta.get("new") else None
                            if (mo and mn) and (mn.lower() == fname.lower() or mo.lower() == v1fname.lower() or mn.lower() == v1fname.lower()):
                                # call report
                                try:
                                    print(f"DEBUG: calling server.report(dataset={key}, old={mo}, new={mn}, pair_id={pid})", file=sys.stderr)
                                    rep = server.report(dataset=key, old=mo, new=mn, pair_id=pid)
                                    repd = rep.dict() if hasattr(rep, "dict") else dict(rep)
                                    be_imp = repd.get("backend_impacts") or []
                                    fe_imp = repd.get("frontend_impacts") or []
                                    ai_expl = repd.get("ai_explanation") or repd.get("ai_explanation", "")
                                    summary = repd.get("summary") or {}
                                    service_risk = float(summary.get("service_risk", repd.get("risk_score", 0.0)))
                                    diffs_from_report = []
                                    # translate report 'details' to our diffs_serial shape if needed
                                    for d in repd.get("details", []) or []:
                                        dd = dict(d)
                                        if "type" in dd and isinstance(dd["type"], str):
                                            dd["type"] = dd["type"].upper()
                                        diffs_from_report.append(dd)
                                    if not diffs_from_report:
                                        # still proceed with empty diffs but with impacts
                                        diffs_from_report = []
                                    return {
                                        "diffs": diffs_from_report,
                                        "analyze": {
                                            "summary": {"service_risk": service_risk, "num_aces": len(diffs_from_report)},
                                            "backend_impacts": be_imp,
                                            "frontend_impacts": fe_imp,
                                            "ai_explanation": ai_expl,
                                            "pair_id": pid,
                                        },
                                    }
                                except Exception as e:
                                    print("DEBUG: server.report call failed during fallback lookup:", e, file=sys.stderr)
                                    continue
                    except Exception:
                        continue
            except Exception as e:
                print("DEBUG: fallback report lookup failed:", e, file=sys.stderr)

    # If we have diffs from earlier, continue to call api_analyze and enrich
    # Call api_analyze to get model score if available
    analy = None
    try:
        baseline_str = json.dumps(o_doc)
        candidate_str = json.dumps(n_doc)
        analy = server.api_analyze(baseline=baseline_str, candidate=candidate_str, dataset=None, options=None)
    except Exception as e:
        print("DEBUG: server.api_analyze failed:", e, file=sys.stderr)
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # extract score, but prefer server-provided summary where present
    try:
        score = float((analy.get("summary") or {}).get("service_risk", 0.0))
    except Exception:
        score = 0.0

    # Prepare graph impacts (best-effort)
    try:
        g = server.load_graph()
    except Exception:
        g = None

    # best effort service guess
    service_guess = None
    if rel_path:
        try:
            rel_norm = normalize_rel_path(rel_path)
            st = Path(rel_norm).stem
            # expected naming: openapi--<service>--<hash>--v2.canonical
            if "--" in st:
                parts = st.split("--")
                if len(parts) > 1:
                    service_guess = parts[1]
                else:
                    service_guess = parts[0]
            else:
                service_guess = st.split("-v")[0] if "-v" in st else st
        except Exception:
            service_guess = None
    if not service_guess:
        service_guess = (n_doc.get("info", {}) or {}).get("title") or "unknown"

    try:
        pfeats = server.producer_features(g, service_guess)
    except Exception:
        pfeats = {}

    try:
        changed_paths = [server.normalize_path(d.get("path")) for d in diffs_serial if d.get("path")]
    except Exception:
        changed_paths = [d.get("path") for d in diffs_serial if d.get("path")]

    try:
        be_imp = server.backend_impacts(g, service_guess, changed_paths)
    except Exception:
        be_imp = []

    try:
        fe_imp = server.ui_impacts(g, service_guess, changed_paths)
    except Exception:
        fe_imp = []

    # ai explanation fallback
    ai_expl = analy.get("explanation") or analy.get("ai_explanation") or ""
    if not ai_expl:
        try:
            ai_expl = server.make_explanation(score, diffs_serial, pfeats, {}, be_imp, fe_imp)
        except Exception:
            ai_expl = ""

    # Build final payload
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
        # Print debug-ish info to stderr so CI logs show it (you asked where to see prints)
        print("DEBUG: original_rel =", rel, file=sys.stderr)
        rel_norm = normalize_rel_path(rel)
        print("DEBUG: normalized_rel =", rel_norm, file=sys.stderr)

        p = Path(rel)
        files_processed.append(rel)
        # if v2 canonical file, try to find v1 on filesystem (including across curated variants)
        # otherwise, try to fetch origin/main copy
        try:
            if p.exists():
                # local workspace file present
                new_doc = read_json_file_if_exists(p)
                # find local v1 counterpart across variants first using normalized rel
                v1cand = find_local_v1_across_variants(Path(rel_norm))
                if v1cand and v1cand.exists():
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
