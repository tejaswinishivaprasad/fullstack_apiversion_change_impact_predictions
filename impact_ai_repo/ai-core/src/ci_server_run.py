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
from typing import Any, Dict, List, Tuple

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
    Heuristic: replace '--v2.' with '--v1.' in filename.
    If file exists on filesystem, return Path. Otherwise return Path with same dir & name.
    """
    name = v2path.name
    if "--v2." in name:
        alt = name.replace("--v2.", "--v1.")
    elif "-v2." in name:
        alt = name.replace("-v2.", "-v1.")
    else:
        # fallback: try replacing 'v2' segments (risky)
        alt = name.replace("v2", "v1", 1)
    cand = v2path.parent / alt
    return cand

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
        return {}

def analyze_pair_files(old_doc: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use server's in-process logic to compute diffs/predictions.
    We'll:
      - diff via server.diff_openapi (gives list of DiffItem)
      - call server.api_analyze with JSON payloads to get model-style predictions/summary
    """
    # run a shallow deref to improve detection of schema changes that use $ref
    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    # the server.diff_openapi returns list of DiffItem pydantic objects (or simple objects)
    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN: server.diff_openapi failed:", e, file=sys.stderr)
        diffs = []

    # convert diff items to plain dicts
    diffs_serial = []
    for d in diffs:
        try:
            # support pydantic model or simple dataclass-like
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
        diffs_serial.append(dd)

    # call server.api_analyze in-process to get predictions & summary
    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy = server.api_analyze(baseline=baseline_str, candidate=candidate_str, dataset=None, options=None)
    except Exception as e:
        print("WARN: server.api_analyze raised:", e, file=sys.stderr)
        analy = {"run_id": None, "predictions": [], "summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # Build combined result
    out = {
        "diffs": diffs_serial,
        "analyze": analy,
    }
    return out

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
    # prefer canonical curated files inside datasets path
    api_files = [f for f in api_files if "datasets/curated" in f or "canonical" in f] or api_files

    results: List[Dict[str, Any]] = []
    files_processed: List[str] = []

    for rel in api_files:
        p = Path(rel)
        files_processed.append(rel)
        # if v2 canonical file, try to find v1 on filesystem
        # otherwise, try to fetch origin/main copy
        try:
            if p.exists():
                new_doc = json.loads(p.read_text(encoding="utf-8"))
                # find local v1 counterpart first
                v1cand = find_counterpart_v1(p)
                if v1cand.exists():
                    old_doc = json.loads(v1cand.read_text(encoding="utf-8"))
                else:
                    # try to fetch from origin/main
                    blob = git_show_file(f"origin/main:{rel}")
                    if blob:
                        old_doc = load_json_text(blob)
                    else:
                        # fallback: try HEAD~1 version
                        code, out = run_cmd(["git", "show", "HEAD~1:" + rel])
                        old_doc = load_json_text(out) if code == 0 else {}
            else:
                # file doesn't exist in workspace (maybe deleted) -> try git show for new and old
                blob_new = git_show_file(f"origin/{os.environ.get('GITHUB_REF_NAME', 'HEAD')}:{rel}") or git_show_file(f"HEAD:{rel}")
                new_doc = load_json_text(blob_new)
                blob_old = git_show_file(f"origin/main:{rel}")
                old_doc = load_json_text(blob_old) if blob_old else {}
        except Exception as e:
            print(f"WARN: failed to load files for {rel}: {e}", file=sys.stderr)
            old_doc, new_doc = {}, {}

        # skip trivial empty pairs
        try:
            pair_res = analyze_pair_files(old_doc, new_doc)
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
    for e in results:
        analy = e["result"].get("analyze", {}) or {}
        summary = analy.get("summary") or {}
        s_r = float(summary.get("service_risk", 0.0))
        max_risk = max(max_risk, s_r)
        naces = int(summary.get("num_aces", 0))
        total_aces += naces
        # collect diffs (atomic change events)
        for d in e["result"].get("diffs", []):
            atomic_aces.append(d)

    compact = {
        "status": full_out["status"],
        "pr": str(args.pr),
        "files_changed": files_processed,
        "api_files_changed": files_processed,
        "atomic_change_events": atomic_aces,
        "impact_assessment": {
            "score": round(float(max_risk), 3),
            "label": "high" if max_risk >= 0.6 else "medium" if max_risk >= 0.25 else "low",
            "breaking_count": sum(1 for a in atomic_aces if a.get("type", "").lower() in ("param_changed", "response_schema_changed", "requestbody_schema_changed", "endpoint_removed")),
            "total_aces": len(atomic_aces)
        }
    }

    Path(args.output_summary).write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print("Wrote summary to", args.output_summary, file=sys.stderr)

    # exit 0 (CI can inspect outputs). If you want to fail on high risk, change policy here.
    return 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
