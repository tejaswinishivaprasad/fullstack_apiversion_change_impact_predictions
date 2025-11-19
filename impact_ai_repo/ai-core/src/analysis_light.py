#!/usr/bin/env python3
"""
analysis_light.py

Lightweight PR analyzer that:
 - finds changed files (git)
 - detects OpenAPI / swagger (.yaml, .yml, .json) files changed in the PR
 - compares HEAD vs origin/main (if available) and emits simple ACEs:
    - endpoint.added / endpoint.removed
    - endpoint.modified (method added/removed)
    - param.schema_changed (type/required)
    - response.removed (response code removed)
 - builds a tiny impact summary and writes JSON to --output

Designed to run in CI without heavy deps. Uses PyYAML if available; falls back to a dumb parser.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Try to import yaml if available; otherwise use json for json files and skip YAML parsing
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return (0, out.decode())
    except subprocess.CalledProcessError as e:
        return (e.returncode, (e.output.decode() if e.output else "")) 
    except Exception:
        return (1, "")

def changed_files() -> List[str]:
    """
    Return a list of changed files for the PR.

    Strategies (in order):
      1. git diff --name-only origin/main...HEAD
         - if empty, attempt to fetch origin/main and retry
      2. git diff --name-only HEAD~1..HEAD (local test fallback)
      3. git ls-files --modified
      4. curated canonical path specific diff (if the above didn't find anything)
      5. fallback: empty list
    """
    candidates: List[str] = []

    def try_cmd(cmd: List[str]) -> Tuple[int, str]:
        code, out = run_cmd(cmd)
        if code == 0 and out.strip():
            return (0, out)
        return (code, out)

    # 1) try origin/main...HEAD
    code, out = try_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        print("DEBUG: git diff origin/main...HEAD detected changes:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    # If nothing found, try to fetch origin/main then retry once (helps shallow checkouts)
    print("DEBUG: no changes found by origin/main...HEAD. Attempting to fetch origin/main and retry.", file=sys.stderr)
    _ = run_cmd(["git", "fetch", "origin", "main", "--depth=1"])
    code, out = try_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        print("DEBUG: After fetching, git diff origin/main...HEAD found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    # 2) HEAD~1..HEAD (local dev fallback)
    code, out = try_cmd(["git", "diff", "--name-only", "HEAD~1..HEAD"])
    if code == 0 and out.strip():
        print("DEBUG: git diff HEAD~1..HEAD found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    # 3) git ls-files --modified
    code, out = try_cmd(["git", "ls-files", "--modified"])
    if code == 0 and out.strip():
        print("DEBUG: git ls-files --modified found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    # 4) curated canonical path specific diff (helpful if your curated files live under a known folder)
    curated_path = "impact_ai_repo/ai-core/src/datasets/curated/canonical"
    print(f"DEBUG: trying curated-path diff for {curated_path}", file=sys.stderr)
    code, out = try_cmd(["git", "diff", "--name-only", f"origin/main...HEAD", "--", curated_path])
    if code == 0 and out.strip():
        print("DEBUG: curated-path git diff found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    # 5) nothing found
    print("DEBUG: no changed files detected by any strategy.", file=sys.stderr)
    return []


def load_spec_from_fs(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf8")
    if path.lower().endswith(".json"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    else:
        if _HAVE_YAML:
            try:
                return yaml.safe_load(text) or {}
            except Exception:
                return {}
        else:
            # very crude fallback: try JSON parse, else empty
            try:
                return json.loads(text)
            except Exception:
                return {}

def load_spec_from_git(ref_path: str) -> Dict[str, Any]:
    # ref_path e.g. "origin/main:apis/petstore.yaml"
    code, out = run_cmd(["git", "show", ref_path])
    if code != 0 or not out:
        return {}
    text = out
    if ref_path.lower().endswith(".json"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    else:
        if _HAVE_YAML:
            try:
                return yaml.safe_load(text) or {}
            except Exception:
                return {}
        else:
            try:
                return json.loads(text)
            except Exception:
                return {}

def path_methods(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # returns dict: path -> method -> operation-object
    paths = spec.get("paths") or {}
    result: Dict[str, Dict[str, Any]] = {}
    for p, methods in (paths.items() if isinstance(paths, dict) else []):
        if not isinstance(methods, dict):
            continue
        method_map: Dict[str, Any] = {}
        for m, op in methods.items():
            if not isinstance(op, dict):
                continue
            method_map[m.lower()] = op
        if method_map:
            result[p] = method_map
    return result

def param_key(p: Dict[str, Any]) -> str:
    # unique key to identify param: in+name
    return f"{p.get('in','')}.{p.get('name','')}"

def compare_params(base_params: List[Dict[str, Any]], head_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # returns list of detected param changes (type/required)
    changes = []
    base_map = {param_key(p): p for p in (base_params or [])}
    head_map = {param_key(p): p for p in (head_params or [])}
    for k in set(base_map) | set(head_map):
        b = base_map.get(k)
        h = head_map.get(k)
        if b and not h:
            changes.append({"type":"param.removed", "param":k, "before": b, "after": None})
        elif h and not b:
            changes.append({"type":"param.added", "param":k, "before": None, "after": h})
        elif b and h:
            # compare schema/type/required
            b_schema = (b.get("schema") or {})
            h_schema = (h.get("schema") or {})
            b_type = b_schema.get("type")
            h_type = h_schema.get("type")
            b_required = bool(b.get("required", False))
            h_required = bool(h.get("required", False))
            if b_type != h_type or b_required != h_required:
                changes.append({
                    "type": "param.schema_changed",
                    "param": k,
                    "before": {"type": b_type, "required": b_required},
                    "after": {"type": h_type, "required": h_required}
                })
    return changes

def compare_responses(b_op: Dict[str, Any], h_op: Dict[str, Any]) -> List[Dict[str, Any]]:
    changes = []
    b_res = b_op.get("responses", {}) if isinstance(b_op, dict) else {}
    h_res = h_op.get("responses", {}) if isinstance(h_op, dict) else {}
    for code in set(b_res.keys()) - set(h_res.keys()):
        changes.append({"type":"response.removed", "code": code, "before": b_res.get(code), "after": None})
    for code in set(h_res.keys()) - set(b_res.keys()):
        changes.append({"type":"response.added", "code": code, "before": None, "after": h_res.get(code)})
    return changes

def make_ace(kind: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": kind,
        "details": details
    }

def analyze_file_change(fname: str) -> List[Dict[str, Any]]:
    aces: List[Dict[str, Any]] = []
    # Load head spec from file system
    head_spec = load_spec_from_fs(fname)
    # Try to load base spec from origin/main:<fname>
    base_spec = load_spec_from_git(f"origin/main:{fname}")
    # If base_spec empty, try to fetch from main branch name (main/master fallback)
    if not base_spec:
        # already tried origin/main, try origin/master
        base_spec = load_spec_from_git(f"origin/master:{fname}")

    head_paths = path_methods(head_spec)
    base_paths = path_methods(base_spec)

    head_set = set(head_paths.keys())
    base_set = set(base_paths.keys())

    added_paths = sorted(list(head_set - base_set))
    removed_paths = sorted(list(base_set - head_set))
    common_paths = sorted(list(head_set & base_set))

    for p in added_paths:
        # list methods added
        methods = list(head_paths.get(p, {}).keys())
        for m in methods:
            aces.append(make_ace("endpoint.added", {"path": p, "method": m, "desc": f"Added {m.upper()} {p}"}))

    for p in removed_paths:
        methods = list(base_paths.get(p, {}).keys())
        for m in methods:
            aces.append(make_ace("endpoint.removed", {"path": p, "method": m, "desc": f"Removed {m.upper()} {p}"}))

    for p in common_paths:
        b_methods = base_paths.get(p, {})
        h_methods = head_paths.get(p, {})
        b_mset = set(b_methods.keys())
        h_mset = set(h_methods.keys())

        # method-level adds/removes
        added_m = sorted(list(h_mset - b_mset))
        removed_m = sorted(list(b_mset - h_mset))
        for m in added_m:
            aces.append(make_ace("endpoint.method_added", {"path": p, "method": m, "desc": f"Added method {m.upper()} on {p}"}))
        for m in removed_m:
            aces.append(make_ace("endpoint.method_removed", {"path": p, "method": m, "desc": f"Removed method {m.upper()} on {p}"}))

        # for shared methods, compare params & responses
        for m in sorted(list(b_mset & h_mset)):
            b_op = b_methods.get(m, {}) or {}
            h_op = h_methods.get(m, {}) or {}

            # params: in OpenAPI both operation-level and path-level params possible
            b_params = (b_op.get("parameters") or []) + (base_spec.get("paths", {}).get(p, {}).get("parameters") or [])
            h_params = (h_op.get("parameters") or []) + (head_spec.get("paths", {}).get(p, {}).get("parameters") or [])

            param_changes = compare_params(b_params, h_params)
            for pc in param_changes:
                # mark breaking if type changed or required became true->true/false? We'll flag type changes as breaking
                breaking = False
                if pc["type"] == "param.schema_changed":
                    before = pc.get("before", {})
                    after = pc.get("after", {})
                    if before.get("type") and after.get("type") and before.get("type") != after.get("type"):
                        breaking = True
                details = {"path": p, "method": m, **pc}
                details["breaking"] = breaking
                aces.append(make_ace(pc["type"], details))

            # responses
            resp_changes = compare_responses(b_op, h_op)
            for rc in resp_changes:
                # if a 2xx code was removed, mark as potentially breaking
                breaking = False
                try:
                    code = int(rc.get("code", "0"))
                    if 200 <= code < 300 and rc["type"] == "response.removed":
                        breaking = True
                except Exception:
                    pass
                details = {"path": p, "method": m, **rc, "breaking": breaking}
                aces.append(make_ace(rc["type"], details))

    return aces

def compute_simple_impact(aces: List[Dict[str, Any]]) -> Dict[str, Any]:
    # naive risk heuristics:
    risk = 0.0
    breaking_count = 0
    for a in aces:
        t = a.get("type", "")
        d = a.get("details", {}) or {}
        if d.get("breaking"):
            breaking_count += 1
    # risk base: each breaking gets heavy weight
    risk += min(1.0, 0.7 * breaking_count)
    # plus small penalty for count of changes
    risk += min(0.3, 0.05 * max(0, len(aces) - breaking_count))
    risk = min(1.0, risk)
    label = "low"
    if risk >= 0.6:
        label = "high"
    elif risk >= 0.25:
        label = "medium"
    return {"score": round(risk, 3), "label": label, "breaking_count": breaking_count, "total_aces": len(aces)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", "unknown"))
    parser.add_argument("--output", default=os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "pr-impact-report.json"))
    args = parser.parse_args()

    files = changed_files()
    # Filter for API-looking files
    api_files = [f for f in files if f.lower().endswith(('.yaml', '.yml', '.json')) and ('openapi' in f.lower() or 'swagger' in f.lower() or 'api' in f.lower())]
    # Also include specific path if a named file like 'apis/petstore.yaml' was changed
    if not api_files:
        # attempt looser match
        api_files = [f for f in files if f.lower().endswith(('.yaml', '.yml', '.json'))]

    aces_total: List[Dict[str, Any]] = []
    for f in api_files:
        try:
            aces = analyze_file_change(f)
            for a in aces:
                a["source_file"] = f
            aces_total.extend(aces)
        except Exception as e:
            # non-fatal: log to stderr for CI
            print(f"analysis error for {f}: {e}", file=sys.stderr)

    impact = compute_simple_impact(aces_total)

    report = {
        "status": "ok" if (aces_total or files) else "partial",
        "pr": args.pr,
        "files_changed": files,
        "api_files_changed": api_files,
        "atomic_change_events": aces_total,
        "impact_assessment": impact
    }

    # write output
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"Wrote {outpath}")

if __name__ == "__main__":
    main()
