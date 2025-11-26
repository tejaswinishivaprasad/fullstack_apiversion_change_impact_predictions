#!/usr/bin/env python3
"""
analysis_light.py - improved schema change detection

Lightweight PR analyzer that:
 - finds changed files (git)
 - detects OpenAPI / swagger (.yaml, .yml, .json) files changed in the PR
 - compares HEAD vs origin/main (if available) and emits ACEs:
    - endpoint.added / endpoint.removed
    - endpoint.method_added / endpoint.method_removed
    - param.schema_changed (type/required)
    - response.added / response.removed
    - schema.property_type_changed (property type changes in request/response/components)
    - schema.property_required_changed (required flag changes)
 - builds a tiny impact summary and writes JSON to --output

Designed to run in CI without heavy deps. Uses PyYAML if available; falls back to a crude JSON-only parser.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
      4. curated canonical path specific diff
      5. fallback: empty list
    """
    def try_cmd(cmd: List[str]) -> Tuple[int, str]:
        code, out = run_cmd(cmd)
        if code == 0 and out.strip():
            return (0, out)
        return (code, out)

    code, out = try_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        print("DEBUG: git diff origin/main...HEAD detected changes:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    print("DEBUG: no changes found by origin/main...HEAD. Attempting to fetch origin/main and retry.", file=sys.stderr)
    _ = run_cmd(["git", "fetch", "origin", "main", "--depth=1"])
    code, out = try_cmd(["git", "diff", "--name-only", "origin/main...HEAD"])
    if code == 0 and out.strip():
        print("DEBUG: After fetching, git diff origin/main...HEAD found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    code, out = try_cmd(["git", "diff", "--name-only", "HEAD~1..HEAD"])
    if code == 0 and out.strip():
        print("DEBUG: git diff HEAD~1..HEAD found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    code, out = try_cmd(["git", "ls-files", "--modified"])
    if code == 0 and out.strip():
        print("DEBUG: git ls-files --modified found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

    curated_path = "impact_ai_repo/ai-core/src/datasets/curated/canonical"
    print(f"DEBUG: trying curated-path diff for {curated_path}", file=sys.stderr)
    code, out = try_cmd(["git", "diff", "--name-only", "origin/main...HEAD", "--", curated_path])
    if code == 0 and out.strip():
        print("DEBUG: curated-path git diff found:", file=sys.stderr)
        print(out, file=sys.stderr)
        return [l.strip() for l in out.splitlines() if l.strip()]

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
            try:
                return json.loads(text)
            except Exception:
                return {}

def load_spec_from_git(ref_path: str) -> Dict[str, Any]:
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
    return f"{p.get('in','')}.{p.get('name','')}"

def compare_params(base_params: List[Dict[str, Any]], head_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

def _pick_first_media_schema(op_or_resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Given an operation or response object, try to extract the first content schema dict
    (e.g. responses['200'].content['application/json'].schema). Returns the schema dict or None.
    """
    if not isinstance(op_or_resp, dict):
        return None
    content = op_or_resp.get("content") or {}
    if not isinstance(content, dict):
        return None
    # pick first media type available
    for mt, spec in content.items():
        if isinstance(spec, dict) and "schema" in spec:
            return spec.get("schema")
    return None

def compare_schemas(base_schema: Dict[str, Any], head_schema: Dict[str, Any], context_path: str) -> List[Dict[str, Any]]:
    """
    Very small schema comparator: looks for property-level type changes and required flag changes.
    context_path is a human-friendly prefix (e.g. "paths./pets.get.requestBody" or "components.schemas.User")
    """
    aces: List[Dict[str, Any]] = []
    if not isinstance(base_schema, dict) or not isinstance(head_schema, dict):
        return aces

    # Simple: if both are primitive type declarations, compare top-level type change
    b_type = base_schema.get("type")
    h_type = head_schema.get("type")
    if b_type and h_type and b_type != h_type:
        aces.append(make_ace("schema.type_changed", {
            "context": context_path,
            "before": b_type,
            "after": h_type,
            "breaking": True
        }))
        # type changed at top-level is already a strong signal; still continue to look for properties if object
    # If object properties exist, walk them
    b_props = base_schema.get("properties") or {}
    h_props = head_schema.get("properties") or {}
    if isinstance(b_props, dict) and isinstance(h_props, dict):
        # required lists
        b_req = set(base_schema.get("required") or [])
        h_req = set(head_schema.get("required") or [])
        # newly required properties (added to required) are breaking
        for prop in sorted(list(h_req - b_req)):
            aces.append(make_ace("schema.property_required_changed", {
                "context": context_path,
                "property": prop,
                "before_required": False,
                "after_required": True,
                "breaking": True
            }))
        # removed required properties (became optional) are non-breaking (or less breaking)
        for prop in sorted(list(b_req - h_req)):
            aces.append(make_ace("schema.property_required_changed", {
                "context": context_path,
                "property": prop,
                "before_required": True,
                "after_required": False,
                "breaking": False
            }))

        # compare each property type if present
        for prop in sorted(set(list(b_props.keys()) + list(h_props.keys()))):
            b_prop = b_props.get(prop)
            h_prop = h_props.get(prop)
            if b_prop is None and h_prop is not None:
                aces.append(make_ace("schema.property_added", {
                    "context": context_path,
                    "property": prop,
                    "before": None,
                    "after": h_prop,
                    "breaking": False
                }))
                continue
            if b_prop is not None and h_prop is None:
                aces.append(make_ace("schema.property_removed", {
                    "context": context_path,
                    "property": prop,
                    "before": b_prop,
                    "after": None,
                    "breaking": True
                }))
                continue
            # both exist — compare basic type and if it's an array compare items.type
            try:
                bp_type = b_prop.get("type")
                hp_type = h_prop.get("type")
                if bp_type != hp_type:
                    # arrays: check item type if both arrays
                    if bp_type == "array" and hp_type == "array":
                        b_items = (b_prop.get("items") or {})
                        h_items = (h_prop.get("items") or {})
                        bi_type = b_items.get("type")
                        hi_type = h_items.get("type")
                        if bi_type and hi_type and bi_type != hi_type:
                            aces.append(make_ace("schema.property_type_changed", {
                                "context": context_path,
                                "property": prop,
                                "before": {"type": bp_type, "items_type": bi_type},
                                "after": {"type": hp_type, "items_type": hi_type},
                                "breaking": True
                            }))
                            continue
                    # generic type change
                    aces.append(make_ace("schema.property_type_changed", {
                        "context": context_path,
                        "property": prop,
                        "before": {"type": bp_type},
                        "after": {"type": hp_type},
                        "breaking": True
                    }))
                else:
                    # same top-level type; for objects, recurse one level (avoid deep recursion to stay fast)
                    if bp_type == "object":
                        nested = compare_schemas(b_prop, h_prop, context_path + f".{prop}")
                        aces.extend(nested)
                    elif bp_type == "array":
                        b_items = (b_prop.get("items") or {})
                        h_items = (h_prop.get("items") or {})
                        bi_type = b_items.get("type")
                        hi_type = h_items.get("type")
                        if bi_type and hi_type and bi_type != hi_type:
                            aces.append(make_ace("schema.property_type_changed", {
                                "context": context_path,
                                "property": prop,
                                "before": {"type": "array", "items_type": bi_type},
                                "after": {"type": "array", "items_type": hi_type},
                                "breaking": True
                            }))
            except Exception:
                continue
    return aces

def analyze_file_change(fname: str) -> List[Dict[str, Any]]:
    aces: List[Dict[str, Any]] = []
    # Load head spec from file system
    head_spec = load_spec_from_fs(fname)
    # Try to load base spec from origin/main:<fname>
    base_spec = load_spec_from_git(f"origin/main:{fname}")
    if not base_spec:
        base_spec = load_spec_from_git(f"origin/master:{fname}")

    head_paths = path_methods(head_spec)
    base_paths = path_methods(base_spec)
    head_set = set(head_paths.keys())
    base_set = set(base_paths.keys())

    added_paths = sorted(list(head_set - base_set))
    removed_paths = sorted(list(base_set - head_set))
    common_paths = sorted(list(head_set & base_set))

    for p in added_paths:
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

        added_m = sorted(list(h_mset - b_mset))
        removed_m = sorted(list(b_mset - h_mset))
        for m in added_m:
            aces.append(make_ace("endpoint.method_added", {"path": p, "method": m, "desc": f"Added method {m.upper()} on {p}"}))
        for m in removed_m:
            aces.append(make_ace("endpoint.method_removed", {"path": p, "method": m, "desc": f"Removed method {m.upper()} on {p}"}))

        for m in sorted(list(b_mset & h_mset)):
            b_op = b_methods.get(m, {}) or {}
            h_op = h_methods.get(m, {}) or {}

            # params: both operation-level and path-level parameters
            b_params = (b_op.get("parameters") or []) + (base_spec.get("paths", {}).get(p, {}).get("parameters") or [])
            h_params = (h_op.get("parameters") or []) + (head_spec.get("paths", {}).get(p, {}).get("parameters") or [])

            param_changes = compare_params(b_params, h_params)
            for pc in param_changes:
                breaking = False
                if pc["type"] == "param.schema_changed":
                    before = pc.get("before", {})
                    after = pc.get("after", {})
                    if before.get("type") and after.get("type") and before.get("type") != after.get("type"):
                        breaking = True
                details = {"path": p, "method": m, **pc}
                details["breaking"] = breaking
                aces.append(make_ace(pc["type"], details))

            # responses (structural)
            resp_changes = compare_responses(b_op, h_op)
            for rc in resp_changes:
                breaking = False
                try:
                    code = int(rc.get("code", "0"))
                    if 200 <= code < 300 and rc["type"] == "response.removed":
                        breaking = True
                except Exception:
                    pass
                details = {"path": p, "method": m, **rc, "breaking": breaking}
                aces.append(make_ace(rc["type"], details))

            # response body schema diffs for common response codes
            b_res_map = b_op.get("responses", {}) or {}
            h_res_map = h_op.get("responses", {}) or {}
            for code in sorted(set(b_res_map.keys()) & set(h_res_map.keys())):
                b_resp = b_res_map.get(code) or {}
                h_resp = h_res_map.get(code) or {}
                b_schema = _pick_first_media_schema(b_resp) or {}
                h_schema = _pick_first_media_schema(h_resp) or {}
                if b_schema and h_schema:
                    schema_aces = compare_schemas(b_schema, h_schema, f"paths.{p}.{m}.responses.{code}")
                    for s in schema_aces:
                        # include path/method context
                        s["details"].update({"path": p, "method": m, "response_code": code})
                        aces.append(s)

            # requestBody schema diffs
            b_req_schema = _pick_first_media_schema(b_op.get("requestBody", {}) or {}) or {}
            h_req_schema = _pick_first_media_schema(h_op.get("requestBody", {}) or {}) or {}
            if b_req_schema and h_req_schema:
                r_aces = compare_schemas(b_req_schema, h_req_schema, f"paths.{p}.{m}.requestBody")
                for ra in r_aces:
                    ra["details"].update({"path": p, "method": m, "location": "requestBody"})
                    aces.append(ra)

    # components/schemas: compare top-level component schema definitions if present
    try:
        b_comps = base_spec.get("components", {}).get("schemas", {}) or {}
        h_comps = head_spec.get("components", {}).get("schemas", {}) or {}
        for comp in sorted(set(list(b_comps.keys()) + list(h_comps.keys()))):
            b_schema = b_comps.get(comp)
            h_schema = h_comps.get(comp)
            if b_schema is None and h_schema is not None:
                aces.append(make_ace("schema.component_added", {"component": comp, "breaking": False}))
            elif b_schema is not None and h_schema is None:
                aces.append(make_ace("schema.component_removed", {"component": comp, "breaking": True}))
            elif b_schema is not None and h_schema is not None:
                comp_aces = compare_schemas(b_schema, h_schema, f"components.schemas.{comp}")
                for ca in comp_aces:
                    ca["details"].update({"component": comp})
                    aces.append(ca)
    except Exception:
        pass

    return aces

def compute_simple_impact(aces: List[Dict[str, Any]]) -> Dict[str, Any]:
    # naive risk heuristics:
    risk = 0.0
    breaking_count = 0
    for a in aces:
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
    if not api_files:
        api_files = [f for f in files if f.lower().endswith(('.yaml', '.yml', '.json'))]

    aces_total: List[Dict[str, Any]] = []
    for f in api_files:
        try:
            aces = analyze_file_change(f)
            for a in aces:
                a["source_file"] = f
            aces_total.extend(aces)
        except Exception as e:
            print(f"analysis error for {f}: {e}", file=sys.stderr)

    impact = compute_simple_impact(aces_total)

    report = {
        "status": "ok" if (aces_total or files) else "partial",
        "pr": args.pr,
        "files_changed": files,
        "api_files_changed": api_files,
        "atomic_change_events": [ { "id": a.get("id"), "type": a.get("type"), "details": a.get("details") } for a in aces_total ],
        "impact_assessment": impact
    }

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"Wrote {outpath}")

if __name__ == "__main__":
    main()
