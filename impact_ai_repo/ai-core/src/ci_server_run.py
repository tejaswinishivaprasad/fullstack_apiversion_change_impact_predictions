#!/usr/bin/env python3
"""
ci_server_run.py

Lightweight CI wrapper that:
 - finds changed files in a PR (git)
 - maps canonical v2 -> v1 or fetches base from origin/main
 - lightly dereferences local '#/components/schemas/...' refs (shallow)
 - invokes server.api_analyze(...) (in-process) to reuse server logic
 - writes pr-impact-full.json (detailed) and pr-impact-report.json (compact)

Enhancements:
 - dataset detection added so backend/frontend impacts are generated
 - includes deterministic artifact/report_id fields
 - preserves multi-line explanation fields
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
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


# -------------------- Helpers --------------------

def detect_dataset_from_path(path: str) -> str | None:
    """
    Infer dataset key from file path.
    Looks for dataset folders such as openapi, petclinic, openrewrite.
    """
    parts = Path(path).parts
    for name in ["openapi", "petclinic", "openrewrite"]:
        if name in parts:
            return name
    return None


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return (0, out.decode())
    except subprocess.CalledProcessError as e:
        return (e.returncode, e.output.decode() if e.output else "")
    except Exception as e:
        return (1, str(e))


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
    return v2path.parent / alt


def read_json_file_if_exists(p: Path) -> Dict[str, Any]:
    try:
        if p.exists():
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
                m2 = {}
                for m, op in (methods or {}).items():
                    m2[m] = resolve(op)
                new_paths[p] = m2
            out["paths"] = new_paths
    except Exception:
        pass
    return out


def analyze_pair_files(old_doc, new_doc, rel_path: str):
    """
    Wrapper around server.diff_openapi + server.api_analyze.
    Now includes correct dataset detection for dependency impacts.
    """
    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN: diff_openapi failed:", e, file=sys.stderr)
        diffs = []

    diffs_serial = []
    for d in diffs:
        try:
            dd = d.dict() if hasattr(d, "dict") else {
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

    dataset = detect_dataset_from_path(rel_path)

    try:
        analy = server.api_analyze(
            baseline=json.dumps(old_doc),
            candidate=json.dumps(new_doc),
            dataset=dataset,
            options=None
        )
    except Exception as e:
        print("WARN: api_analyze failed:", e, file=sys.stderr)
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # ai explanation
    ai_expl = analy.get("ai_explanation") or analy.get("explanation") or ""

    # pack
    analy_out = dict(analy)
    analy_out.setdefault("summary", {})
    analy_out["summary"].setdefault("num_aces", len(diffs_serial))
    analy_out["ai_explanation"] = ai_expl
    analy_out["ai_explanation_lines"] = ai_expl.splitlines() if ai_expl else []

    return {
        "diffs": diffs_serial,
        "analyze": analy_out,
    }


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", "unknown"))
    parser.add_argument("--output-full", default="pr-impact-full.json")
    parser.add_argument("--output-summary", default="pr-impact-report.json")
    args = parser.parse_args()

    changed = git_changed_files()
    print("CI: changed files:", changed, file=sys.stderr)

    api_files = [f for f in changed if f.lower().endswith((".json", ".yaml", ".yml"))]
    results = []
    files_processed = []

    for rel in api_files:
        p = Path(rel)
        files_processed.append(rel)

        # load baseline and candidate
        if p.exists():
            new_doc = read_json_file_if_exists(p)
            v1cand = find_counterpart_v1(p)
            if v1cand.exists():
                old_doc = read_json_file_if_exists(v1cand)
            else:
                blob = git_show_file(f"origin/main:{rel}")
                old_doc = json.loads(blob) if blob else {}
        else:
            ref_branch = os.environ.get("GITHUB_REF_NAME", "HEAD")
            blob_new = git_show_file(f"origin/{ref_branch}:{rel}") or git_show_file(f"HEAD:{rel}")
            new_doc = json.loads(blob_new) if blob_new else {}
            blob_old = git_show_file(f"origin/main:{rel}")
            old_doc = json.loads(blob_old) if blob_old else {}

        pair_res = analyze_pair_files(old_doc, new_doc, rel)
        results.append({"file": rel, "result": pair_res})

    # full report
    full_out = {
        "status": "ok" if results else "partial",
        "pr": str(args.pr),
        "files_changed": files_processed,
        "entries": results
    }
    Path(args.output_full).write_text(json.dumps(full_out, indent=2, ensure_ascii=False))

    # compact summary
    atomic = []
    max_risk = 0.0
    be_all = []
    fe_all = []
    ai_lines = []
    ai_str = ""
    pair_id = ""

    for entry in results:
        res = entry["result"]
        analy = res.get("analyze", {})
        summary = analy.get("summary", {})

        risk = float(summary.get("service_risk", 0.0))
        max_risk = max(max_risk, risk)

        # collect ACEs
        for d in res.get("diffs", []):
            atomic.append(d)

        # collect impacts
        be_all.extend(analy.get("backend_impacts", []) or [])
        fe_all.extend(analy.get("frontend_impacts", []) or [])

        if not ai_str:
            ai_str = analy.get("ai_explanation", "") or ""
        if not ai_lines:
            ai_lines = analy.get("ai_explanation_lines", []) or []

    band = "High" if max_risk >= 0.7 else "Medium" if max_risk >= 0.4 else "Low"
    level = "BLOCK" if max_risk >= 0.7 else "WARN" if max_risk >= 0.4 else "PASS"

    safe_pr = str(args.pr).replace("/", "-")
    artifact_id = f"pr-impact-report-{safe_pr}"

    compact = {
        "status": full_out["status"],
        "pr": str(args.pr),
        "files_changed": files_processed,
        "api_files_changed": files_processed,
        "atomic_change_events": atomic,
        "impact_assessment": {
            "score": round(max_risk, 3),
            "label": level.lower(),
            "breaking_count": sum(1 for a in atomic if a.get("type", "").lower() in ("endpoint_removed", "param_changed", "requestbody_schema_changed", "response_schema_changed")),
            "total_aces": len(atomic)
        },
        "risk_score": round(max_risk, 3),
        "risk_band": band,
        "risk_level": level,
        "backend_impacts": be_all,
        "frontend_impacts": fe_all,
        "ai_explanation": ai_str,
        "ai_explanation_lines": ai_lines,
        "artifact": artifact_id,
        "report_id": artifact_id,
    }

    Path(args.output_summary).write_text(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
