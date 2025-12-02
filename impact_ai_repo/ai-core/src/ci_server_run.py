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
 - includes deterministic artifact/report_id fields (pr-based) to help dedupe/update PR comments
 - exposes 'ai_explanation_lines' (list[str]) to allow safe reconstitution of multi-line explanations
 - ensures pair_id present under top-level and metadata for downstream consumers
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            can = Path(dp.get("canonical", dp.get("canonical")))
            p = can / name_alt
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
            if "$ref" in obj:
                ref = obj["$ref"]
                if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
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
            return {}
        except Exception:
            return {}

def analyze_pair_files(old_doc: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN: diff failed:", e, file=sys.stderr)
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

    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy = server.api_analyze(baseline=baseline_str, candidate=candidate_str, dataset=None, options=None)
    except Exception:
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    pair_id = None
    try:
        v = analy.get("versioning") or {}
        m = analy.get("metadata") or {}
        b = analy.get("backend") or {}
        pair_id = v.get("pair_id") or m.get("pair_id") or b.get("pair_id")
    except Exception:
        pass

    if not pair_id:
        for d in diffs_serial:
            aid = d.get("ace_id") or ""
            if isinstance(aid, str) and "::" in aid and aid.startswith("pair-"):
                pair_id = aid.split("::")[0]
                break

    ai_expl = analy.get("ai_explanation") or analy.get("explanation")
    if not ai_expl:
        try:
            score = float((analy.get("summary") or {}).get("service_risk", 0.0))
            be_imp = analy.get("backend_impacts") or []
            fe_imp = analy.get("frontend_impacts") or []
            ai_expl = server.make_explanation(score, diffs_serial, {}, analy.get("versioning") or {}, be_imp, fe_imp)
        except Exception:
            ai_expl = ""

    analy_out = dict(analy)
    analy_out.setdefault("summary", analy_out.get("summary") or {})
    analy_out["summary"].setdefault("service_risk", float(analy_out["summary"].get("service_risk") or 0.0))
    analy_out["summary"].setdefault("num_aces", len(diffs_serial))
    analy_out["pair_id"] = pair_id or analy_out.get("pair_id") or ""
    analy_out["ai_explanation"] = ai_expl or ""
    analy_out["ai_explanation_lines"] = ai_expl.splitlines() if ai_expl else []

    return {"diffs": diffs_serial, "analyze": analy_out}

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
    api_files = [f for f in api_files if any(tok in f for tok in curated_tokens)] or api_files

    results: List[Dict[str, Any]] = []
    files_processed: List[str] = []

    # -------------------------------
    # NEW: aggregation lists for BE/FE
    # -------------------------------
    BE_IMPACTS_ALL: List[Dict[str, Any]] = []
    FE_IMPACTS_ALL: List[Dict[str, Any]] = []

    for rel in api_files:
        p = Path(rel)
        files_processed.append(rel)

        try:
            if p.exists():
                new_doc = read_json_file_if_exists(p)
                v1cand = find_local_v1_across_variants(p)
                if v1cand.exists():
                    old_doc = read_json_file_if_exists(v1cand)
                else:
                    blob = git_show_file(f"origin/main:{rel}")
                    old_doc = load_json_text(blob) if blob else {}
            else:
                ref_branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("BRANCH") or "HEAD"
                blob_new = (git_show_file(f"origin/{ref_branch}:{rel}")
                            or git_show_file(f"HEAD:{rel}")
                            or git_show_file(f"origin/main:{rel}"))
                new_doc = load_json_text(blob_new)
                blob_old = git_show_file(f"origin/main:{rel}")
                old_doc = load_json_text(blob_old) if blob_old else {}
        except Exception as e:
            print("WARN: failed to load files:", e, file=sys.stderr)
            old_doc, new_doc = {}, {}

        try:
            pair_res = analyze_pair_files(old_doc, new_doc)
        except Exception as e:
            print("WARN: analyze_pair_files exception:", e, file=sys.stderr)
            pair_res = {"diffs": [], "analyze": {"summary": {"service_risk": 0.0, "num_aces": 0}}}

        # -------------------------------
        # NEW: aggregate BE/FE impacts
        # -------------------------------
        try:
            analy = pair_res.get("analyze", {})
            be = analy.get("backend_impacts") or []
            fe = analy.get("frontend_impacts") or []
            if isinstance(be, list) and be:
                BE_IMPACTS_ALL.extend(be)
            if isinstance(fe, list) and fe:
                FE_IMPACTS_ALL.extend(fe)
        except Exception:
            pass

        results.append({"file": rel, "result": pair_res})

    full_out = {
        "status": "ok" if results else "partial",
        "pr": str(args.pr),
        "files_changed": files_processed,
        "files_analyzed": len(results),
        "entries": results
    }
    Path(args.output_full).write_text(json.dumps(full_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote full output to", args.output_full, file=sys.stderr)

    max_risk = 0.0
    atomic_aces = []
    pair_id_top = None
    ai_expl_top = None
    ai_expl_lines_top: List[str] = []

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
            pair_id_top = (
                analy.get("pair_id")
                or (analy.get("versioning") or {}).get("pair_id")
                or (analy.get("metadata") or {}).get("pair_id")
            )
        if not ai_expl_top:
            ai_expl_top = analy.get("ai_explanation") or analy.get("explanation")
            if not ai_expl_lines_top:
                ai_expl_lines_top = analy.get("ai_explanation_lines") or []

    def _band_label(score: float):
        if score >= 0.7:
            return "High", "BLOCK"
        if score >= 0.4:
            return "Medium", "WARN"
        return "Low", "PASS"

    band, level = _band_label(max_risk)

    safe_pr = str(args.pr).replace("/", "-")
    artifact_id = f"pr-impact-report-{safe_pr}"
    report_id = artifact_id

    compact = {
        "status": full_out["status"],
        "pr": str(args.pr),
        "files_changed": files_processed,
        "api_files_changed": files_processed,
        "atomic_change_events": atomic_aces,
        "impact_assessment": {
            "score": round(float(max_risk), 3),
            "label": "high" if max_risk >= 0.6 else "medium" if max_risk >= 0.25 else "low",
            "breaking_count": sum(
                1 for a in atomic_aces
                if a.get("type", "").lower() in (
                    "param_changed",
                    "response_schema_changed",
                    "requestbody_schema_changed",
                    "endpoint_removed"
                )
            ),
            "total_aces": len(atomic_aces)
        },
        "risk_score": round(float(max_risk), 3),
        "risk_band": band,
        "risk_level": level,
        "ai_explanation": ai_expl_top or "",
        "ai_explanation_lines": ai_expl_lines_top or (ai_expl_top.splitlines() if ai_expl_top else []),
        "artifact": artifact_id,
        "report_id": report_id,
        "pair_id": pair_id_top or "",
        "metadata": {"pair_id": pair_id_top or ""}
    }

    # -------------------------------
    # NEW: include aggregated BE/FE in summary
    # -------------------------------
    try:
        compact["backend_impacts"] = BE_IMPACTS_ALL
        compact["frontend_impacts"] = FE_IMPACTS_ALL
    except Exception:
        compact.setdefault("backend_impacts", [])
        compact.setdefault("frontend_impacts", [])

    Path(args.output_summary).write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote summary to", args.output_summary, file=sys.stderr)

    return 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
