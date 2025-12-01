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

def analyze_pair_files(old_doc: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use server's in-process logic to compute diffs/predictions.
    We'll:
      - diff via server.diff_openapi (gives list of DiffItem)
      - call server.api_analyze with JSON payloads to get model-style predictions/summary

    IMPORTANT: produce 'pair_id' and 'ai_explanation' in the 'analyze' payload if available,
    so the downstream CI summary can pick them up cleanly.
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
        # normalize type casing to upper (helps UIs show uniform case)
        if "type" in dd and isinstance(dd["type"], str):
            dd["type"] = dd["type"].upper()
        diffs_serial.append(dd)

    # call server.api_analyze in-process to get predictions & summary
    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy = server.api_analyze(baseline=baseline_str, candidate=candidate_str, dataset=None, options=None)
    except Exception as e:
        print("WARN: server.api_analyze raised:", e, file=sys.stderr)
        analy = {"run_id": None, "predictions": [], "summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # Ensure analy is a dict we can augment
    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    # Try to propagate pair_id from various places (versioning / metadata / backend)
    pair_id = None
    try:
        if isinstance(analy.get("versioning"), dict):
            pair_id = analy["versioning"].get("pair_id") or pair_id
        if isinstance(analy.get("metadata"), dict):
            pair_id = analy["metadata"].get("pair_id") or pair_id
        if isinstance(analy.get("backend"), dict):
            pair_id = analy["backend"].get("pair_id") or pair_id
    except Exception:
        pass

    # If still not found and diffs contain ace_id with pair prefix, try to extract
    if not pair_id:
        for d in diffs_serial:
            aid = d.get("ace_id") or ""
            if isinstance(aid, str) and "::" in aid:
                maybe = aid.split("::")[0]
                if maybe.startswith("pair-"):
                    pair_id = maybe
                    break

    # Prepare ai_explanation: use analy.ai_explanation if available, else call server.make_explanation fallback
    ai_expl = analy.get("ai_explanation") or analy.get("explanation") or None
    if not ai_expl:
        # attempt to build using server.make_explanation if available
        try:
            # need pfeats/vfeats/be_imp/fe_imp for server.make_explanation; build minimal defaults
            fake_pfeats = {}
            fake_vfeats = analy.get("versioning") or {}
            # compute some impacts counts heuristically
            be_imp = analy.get("backend_impacts") or []
            fe_imp = analy.get("frontend_impacts") or []
            score = 0.0
            try:
                score = float((analy.get("summary") or {}).get("service_risk", 0.0))
            except Exception:
                score = 0.0
            try:
                ai_expl = server.make_explanation(score, diffs_serial, fake_pfeats, fake_vfeats, be_imp, fe_imp)
            except Exception:
                ai_expl = ""
        except Exception:
            ai_expl = ""

    # make sure analy contains canonical entries downstream expects
    analy_out = dict(analy)
    analy_out.setdefault("summary", analy_out.get("summary") or {})
    analy_out["summary"].setdefault("service_risk", float((analy_out["summary"].get("service_risk") or 0.0)))
    analy_out["summary"].setdefault("num_aces", len(diffs_serial))
    analy_out["pair_id"] = pair_id
    analy_out["ai_explanation"] = ai_expl

    # Build combined result
    out = {
        "diffs": diffs_serial,
        "analyze": analy_out,
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
