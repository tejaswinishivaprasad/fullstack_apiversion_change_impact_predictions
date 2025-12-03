#!/usr/bin/env python3
"""
analysis_light.py

CI-only impact analysis script that does NOT use server.report.

It:
 - detects changed OpenAPI canonical files in any dataset variant:
   curated_clean / curated_noisy_light / curated_noisy_heavy
 - finds the local v1 counterpart in the same canonical folder
 - runs diff_openapi + api_analyze
 - loads graph.json for backend/frontend impact heuristics
 - writes:
     pr-impact-full.json  (detailed)
     pr-impact-report.json (compact summary)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve repo root and ai-core/src
REPO_ROOT = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve()
AI_CORE_DIR_ENV = os.environ.get("AI_CORE_DIR", "impact_ai_repo/ai-core/src")
AI_CORE_SRC = (REPO_ROOT / AI_CORE_DIR_ENV).resolve()

HERE = AI_CORE_SRC if AI_CORE_SRC.exists() else Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import server  # type: ignore  # noqa: E402
    print(f"DEBUG[analysis_light]: imported server from {getattr(server, '__file__', 'unknown')}", file=sys.stderr)
except Exception as e:
    print("ERROR[analysis_light]: failed to import server:", e, file=sys.stderr)
    raise


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return 0, out.decode()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output.decode() if e.output else ""
    except Exception as e:
        return 1, str(e)


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


def read_json_file_if_exists(p: Path) -> Dict[str, Any]:
    try:
        if not p.exists():
            return {}
        txt = p.read_text(encoding="utf-8")
        try:
            return json.loads(txt)
        except Exception:
            try:
                return server._load_json_or_yaml(p)  # type: ignore[attr-defined]
            except Exception:
                return {}
    except Exception:
        return {}


def load_json_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        try:
            # text may actually be path content; be forgiving
            return server._load_json_or_yaml(Path(text))  # type: ignore[attr-defined]
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
            new_paths: Dict[str, Any] = {}
            for pth, methods in spec["paths"].items():
                if not isinstance(methods, dict):
                    new_paths[pth] = methods
                    continue
                nm: Dict[str, Any] = {}
                for m, op in methods.items():
                    nm[m] = resolve(op)
                new_paths[pth] = nm
            out["paths"] = new_paths
    except Exception:
        pass
    try:
        out["components"] = dict(out.get("components") or {})
    except Exception:
        pass
    return out


def find_v1_from_v2_path(v2path: Path) -> Path:
    name = v2path.name
    if "--v2." in name:
        alt = name.replace("--v2.", "--v1.")
    elif "-v2." in name:
        alt = name.replace("-v2.", "-v1.")
    else:
        alt = name.replace("v2", "v1", 1)
    return v2path.parent / alt


def detect_variant_root(rel: str) -> Optional[Path]:
    """
    From a changed path like:
      impact_ai_repo/ai-core/src/datasets/curated_clean/openapi/canonical/...
    we derive variant_root = HERE/datasets/curated_clean
    """
    parts = Path(rel).parts
    if "datasets" not in parts:
        return None
    idx = parts.index("datasets")
    if idx + 1 >= len(parts):
        return None
    variant = parts[idx + 1]  # curated_clean / curated_noisy_light / curated_noisy_heavy / raw
    vr = HERE / "datasets" / variant
    return vr.resolve() if vr.exists() else None


def ensure_graph_loaded_for_variant(variant_root: Optional[Path]) -> Optional[Any]:
    """
    Try to load graph.json for the given variant, with sane fallbacks.
    """
    g = None
    try:
        # Try existing graph first
        g = server.load_graph()
        if g is not None:
            print("DEBUG[analysis_light]: server.load_graph() already returns", type(g), file=sys.stderr)
            return g
    except Exception:
        g = None

    candidates: List[Path] = []
    if variant_root is not None:
        candidates.append(variant_root / "graph.json")
    candidates.append(HERE / "datasets" / "graph.json")

    seen = set()
    for c in candidates:
        cp = c.resolve()
        if str(cp) in seen:
            continue
        seen.add(str(cp))
        if cp.exists():
            try:
                server.GRAPH_PATH = str(cp)
                print(f"DEBUG[analysis_light]: forcing server.GRAPH_PATH={server.GRAPH_PATH}", file=sys.stderr)
                g = server.load_graph()
                print("DEBUG[analysis_light]: server.load_graph() after forcing ->", type(g), file=sys.stderr)
                return g
            except Exception as e:
                print("WARN[analysis_light]: load_graph failed for candidate", cp, "error:", e, file=sys.stderr)

    print("DEBUG[analysis_light]: no usable graph.json found", file=sys.stderr)
    return None


def analyze_pair_files(old_doc: Dict[str, Any],
                       new_doc: Dict[str, Any],
                       rel_path: Optional[str]) -> Dict[str, Any]:
    # deref
    try:
        old2 = dereference_components(old_doc)
        new2 = dereference_components(new_doc)
    except Exception:
        old2, new2 = old_doc, new_doc

    # diffs
    try:
        diffs = server.diff_openapi(old2, new2)
    except Exception as e:
        print("WARN[analysis_light]: server.diff_openapi failed:", e, file=sys.stderr)
        diffs = []

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
        if isinstance(dd.get("type"), str):
            dd["type"] = dd["type"].upper()
        diffs_serial.append(dd)

    # api_analyze (ML model)
    try:
        baseline_str = json.dumps(old_doc)
        candidate_str = json.dumps(new_doc)
        analy = server.api_analyze(
            baseline=baseline_str,
            candidate=candidate_str,
            dataset=None,
            options=None,
        )
    except Exception as e:
        print("WARN[analysis_light]: api_analyze raised:", e, file=sys.stderr)
        analy = {"run_id": None, "predictions": [], "summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    if not isinstance(analy, dict):
        analy = {"summary": {"service_risk": 0.0, "num_aces": len(diffs_serial)}}

    try:
        score = float((analy.get("summary") or {}).get("service_risk", 0.0))
    except Exception:
        score = 0.0

    # load graph & enrich
    variant_root = detect_variant_root(rel_path) if rel_path else None
    g = ensure_graph_loaded_for_variant(variant_root)

    # best-effort service guess
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

    pfeats: Dict[str, Any] = {}
    be_imp: List[Dict[str, Any]] = []
    fe_imp: List[Dict[str, Any]] = []

    try:
        if g is not None:
            pfeats = server.producer_features(g, service_guess)
            changed_paths = [server.normalize_path(d.get("path"))
                             for d in diffs_serial if d.get("path")]
            be_imp = server.backend_impacts(g, service_guess, changed_paths)
            fe_imp = server.ui_impacts(g, service_guess, changed_paths)
            print(f"DEBUG[analysis_light]: enriched impacts -> backend={len(be_imp)} frontend={len(fe_imp)}",
                  file=sys.stderr)
        else:
            print("DEBUG[analysis_light]: skipping backend/ui impact enrichment (no graph)", file=sys.stderr)
    except Exception as e:
        print("WARN[analysis_light]: impact enrichment failed:", e, file=sys.stderr)

    ai_expl = analy.get("ai_explanation") or analy.get("explanation") or ""
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


def main() -> int:
    changed = git_changed_files()
    print("CI[analysis_light]: changed files:", changed, file=sys.stderr)

    api_files = [f for f in changed if f.lower().endswith((".json", ".yaml", ".yml"))]

    # focus only on dataset canonical files (any curated variant)
    curated_markers = ["datasets/curated_clean", "datasets/curated_noisy_light",
                       "datasets/curated_noisy_heavy", "/canonical/"]
    api_files = [f for f in api_files if any(tok in f for tok in curated_markers)] or api_files

    results: List[Dict[str, Any]] = []
    files_processed: List[str] = []

    for rel in api_files:
        print("DEBUG[analysis_light]: rel_path =", rel, file=sys.stderr)
        p = REPO_ROOT / rel
        files_processed.append(rel)

        try:
            if p.exists():
                new_doc = read_json_file_if_exists(p)
                v1cand = find_v1_from_v2_path(p)
                print(f"DEBUG[analysis_light]: v1cand guessed as {v1cand}", file=sys.stderr)
                if v1cand.exists():
                    print(f"DEBUG[analysis_light]: using local v1 counterpart at {v1cand}", file=sys.stderr)
                    old_doc = read_json_file_if_exists(v1cand)
                else:
                    blob = git_show_file(f"origin/main:{rel}")
                    old_doc = load_json_text(blob) if blob else {}
            else:
                # purely historical file (unlikely in your workflow, but keep it safe)
                blob_new = git_show_file(f"HEAD:{rel}")
                new_doc = load_json_text(blob_new) if blob_new else {}
                blob_old = git_show_file(f"origin/main:{rel}")
                old_doc = load_json_text(blob_old) if blob_old else {}
        except Exception as e:
            print(f"WARN[analysis_light]: failed to load files for {rel}: {e}", file=sys.stderr)
            old_doc, new_doc = {}, {}

        try:
            pair_res = analyze_pair_files(old_doc, new_doc, rel_path=rel)
        except Exception as e:
            print("WARN[analysis_light]: analyze_pair_files exception:", e, file=sys.stderr)
            pair_res = {"diffs": [], "analyze": {"summary": {"service_risk": 0.0, "num_aces": 0}}}

        results.append({"file": rel, "result": pair_res})

    full_out = {
        "status": "ok" if results else "partial",
        "files_changed": files_processed,
        "files_analyzed": len(results),
        "entries": results,
    }
    (REPO_ROOT / "pr-impact-full.json").write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print("CI[analysis_light]: wrote full output to pr-impact-full.json", file=sys.stderr)

    max_risk = 0.0
    atomic_aces: List[Dict[str, Any]] = []
    ai_expl_top: Optional[str] = None

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
                    "endpoint_removed",
                )
            ),
            "total_aces": len(atomic_aces),
        },
        "risk_score": round(float(max_risk), 3),
        "risk_band": band,
        "risk_level": level,
        "ai_explanation": ai_expl_top or "",
    }

    (REPO_ROOT / "pr-impact-report.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print("CI[analysis_light]: wrote summary to pr-impact-report.json", file=sys.stderr)
    return 0


def git_show_file(ref_path: str) -> str:
    code, out = run_cmd(["git", "show", ref_path])
    return out if code == 0 else ""


if __name__ == "__main__":
    sys.exit(main())
