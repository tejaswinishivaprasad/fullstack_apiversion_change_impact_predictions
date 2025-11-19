#!/usr/bin/env bash
# .github/scripts/ci_analyze_pr.sh
# Lightweight CI runner for AI Core analysis — hardened
set -euo pipefail
IFS=$'\n\t'

echo "CI: starting ai-core analysis runner"

# allow override from workflow or environment; default path used in your repo
AI_CORE_DIR="${AI_CORE_DIR:-impact_ai_repo/ai-core/src}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
OUTFILE="${REPO_ROOT}/pr-impact-report.json"
PR_NUMBER="${PR_NUMBER:-unknown}"

echo "CI: using AI_CORE_DIR=${AI_CORE_DIR}"
echo "CI: repo root = ${REPO_ROOT}"
echo "CI: output file = ${OUTFILE}"

if [ ! -d "${AI_CORE_DIR}" ]; then
  echo "ERROR: AI core directory not found: ${AI_CORE_DIR}" >&2
  echo "CI: listing repo root for debugging:"
  ls -la "${REPO_ROOT}" || true
  # create a minimal fallback report so the workflow doesn't fail silently
  cat > "${OUTFILE}" <<EOF
{
  "status": "failed",
  "pr": "${PR_NUMBER}",
  "error": "ai_core_dir_not_found",
  "files_changed": []
}
EOF
  exit 0
fi

pushd "${AI_CORE_DIR}" > /dev/null

echo "CI: entering ${AI_CORE_DIR}"

# Create a venv to isolate installations
VENV_DIR=".ci-venv"
PYBIN="${VENV_DIR}/bin/python"
PIPBIN="${VENV_DIR}/bin/pip"

if [ ! -x "${PYBIN}" ]; then
  echo "CI: creating python venv at ${VENV_DIR}"
  python -m venv "${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Upgrade pip quietly (use prefer-binary to prefer wheels if available)
python -m pip install --upgrade pip setuptools wheel >/dev/null

# Minimal CI requirements: prefer a repo-local requirements-ci.txt if present
if [ -f "requirements-ci.txt" ]; then
  echo "CI: installing minimal CI requirements from requirements-ci.txt (fast deps only)"
  # Use --prefer-binary to avoid building from source when possible, but do NOT install heavy deps here.
  pip install --prefer-binary -r requirements-ci.txt || true
else
  echo "CI: no requirements-ci.txt found. Installing a tiny set of safe, fast packages."
  pip install --prefer-binary fastapi uvicorn networkx pyyaml || true
fi

# Attempt to run supported CI/demo entrypoints.
# Prefer non-blocking approach: we try server.py --ci-scan, fallback to server.py, fallback to analysis_light
SUCCESS=0

if [ -f "server.py" ]; then
  echo "CI: found server.py — trying --ci-scan (non-fatal)"
  if python -u server.py --ci-scan; then
    echo "CI: server.py --ci-scan succeeded"
    SUCCESS=1
  else
    echo "CI: server.py --ci-scan returned non-zero (continuing to fallback)"
  fi
fi

if [ $SUCCESS -eq 0 ] && [ -f "server.py" ]; then
  echo "CI: attempting plain server.py run (non-fatal)"
  if python -u server.py; then
    echo "CI: server.py run succeeded"
    SUCCESS=1
  else
    echo "CI: server.py run failed (continuing to fallback)"
  fi
fi

# Fallback: run analysis_light if present in package layout or as script
if [ $SUCCESS -eq 0 ]; then
  echo "CI: running fallback lightweight analyzer"
  # Try module entry first (if impact_ai_repo package is importable)
  if python -c "import importlib, sys; importlib.import_module('impact_ai_repo.ai_core.analysis_light')" 2>/dev/null; then
    python -m impact_ai_repo.ai_core.analysis_light --pr "${PR_NUMBER}" --output "${OUTFILE}" || true
  elif [ -f "analysis_light.py" ]; then
    python analysis_light.py --pr "${PR_NUMBER}" --output "${OUTFILE}" || true
  else
    echo "CI: no analysis_light found. Will search for any generated report files."
  fi
fi

# Search for generated impact-report*.json (limit depth)
REPORT_FILE="$(find . -maxdepth 3 -type f -iname 'impact-report*.json' -print -quit || true)"

if [ -n "${REPORT_FILE}" ]; then
  echo "CI: found report ${REPORT_FILE}. Copying to repo root as pr-impact-report.json"
  cp "${REPORT_FILE}" "${OUTFILE}"
else
  # If fallback wrote to OUTFILE already, keep it. Otherwise create minimal fallback report.
  if [ ! -f "${OUTFILE}" ]; then
    echo "CI: no impact report found; writing fallback minimal report"
    cat > "${OUTFILE}" <<EOF
{
  "status": "partial",
  "pr": "${PR_NUMBER}",
  "note": "no_impact_report_generated",
  "files_changed": []
}
EOF
  else
    echo "CI: using existing ${OUTFILE}"
  fi
fi

# Print preview
echo "=== pr-impact-report.json preview ==="
if command -v jq >/dev/null 2>&1; then
  jq -C . "${OUTFILE}" || cat "${OUTFILE}"
else
  cat "${OUTFILE}"
fi
echo "=== end preview ==="

popd > /dev/null

echo "CI: done — report at ${OUTFILE}"
