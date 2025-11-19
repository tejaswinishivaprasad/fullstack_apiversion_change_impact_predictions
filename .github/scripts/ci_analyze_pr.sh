#!/usr/bin/env bash
# CI runner for AI Core analysis — robust to runner cwd and fetch-depth issues.
# - runs from repo root so git diff paths and filesystem paths align
# - creates a tiny venv for minimal deps if needed (uses requirements-ci.txt if present)
# - runs analysis_light.py (fast) or server.py --ci-scan as fallback
# - copies any generated impact-report*.json to $GITHUB_WORKSPACE/pr-impact-report.json
set -euo pipefail

echo "CI: starting ai-core analysis runner"

# allow override from workflow or environment; default path used in your repo
AI_CORE_DIR="${AI_CORE_DIR:-impact_ai_repo/ai-core/src}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"

echo "CI: AI_CORE_DIR=${AI_CORE_DIR}"
echo "CI: repo root = ${REPO_ROOT}"

# ensure we run from repo root (git diffs produced by checkout are repo-root-relative)
cd "${REPO_ROOT}"

# sanity: confirm the AI_CORE_DIR exists (relative to repo root)
if [ ! -d "${AI_CORE_DIR}" ]; then
  echo "ERROR: AI core directory not found: ${AI_CORE_DIR}" >&2
  echo "CI: listing repo root for debugging:"
  ls -la "${REPO_ROOT}" || true
  exit 1
fi

# Prefer a fast "analysis_light.py" if present; otherwise fallback to server.py --ci-scan
ANALYSIS_LIGHT="${AI_CORE_DIR}/analysis_light.py"
SERVER_PY="${AI_CORE_DIR}/server.py"

# create minimal venv to avoid trying to build heavy ML deps in PR
if ! python -c "import yaml,uvicorn,fastapi,networkx" 2>/dev/null; then
  echo "CI: creating local venv for minimal runtime deps"
  python -m venv .ci-venv
  # shellcheck disable=SC1091
  source .ci-venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  if [ -f "${AI_CORE_DIR}/requirements-ci.txt" ]; then
    echo "CI: installing requirements-ci.txt (fast deps)"
    pip install --prefer-binary -r "${AI_CORE_DIR}/requirements-ci.txt" || true
  else
    # install just the tiny parser/runtime deps we need
    pip install --prefer-binary pyyaml || true
  fi
fi

# Helper to capture any generated report
copy_report() {
  # search repo for any impact-report JSON with reasonable depth
  REPORT_FILE="$(find "${REPO_ROOT}" -maxdepth 3 -type f -iname 'pr-impact-report.json' -o -iname 'impact-report*.json' -print -quit || true)"
  if [ -n "${REPORT_FILE}" ]; then
    OUTPUT_PATH="${REPO_ROOT}/pr-impact-report.json"
    cp "${REPORT_FILE}" "${OUTPUT_PATH}"
    echo "CI: copied ${REPORT_FILE} -> ${OUTPUT_PATH}"
    # set GITHUB_OUTPUT if available (modern actions)
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
      echo "report=pr-impact-report.json" >> "${GITHUB_OUTPUT}" || true
    else
      echo "::set-output name=report::pr-impact-report.json" || true
    fi
    return 0
  fi
  return 1
}

# Run the lightweight analyzer first (recommended). Run from repo root so git paths align.
if [ -f "${ANALYSIS_LIGHT}" ]; then
  echo "CI: running analysis_light.py (repo-root execution)"
  # make sure it's executable
  chmod +x "${ANALYSIS_LIGHT}" || true

  # run with explicit python so venv is respected if activated
  if python3 "${ANALYSIS_LIGHT}" --pr "${PR_NUMBER:-unknown}" --output "${REPO_ROOT}/pr-impact-report.json"; then
    echo "CI: analysis_light.py finished"
    exit 0
  else
    echo "CI: analysis_light.py returned non-zero; will attempt server.py fallback"
  fi
fi

# Fallback: try server.py --ci-scan (run from AI_CORE_DIR but still referencing repo root for files)
if [ -f "${SERVER_PY}" ]; then
  echo "CI: attempting fallback server.py --ci-scan"
  # run from AI_CORE_DIR but preserve repo-root env var so server can find datasets if needed
  pushd "${AI_CORE_DIR}" > /dev/null || true
  if python3 -u server.py --ci-scan --output "${REPO_ROOT}/pr-impact-report.json"; then
    echo "CI: server.py --ci-scan completed"
    popd > /dev/null || true
    copy_report || true
    exit 0
  else
    echo "CI: server.py --ci-scan failed"
    popd > /dev/null || true
  fi
fi

# If we reach here, try to copy any generated report (maybe analysis wrote somewhere unexpected)
if copy_report; then
  echo "CI: found report after fallback attempts"
  exit 0
fi

echo "CI: no impact report generated"
# produce a minimal placeholder for tooling so workflow steps downstream see something
cat > "${REPO_ROOT}/pr-impact-report.json" <<'JSON'
{
  "status": "partial",
  "pr": "'"${PR_NUMBER:-unknown}"'",
  "note": "no_impact_report_generated",
  "files_changed": []
}
JSON

echo "CI: done (no real report)"
exit 0
