#!/usr/bin/env bash
# CI runner for AI Core analysis — prefer in-process server wrapper (ci_server_run.py)
# - runs from repo root so git diff paths and filesystem paths align
# - creates a tiny venv for minimal deps if needed (uses requirements-ci.txt if present)
# - runs ci_server_run.py (recommended). Falls back to analysis_light.py or server.py if present.
# - copies any generated impact-report*.json / pr-impact-full.json to $GITHUB_WORKSPACE/pr-impact-report.json
set -euo pipefail

echo "CI: starting ai-core analysis runner"

AI_CORE_DIR="${AI_CORE_DIR:-impact_ai_repo/ai-core/src}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"

echo "CI: AI_CORE_DIR=${AI_CORE_DIR}"
echo "CI: repo root = ${REPO_ROOT}"

cd "${REPO_ROOT}"

if [ ! -d "${AI_CORE_DIR}" ]; then
  echo "ERROR: AI core directory not found: ${AI_CORE_DIR}" >&2
  ls -la "${REPO_ROOT}" || true
  exit 1
fi

CI_WRAPPER="${AI_CORE_DIR}/ci_server_run.py"
ANALYSIS_LIGHT="${AI_CORE_DIR}/analysis_light.py"
SERVER_PY="${AI_CORE_DIR}/server.py"

# Create minimal venv to avoid heavy installs in PR runs
create_venv_if_needed() {
  # quick check: try importing pyyaml and networkx which our wrapper may need
  if python -c "import yaml, json, sys" 2>/dev/null; then
    return 0
  fi
  echo "CI: creating local venv for minimal runtime deps"
  python -m venv .ci-venv
  # shellcheck disable=SC1091
  source .ci-venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  if [ -f "${AI_CORE_DIR}/requirements-ci.txt" ]; then
    echo "CI: installing requirements-ci.txt (fast deps)"
    pip install --prefer-binary -r "${AI_CORE_DIR}/requirements-ci.txt" || true
  else
    # minimal installs we need: PyYAML and networkx for graph helpers
    pip install --prefer-binary pyyaml networkx || true
  fi
}

# copy any generated report(s) to repo root and set github outputs when possible
copy_report() {
  # prefer explicit names we expect
  REPORT_FULL="$(find "${REPO_ROOT}" -maxdepth 4 -type f -name 'pr-impact-full.json' -print -quit || true)"
  REPORT_SUM="$(find "${REPO_ROOT}" -maxdepth 4 -type f -name 'pr-impact-report.json' -print -quit || true)"

  # fallback to any impact-report* files (older naming)
  if [ -z "${REPORT_SUM}" ]; then
    REPORT_SUM="$(find "${REPO_ROOT}" -maxdepth 4 -type f -iname 'impact-report*.json' -print -quit || true)"
  fi

  HANDLED=0
  if [ -n "${REPORT_FULL}" ]; then
    cp "${REPORT_FULL}" "${REPO_ROOT}/pr-impact-full.json" || true
    echo "CI: copied full report ${REPORT_FULL} -> ${REPO_ROOT}/pr-impact-full.json"
    HANDLED=1
  fi

  if [ -n "${REPORT_SUM}" ]; then
    cp "${REPORT_SUM}" "${REPO_ROOT}/pr-impact-report.json" || true
    echo "CI: copied summary ${REPORT_SUM} -> ${REPO_ROOT}/pr-impact-report.json"
    # set github output (modern Actions)
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
      echo "report=pr-impact-report.json" >> "${GITHUB_OUTPUT}" || true
    else
      echo "::set-output name=report::pr-impact-report.json" || true
    fi
    HANDLED=1
  fi

  return "${HANDLED}"
}

# Activate venv if exists
if [ -d ".ci-venv" ]; then
  # shellcheck disable=SC1091
  source .ci-venv/bin/activate || true
fi

# create venv if wrapper exists and dependencies look missing
if [ -f "${CI_WRAPPER}" ]; then
  create_venv_if_needed
  chmod +x "${CI_WRAPPER}" || true
  echo "CI: running ci_server_run.py (preferred wrapper)"
  if python -u "${CI_WRAPPER}" --pr "${PR_NUMBER:-unknown}" --output-full "${REPO_ROOT}/pr-impact-full.json" --output-summary "${REPO_ROOT}/pr-impact-report.json"; then
    echo "CI: ci_server_run.py finished"
    copy_report || true
    exit 0
  else
    echo "CI: ci_server_run.py returned non-zero; will attempt fallbacks" >&2
  fi
fi

# Fallback 1: analysis_light.py (old fast script)
if [ -f "${ANALYSIS_LIGHT}" ]; then
  echo "CI: running analysis_light.py (repo-root execution)"
  chmod +x "${ANALYSIS_LIGHT}" || true
  if python3 "${ANALYSIS_LIGHT}" --pr "${PR_NUMBER:-unknown}" --output "${REPO_ROOT}/pr-impact-report.json"; then
    echo "CI: analysis_light.py finished"
    copy_report || true
    exit 0
  else
    echo "CI: analysis_light.py returned non-zero; will attempt server.py fallback" >&2
  fi
fi

# Fallback 2: server.py --ci-scan (if you implemented that interface)
if [ -f "${SERVER_PY}" ]; then
  echo "CI: attempting fallback server.py --ci-scan"
  pushd "${AI_CORE_DIR}" > /dev/null || true
  if python3 -u server.py --ci-scan --output "${REPO_ROOT}/pr-impact-report.json"; then
    echo "CI: server.py --ci-scan completed"
    popd > /dev/null || true
    copy_report || true
    exit 0
  else
    echo "CI: server.py --ci-scan failed" >&2
    popd > /dev/null || true
  fi
fi

# final attempt to copy any report
if copy_report; then
  echo "CI: found report after fallback attempts"
  exit 0
fi

echo "CI: no impact report generated, writing placeholder"
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
