#!/usr/bin/env bash
# Simple CI runner for AI Core analysis
# - finds the ai-core folder
# - runs server.py in "ci-scan" / demo mode (if supported)
# - copies any generated impact-report*.json to $GITHUB_WORKSPACE/pr-impact-report.json
set -euo pipefail

echo "CI: starting ai-core analysis runner"

# allow override from workflow or environment; default path used in your repo
AI_CORE_DIR="${AI_CORE_DIR:-impact_ai_repo/ai-core/src}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"

echo "CI: using AI_CORE_DIR=${AI_CORE_DIR}"
echo "CI: repo root = ${REPO_ROOT}"

if [ ! -d "${AI_CORE_DIR}" ]; then
  echo "ERROR: AI core directory not found: ${AI_CORE_DIR}" >&2
  echo "CI: listing repo root for debugging:"
  ls -la "${REPO_ROOT}" || true
  exit 1
fi

pushd "${AI_CORE_DIR}" > /dev/null

echo "CI: running server.py in CI/demo mode (if supported)"

# If basic deps missing, create a small venv and install minimal runtime deps (fast, avoids heavy ML builds)
if ! python -c "import fastapi,uvicorn,networkx" 2>/dev/null; then
  echo "CI: python deps not present — creating local venv and installing minimal deps"
  python -m venv .ci-venv
  source .ci-venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  if [ -f requirements.txt ]; then
    pip install --prefer-binary -r requirements.txt || true
  else
    pip install fastapi uvicorn networkx pyyaml || true
  fi
fi

# Prefer --ci-scan; run non-blocking so failures don't kill the script (workflow handles continue-on-error)
if python3 -u server.py --ci-scan; then
  echo "CI: server.py --ci-scan completed (exit 0)"
else
  echo "CI: server.py --ci-scan returned non-zero or is unsupported; attempting fallback run"
  if python3 -u server.py; then
    echo "CI: server.py (no args) completed"
  else
    echo "CI: server.py fallback also failed — continuing to search for any generated report"
  fi
fi

# locate any generated impact report (search depth limited)
REPORT_FILE="$(find . -maxdepth 3 -type f -iname 'impact-report*.json' -print -quit || true)"

if [ -n "${REPORT_FILE}" ]; then
  OUTPUT_PATH="${REPO_ROOT}/pr-impact-report.json"
  cp "${REPORT_FILE}" "${OUTPUT_PATH}"
  echo "CI: copied ${REPORT_FILE} -> ${OUTPUT_PATH}"

  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "report=pr-impact-report.json" >> "${GITHUB_OUTPUT}" || true
  else
    echo "::set-output name=report::pr-impact-report.json" || true
  fi
else
  echo "CI: no impact-report JSON found in ${AI_CORE_DIR}"
fi

popd > /dev/null
echo "CI: done"
