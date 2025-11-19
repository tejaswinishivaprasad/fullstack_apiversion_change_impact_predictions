#!/usr/bin/env bash
# Simple CI runner for AI Core analysis
# - finds the ai-core folder
# - runs server.py in "ci-scan" / demo mode
# - copies any generated impact-report*.json to $GITHUB_WORKSPACE/pr-impact-report.json
set -euo pipefail

echo "CI: starting ai-core analysis runner"

# Allow override from workflow, otherwise use the typical path seen in your repo
AI_CORE_DIR="${AI_CORE_DIR:-impact_ai_repo/ai-core/src}"
echo "CI: using AI_CORE_DIR=${AI_CORE_DIR}"

if [ ! -d "${AI_CORE_DIR}" ]; then
  echo "ERROR: AI core directory not found: ${AI_CORE_DIR}" >&2
  ls -la || true
  exit 1
fi

pushd "${AI_CORE_DIR}" > /dev/null

echo "CI: running server.py in CI/demo mode"
# prefer a --ci-scan flag if your server supports it; otherwise run a safe API call
if python3 server.py --ci-scan; then
  echo "CI: server.py --ci-scan completed"
else
  echo "CI: server.py --ci-scan failed or returned non-zero"
fi

# find any generated impact report
REPORT_FILE="$(find . -maxdepth 2 -type f -iname 'impact-report*.json' -print -quit || true)"

if [ -n "${REPORT_FILE}" ]; then
  TARGET="${GITHUB_WORKSPACE:-$(pwd)/../..}/pr-impact-report.json"
  # ensure workspace path exists
  if [ -n "${GITHUB_WORKSPACE:-}" ]; then
    cp "${REPORT_FILE}" "${GITHUB_WORKSPACE}/pr-impact-report.json"
    echo "CI: copied ${REPORT_FILE} -> ${GITHUB_WORKSPACE}/pr-impact-report.json"
  else
    # fallback to repo root
    cp "${REPORT_FILE}" "${TARGET}"
    echo "CI: copied ${REPORT_FILE} -> ${TARGET}"
  fi
  echo "::set-output name=report::pr-impact-report.json" || true
else
  echo "CI: no impact-report JSON found in ${AI_CORE_DIR}"
fi

popd > /dev/null
echo "CI: done"
