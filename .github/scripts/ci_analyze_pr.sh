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

# run from inside ai core folder
pushd "${AI_CORE_DIR}" > /dev/null

echo "CI: running server.py in CI/demo mode (if supported)"
# Prefer --ci-scan if supported. Use "-u" to disable python buffering in logs.
if python3 -u server.py --ci-scan; then
  echo "CI: server.py --ci-scan completed (exit 0)"
else
  echo "CI: server.py --ci-scan returned non-zero or is unsupported; attempting fallback run"

  # Fallback: try running with no args (server might output a report) — tolerant
  if python3 -u server.py; then
    echo "CI: server.py (no args) completed"
  else
    echo "CI: server.py fallback also failed — continuing to search for any generated report"
  fi
fi

# locate any generated impact report (search depth limited to keep it quick)
REPORT_FILE="$(find . -maxdepth 3 -type f -iname 'impact-report*.json' -print -quit || true)"

if [ -n "${REPORT_FILE}" ]; then
  # copy into the repository root / GITHUB_WORKSPACE so Actions can pick it up
  OUTPUT_PATH="${REPO_ROOT}/pr-impact-report.json"
  cp "${REPORT_FILE}" "${OUTPUT_PATH}"
  echo "CI: copied ${REPORT_FILE} -> ${OUTPUT_PATH}"

  # modern Actions way: write key to GITHUB_OUTPUT (if present) for downstream steps
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "report=pr-impact-report.json" >> "${GITHUB_OUTPUT}" || true
  else
    # fallback for older runners (deprecated ::set-output)
    echo "::set-output name=report::pr-impact-report.json" || true
  fi
else
  echo "CI: no impact-report JSON found in ${AI_CORE_DIR}"
fi

popd > /dev/null
echo "CI: done"
