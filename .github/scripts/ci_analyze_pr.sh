#!/usr/bin/env bash
# Simple CI runner for AI Core analysis (improved)
# - finds the ai-core folder
# - optionally creates a venv & installs requirements.txt (only if needed or forced)
# - runs server.py in "ci-scan" / demo mode (if supported)
# - copies any generated impact-report*.json to $GITHUB_WORKSPACE/pr-impact-report.json
set -euo pipefail

echo "CI: starting ai-core analysis runner"

# allow override from workflow or environment; default path used in your repo
AI_CORE_DIR="${AI_CORE_DIR:-impact_ai_repo/ai-core/src}"
REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
FORCE_INSTALL="${FORCE_INSTALL:-0}"   # set=1 to force pip install in CI (optional)

echo "CI: using AI_CORE_DIR=${AI_CORE_DIR}"
echo "CI: repo root = ${REPO_ROOT}"
echo "CI: FORCE_INSTALL=${FORCE_INSTALL}"

if [ ! -d "${AI_CORE_DIR}" ]; then
  echo "ERROR: AI core directory not found: ${AI_CORE_DIR}" >&2
  echo "CI: listing repo root for debugging:"
  ls -la "${REPO_ROOT}" || true
  exit 1
fi

# ensure we return to original dir on exit
pushd "${AI_CORE_DIR}" > /dev/null
trap 'popd > /dev/null' EXIT

echo "CI: current working dir: $(pwd)"

# helper: test for presence of a required module (networkx is a good sentinel)
python_import_ok() {
  python3 - <<'PY' 2>/dev/null || true
import importlib, sys
mod="$1"
try:
    importlib.import_module(mod)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  return $?
}

# If FORCE_INSTALL set, install. Otherwise only install when a sanity import fails.
NEEDS_INSTALL=0
if [ "${FORCE_INSTALL}" = "1" ]; then
  NEEDS_INSTALL=1
else
  # check for a commonly used module that caused the earlier failure
  if ! python3 -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('networkx') else 1)"; then
    echo "CI: networkx not importable - will attempt to install requirements if available"
    NEEDS_INSTALL=1
  else
    echo "CI: networkx import appears available in runner's environment"
  fi
fi

# if requirements.txt exists and install is needed, create venv and pip install there
if [ "${NEEDS_INSTALL}" -eq 1 ] && [ -f "requirements.txt" ]; then
  echo "CI: preparing venv and installing requirements from requirements.txt"
  VENV_DIR=".ci-venv"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
  echo "CI: pip install finished (venv: ${VENV_DIR})"
else
  if [ "${NEEDS_INSTALL}" -eq 1 ]; then
    echo "CI: install requested but no requirements.txt found; continuing without install"
  fi
fi

echo "CI: running server.py in CI/demo mode (if supported)"
# prefer --ci-scan flag if your server supports it; use -u to force unbuffered output in logs
if command -v python3 >/dev/null 2>&1; then
  if python3 -u server.py --ci-scan; then
    echo "CI: server.py --ci-scan completed (exit 0)"
  else
    echo "CI: server.py --ci-scan returned non-zero or is unsupported; attempting fallback run" >&2
    if python3 -u server.py; then
      echo "CI: server.py (no args) completed"
    else
      echo "CI: server.py fallback also failed — continuing to search for any generated report" >&2
    fi
  fi
else
  echo "ERROR: python3 not found in PATH" >&2
fi

# locate any generated impact report (search depth limited to keep it quick)
REPORT_FILE="$(find . -maxdepth 3 -type f -iname 'impact-report*.json' -print -quit || true)"

if [ -n "${REPORT_FILE}" ]; then
  OUTPUT_PATH="${REPO_ROOT}/pr-impact-report.json"
  cp "${REPORT_FILE}" "${OUTPUT_PATH}"
  echo "CI: copied ${REPORT_FILE} -> ${OUTPUT_PATH}"

  # modern Actions way: write to GITHUB_OUTPUT if available
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "report=pr-impact-report.json" >> "${GITHUB_OUTPUT}" || true
  else
    echo "::set-output name=report::pr-impact-report.json" || true
  fi
else
  echo "CI: no impact-report JSON found in ${AI_CORE_DIR}"
fi

echo "CI: done"
