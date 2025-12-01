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

# Try to ensure jq is available (best effort). We can't assume sudo in hosted runner.
ensure_jq() {
  if command -v jq >/dev/null 2>&1; then
    return 0
  fi
  echo "CI: jq not found in PATH."
  # Attempt install if apt-get exists and we have sudo
  if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
    echo "CI: attempting to install jq via apt-get"
    sudo apt-get update -y && sudo apt-get install -y jq || true
  elif command -v yum >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
    echo "CI: attempting to install jq via yum"
    sudo yum install -y jq || true
  else
    echo "CI: cannot auto-install jq (no apt-get/yum or no sudo). Ensure runner has jq or parsing will be degraded."
  fi
}

# Post comment to PR and gate the job based on report
# Post comment to PR and gate the job based on report
post_and_gate() {
  # $1 = path to JSON report (OUT)
  OUT="${1:-"${REPO_ROOT}/pr-impact-report.json"}"
  if [ ! -f "$OUT" ]; then
    echo "post_and_gate: report not found at $OUT, skipping posting/gating"
    return 0
  fi

  # Check prerequisites
  if ! command -v jq >/dev/null 2>&1; then
    echo "post_and_gate: jq not found; please install jq in the runner for full functionality"
    # continue but parsing will be weak
  fi

  # read canonical fields with sane fallbacks
  RISK=$(jq -r 'if .risk_score!=null then .risk_score else ( .predicted_risk // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .impact_assessment.score // .score // 0 ) end' "$OUT" 2>/dev/null || echo "0")
  # normalize numeric formatting (3 decimals)
  RISK_FMT=$(awk -v r="$RISK" 'BEGIN{printf "%.3f", (r+0)}')

  # prefer explicit band fields; fallback to impact_assessment.label or compute from score
  BAND=$(jq -r '.risk_band // .impact_assessment.label // .impact_assessment.label // empty' "$OUT" 2>/dev/null || echo "")
  LEVEL=$(jq -r '.risk_level // .risk_level // empty' "$OUT" 2>/dev/null || echo "")

  if [ -z "$LEVEL" ]; then
    SCORE_NUM=$(awk -v r="$RISK" 'BEGIN{print (r+0)}')
    if (( $(awk "BEGIN {print ($SCORE_NUM >= 0.7)}") )); then
      LEVEL="BLOCK"
      BAND=${BAND:-"High"}
    elif (( $(awk "BEGIN {print ($SCORE_NUM >= 0.4)}") )); then
      LEVEL="WARN"
      BAND=${BAND:-"Medium"}
    else
      LEVEL="PASS"
      BAND=${BAND:-"Low"}
    fi
  fi

  # prefer metadata.pair_id, then backend.pair_id, then scan logs if necessary
  PAIR_ID=$(jq -r '.metadata.pair_id // .backend.pair_id // .pair_id // empty' "$OUT" 2>/dev/null || echo "")
  if [ -z "$PAIR_ID" ]; then
    PAIR_ID=$(jq -r '[.logs[]? // empty] | map(select(. != "")) | .[]? as $l | ($l | capture("pair_id=(?<pid>[^\\s,;]+)"))?.pid // empty' "$OUT" 2>/dev/null || echo "")
  fi

  EXPL=$(jq -r '.ai_explanation // .explanation // empty' "$OUT" 2>/dev/null || echo "")

  # other handy fields
  ACES=$(jq -r '.details | length // (.summary_counts?.aces // .summary?.aces // .impact_assessment?.total_aces // .atomic_change_events | length // 0) // 0' "$OUT" 2>/dev/null || echo "0")
  BACKEND_IMP=$(jq -r '.backend_impacts | length // 0' "$OUT" 2>/dev/null || echo "0")
  FRONTEND_IMP=$(jq -r '.frontend_impacts | length // 0' "$OUT" 2>/dev/null || echo "0")
  FILES_CHANGED=$(jq -r '.metadata.files_changed // .files_changed // .api_files_changed // empty' "$OUT" 2>/dev/null || echo "")

  # Make a small badge and result string
  if [ "$LEVEL" = "BLOCK" ]; then
    BADGE="🔴 **BLOCK**"
  elif [ "$LEVEL" = "WARN" ]; then
    BADGE="🟡 **WARN**"
  else
    BADGE="🟢 **PASS**"
  fi

  RISK_LINE="**Risk:** ${RISK_FMT} (${BAND:-n/a}) — ${BADGE}"

  # Quick one-line summary for logs & top of PR comment (this is the 'spark' line)
  QUICK_LINE="Quick: Risk: ${RISK_FMT} | Band: ${BAND:-n/a} | ${BADGE} | Files: ${FILES_CHANGED:-0} | API changes: ${ACES:-0}"
  # also echo to runner log so it's visible in Actions console
  echo "$QUICK_LINE"

  META_LINE=""
  if [ -n "$PAIR_ID" ]; then
    META_LINE="$META_LINE • pair_id=${PAIR_ID}"
  fi
  if [ -n "$FILES_CHANGED" ]; then
    META_LINE="$META_LINE • files=${FILES_CHANGED}"
  fi

  # Build markdown body (include quick line prominently)
  BODY="### Impact AI — Analysis result\n\n**${QUICK_LINE}**\n\n${RISK_LINE}\n\n"
  BODY="${BODY}- ACES detected: ${ACES:-0}\n- Backend impacts: ${BACKEND_IMP:-0}\n- Frontend impacts: ${FRONTEND_IMP:-0}\n"
  if [ -n "$META_LINE" ]; then
    BODY="${BODY}\n${META_LINE}\n"
  fi

  if [ -n "$EXPL" ]; then
    BODY="${BODY}\n**Explanation:**\n\n\`\`\`\n${EXPL}\n\`\`\`\n"
  fi

  # include compact JSON summary for debugging / link back
  BODY="${BODY}\n<details>\n<summary>Raw report (click to expand)</summary>\n\n\`\`\`json\n$(jq -c '.' "$OUT" | sed 's/`/`/g')\n\`\`\`\n</details>\n"

  # Post comment: prefer gh CLI; fallback to curl with GITHUB_TOKEN
  if command -v gh >/dev/null 2>&1; then
    if [ -n "${PR_NUMBER:-}" ]; then
      echo "post_and_gate: posting PR comment via gh for PR ${PR_NUMBER}"
      gh pr comment "${PR_NUMBER}" --body "$BODY" || echo "gh comment failed"
    else
      if [ -n "$GITHUB_REF" ] && echo "$GITHUB_REF" | grep -q "refs/pull/"; then
        PR_NUM=$(echo "$GITHUB_REF" | sed -n 's@refs/pull/\([0-9]\+\)/.*@\1@p')
        if [ -n "$PR_NUM" ]; then
          echo "post_and_gate: posting PR comment via gh for PR ${PR_NUM}"
          gh pr comment "${PR_NUM}" --body "$BODY" || echo "gh comment failed"
        fi
      else
        echo "post_and_gate: cannot determine PR number for gh comment (set PR_NUMBER or run on pull_request event)"
      fi
    fi
  else
    if [ -z "${GITHUB_TOKEN:-}" ]; then
      echo "Warning: gh not found and GITHUB_TOKEN not set — skipping PR comment."
    else
      if [ -z "${PR_NUMBER:-}" ]; then
        if [ -n "$GITHUB_REF" ] && echo "$GITHUB_REF" | grep -q "refs/pull/"; then
          PR_NUMBER=$(echo "$GITHUB_REF" | sed -n 's@refs/pull/\([0-9]\+\)/.*@\1@p')
        fi
      fi
      if [ -n "${PR_NUMBER:-}" ]; then
        REPO=${GITHUB_REPOSITORY:-}
        API_URL="https://api.github.com/repos/${REPO}/issues/${PR_NUMBER}/comments"
        echo "post_and_gate: posting PR comment via REST API for PR ${PR_NUMBER}"
        curl -s -H "Authorization: token ${GITHUB_TOKEN}" -X POST -d "$(jq -nc --arg body "$BODY" '{body:$body}')" "$API_URL" || echo "curl post failed"
      else
        echo "post_and_gate: Cannot determine PR number to post comment. Set PR_NUMBER env or let workflow run on pull_request event."
      fi
    fi
  fi

  # Fail the job if BLOCK (this prevents merging if branch protection requires this job)
  if [ "$LEVEL" = "BLOCK" ]; then
    echo "Impact AI gating: BLOCK (risk ${RISK_FMT}). Failing job to prevent merge."
    # exit non-zero to fail workflow and block PR merge when job is required by branch protection
    exit 1
  fi

  echo "Impact AI gating: ${LEVEL} (risk ${RISK_FMT}). Continuing."
  return 0
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
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
      echo "report=pr-impact-report.json" >> "${GITHUB_OUTPUT}" || true
    else
      # fallback for older runners
      echo "::set-output name=report::pr-impact-report.json" || true
    fi
    HANDLED=1
  fi

  if [ "${HANDLED}" -gt 0 ]; then
    return 0
  else
    return 1
  fi
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
    if copy_report; then
      post_and_gate "${REPO_ROOT}/pr-impact-report.json" || true
    else
      echo "CI: no report produced by ci_server_run.py"
    fi
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
    if copy_report; then
      post_and_gate "${REPO_ROOT}/pr-impact-report.json" || true
    else
      echo "CI: no report produced by analysis_light.py"
    fi
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
    if copy_report; then
      post_and_gate "${REPO_ROOT}/pr-impact-report.json" || true
    else
      echo "CI: no report produced by server.py --ci-scan"
    fi
    exit 0
  else
    echo "CI: server.py --ci-scan failed" >&2
    popd > /dev/null || true
  fi
fi

# final attempt to copy any report
if copy_report; then
  echo "CI: found report after fallback attempts"
  post_and_gate "${REPO_ROOT}/pr-impact-report.json" || true
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
