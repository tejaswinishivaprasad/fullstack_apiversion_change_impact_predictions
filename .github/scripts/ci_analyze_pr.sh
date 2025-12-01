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
post_and_gate() {
  OUT="${1:-"${REPO_ROOT}/pr-impact-report.json"}"
  if [ ! -f "$OUT" ]; then
    echo "post_and_gate: report not found at $OUT, skipping posting/gating"
    return 0
  fi

  # helpers
  repo="${GITHUB_REPOSITORY:-}"
  pr="${PR_NUMBER:-}"
  if [ -z "$pr" ] && [ -n "$GITHUB_REF" ]; then
    pr=$(printf "%s" "$GITHUB_REF" | sed -n 's@refs/pull/\([0-9]\+\)/.*@\1@p')
  fi

  if [ -z "$repo" ] || [ -z "$pr" ]; then
    echo "post_and_gate: cannot determine repo/pr (repo=${repo}, pr=${pr})"
    return 0
  fi

  # parse fields safely with jq if available
  RISK=$(jq -r '( .risk_score // .predicted_risk // .impact_assessment.score // 0 )' "$OUT" 2>/dev/null || echo "0")
  RISK_FMT=$(awk -v r="$RISK" 'BEGIN{printf "%.3f", (r+0)}')

  BAND=$(jq -r '.risk_band // .impact_assessment.label // empty' "$OUT" 2>/dev/null || echo "")
  LEVEL=$(jq -r '.risk_level // empty' "$OUT" 2>/dev/null || echo "")

  if [ -z "$LEVEL" ]; then
    SCORE_NUM=$(awk -v r="$RISK" 'BEGIN{printf("%.6f", (r+0))}')
    if (( $(awk "BEGIN {print ($SCORE_NUM >= 0.7)}") )); then
      LEVEL="BLOCK"; BAND=${BAND:-"High"}
    elif (( $(awk "BEGIN {print ($SCORE_NUM >= 0.4)}") )); then
      LEVEL="WARN"; BAND=${BAND:-"Medium"}
    else
      LEVEL="PASS"; BAND=${BAND:-"Low"}
    fi
  fi

  PAIR_ID=$(jq -r '.metadata.pair_id // .backend.pair_id // .pair_id // empty' "$OUT" 2>/dev/null || echo "")
  EXPL_RAW=$(jq -r '.ai_explanation // .explanation // empty' "$OUT" 2>/dev/null || echo "")

  # truncate explanation for top-level display, keep full version in raw JSON block
  EXPL_MAX_CHARS=${EXPL_MAX_CHARS:-600}
  if [ -n "$EXPL_RAW" ] && [ "${#EXPL_RAW}" -gt "$EXPL_MAX_CHARS" ]; then
    EXPL_TRIMMED="$(printf '%s' "$EXPL_RAW" | cut -c1-$EXPL_MAX_CHARS) ... (truncated)"
  else
    EXPL_TRIMMED="$EXPL_RAW"
  fi

  ACES=$(jq -r '.details | length // (.summary_counts?.aces // .summary?.aces // .impact_assessment?.total_aces // (.atomic_change_events | length) // 0) // 0' "$OUT" 2>/dev/null || echo "0")
  BACKEND_IMP=$(jq -r '.backend_impacts | length // 0' "$OUT" 2>/dev/null || echo "0")
  FRONTEND_IMP=$(jq -r '.frontend_impacts | length // 0' "$OUT" 2>/dev/null || echo "0")

  # files list as markdown
  FILES_MD=$(jq -r '.metadata.files_changed // .files_changed // .api_files_changed
    | if . == null then "" elif type=="array" then map("- "+.)|.[] elif type=="string" then "- "+. else tostring end' "$OUT" 2>/dev/null || echo "")
  if [ -z "$FILES_MD" ]; then
    FILES_MD="- (none)"
  fi

  # badge
  if [ "$LEVEL" = "BLOCK" ]; then
    BADGE="🔴 BLOCK (${BAND:-n/a})"
  elif [ "$LEVEL" = "WARN" ]; then
    BADGE="🟡 WARN (${BAND:-n/a})"
  else
    BADGE="🟢 PASS (${BAND:-n/a})"
  fi

  QUICK_LINE="Risk: ${RISK_FMT} | Band: ${BAND:-n/a} | ${BADGE}"
  echo "post_and_gate: ${QUICK_LINE}"

  # prepare body file
  BODY_FILE="$(mktemp --tmpdir pr-impact-body.XXXXXX.md)"
  {
    printf "### Impact AI — Analysis result\n\n"
    printf "**Quick:** %s\n\n" "$QUICK_LINE"
    printf "**Summary**\n\n"
    printf "%s\n\n" "$FILES_MD"
    printf "- ACES: %s\n" "$ACES"
    printf "- Backend impacts: %s\n" "$BACKEND_IMP"
    printf "- Frontend impacts: %s\n\n" "$FRONTEND_IMP"
    [ -n "$PAIR_ID" ] && printf "pair_id: %s\n\n" "$PAIR_ID"

    if [ -n "$EXPL_TRIMMED" ]; then
      printf "**Explanation**\n\n"
      printf '```\n%s\n```\n\n' "$EXPL_TRIMMED"
    fi

    printf "<details>\n<summary>Raw report (click to expand)</summary>\n\n"
    printf "```json\n"
    MAX_LINES=${MAX_LINES:-800}
    if command -v jq >/dev/null 2>&1; then
      # pretty print and cap to MAX_LINES
      jq . "$OUT" 2>/dev/null | sed -n "1,${MAX_LINES}p" || cat "$OUT" | sed -n "1,${MAX_LINES}p"
    else
      cat "$OUT" | sed -n "1,${MAX_LINES}p"
    fi
    printf "\n```\n</details>\n"
  } > "${BODY_FILE}"

  # find existing comment by bot that contains our marker text
  existing_comment_id=""
  if command -v gh >/dev/null 2>&1; then
    # List comments and filter by body contents and by bot user
    # We use gh api to fetch comments and jq to find the id
    existing_comment_id=$(gh api -H "Accept: application/vnd.github.v3+json" \
      "/repos/${repo}/issues/${pr}/comments" --paginate -q \
      '.[] | select(.user.login=="github-actions[bot]" or .user.login=="github-actions") | select(.body | test("Impact AI — Analysis result")) | .id' 2>/dev/null || true)
  else
    # fallback: use curl + jq if available
    if [ -n "${GITHUB_TOKEN:-}" ] && command -v jq >/dev/null 2>&1; then
      existing_comment_id=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/${repo}/issues/${pr}/comments" | jq -r '.[] | select(.user.login=="github-actions[bot]" or .user.login=="github-actions") | select(.body | test("Impact AI — Analysis result")) | .id' 2>/dev/null || true)
    fi
  fi

  # Post or update: prefer gh api update if comment exists
  if [ -n "$existing_comment_id" ]; then
    echo "post_and_gate: updating existing comment id=${existing_comment_id}"
    if command -v gh >/dev/null 2>&1; then
      gh api -X PATCH -H "Accept: application/vnd.github.v3+json" "/repos/${repo}/issues/comments/${existing_comment_id}" -F body="$(cat "${BODY_FILE}")" >/dev/null 2>&1 || echo "gh update failed"
    else
      # curl patch fallback
      if [ -n "${GITHUB_TOKEN:-}" ]; then
        jq -nc --arg body "$(cat "${BODY_FILE}")" '{body:$body}' | \
          curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Content-Type: application/json" -X PATCH --data @- "https://api.github.com/repos/${repo}/issues/comments/${existing_comment_id}" >/dev/null 2>&1 || echo "curl patch failed"
      else
        echo "post_and_gate: cannot update comment (no gh and no GITHUB_TOKEN)"
      fi
    fi
  else
    echo "post_and_gate: posting new comment"
    if command -v gh >/dev/null 2>&1; then
      gh pr comment "${pr}" --body-file "${BODY_FILE}" >/dev/null 2>&1 || echo "gh comment failed"
    else
      if [ -n "${GITHUB_TOKEN:-}" ]; then
        jq -nc --arg body "$(cat "${BODY_FILE}")" '{body:$body}' | \
          curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Content-Type: application/json" -X POST --data @- "https://api.github.com/repos/${repo}/issues/${pr}/comments" >/dev/null 2>&1 || echo "curl post failed"
      else
        echo "post_and_gate: cannot post comment (no gh and no GITHUB_TOKEN)"
      fi
    fi
  fi

  rm -f "${BODY_FILE}" || true

  # export outputs for downstream steps
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "impact_level=${LEVEL}" >> "${GITHUB_OUTPUT}"
    echo "impact_risk=${RISK_FMT}" >> "${GITHUB_OUTPUT}"
  fi

  # gating behavior (unchanged)
  FAIL_ON_BLOCK="${FAIL_ON_BLOCK:-true}"
  if [ "$LEVEL" = "BLOCK" ]; then
    if [ "$FAIL_ON_BLOCK" = "true" ] || [ "$FAIL_ON_BLOCK" = "1" ]; then
      echo "Impact AI gating: BLOCK (risk ${RISK_FMT}). Failing job to prevent merge."
      exit 1
    else
      echo "Impact AI gating: BLOCK (risk ${RISK_FMT}). NOT failing job (FAIL_ON_BLOCK=${FAIL_ON_BLOCK}). Leaving advisory comment only."
      return 0
    fi
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
