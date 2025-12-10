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

# Check if a similar PR comment already exists (dedupe) using gh or REST API
comment_exists() {
  local sentinel="$1"  # the unique sentinel string to search for (HTML comment)
  local prnum="${PR_NUMBER:-}"
  if [ -z "$prnum" ]; then
    # try to extract from GITHUB_REF
    if [ -n "${GITHUB_REF:-}" ] && echo "$GITHUB_REF" | grep -q "refs/pull/"; then
      prnum=$(echo "$GITHUB_REF" | sed -n 's@refs/pull/\([0-9]\+\)/.*@\1@p')
    fi
  fi
  if [ -z "$prnum" ]; then
    echo "comment_exists: PR number unknown; cannot dedupe"
    return 1
  fi

  if command -v gh >/dev/null 2>&1; then
    # list comments bodies and search sentinel
    if gh api repos/"${GITHUB_REPOSITORY}"/issues/"${prnum}"/comments --jq '.[].body' 2>/dev/null | grep -qF "$sentinel"; then
      return 0
    else
      return 1
    fi
  fi

  # fallback to REST API if GITHUB_TOKEN present
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    local api_url="https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${prnum}/comments"
    # Use curl to fetch comment list (may be paginated but usually small)
    if curl -s -H "Authorization: token ${GITHUB_TOKEN}" "$api_url" | grep -qF "$sentinel"; then
      return 0
    else
      return 1
    fi
  fi

  echo "comment_exists: no mechanism to check existing comments (no gh and no GITHUB_TOKEN)."
  return 1
}



# Post comment to PR and gate the job based on report
post_and_gate() {
  OUT="${1:-"${REPO_ROOT}/pr-impact-report.json"}"

  # --- NEW GUARD: allow workflow to opt-out of in-script posting
  if [ "${SKIP_POST:-false}" = "true" ] || [ "${POST_IMPACT_COMMENT:-true}" = "false" ]; then
    echo "post_and_gate: posting skipped (SKIP_POST=${SKIP_POST:-}, POST_IMPACT_COMMENT=${POST_IMPACT_COMMENT:-})"
    # still export gating outputs if present so workflow can read them
    if [ -n "${GITHUB_OUTPUT:-}" ] && [ -f "$OUT" ]; then
      RISK=$(jq -r '( .risk_score // .predicted_risk // .impact_assessment.score // 0 )' "$OUT" 2>/dev/null || echo "0")
      RISK_FMT=$(awk -v r="$RISK" 'BEGIN{printf "%.3f", (r+0)}')
      LEVEL=$(jq -r '.risk_level // empty' "$OUT" 2>/dev/null || echo "")
      echo "impact_level=${LEVEL}" >> "${GITHUB_OUTPUT}" || true
      echo "impact_risk=${RISK_FMT}" >> "${GITHUB_OUTPUT}" || true
    fi
    return 0
  fi

  if [ ! -f "$OUT" ]; then
    echo "post_and_gate: report not found at $OUT, skipping posting/gating"
    return 0
  fi

  ensure_jq || true

  # extract core fields with safe fallbacks
  RISK=$(jq -r '( .risk_score // .predicted_risk // .impact_assessment.score // 0 )' "$OUT" 2>/dev/null || echo "0")
  RISK_FMT=$(awk -v r="$RISK" 'BEGIN{printf "%.3f", (r+0)}')

  BAND=$(jq -r '.risk_band // .impact_assessment.label // empty' "$OUT" 2>/dev/null || echo "")
  LEVEL=$(jq -r '.risk_level // empty' "$OUT" 2>/dev/null || echo "")

  # derive level if missing
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

  # artifact id sentinel (attempt from report or fallback)
  ARTIFACT=$(jq -r '.artifact // .report_id // .artifact_id // "pr-impact-report"' "$OUT" 2>/dev/null || echo "pr-impact-report")
  PAIR_ID=$(jq -r '.metadata.pair_id // .backend.pair_id // .pair_id // empty' "$OUT" 2>/dev/null || echo "")
  EXPL=$(jq -r '.ai_explanation // .explanation // empty' "$OUT" 2>/dev/null || echo "")

  # safe truncation for explanation (avoid huge comments)
  EXPL_MAX_CHARS=${EXPL_MAX_CHARS:-800}
  if [ -n "$EXPL" ]; then
    if [ "${#EXPL}" -gt "$EXPL_MAX_CHARS" ]; then
      EXPL_TRIMMED="$(printf '%s' "$EXPL" | cut -c1-$EXPL_MAX_CHARS) ... (truncated)"
    else
      EXPL_TRIMMED="$EXPL"
    fi
  else
    EXPL_TRIMMED=""
  fi

  # counts: ACES, then BE/FE count (we'll recompute from lists)
  ACES=$(jq -r '.details | length // (.summary_counts?.aces // .summary?.aces // .impact_assessment?.total_aces // (.atomic_change_events | length) // 0) // 0' "$OUT" 2>/dev/null || echo "0")

  # --- NEW: resolve backend/frontend impact *lists* robustly (any nesting) ---
  FULL_JSON="${REPO_ROOT:-.}/pr-impact-full.json"

  BACKEND_IMPACT_LINES=$(jq -r '
    [ .. | objects | .backend_impacts? | select(type=="array") ][0] // [] |
    .[0:5][]? |
      "- " + (
        .service // .producer // .target // .name // "Unknown"
      ) + " (risk ~ " + (
        (.risk_score // .score // .risk // 0) | tostring
      ) + ")"
  ' "$OUT" 2>/dev/null || echo "")

  FRONTEND_IMPACT_LINES=$(jq -r '
    [ .. | objects | .frontend_impacts? | select(type=="array") ][0] // [] |
    .[0:5][]? |
      "- " + (
        .service // .consumer // .target // .name // "Unknown"
      ) + " (risk ~ " + (
        (.risk_score // .score // .risk // 0) | tostring
      ) + ")"
  ' "$OUT" 2>/dev/null || echo "")

  # fallback to full JSON if summary doesn't contain impacts
  if [ -z "$BACKEND_IMPACT_LINES" ] && [ -f "$FULL_JSON" ]; then
    BACKEND_IMPACT_LINES=$(jq -r '
      [ .. | objects | .backend_impacts? | select(type=="array") ][0] // [] |
      .[0:5][]? |
        "- " + (
          .service // .producer // .target // .name // "Unknown"
        ) + " (risk ~ " + (
          (.risk_score // .score // .risk // 0) | tostring
        ) + ")"
    ' "$FULL_JSON" 2>/dev/null || echo "")
  fi

  if [ -z "$FRONTEND_IMPACT_LINES" ] && [ -f "$FULL_JSON" ]; then
    FRONTEND_IMPACT_LINES=$(jq -r '
      [ .. | objects | .frontend_impacts? | select(type=="array") ][0] // [] |
      .[0:5][]? |
        "- " + (
          .service // .consumer // .target // .name // "Unknown"
        ) + " (risk ~ " + (
          (.risk_score // .score // .risk // 0) | tostring
        ) + ")"
    ' "$FULL_JSON" 2>/dev/null || echo "")
  fi

  echo "DEBUG: backend impact lines:"
  printf '%s\n' "${BACKEND_IMPACT_LINES:-<empty>}"
  echo "DEBUG: frontend impact lines:"
  printf '%s\n' "${FRONTEND_IMPACT_LINES:-<empty>}"

  # derive counts from lines (more robust than trusting schema)
  BACKEND_IMP="$(printf '%s\n' "${BACKEND_IMPACT_LINES}" | sed '/^\s*$/d' | wc -l | tr -d ' ' || echo 0)"
  FRONTEND_IMP="$(printf '%s\n' "${FRONTEND_IMPACT_LINES}" | sed '/^\s*$/d' | wc -l | tr -d ' ' || echo 0)"

  # format files list as markdown bullets
  FILES_MD=""
  if jq -e '.metadata.files_changed // .files_changed // .api_files_changed' "$OUT" >/dev/null 2>&1; then
    FILES_MD=$(jq -r '.metadata.files_changed // .files_changed // .api_files_changed
      | if . == null then "" elif type=="array" then map("- "+.)|.[] elif type=="string" then "- "+. else tostring end' "$OUT" 2>/dev/null || echo "")
  fi
  if [ -z "$FILES_MD" ]; then
    FILES_MD=$(jq -r '.files_changed // [] | if type=="array" then map("- "+.)|.[] elif type=="string" then "- "+. else empty end' "$OUT" 2>/dev/null || echo "")
  fi
  if [ -z "$FILES_MD" ]; then
    FILES_MD="- (none)"
  fi

  # badge text
  if [ "$LEVEL" = "BLOCK" ]; then
    BADGE="🔴 BLOCK (${BAND:-n/a})"
  elif [ "$LEVEL" = "WARN" ]; then
    BADGE="🟡 WARN (${BAND:-n/a})"
  else
    BADGE="🟢 PASS (${BAND:-n/a})"
  fi

  QUICK_LINE="Risk: ${RISK_FMT} | Band: ${BAND:-n/a} | ${BADGE} | ACE Count: ${ACES}"

  echo "post_and_gate: $QUICK_LINE"

  # Convert any literal "\n" in EXPL to real newlines (normalize)
  EXPL_PRETTY="$(printf '%b' "${EXPL_TRIMMED:-}")"

  # Build markdown body file (preserves real newlines)
  BODY_FILE="$(mktemp --tmpdir pr-impact-body.XXXXXX.md)"
  SENTINEL="<!-- impact-report-id: ${ARTIFACT} | risk:${RISK_FMT} | aces:${ACES} | be:${BACKEND_IMP} | fe:${FRONTEND_IMP} -->"
  echo "post_and_gate: using sentinel: ${SENTINEL}"

  {
    printf "### Impact AI — Analysis result\n\n"
    printf "**Quick:** %s\n\n" "$QUICK_LINE"

    printf "**Summary**\n\n"
    printf "%s\n\n" "$FILES_MD"

    printf "- ACES: %s\n" "$ACES"
    printf "- Backend impacts: %s\n" "$BACKEND_IMP"
    printf "- Frontend impacts: %s\n\n" "$FRONTEND_IMP"

    # pretty-print backend/frontend impact samples
    if [ -n "$BACKEND_IMPACT_LINES" ]; then
      printf "**Backend impacts (sample)**\n\n"
      printf "%s\n\n" "$BACKEND_IMPACT_LINES"
    fi

    if [ -n "$FRONTEND_IMPACT_LINES" ]; then
      printf "**Frontend impacts (sample)**\n\n"
      printf "%s\n\n" "$FRONTEND_IMPACT_LINES"
    fi

    [ -n "$PAIR_ID" ] && printf "pair_id: %s\n\n" "$PAIR_ID"

    if [ -n "$EXPL_PRETTY" ]; then
      printf "**Explanation**\n\n"
      printf '```\n%s\n```\n\n' "$EXPL_PRETTY"
    fi

       # Raw report in collapsible block; pretty-print and limit lines
    MAX_LINES=${MAX_LINES:-500}
    RAW_PRETTY=""

    if [ -f "$OUT" ]; then
      if command -v jq >/dev/null 2>&1; then
        RAW_PRETTY="$(jq . "$OUT" 2>/dev/null | sed -n "1,${MAX_LINES}p" || sed -n "1,${MAX_LINES}p" "$OUT")"
      else
        RAW_PRETTY="$(sed -n "1,${MAX_LINES}p" "$OUT")"
      fi
    else
      RAW_PRETTY="{ \"error\": \"report file not found at $OUT\" }"
    fi

    printf "<details>\n<summary>Raw report (click to expand)</summary>\n\n"
    printf '```json\n%s\n```\n' "$RAW_PRETTY"
    printf "\n</details>\n\n"

    printf "%s\n" "$SENTINEL"
  } > "${BODY_FILE}"


  # Dedupe: skip posting if comment with identical sentinel already exists
  if comment_exists "$SENTINEL"; then
    echo "post_and_gate: similar impact comment already exists — skipping duplicate post"
  else
    # Post using gh if available, else fallback to safe python+curl approach
    if command -v gh >/dev/null 2>&1; then
      if [ -z "${GH_TOKEN:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
        export GH_TOKEN="${GITHUB_TOKEN}"
      fi
      if [ -n "${PR_NUMBER:-}" ]; then
        echo "post_and_gate: posting PR comment via gh for PR ${PR_NUMBER}"
        gh pr comment "${PR_NUMBER}" --body-file "${BODY_FILE}" || echo "gh comment failed"
      else
        if [ -n "$GITHUB_REF" ] && echo "$GITHUB_REF" | grep -q "refs/pull/"; then
          PR_NUM=$(echo "$GITHUB_REF" | sed -n 's@refs/pull/\([0-9]\+\)/.*@\1@p')
          [ -n "$PR_NUM" ] && gh pr comment "${PR_NUM}" --body-file "${BODY_FILE}" || echo "gh comment failed"
        else
          echo "post_and_gate: cannot determine PR number for gh comment"
        fi
      fi
    else
      if [ -z "${GITHUB_TOKEN:-}" ]; then
        echo "post_and_gate: gh not found and GITHUB_TOKEN not set — skipping PR comment."
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

          PAYLOAD="$(python3 - <<PY
import json,sys
body_path = "${BODY_FILE}"
with open(body_path, "r", encoding="utf-8") as fh:
    body = fh.read()
print(json.dumps({"body": body}))
PY
)"
          curl -s -H "Authorization: token ${GITHUB_TOKEN}" -H "Content-Type: application/json" -X POST -d "${PAYLOAD}" "${API_URL}" || echo "curl post failed"
        else
          echo "post_and_gate: Cannot determine PR number to post comment."
        fi
      fi
    fi
  fi

  # cleanup
  rm -f "${BODY_FILE}" || true

  # export outputs for downstream steps
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "impact_level=${LEVEL}" >> "${GITHUB_OUTPUT}"
    echo "impact_risk=${RISK_FMT}" >> "${GITHUB_OUTPUT}"
  fi

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
  REPORT_FULL="$(find "${REPO_ROOT}" -maxdepth 4 -type f -name 'pr-impact-full.json' -print -quit || true)"
  REPORT_SUM="$(find "${REPO_ROOT}" -maxdepth 4 -type f -name 'pr-impact-report.json' -print -quit || true)"

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
