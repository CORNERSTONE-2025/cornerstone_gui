#!/usr/bin/env bash
# Morning report of the nightly campaign (started by cron, e.g. at 07:30). See docs/NIGHTLY_TESTS.md.
#
#   scripts/nightly/report.sh [RUN_DIR]     # default: the most recent run in $NIGHTLY_DIR
#
# 1. Claude Code (headless) reads the results and writes REPORT.md in the run folder; it may only
#    read files and write in that folder. Without Claude, summary.md (automatic) is the report.
# 2. The report, summary, results and figures are committed to the branch `test-reports`
#    (git worktree in $NIGHTLY_DIR/reports) with the git identity of this machine, and pushed,
#    unless NIGHTLY_PUSH=0.
set -uo pipefail

# the whole script is parsed before it runs: `git pull` may replace this file meanwhile
main() {

    REPO="$(cd "$(dirname "$0")/../.." && pwd)"
    BASE="${NIGHTLY_DIR:-$HOME/cornerstone_nightly}"
    RUN="${1:-$(ls -1d "$BASE"/20*/ 2>/dev/null | sort | tail -n 1)}"
    RUN="${RUN%/}"
    BRANCH=test-reports
    [ -f "$RUN/results.json" ] || { echo "No results in '$RUN'"; exit 1; }
    RUN_ID="$(basename "$RUN")"
    exec >>"$RUN/report.log" 2>&1
    echo "===== $(date '+%F %T') report of $RUN_ID ====="

    # 1. Claude writes REPORT.md (at most 30 minutes)
    if command -v claude >/dev/null; then
        prompt="$(sed -e "s|{{RUN}}|$RUN|g" -e "s|{{REPO}}|$REPO|g" "$REPO/scripts/nightly/report_prompt.md")"
        (cd "$RUN" && timeout 30m claude -p "$prompt" \
            --allowedTools "Read" "Glob" "Grep" "Write" "Edit" \
            --permission-mode acceptEdits) || echo "!! claude ended with an error"
    else
        echo "!! claude not found in PATH: only the automatic summary"
    fi

    # 2. publish on the test-reports branch
    [ "${NIGHTLY_PUSH:-1}" = "0" ] && { echo "push disabled"; exit 0; }
    WT="$BASE/reports"
    cd "$REPO"
    git fetch -q origin "$BRANCH" 2>/dev/null
    if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
        if git rev-parse -q --verify "origin/$BRANCH" >/dev/null; then
            git worktree add -B "$BRANCH" "$WT" "origin/$BRANCH"
        else                                    # first report: branch without the code history
            git worktree add --detach "$WT"
            (cd "$WT" && git checkout --orphan "$BRANCH" && git rm -rqf . && \
             printf '# CORNERSTONE nightly test reports\n\nOne folder per night: REPORT.md (Claude), summary.md (automatic), results.json, figures.\n' > README.md)
        fi
    fi
    cd "$WT"
    git pull -q --ff-only origin "$BRANCH" 2>/dev/null
    dst="reports/$RUN_ID"
    mkdir -p "$dst"
    for f in REPORT.md summary.md results.json build.json; do
        [ -f "$RUN/$f" ] && cp "$RUN/$f" "$dst/"
    done
    [ -d "$RUN/figures" ] && cp -r "$RUN/figures" "$dst/"
    git add -A
    git commit -qm "Nightly test report $RUN_ID" && git push -q -u origin "$BRANCH" \
        && echo "pushed to $BRANCH: $dst" || echo "!! nothing to commit or push failed"
}

main "$@"
exit $?
