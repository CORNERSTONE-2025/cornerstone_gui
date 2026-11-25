#!/usr/bin/env bash
# Nightly test campaign (started by cron, e.g. at 21:00). See docs/NIGHTLY_TESTS.md.
#
#   scripts/nightly/run.sh            # full campaign, stops starting tests at $NIGHTLY_DEADLINE
#   scripts/nightly/run.sh --quick    # short version (~1 h) to check the set-up
#
# Environment (optional): NIGHTLY_DIR (default ~/cornerstone_nightly), NIGHTLY_DEADLINE (07:30),
# NIGHTLY_KEEP (workspaces of the last N runs kept, default 3), NIGHTLY_ONLY (test ids/phases).
set -uo pipefail

# the whole script is parsed before it runs: `git pull` may replace this file meanwhile
main() {

    REPO="$(cd "$(dirname "$0")/../.." && pwd)"
    BASE="${NIGHTLY_DIR:-$HOME/cornerstone_nightly}"
    DEADLINE="${NIGHTLY_DEADLINE:-07:30}"
    KEEP="${NIGHTLY_KEEP:-3}"
    RUN_ID="$(date +%F)"
    [ "${1:-}" = "--quick" ] && RUN_ID="${RUN_ID}_quick"
    RUN="$BASE/$RUN_ID"
    mkdir -p "$RUN"
    exec >>"$RUN/run.log" 2>&1

    # one campaign at a time
    exec 9>"$BASE/.lock"
    if ! flock -n 9; then
        echo "$(date '+%F %T') another campaign is running: exit"
        exit 1
    fi

    echo "===== $(date '+%F %T') nightly campaign $RUN_ID (deadline $DEADLINE) ====="
    cd "$REPO"

    # 1. latest code: GUI from git, package from GitHub (fetched again by the image build)
    git pull --ff-only || echo "!! git pull failed: testing the local version"
    build_start=$(date +%s)
    if docker compose build; then build_ok=true; else build_ok=false; fi
    printf '{"ok": %s, "seconds": %d, "gui_commit": "%s", "finished_at": "%s"}\n' \
        "$build_ok" $(( $(date +%s) - build_start )) "$(git rev-parse HEAD)" "$(date -Is)" >"$RUN/build.json"
    if [ "$build_ok" != true ]; then
        echo "!! docker compose build failed: the campaign runs on the previous image"
    fi

    # 2. the campaign, in a container of the GUI image with the same data folder as the GUI.
    #    Hard stop 15 minutes after the deadline (the campaign itself stops starting tests before it).
    now=$(date +%s)
    stop=$(date -d "$DEADLINE" +%s)
    [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
    limit=$(( stop - now + 900 ))
    name="cornerstone-nightly-$RUN_ID"
    args=(--out /nightly --deadline "$DEADLINE" --deadline-epoch "$stop")
    [ "${1:-}" = "--quick" ] && args+=(--quick)
    [ -n "${NIGHTLY_ONLY:-}" ] && args+=(--only "$NIGHTLY_ONLY")

    docker rm -f "$name" >/dev/null 2>&1
    timeout --foreground "$limit" docker compose run --rm --no-deps -T --name "$name" \
        --user "$(id -u):$(id -g)" \
        -v "$RUN:/nightly" \
        -v /etc/localtime:/etc/localtime:ro \
        -e HOME=/nightly -e MPLCONFIGDIR=/nightly/.mpl \
        -e CS_GUI_WORKSPACE=/nightly/workspace -e CS_GUI_DEMO_WORKSPACE=/nightly/workspace_demo \
        -e CS_GUI_SETTINGS_FILE=/nightly/gui_settings.json \
        gui python scripts/nightly/campaign.py "${args[@]}"
    rc=$?
    if [ $rc -eq 124 ]; then
        echo "!! hard time limit reached: stopping the container"
        docker stop -t 30 "$name" >/dev/null 2>&1
    fi
    echo "campaign exit code $rc"

    # 3. keep the results of every run, the (large) artifact workspaces only of the last $KEEP
    ls -1d "$BASE"/20*/ 2>/dev/null | sort | head -n -"$KEEP" | while read -r old; do
        rm -rf "${old}workspace" "${old}workspace_demo"
    done
    echo "===== $(date '+%F %T') done: $RUN/summary.md ====="
}

main "$@"
exit $?
