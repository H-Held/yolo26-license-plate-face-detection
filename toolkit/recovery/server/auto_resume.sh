#!/usr/bin/env bash
# ONE command for a sporadic check-in (phone browser -> JupyterHub web terminal,
# or an agent running a shell tool): print status, and if the run has stopped
# but isn't complete, resume it. Safe to run as often as you like — it never
# duplicates a running watchdog/keep-alive (start_robust.sh is idempotent) and
# never touches anything if the pipeline is already complete or already running.
#
#   bash recovery/server/auto_resume.sh
#
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> toolkit/
PY="${PYTHON:-python3}"

echo "############ BEFORE ############"
STATUS_TEXT="$("$PY" run.py status 2>&1)"
echo "$STATUS_TEXT"

if echo "$STATUS_TEXT" | grep -q "NOTHING IS RUNNING"; then
  echo
  echo "############ RESUMING ############"
  bash recovery/server/start_robust.sh
  sleep 5
  echo
  echo "############ AFTER ############"
  "$PY" run.py status
else
  echo
  echo "############ NO ACTION NEEDED ############"
fi
