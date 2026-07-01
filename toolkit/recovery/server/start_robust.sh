#!/usr/bin/env bash
# Idempotent launcher for the in-pod self-healing pair (keep-alive + watchdog).
# Safe to run repeatedly: it only starts what is not already running.
# The bracket globs in pgrep patterns stop pgrep from matching THIS script.
#
#   bash recovery/server/start_robust.sh
#
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> toolkit/
mkdir -p runs
PY="${PYTHON:-python3}"

start_if_absent () {   # $1 = pgrep pattern, $2.. = command
  local pat="$1"; shift
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "already running: $pat"
  else
    nohup "$@" >> "runs/$(basename "${!#}" .py).log" 2>&1 &
    echo "started: $* (pid $!)"
  fi
}

start_if_absent "[h]ub_keepalive.py" "$PY" recovery/server/hub_keepalive.py
start_if_absent "[w]atchdog.py"      "$PY" recovery/server/watchdog.py

echo "--- status ---"
pgrep -af "[h]ub_keepalive.py" || echo "keepalive: NOT running"
pgrep -af "[w]atchdog.py"      || echo "watchdog: NOT running"
