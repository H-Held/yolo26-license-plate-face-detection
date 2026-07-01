"""Anti-cull keep-alive (in-pod). Posts JupyterHub activity so the hub does not
cull the server while a long nohup training runs with no browser attached.

Start once (start_robust.sh does this for you):
    nohup python recovery/server/hub_keepalive.py >> runs/keepalive.log 2>&1 &
"""
from __future__ import annotations
import os
import time
import urllib.request

INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "120"))


def _post():
    url = os.environ.get("JUPYTERHUB_ACTIVITY_URL")
    token = os.environ.get("JUPYTERHUB_API_TOKEN")
    server = os.environ.get("JUPYTERHUB_SERVER_NAME", "")
    if not url or not token:
        return "no JUPYTERHUB_ACTIVITY_URL/API_TOKEN in env (nothing to post)"
    import json
    body = json.dumps({"servers": {server: {"last_activity": _now_iso()}},
                       "last_activity": _now_iso()}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=15) as r:
        return f"HTTP {r.status}"


def _now_iso():
    # activity endpoint wants an ISO8601 UTC timestamp
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def main():
    while True:
        try:
            print(time.strftime("%H:%M:%S"), "keepalive", _post(), flush=True)
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "keepalive ERROR", e, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
