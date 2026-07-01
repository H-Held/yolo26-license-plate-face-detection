#!/usr/bin/env python
"""
PROGRAM 2 — client-side resume watchdog (restart layer #2, runs on YOUR Windows PC).
=====================================================================================
GENERIC, CONFIG-DRIVEN version for the toolkit. It survives JupyterHub pod culls:
when the hub tears the pod down, every in-pod process dies (keep-alive + in-pod
watchdog). This runs on your PC, independent of the pod, and:

  1. ensures the HTW VPN is up + the hub is reachable,
  2. (re)starts the GPU server profile if it is not running,
  3. once ready, (re)launches the in-pod robust system (recovery/server/start_robust.sh),
     which RESUMES training from the last checkpoint,
  4. stops itself once the run reports pipeline_state.json stages.complete.

ALL run-specific paths come from `recovery.json` (copy recovery.json.example).
Nothing here is hard-coded to a particular run — change the json, not the code.

Run:
  <mcp-venv-python> program2_resume_watchdog.py [path\\to\\recovery.json]
Stop: Ctrl+C (in-pod training keeps running).
"""
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent
CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "recovery.json"
LOGFILE = HERE / "program2_resume_watchdog.log"


def _load_recovery() -> dict:
    if not CFG_PATH.exists():
        print(f"FATAL: {CFG_PATH} not found (copy recovery.json.example).")
        sys.exit(2)
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


RC = _load_recovery()
REMOTE_ROOT = RC["remote_root"]
TOOLKIT_DIR = RC.get("toolkit_dir", f"{REMOTE_ROOT}/toolkit")
DATA_ROOT = RC.get("data_root", f"{REMOTE_ROOT}/deepseek")
RUN_NAME = RC["run_name"]
START_SCRIPT = f"{TOOLKIT_DIR}/recovery/server/start_robust.sh"
STATE_FILE = f"{DATA_ROOT}/runs/faces/{RUN_NAME}/pipeline_state.json"
WATCHDOG_PATTERN = "[w]atchdog.py"
PROFILE_SLUG = RC.get("profile_slug", "gpu-v100-48gb")
POLL_SECONDS = int(RC.get("poll_seconds", 120))
VPN_TASK_NAME = RC.get("vpn_task_name", "HTW-VPN-AutoConnect")

# MCP config supplies hub_url + token (reused, so there is only one copy of the token)
MCP_CONFIG = (HERE / RC["mcp_config"]).resolve() if RC.get("mcp_config") else None
CONNECT_DIR = HERE.parent.parent.parent / "connect"   # best-effort default
KERNEL_NAME_FALLBACK = "python3"


def load_cfg() -> dict:
    cfg = {"hub_url": RC.get("hub_url", ""), "token": "", "verify_tls": True,
           "http_timeout": 30, "server_start_timeout": 300, "vpn_connect_timeout": 90}
    if MCP_CONFIG and MCP_CONFIG.exists():
        cfg.update(json.loads(MCP_CONFIG.read_text(encoding="utf-8")))
    if os.environ.get("JUPYTERHUB_TOKEN"):
        cfg["token"] = os.environ["JUPYTERHUB_TOKEN"]
    if RC.get("hub_url"):
        cfg["hub_url"] = RC["hub_url"]
    cfg["hub_url"] = cfg["hub_url"].rstrip("/")
    return cfg


CFG = load_cfg()
KERNEL_NAME = KERNEL_NAME_FALLBACK


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(msg: str) -> None:
    line = f"{now_iso()} {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"Authorization": f"token {CFG['token']}"},
                             verify=CFG.get("verify_tls", True),
                             timeout=CFG.get("http_timeout", 30), follow_redirects=True)


def _hub_api(path: str) -> str:
    return f"{CFG['hub_url']}/hub/api{path}"


def _user_api(user: str, path: str) -> str:
    return f"{CFG['hub_url']}/user/{user}{path}"


# ---- VPN / reachability (Windows scheduled-task based) ----
async def hub_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=8, verify=CFG.get("verify_tls", True),
                                     follow_redirects=True) as c:
            r = await c.get(f"{CFG['hub_url']}/hub/api")
            return r.status_code < 500
    except Exception:
        return False


def _run(cmd, timeout=30):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _task_exists() -> bool:
    try:
        return _run(["schtasks", "/query", "/tn", VPN_TASK_NAME]).returncode == 0
    except Exception:
        return False


def vpn_stop() -> None:
    try:
        if _task_exists():
            _run(["schtasks", "/end", "/tn", VPN_TASK_NAME])
    except Exception as e:
        log(f"vpn_stop error: {e!r}")


def vpn_start() -> str:
    if _task_exists():
        try:
            _run(["schtasks", "/run", "/tn", VPN_TASK_NAME])
            return f"scheduled task {VPN_TASK_NAME}"
        except Exception as e:
            log(f"vpn_start error: {e!r}")
    return "no VPN scheduled task found (set vpn_task_name in recovery.json)"


async def vpn_full_restart() -> None:
    log("VPN full restart: stopping tunnel ...")
    vpn_stop()
    await asyncio.sleep(4)
    log(f"VPN full restart: starting ({vpn_start()})")


async def ensure_reachable() -> bool:
    if await hub_reachable():
        return True
    for attempt in range(1, 4):
        log(f"hub not reachable -> VPN full restart ({attempt}/3)")
        await vpn_full_restart()
        deadline = asyncio.get_event_loop().time() + CFG.get("vpn_connect_timeout", 90)
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(5)
            if await hub_reachable():
                log("hub reachable after VPN restart")
                return True
        log(f"still unreachable after attempt {attempt}/3")
    return False


# ---- hub: whoami / server state / start ----
async def whoami() -> dict:
    async with _client() as c:
        r = await c.get(_hub_api("/user"))
        r.raise_for_status()
        return r.json()


def default_server(model: dict) -> dict:
    servers = model.get("servers") or {}
    d = servers.get("") or servers.get(model.get("name", ""), {})
    if not d and servers:
        d = next(iter(servers.values()))
    return d or {}


def server_ready(model: dict) -> bool:
    return bool(default_server(model).get("ready"))


def server_pending(model: dict) -> bool:
    return bool(default_server(model).get("pending"))


async def start_server(user: str) -> None:
    log(f"starting server for {user} with profile '{PROFILE_SLUG}' ...")
    async with _client() as c:
        try:
            r = await c.post(_hub_api(f"/users/{user}/server"),
                             json={"profile": PROFILE_SLUG})
            log(f"  start POST -> HTTP {r.status_code}")
        except Exception as e:
            log(f"  start POST error: {e!r}")
    await wait_ready(user)


async def wait_ready(user: str) -> bool:
    deadline = asyncio.get_event_loop().time() + CFG.get("server_start_timeout", 300)
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with _client() as c:
                async with c.stream("GET", _hub_api(f"/users/{user}/server/progress"),
                                    timeout=30) as r:
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        evt = json.loads(line[5:].strip())
                        if evt.get("ready"):
                            log("  server ready"); return True
                        if evt.get("failed"):
                            log(f"  server start FAILED: {evt.get('message','')}"); return False
        except Exception:
            pass
        if server_ready(await whoami()):
            log("  server ready"); return True
        await asyncio.sleep(3)
    log("  server start timed out")
    return False


# ---- in-pod command execution via a Jupyter kernel websocket ----
_KID = {"id": None}


async def _kernel_alive(user: str, kid: str) -> bool:
    try:
        async with _client() as c:
            r = await c.get(_user_api(user, f"/api/kernels/{kid}"))
            return r.status_code == 200
    except Exception:
        return False


async def _default_kernelspec(user: str) -> str:
    try:
        async with _client() as c:
            r = await c.get(_user_api(user, "/api/kernelspecs"))
            if r.status_code == 200:
                ks = r.json()
                return ks.get("default") or next(iter(ks.get("kernelspecs") or {}), "python3")
    except Exception:
        pass
    return "python3"


async def ensure_kernel(user: str) -> str:
    kid = _KID["id"]
    if kid and await _kernel_alive(user, kid):
        return kid
    for name in (KERNEL_NAME, await _default_kernelspec(user)):
        try:
            async with _client() as c:
                r = await c.post(_user_api(user, "/api/kernels"), json={"name": name})
            if r.status_code in (200, 201):
                _KID["id"] = r.json()["id"]
                return _KID["id"]
            log(f"kernel start '{name}' -> HTTP {r.status_code}; trying fallback")
        except Exception as e:
            log(f"kernel start '{name}' error: {e!r}; fallback")
    raise RuntimeError("could not start any kernel")


def _msg(msg_type: str, content: dict, session: str) -> dict:
    return {"header": {"msg_id": uuid.uuid4().hex, "username": "program2",
                       "session": session, "date": now_iso(),
                       "msg_type": msg_type, "version": "5.3"},
            "parent_header": {}, "metadata": {}, "content": content,
            "channel": "shell", "buffers": []}


async def exec_code(user: str, code: str, timeout: int = 150) -> str:
    kid = await ensure_kernel(user)
    ws_base = CFG["hub_url"].replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/user/{user}/api/kernels/{kid}/channels?token={CFG['token']}"
    session = uuid.uuid4().hex
    req = _msg("execute_request", {"code": code, "silent": False, "store_history": False,
                                   "user_expressions": {}, "allow_stdin": False,
                                   "stop_on_error": True}, session)
    parent_id = req["header"]["msg_id"]
    out = []

    async def run():
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps(req))
            idle = reply = False
            while not (idle and reply):
                m = json.loads(await ws.recv())
                if m.get("parent_header", {}).get("msg_id") != parent_id:
                    continue
                mt = m.get("msg_type") or m.get("header", {}).get("msg_type")
                ch, c = m.get("channel"), m.get("content", {})
                if ch == "iopub" and mt == "stream":
                    out.append(c.get("text", ""))
                elif ch == "iopub" and mt == "error":
                    out.append("\n".join(c.get("traceback", [])))
                elif ch == "iopub" and mt == "status" and c.get("execution_state") == "idle":
                    idle = True
                elif ch == "shell" and mt == "execute_reply":
                    reply = True

    try:
        await asyncio.wait_for(run(), timeout=timeout)
    except asyncio.TimeoutError:
        out.append(f"[exec timeout after {timeout}s]")
    return "".join(out)


HEAL_CODE = f"""
import json, os, subprocess
st = {STATE_FILE!r}
done = False
if os.path.exists(st):
    try:
        done = bool(json.load(open(st)).get('stages', {{}}).get('complete'))
    except Exception:
        done = False
print('CAMPAIGN_DONE' if done else 'CAMPAIGN_RUNNING')
if not done:
    subprocess.run(
        'pgrep -f "{WATCHDOG_PATTERN}" >/dev/null && echo WATCHDOG_ALIVE '
        '|| bash {START_SCRIPT}',
        shell=True, executable='/bin/bash')
"""


async def cycle() -> bool:
    if not await ensure_reachable():
        return False
    model = await whoami()
    user = model.get("name", "")
    if not user:
        log("could not determine user from token"); return False
    if not server_ready(model):
        if server_pending(model):
            log("server pending -> waiting"); await wait_ready(user)
        else:
            log("server NOT running -> (re)starting GPU profile"); await start_server(user)
        model = await whoami()
        if not server_ready(model):
            log("server still not ready; retry next cycle"); return False
    out = await exec_code(user, HEAL_CODE)
    out_s = out.strip().replace("\n", " | ")
    if "CAMPAIGN_DONE" in out:
        log(f"run COMPLETE. in-pod: {out_s}"); return True
    if "WATCHDOG_ALIVE" in out:
        log("ok: in-pod watchdog alive")
    else:
        log(f"(re)launched in-pod robust system: {out_s[:300]}")
    return False


async def main() -> None:
    log("================ PROGRAM 2 START ================")
    log(f"hub={CFG['hub_url']} profile={PROFILE_SLUG} run={RUN_NAME} poll={POLL_SECONDS}s")
    if not CFG.get("token"):
        log("FATAL: no token (mcp_config / JUPYTERHUB_TOKEN)"); sys.exit(2)
    backoff = POLL_SECONDS
    while True:
        try:
            if await cycle():
                log("================ PROGRAM 2 DONE ================"); return
            backoff = POLL_SECONDS
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"cycle error: {e!r} (retry in {min(backoff,60)}s)")
            backoff = min(backoff, 60)
        await asyncio.sleep(backoff)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("interrupted — exiting (in-pod training keeps running)")
