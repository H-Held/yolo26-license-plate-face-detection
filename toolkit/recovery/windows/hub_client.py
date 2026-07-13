#!/usr/bin/env python
"""
Shared JupyterHub connection helper for the local (Windows) log tools
(tail_log.py, fetch_log.py). Factored out of program2_resume_watchdog.py so
both scripts share one connection/auth path instead of duplicating it.

Config precedence is the same as program2_resume_watchdog.py: recovery.json
next to this file (or a path given on the command line) supplies hub_url +
remote paths; the token comes from (lowest to highest precedence) mcp_config
file < recovery.json "token" < JUPYTERHUB_TOKEN env var.
"""
from __future__ import annotations
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent


def load_recovery(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        print(f"FATAL: {cfg_path} not found (copy recovery.json.example).")
        sys.exit(2)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class HubClient:
    def __init__(self, cfg_path: Path | None = None):
        self.cfg_path = cfg_path or (HERE / "recovery.json")
        self.rc = load_recovery(self.cfg_path)
        self.remote_root = self.rc["remote_root"]
        self.toolkit_dir = self.rc.get("toolkit_dir", f"{self.remote_root}/toolkit")
        self.data_root = self.rc.get("data_root", f"{self.remote_root}/deepseek")
        self.run_name = self.rc.get("run_name", "")
        self.cfg = self._load_cfg()
        if not self.cfg.get("token"):
            print("FATAL: no token (set it in recovery.json, mcp_config, or "
                  "JUPYTERHUB_TOKEN env var).")
            sys.exit(2)
        self._kid = None
        self._user = None

    def _load_cfg(self) -> dict:
        cfg = {"hub_url": self.rc.get("hub_url", ""), "token": "", "verify_tls": True,
               "http_timeout": 30}
        mcp_config = (HERE / self.rc["mcp_config"]).resolve() if self.rc.get("mcp_config") else None
        if mcp_config and mcp_config.exists():
            cfg.update(json.loads(mcp_config.read_text(encoding="utf-8")))
        if self.rc.get("token"):
            cfg["token"] = self.rc["token"]
        if os.environ.get("JUPYTERHUB_TOKEN"):
            cfg["token"] = os.environ["JUPYTERHUB_TOKEN"]
        if self.rc.get("hub_url"):
            cfg["hub_url"] = self.rc["hub_url"]
        cfg["hub_url"] = cfg["hub_url"].rstrip("/")
        return cfg

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers={"Authorization": f"token {self.cfg['token']}"},
                                 verify=self.cfg.get("verify_tls", True),
                                 timeout=self.cfg.get("http_timeout", 30),
                                 follow_redirects=True)

    def _hub_api(self, path: str) -> str:
        return f"{self.cfg['hub_url']}/hub/api{path}"

    def _user_api(self, user: str, path: str) -> str:
        return f"{self.cfg['hub_url']}/user/{user}{path}"

    async def whoami(self) -> str:
        if self._user:
            return self._user
        async with self._client() as c:
            r = await c.get(self._hub_api("/user"))
            r.raise_for_status()
            self._user = r.json()["name"]
            return self._user

    async def server_ready(self) -> bool:
        async with self._client() as c:
            r = await c.get(self._hub_api("/user"))
            r.raise_for_status()
            model = r.json()
            servers = model.get("servers") or {}
            d = servers.get("") or {}
            return bool(d.get("ready"))

    async def _kernel_alive(self, user: str, kid: str) -> bool:
        try:
            async with self._client() as c:
                r = await c.get(self._user_api(user, f"/api/kernels/{kid}"))
                return r.status_code == 200
        except Exception:
            return False

    async def ensure_kernel(self) -> str:
        user = await self.whoami()
        if self._kid and await self._kernel_alive(user, self._kid):
            return self._kid
        async with self._client() as c:
            r = await c.post(self._user_api(user, "/api/kernels"), json={"name": "python3"})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"could not start kernel: HTTP {r.status_code} {r.text[:200]}")
        self._kid = r.json()["id"]
        return self._kid

    async def exec_code(self, code: str, timeout: int = 60) -> str:
        import asyncio
        user = await self.whoami()
        kid = await self.ensure_kernel()
        ws_base = self.cfg["hub_url"].replace("https://", "wss://").replace("http://", "ws://")
        url = f"{ws_base}/user/{user}/api/kernels/{kid}/channels?token={self.cfg['token']}"
        session = uuid.uuid4().hex
        req = {"header": {"msg_id": uuid.uuid4().hex, "username": "tool", "session": session,
                          "date": now_iso(), "msg_type": "execute_request", "version": "5.3"},
               "parent_header": {}, "metadata": {},
               "content": {"code": code, "silent": False, "store_history": False,
                          "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
               "channel": "shell", "buffers": []}
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


def default_cfg_path() -> Path:
    """Fallback recovery.json location when the caller's --cfg arg is None."""
    return HERE / "recovery.json"
