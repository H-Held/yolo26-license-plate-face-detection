#!/usr/bin/env python
"""
Local companion to tail_log.py: download the full remote log to disk, or
reset it (archive the current content server-side, then start it fresh) —
useful before/after a long run so tail_log.py starts from a clean file.

Run:
    python fetch_log.py download [path\\to\\recovery.json] [--file runs/train_live.log] [--out local_path]
    python fetch_log.py reset    [path\\to\\recovery.json] [--file runs/train_live.log]

"reset" does NOT delete history: the server-side file is renamed to
<name>.<UTC timestamp>.bak in the same directory, then an empty file with the
original name is created so training/the watchdog keep appending to it and
tail_log.py has a clean stream to follow.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

from hub_client import HubClient, default_cfg_path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["download", "reset"])
    p.add_argument("cfg", nargs="?", default=None)
    p.add_argument("--file", default="runs/train_live.log",
                   help="path relative to toolkit_dir (default: runs/train_live.log)")
    p.add_argument("--out", default=None,
                   help="local destination for 'download' (default: ./<basename>)")
    return p.parse_args()


async def download(client: HubClient, remote_path: str, out_path: Path):
    code = f"""
import base64, os
p = {remote_path!r}
if not os.path.exists(p):
    print("MISSING")
else:
    with open(p, "rb") as f:
        data = f.read()
    print(f"SIZE:{{len(data)}}")
    import sys
    sys.stdout.write(base64.b64encode(data).decode("ascii"))
"""
    print(f"downloading {remote_path} ...")
    out = await client.exec_code(code, timeout=300)
    if out.startswith("MISSING"):
        print(f"remote file does not exist: {remote_path}")
        sys.exit(1)
    nl = out.find("\n")
    size_line, b64 = out[:nl], out[nl + 1:].strip()
    import base64
    data = base64.b64decode(b64)
    out_path.write_bytes(data)
    print(f"wrote {out_path} ({len(data)} bytes, remote reported {size_line})")


async def reset(client: HubClient, remote_path: str):
    code = f"""
import os, time
p = {remote_path!r}
if os.path.exists(p):
    bak = p + "." + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".bak"
    os.rename(p, bak)
    open(p, "w").close()
    print(f"OK:{{bak}}")
else:
    open(p, "w").close()
    print("OK:(no previous file)")
"""
    print(f"resetting {remote_path} ...")
    out = (await client.exec_code(code, timeout=30)).strip()
    if out.startswith("OK:"):
        print(f"done — previous content archived as: {out[3:]}")
    else:
        print(f"unexpected response: {out}")
        sys.exit(1)


async def main():
    args = parse_args()
    cfg_path = Path(args.cfg) if args.cfg else default_cfg_path()
    client = HubClient(cfg_path)
    remote_path = args.file if args.file.startswith("/") else f"{client.toolkit_dir}/{args.file}"

    if args.action == "download":
        out_path = Path(args.out) if args.out else Path(remote_path).name
        await download(client, remote_path, out_path)
    else:
        await reset(client, remote_path)


if __name__ == "__main__":
    asyncio.run(main())
