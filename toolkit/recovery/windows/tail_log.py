#!/usr/bin/env python
"""
Local live-tail for the training log on the JupyterHub server ("immer die
aktuellste Zeile ausgeben" — like `tail -f`, but over the hub's REST/websocket
API since there is no SSH to the pod).

Polls the remote log file's size via a kernel exec, fetches only the bytes
appended since the last poll, and prints them — so every line the training
process (or the watchdog) prints eventually shows up here, without re-reading
the whole file each time.

Run:
    python tail_log.py [path\\to\\recovery.json] [--file runs/train_live.log] [--interval 3]

Files worth tailing (relative to toolkit_dir unless they start with /):
    runs/train_live.log   - raw training stdout incl. the progress bar (default)
    runs/watchdog.log     - watchdog's own restart/crash decisions
Stop: Ctrl+C.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
import time
from pathlib import Path

from hub_client import HubClient, default_cfg_path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def now_prefix() -> str:
    """Current local date/time as a bracketed log prefix, e.g. '[2026-07-08 14:03:21] '."""
    return time.strftime("[%Y-%m-%d %H:%M:%S] ")


def log(msg: str = "", **kwargs):
    """print() but with a date/time stamp prepended."""
    print(f"{now_prefix()}{msg}", **kwargs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cfg", nargs="?", default=None)
    p.add_argument("--file", default="runs/train_live.log",
                   help="path relative to toolkit_dir (default: runs/train_live.log)")
    p.add_argument("--interval", type=float, default=60.0, help="poll interval, seconds")
    p.add_argument("--max-chunk", type=int, default=20000,
                   help="max bytes to pull per poll, to avoid tripping the hub's "
                        "IOPub data-rate limit when catching up on a large backlog")
    p.add_argument("--skip-threshold", type=int, default=2000000,
                   help="if the unread backlog exceeds this many bytes (typically "
                        "\\r-overwritten progress-bar spam), jump straight to the "
                        "last --max-chunk bytes instead of streaming through it")
    return p.parse_args()


async def main():
    args = parse_args()
    cfg_path = Path(args.cfg) if args.cfg else default_cfg_path()
    client = HubClient(cfg_path)
    remote_path = args.file if args.file.startswith("/") else f"{client.toolkit_dir}/{args.file}"

    log(f"tailing {remote_path} on {client.cfg['hub_url']} (Ctrl+C to stop)")
    offset = 0
    last_status = None
    catching_up = False
    at_line_start = True

    def write_stamped(text: str):
        """Write tailed log text to stdout, prefixing each new line with a timestamp."""
        nonlocal at_line_start
        while text:
            idx = text.find("\n")
            if idx == -1:
                if at_line_start:
                    sys.stdout.write(now_prefix())
                sys.stdout.write(text)
                at_line_start = False
                break
            chunk = text[:idx + 1]
            if at_line_start:
                sys.stdout.write(now_prefix())
            sys.stdout.write(chunk)
            at_line_start = True
            text = text[idx + 1:]
        sys.stdout.flush()

    while True:
        code = f"""
import os
p = {remote_path!r}
if not os.path.exists(p):
    print("STATUS:MISSING")
else:
    sz = os.path.getsize(p)
    off = {offset}
    if sz < off:
        off = 0  # file was reset/rotated underneath us
    skipped = 0
    if sz - off > {args.skip_threshold}:
        skipped = (sz - off) - {args.max_chunk}
        off = max(off, sz - {args.max_chunk})
    read_to = min(sz, off + {args.max_chunk})
    print(f"STATUS:OK:{{sz}}:{{off}}:{{read_to}}:{{skipped}}")
    if read_to > off:
        with open(p, "rb") as f:
            f.seek(off)
            data = f.read(read_to - off)
        import sys as _s
        _s.stdout.write(data.decode("utf-8", "replace"))
"""
        try:
            out = await client.exec_code(code, timeout=30)
        except Exception as e:
            log(f"[tail_log: connection error: {e!r}; retrying in {args.interval}s]",
                file=sys.stderr, flush=True)
            await asyncio.sleep(args.interval)
            continue

        nl = out.find("\n")
        status_line = out[:nl] if nl != -1 else out
        body = out[nl + 1:] if nl != -1 else ""

        if status_line == "STATUS:MISSING":
            if last_status != status_line:
                log(f"[tail_log: {remote_path} does not exist yet]", flush=True)
            last_status = status_line
            offset = 0
        elif status_line.startswith("STATUS:OK:"):
            _, _, sz, off, read_to, skipped = status_line.split(":")
            sz = int(sz)
            read_to = int(read_to)
            skipped = int(skipped)
            if int(off) == 0 and offset != 0 and skipped == 0:
                log(f"[tail_log: {remote_path} shrank/rotated — resuming from start]",
                    flush=True)
            if skipped:
                log(f"[tail_log: backlog too large — skipped {skipped} bytes "
                    f"(likely progress-bar spam), jumping to the tail]",
                    file=sys.stderr, flush=True)
            if body:
                write_stamped(body)
            catching_up = read_to < sz
            if catching_up:
                log(f"[tail_log: catching up — {sz - read_to} bytes still behind]",
                    file=sys.stderr, flush=True)
            offset = read_to
            last_status = status_line
        else:
            catching_up = False
            log(f"[tail_log: unexpected response: {out[:200]!r}]", file=sys.stderr, flush=True)

        if not catching_up:
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        log("[tail_log: stopped]")
