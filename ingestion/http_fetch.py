"""HTTP fetch helper.

Shells out to curl rather than using requests/urllib: empirically, in this
environment's network path, Python's http stack gets 403'd or hangs against
Cloudflare-fronted hosts (TLS fingerprinting) while curl consistently works.
"""
from __future__ import annotations

import subprocess
import time

import config


def get(url: str, timeout: int = 20, retries: int = 3, backoff: float = 2.0) -> tuple[int, bytes]:
    """Returns (status_code, body_bytes). status_code is 0 on total failure (timeout/DNS/etc)."""
    last_status = 0
    for attempt in range(retries):
        proc = subprocess.run(
            [
                "curl", "-sL", "--max-time", str(timeout),
                "-A", config.GUTENBERG_USER_AGENT,
                "-w", "\n__STATUS__%{http_code}",
                url,
            ],
            capture_output=True,
        )
        out = proc.stdout
        marker = b"\n__STATUS__"
        idx = out.rfind(marker)
        if idx == -1:
            last_status = 0
            body = b""
        else:
            body = out[:idx]
            try:
                last_status = int(out[idx + len(marker):].decode().strip())
            except ValueError:
                last_status = 0
        if last_status == 200:
            return last_status, body
        time.sleep(backoff * (attempt + 1))
    return last_status, b""
