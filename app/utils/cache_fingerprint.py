"""Cache-aligned fingerprint of a Claude API request body.

Produces short cumulative SHA-256 prefixes at each cache_control breakpoint,
mirroring the order Anthropic hashes prompt-cache keys (tools → system →
messages). Two consecutive requests can be compared label-by-label; the first
differing label is the boundary that invalidated the cache.
"""

import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _snap(hasher: "hashlib._Hash") -> str:
    return hasher.copy().hexdigest()[:8]


def _is_billing_header(blk: Any) -> bool:
    """CC emits a volatile billing block (no cache_control, contains cch=)
    which Anthropic appears to exclude from cache-key computation. Skip it
    so cross-request fingerprints remain comparable.
    """
    if not isinstance(blk, dict) or blk.get("cache_control"):
        return False
    text = blk.get("text", "")
    return isinstance(text, str) and ("cch=" in text or "cc_version=" in text)


def fingerprint_body(body: "str | bytes | None") -> str | None:
    """Return a space-joined `label=hex8` fingerprint, or None if body is not JSON."""
    if body is None:
        return None
    try:
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        data = json.loads(body)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    h = hashlib.sha256()
    parts: list[str] = []

    model = data.get("model")
    if model is not None:
        h.update(_canonical({"model": model}))
        parts.append(f"model={_snap(h)}")

    tools = data.get("tools") or []
    bp = 0
    for tool in tools:
        h.update(_canonical(tool))
        if isinstance(tool, dict) and tool.get("cache_control"):
            parts.append(f"tools_bp{bp}={_snap(h)}")
            bp += 1
    if tools:
        parts.append(f"tools_end={_snap(h)}")

    system = data.get("system")
    sys_blocks: list[Any]
    if isinstance(system, str):
        sys_blocks = [{"type": "text", "text": system}]
    elif isinstance(system, list):
        sys_blocks = system
    else:
        sys_blocks = []
    bp = 0
    for blk in sys_blocks:
        if _is_billing_header(blk):
            continue
        h.update(_canonical(blk))
        if isinstance(blk, dict) and blk.get("cache_control"):
            parts.append(f"system_bp{bp}={_snap(h)}")
            bp += 1
    if sys_blocks:
        parts.append(f"system_end={_snap(h)}")

    messages = data.get("messages") or []
    bp = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        h.update(_canonical({"role": msg.get("role")}))
        content = msg.get("content")
        if isinstance(content, str):
            h.update(_canonical({"type": "text", "text": content}))
        elif isinstance(content, list):
            for blk in content:
                h.update(_canonical(blk))
                if isinstance(blk, dict) and blk.get("cache_control"):
                    parts.append(f"msg_bp{bp}={_snap(h)}")
                    bp += 1
    if messages:
        parts.append(f"msgs_end={_snap(h)}")

    return " ".join(parts) if parts else None
