"""Shape outgoing OAuth requests to look like a Claude Code CLI request.

Called from ``ClaudeAPIProcessor`` once an OAuth account has been acquired and
before the request body is serialised. Not a pipeline processor — it has one
job (rewrite ``system[]``) and only makes sense on the OAuth path, so it lives
here as a plain util.

Why this exists
---------------
On 2026-04-04 Anthropic deployed server-side validation on OAuth requests:
unless ``system[]`` is shaped like a Claude Code CLI request, the request is
rejected with HTTP 400 even when the account has remaining quota.

A Claude Code CLI request has this shape::

    system[0] = {"type": "text", "text": "x-anthropic-billing-header: ..."}
    system[1] = {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}
    system[2..] = anything the client put there

The billing header is a per-request signature: SHA-256 derivations over the
first user message's first text block and a salt extracted from the CLI
binary. It changes per request, so the billing block must carry no
``cache_control`` and sits at ``system[0]`` ahead of any cacheable blocks.

Passthrough rule
----------------
``inject_claude_code_prefix`` is a no-op when ``system[0]`` already looks like
a billing header — i.e. when the client is Claude Code itself proxying through
clove, in which case Claude Code has already signed its own request with
*its* view of the first user message. Recomputing would just duplicate work
(and for a proxy that rewrites messages could actually invalidate the
signature). Otherwise both the billing and identity blocks are prepended.

Ported from hermes-claude-auth's ``anthropic_billing_bypass`` (MIT), which in
turn ports the logic from opencode-claude-auth.
"""

from __future__ import annotations

import hashlib
from typing import List, Union

from loguru import logger

from app.models.claude import InputMessage, MessagesAPIRequest, TextContent


_BILLING_SALT = "59cf53e54c78"
_BILLING_PREFIX = "x-anthropic-billing-header"
_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
_CLAUDE_CODE_VERSION = "2.1.90"
_ENTRYPOINT = "cli"


def _extract_first_user_message_text(messages: List[InputMessage]) -> str:
    """Text of the first user message's first text block.

    Mirrors Claude Code's K19(): scan for the first ``role="user"`` message,
    then return the text of its first text block (empty string if none).
    """
    for msg in messages:
        if msg.role.value != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, TextContent):
                    return block.text or ""
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        return text
        return ""
    return ""


def _compute_cch(message_text: str) -> str:
    return hashlib.sha256(message_text.encode("utf-8")).hexdigest()[:5]


def _compute_version_suffix(message_text: str, version: str) -> str:
    """3-hex suffix: SHA-256(salt + chars[4,7,20] + version)[:3].

    Short messages pad missing indices with the literal character ``"0"``.
    """
    sampled = "".join(
        message_text[i] if i < len(message_text) else "0" for i in (4, 7, 20)
    )
    return hashlib.sha256(
        f"{_BILLING_SALT}{sampled}{version}".encode("utf-8")
    ).hexdigest()[:3]


def build_billing_header_value(
    messages: List[InputMessage],
    version: str = _CLAUDE_CODE_VERSION,
    entrypoint: str = _ENTRYPOINT,
) -> str:
    text = _extract_first_user_message_text(messages)
    return (
        f"{_BILLING_PREFIX}: "
        f"cc_version={version}.{_compute_version_suffix(text, version)}; "
        f"cc_entrypoint={entrypoint}; "
        f"cch={_compute_cch(text)};"
    )


def _is_billing_header_block(block: object) -> bool:
    return isinstance(block, TextContent) and block.text.startswith(_BILLING_PREFIX)


def inject_claude_code_prefix(request: MessagesAPIRequest) -> None:
    """Ensure ``system[]`` starts with a Claude Code billing header + identity.

    If ``system[0]`` is already a billing header, the request was built by
    Claude Code itself (proxying through clove) and carries its own signed
    header — leave it untouched. Otherwise prepend ``[billing, identity]`` so
    the request passes Anthropic's OAuth content validation.
    """
    if request is None or not request.messages:
        return

    raw: Union[str, List[TextContent], None] = request.system

    # Already Claude-Code-shaped: pass through.
    if isinstance(raw, list) and raw and _is_billing_header_block(raw[0]):
        return

    # Normalise system to List[TextContent] so we can prepend.
    if raw is None or raw == "":
        kept: List[TextContent] = []
    elif isinstance(raw, str):
        kept = [TextContent(type="text", text=raw)]
    elif isinstance(raw, list):
        kept = list(raw)
    else:
        logger.warning(
            f"claude_code_prefix: unexpected system type {type(raw).__name__}; skipping"
        )
        return

    try:
        billing_text = build_billing_header_value(request.messages)
    except Exception as exc:
        logger.warning(f"claude_code_prefix: failed to build billing header: {exc}")
        return

    billing_block = TextContent(type="text", text=billing_text)
    identity_block = TextContent(type="text", text=_SYSTEM_IDENTITY)

    request.system = [billing_block, identity_block] + kept
