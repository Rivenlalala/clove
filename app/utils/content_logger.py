"""Dedicated loguru sink for writing detailed HTTP content log entries to a separate file."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.core.exceptions import AppError

# Module-level bound logger instance.
# Set to a bound logger after configure_content_logger() when enabled.
# Remains None when content logging is disabled.
content_log = None

# Separator constants
_OUTER_SEP = "=" * 80
_INNER_SEP = "-" * 80


def _content_log_filter(record: dict) -> bool:
    """Loguru sink filter: accept only records bound with content_log=True."""
    return record["extra"].get("content_log") is True


def configure_content_logger() -> None:
    """Initialize the content log sink. No-op when content_log_enabled is False."""
    global content_log

    if not settings.content_log_enabled:
        return

    log_path = Path(settings.content_log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        filter=_content_log_filter,
        format="{message}",
        rotation=settings.content_log_file_rotation,
        retention=settings.content_log_file_retention,
        compression=settings.content_log_file_compression,
        enqueue=True,
        encoding="utf-8",
    )

    content_log = logger.bind(content_log=True)


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------


def _format_body(body: "str | bytes | None") -> str:
    """Decode bytes, pretty-print valid JSON, fall back to raw string."""
    if body is None:
        return ""

    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")

    try:
        parsed = json.loads(body)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, ValueError):
        return body


def _format_headers(headers: dict) -> str:
    """Format a headers dict as 'key: value' lines."""
    if not headers:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in headers.items())


def _timestamp() -> str:
    return datetime.now().isoformat(sep=" ", timespec="milliseconds")


# ---------------------------------------------------------------------------
# Public writing functions
# ---------------------------------------------------------------------------


def log_summary(request_id: "str | None", message: str) -> None:
    """Emit a single [SUMMARY] line (e.g., per-request token/cache/duration stats)."""
    if content_log is None:
        return
    try:
        prefix = f"[{request_id}] " if request_id else ""
        content_log.info(f"{prefix}[SUMMARY] {message}")
    except Exception as exc:
        logger.warning(f"content_logger: failed to write summary: {exc}")


def log_fingerprint(request_id: str, direction: str, fingerprint: "str | None") -> None:
    """Emit a single [FINGERPRINT] line tagged by request_id and direction.

    direction is a short label like "INBOUND" or "OUTBOUND".
    """
    if content_log is None or not fingerprint:
        return
    try:
        content_log.info(f"[{request_id}] {direction} [FINGERPRINT] {fingerprint}")
    except Exception as exc:
        logger.warning(f"content_logger: failed to write fingerprint: {exc}")


def log_request_entry(
    direction: str,
    request_id: str,
    status_line: str,
    headers: dict,
    body: "str | bytes | None",
    include_body: bool = True,
) -> None:
    """Format and write a request log entry (inbound or outbound request).

    No-op if content logging is disabled (content_log is None).

    Parameters:
        direction: One of ">>> INBOUND REQUEST", ">>> OUTBOUND REQUEST"
        request_id: Correlation identifier for the request
        status_line: First line (e.g., "POST /v1/messages HTTP/1.1")
        headers: HTTP headers as key-value pairs
        body: Raw body content (str or bytes); pretty-printed if valid JSON
        include_body: When False, only headers are logged (body omitted)
    """
    if content_log is None:
        return

    try:
        ts = _timestamp()
        formatted_headers = _format_headers(headers)
        formatted_body = _format_body(body) if include_body else ""

        parts = [
            _OUTER_SEP,
            f"[{ts}] [{request_id}] {direction}",
            _INNER_SEP,
            status_line,
        ]

        if formatted_headers:
            parts.append(formatted_headers)

        if include_body:
            parts.append("")  # blank line before body

            if formatted_body:
                parts.append(formatted_body)

        parts.append(_OUTER_SEP)

        entry = "\n".join(parts)
        content_log.info(entry)

    except Exception as exc:
        logger.warning(f"content_logger: failed to write request entry: {exc}")


def log_response_entry(
    request_id: str,
    status_line: str,
    outbound_headers: dict,
    inbound_headers: dict,
    body: str | None,
    include_body: bool = True,
) -> None:
    """Format and write the response log entry with both sets of headers.

    No-op if content logging is disabled (content_log is None).

    Parameters:
        request_id: Correlation identifier for the request
        status_line: HTTP status line (e.g., "200 OK")
        outbound_headers: Response headers from Anthropic
        inbound_headers: Response headers sent to client
        body: Response body text (extracted text for streaming, full body for non-streaming)
        include_body: When False, only headers are logged (body omitted)
    """
    if content_log is None:
        return

    try:
        ts = _timestamp()

        parts = [
            _OUTER_SEP,
            f"[{ts}] [{request_id}] <<< RESPONSE",
            _INNER_SEP,
            status_line,
            "",
            "--- Outbound Response Headers (Anthropic \u2192 Clove) ---",
        ]

        if outbound_headers:
            parts.append(_format_headers(outbound_headers))

        parts.append("")
        parts.append("--- Inbound Response Headers (Clove \u2192 Client) ---")

        if inbound_headers:
            parts.append(_format_headers(inbound_headers))

        if include_body:
            parts.append("")
            parts.append("--- Body ---")

            if body:
                parts.append(body)

        parts.append("")
        parts.append(_OUTER_SEP)

        entry = "\n".join(parts)
        content_log.info(entry)

    except Exception as exc:
        logger.warning(f"content_logger: failed to write response entry: {exc}")


def _extract_error_details(exc: Exception) -> dict:
    """Extract structured error details from an exception.

    For AppError subclasses: extracts status_code, message_key, and context.copy().
    For generic exceptions: falls back to type(exc).__name__, str(exc), and None.

    Returns:
        dict with keys: error_class (str), error_code (int|None),
                        error_message (str), context (dict|None)
    """
    error_class = type(exc).__name__

    if isinstance(exc, AppError):
        return {
            "error_class": error_class,
            "error_code": exc.status_code,
            "error_message": exc.message_key,
            "context": exc.context.copy() if exc.context else None,
        }

    return {
        "error_class": error_class,
        "error_code": None,
        "error_message": str(exc),
        "context": None,
    }


def log_error_entry(
    request_id: str,
    error_class: str,
    error_code: Optional[int],
    error_message: str,
    context: Optional[dict] = None,
) -> None:
    """Format and write an error log entry for failed requests.

    No-op if content logging is disabled (content_log is None).

    Parameters:
        request_id: Correlation identifier for the request
        error_class: Exception class name (e.g., "NoAccountsAvailableError")
        error_code: HTTP status code or error code (e.g., 503, 429)
        error_message: Human-readable error description
        context: Optional additional context (e.g., account_id, model)
    """
    if content_log is None:
        return

    try:
        ts = _timestamp()

        parts = [
            _OUTER_SEP,
            f"[{ts}] [{request_id}] !!! ERROR",
            _INNER_SEP,
            f"Exception: {error_class}",
        ]

        if error_code is not None:
            parts.append(f"Status Code: {error_code}")

        parts.append(f"Message: {error_message}")
        parts.append("")
        parts.append("Context:")

        if context:
            for key, value in context.items():
                parts.append(f"  {key}: {value}")
        else:
            parts.append("  (none)")

        parts.append(_OUTER_SEP)

        entry = "\n".join(parts)
        content_log.info(entry)

    except Exception as exc:
        logger.warning(f"content_logger: failed to write error entry: {exc}")
