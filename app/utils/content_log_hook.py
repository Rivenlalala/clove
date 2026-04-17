"""Route-level content logging hook for failed requests.

Invoked in a try/finally block in the route handler to ensure content logging
happens regardless of pipeline success or failure.
"""

from typing import Optional

from loguru import logger

from app.core.config import settings
from app.processors.claude_ai.context import ClaudeAIContext
from app.utils.cache_fingerprint import fingerprint_body
from app.utils.content_logger import (
    log_fingerprint,
    log_request_entry,
    log_error_entry,
    _extract_error_details,
)

_INBOUND = ">>> INBOUND REQUEST"
_OUTBOUND = ">>> OUTBOUND REQUEST"


async def content_log_hook(
    context: ClaudeAIContext,
    request_id: str,
    error: Optional[Exception] = None,
) -> None:
    """Route-level content logging hook, called in a finally block.

    Ensures the inbound request is always logged and error details are captured
    when the pipeline fails.

    Deduplication logic:
      - If context.metadata.get("content_inbound_logged") is True, ContentLogProcessor
        already ran and logged the inbound request. Only log an error entry
        if context.response is None or an error was caught.
      - If the flag is absent, ContentLogProcessor never completed inbound logging.
        Log the inbound request and any error entry.

    Error extraction:
      - Calls _extract_error_details(error) to extract structured fields from
        AppError subclasses (status_code, message_key, context).
      - Falls back to type().__name__ and str() for generic exceptions.

    Parameters:
        context: The ClaudeAIContext (may have response=None on failure)
        request_id: Pre-assigned request ID from the route handler
        error: The exception that caused the failure (if any)

    Invariants:
      - Never raises exceptions
      - No-op when content logging is disabled
    """
    if not settings.content_log_enabled:
        return

    try:
        already_logged = context.metadata.get("content_inbound_logged", False)

        if not already_logged:
            # ContentLogProcessor never completed inbound logging — log it now
            await _log_inbound_from_context(context, request_id)
            # Also log outbound request if it was stashed before the failure
            _log_outbound_from_context(context, request_id)

        # Log error entry if there's no response or an exception was caught
        if context.response is None or error is not None:
            if error is not None:
                details = _extract_error_details(error)
            else:
                details = {
                    "error_class": "NoResponseError",
                    "error_code": None,
                    "error_message": "Pipeline completed without producing a response",
                    "context": None,
                }

            log_error_entry(
                request_id=request_id,
                error_class=details["error_class"],
                error_code=details["error_code"],
                error_message=details["error_message"],
                context=details["context"],
            )

    except Exception as exc:
        logger.warning(f"ContentLogHook: failed to log content: {exc}")


async def _log_inbound_from_context(
    context: ClaudeAIContext,
    request_id: str,
) -> None:
    """Log the inbound request from context.original_request.

    Reads method, URL path, headers, and body from the original request.
    """
    req = context.original_request
    method = req.method
    path = req.url.path

    try:
        headers = dict(req.headers)
    except Exception:
        headers = {}

    try:
        body = await req.body()
    except Exception:
        body = None

    status_line = f"{method} {path}"
    log_request_entry(
        _INBOUND,
        request_id,
        status_line,
        headers,
        body,
        include_body=settings.content_log_include_body,
    )
    log_fingerprint(request_id, "INBOUND", fingerprint_body(body))


def _log_outbound_from_context(
    context: ClaudeAIContext,
    request_id: str,
) -> None:
    """Log the outbound request if it was stashed before the failure.

    Reads from context.metadata["outbound_request"] dict.
    """
    outbound = context.metadata.get("outbound_request")
    if outbound is None:
        return

    method = outbound.get("method", "POST")
    url = outbound.get("url", "")
    headers = outbound.get("headers", {})
    body = outbound.get("body")

    status_line = f"{method} {url}"
    log_request_entry(
        _OUTBOUND,
        request_id,
        status_line,
        headers,
        body,
        include_body=settings.content_log_include_body,
    )
    log_fingerprint(request_id, "OUTBOUND", fingerprint_body(body))
