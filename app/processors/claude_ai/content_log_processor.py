"""Processor that writes detailed HTTP content log entries for each proxied request."""

import json
from typing import AsyncIterator
from uuid import uuid4

from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from app.processors.base import BaseProcessor
from app.processors.claude_ai.context import ClaudeAIContext
from app.core.config import settings
from app.utils.content_logger import log_request_entry, log_response_entry

_INBOUND = ">>> INBOUND REQUEST"
_OUTBOUND = ">>> OUTBOUND REQUEST"


class ContentLogProcessor(BaseProcessor):
    """Processor that writes detailed HTTP content log entries for each proxied request.

    Produces three entries per request: inbound request, outbound request, response.
    Response entry includes both outbound (Anthropic) and inbound (client) response headers.
    For streaming responses, text is extracted from SSE text_delta events into a per-request
    buffer, and the accumulated text is logged when the stream completes.

    Positioned after RequestLogProcessor in the pipeline.
    No-op when settings.content_log_enabled is False.
    """

    async def process(self, context: ClaudeAIContext) -> ClaudeAIContext:
        """Log inbound request, outbound request, and response for each proxied call.

        Invariants:
            - Never modifies request or response data
            - Never raises exceptions to the pipeline
        """
        if not settings.content_log_enabled:
            return context

        request_id = uuid4().hex[:8]
        context.metadata["content_request_id"] = request_id

        try:
            await self._log_inbound_request(context, request_id)
        except Exception as exc:
            logger.warning(f"ContentLogProcessor: failed to log inbound request: {exc}")

        try:
            self._log_outbound_request(context, request_id)
        except Exception as exc:
            logger.warning(f"ContentLogProcessor: failed to log outbound request: {exc}")

        # Log response
        if context.response is None:
            return context

        if isinstance(context.response, JSONResponse):
            try:
                self._log_response(context, request_id)
            except Exception as exc:
                logger.warning(f"ContentLogProcessor: failed to log JSON response: {exc}")

        elif isinstance(context.response, StreamingResponse):
            context.response.body_iterator = self._wrap_stream(
                context.response.body_iterator, context, request_id
            )

        return context

    async def _log_inbound_request(
        self, context: ClaudeAIContext, request_id: str
    ) -> None:
        """Log the original client request."""
        req = context.original_request
        method = req.method
        path = req.url.path

        # No redaction — authorization and cookie headers are logged as-is
        try:
            headers = dict(req.headers)
        except Exception:
            headers = {}

        try:
            body = await req.body()
        except Exception:
            body = None

        status_line = f"{method} {path}"
        log_request_entry(_INBOUND, request_id, status_line, headers, body)

    def _log_outbound_request(
        self, context: ClaudeAIContext, request_id: str
    ) -> None:
        """Log the outbound request to Anthropic. Skips gracefully if metadata is absent."""
        outbound = context.metadata.get("outbound_request")
        if outbound is None:
            return

        method = outbound.get("method", "POST")
        url = outbound.get("url", "")
        headers = outbound.get("headers", {})
        body = outbound.get("body")

        status_line = f"{method} {url}"
        log_request_entry(_OUTBOUND, request_id, status_line, headers, body)

    def _log_response(
        self, context: ClaudeAIContext, request_id: str, body: str | None = None
    ) -> None:
        """Log the response entry with both outbound and inbound response headers."""
        response = context.response

        try:
            inbound_headers = dict(response.headers)
        except Exception:
            inbound_headers = {}

        # Outbound (Anthropic → Clove) headers stashed in metadata; OAuth path uses same headers
        outbound_headers = context.metadata.get("outbound_response_headers", inbound_headers)

        try:
            status_line = str(response.status_code)
        except Exception:
            status_line = "unknown"

        if body is None and isinstance(response, JSONResponse):
            try:
                body_bytes = response.body
                body = body_bytes.decode("utf-8", errors="replace") if isinstance(body_bytes, bytes) else str(body_bytes)
            except Exception:
                body = None

        log_response_entry(request_id, status_line, outbound_headers, inbound_headers, body)

    async def _wrap_stream(
        self,
        body_iterator: AsyncIterator,
        context: ClaudeAIContext,
        request_id: str,
    ) -> AsyncIterator:
        """Async generator wrapping the response body iterator.

        Yields every chunk byte-identical to the original.
        Accumulates extracted text per request_id (preventing interleaving).
        Logs the response when the stream completes or errors.
        """
        text_parts: list[str] = []
        try:
            async for chunk in body_iterator:
                delta = self._extract_text_delta(chunk)
                if delta:
                    text_parts.append(delta)
                yield chunk
        except Exception:
            try:
                self._log_response(context, request_id, body="".join(text_parts))
            except Exception as exc:
                logger.warning(f"ContentLogProcessor: failed to log streaming response on error: {exc}")
            raise
        else:
            try:
                self._log_response(context, request_id, body="".join(text_parts))
            except Exception as exc:
                logger.warning(f"ContentLogProcessor: failed to log streaming response: {exc}")

    def _extract_text_delta(self, chunk: str | bytes) -> str:
        """Parse an SSE chunk and return concatenated text from text_delta events."""
        if not chunk:
            return ""

        if isinstance(chunk, bytes):
            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                return ""
        else:
            text = chunk

        # Cheap pre-check: skip JSON parsing entirely for non-text-delta chunks
        if "text_delta" not in text:
            return ""

        extracted: list[str] = []
        for event_block in text.split("\n\n"):
            data_line = None
            for line in event_block.split("\n"):
                if line.startswith("data: "):
                    data_line = line[len("data: "):]
                    break
            if data_line is None:
                continue
            try:
                data = json.loads(data_line)
            except (json.JSONDecodeError, ValueError):
                continue

            if data.get("type") == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    extracted.append(delta.get("text", ""))

        return "".join(extracted)
