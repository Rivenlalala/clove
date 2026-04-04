import json
import time
from typing import AsyncIterator, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from app.processors.base import BaseProcessor
from app.processors.claude_ai.context import ClaudeAIContext


class RequestLogProcessor(BaseProcessor):
    """Processor that emits a single INFO-level log line summarizing a completed request."""

    async def process(self, context: ClaudeAIContext) -> ClaudeAIContext:
        if context.response is None:
            return context

        if isinstance(context.response, JSONResponse):
            self._emit_log(context)
        elif isinstance(context.response, StreamingResponse):
            context.response.body_iterator = self._wrap_stream(
                context.response.body_iterator, context
            )

        return context

    async def _wrap_stream(
        self,
        body_iterator: AsyncIterator,
        context: "ClaudeAIContext",
    ) -> AsyncIterator:
        """
        Pass-through generator that yields chunks unchanged.
        For OAuth: parses SSE data lines for usage events.
        On completion or error: calls _emit_log().
        """
        is_oauth = context.metadata.get("oauth_path", False)
        oauth_usage = (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
            if is_oauth
            else None
        )
        try:
            async for chunk in body_iterator:
                if is_oauth:
                    chunk_bytes = (
                        chunk
                        if isinstance(chunk, bytes)
                        else chunk.encode("utf-8", errors="replace")
                    )
                    self._extract_usage_from_sse(chunk_bytes, oauth_usage)
                yield chunk
        finally:
            self._emit_log(context, oauth_usage=oauth_usage)

    def _extract_usage_from_sse(self, chunk: bytes, usage: dict) -> None:
        """
        Parse SSE data lines from raw bytes to extract token usage.
        Looks for 'message_start' (input_tokens) and 'message_delta' (output_tokens).
        Silently ignores parse failures.
        """
        try:
            text = chunk.decode("utf-8", errors="replace")
        except Exception:
            return
        for event_block in text.split("\n\n"):
            try:
                data_line = None
                for line in event_block.split("\n"):
                    if line.startswith("data: "):
                        data_line = line[len("data: ") :]
                        break
                if data_line is None:
                    continue
                data = json.loads(data_line)
                event_type = data.get("type")
                if event_type == "message_start":
                    msg_usage = data.get("message", {}).get("usage", {})
                    if "input_tokens" in msg_usage:
                        usage["input_tokens"] = msg_usage["input_tokens"]
                    if "cache_read_input_tokens" in msg_usage:
                        usage["cache_read_input_tokens"] = msg_usage[
                            "cache_read_input_tokens"
                        ]
                    if "cache_creation_input_tokens" in msg_usage:
                        usage["cache_creation_input_tokens"] = msg_usage[
                            "cache_creation_input_tokens"
                        ]
                elif event_type == "message_delta":
                    delta_usage = data.get("usage", {})
                    if "output_tokens" in delta_usage:
                        usage["output_tokens"] = delta_usage["output_tokens"]
            except Exception:
                continue

    def _emit_log(
        self,
        context: ClaudeAIContext,
        oauth_usage: Optional[dict] = None,
    ) -> None:
        model = "unknown"
        if context.messages_api_request:
            model = context.messages_api_request.model or "unknown"

        input_tokens = 0
        output_tokens = 0
        if oauth_usage is not None:
            input_tokens = oauth_usage.get("input_tokens", 0)
            output_tokens = oauth_usage.get("output_tokens", 0)
        elif context.collected_message and context.collected_message.usage:
            input_tokens = context.collected_message.usage.input_tokens
            output_tokens = context.collected_message.usage.output_tokens

        if oauth_usage is not None:
            cache_read = oauth_usage.get("cache_read_input_tokens", 0)
            cache_write = oauth_usage.get("cache_creation_input_tokens", 0)
        else:
            cache_read = str(context.metadata.get("cache_read", False)).lower()
            cache_write = str(context.metadata.get("cache_write", False)).lower()

        start = context.metadata.get("request_start_time")
        duration = time.monotonic() - start if start is not None else 0.0
        duration_str = f"{duration:.1f}s"

        account = context.metadata.get("account_id")
        if not account:
            try:
                account = context.claude_session.account.organization_uuid
            except AttributeError:
                account = None
        if not account:
            account = "unknown"

        stream = "false"
        if context.messages_api_request and context.messages_api_request.stream:
            stream = "true"

        logger.info(
            f"model={model} input_tokens={input_tokens} output_tokens={output_tokens} "
            f"cache_read={cache_read} cache_write={cache_write} "
            f"duration={duration_str} account={account} stream={stream}"
        )
