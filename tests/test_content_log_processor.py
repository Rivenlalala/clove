"""Tests for ContentLogProcessor.

Task 4: Implement the content log processor.
Subtasks: 4.1 (inbound/outbound request logging), 4.2 (JSON response logging),
          4.3 (streaming response logging via body iterator wrapping)
Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 6.1, 7.1
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import JSONResponse, StreamingResponse

from app.processors.claude_ai.context import ClaudeAIContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    body: bytes = b'{"model": "claude-3", "messages": []}',
    method: str = "POST",
    path: str = "/v1/messages",
):
    """Create a mock FastAPI Request."""
    req = MagicMock()
    req.method = method
    req.url.path = path
    req.headers = {
        "content-type": "application/json",
        "authorization": "Bearer sk-ant-test",
    }
    req.body = AsyncMock(return_value=body)
    return req


def _make_context(
    body: bytes = b'{"model": "claude-3"}',
    method: str = "POST",
    path: str = "/v1/messages",
):
    """Create a ClaudeAIContext with a mock request."""
    return ClaudeAIContext(original_request=_make_request(body, method, path))


def _make_json_response(status_code: int = 200, body: dict = None):
    """Create a JSONResponse."""
    if body is None:
        body = {
            "id": "msg_abc",
            "type": "message",
            "content": [{"type": "text", "text": "Hello!"}],
        }
    return JSONResponse(content=body, status_code=status_code)


async def _iter_chunks(chunks):
    """Async generator yielding chunks."""
    for chunk in chunks:
        yield chunk


def _make_streaming_response(chunks=None, status_code: int = 200):
    """Create a StreamingResponse with SSE chunks."""
    if chunks is None:
        chunks = [
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}\n\n',
            b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]
    resp = StreamingResponse(
        _iter_chunks(chunks),
        status_code=status_code,
        media_type="text/event-stream",
        headers={"content-type": "text/event-stream", "x-request-id": "req_abc123"},
    )
    return resp


def _make_settings(enabled: bool = True):
    """Create mock settings."""
    s = MagicMock()
    s.content_log_enabled = enabled
    return s


# ---------------------------------------------------------------------------
# Task 4.1: Processor creation and inbound/outbound request logging
# ---------------------------------------------------------------------------


class TestContentLogProcessorImport(unittest.TestCase):
    """Verify the processor class is importable."""

    def test_import(self):
        """ContentLogProcessor must be importable from the processors package."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        self.assertTrue(callable(ContentLogProcessor))

    def test_is_base_processor(self):
        """ContentLogProcessor must extend BaseProcessor."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor
        from app.processors.base import BaseProcessor

        self.assertTrue(issubclass(ContentLogProcessor, BaseProcessor))

    def test_has_process_method(self):
        """ContentLogProcessor must have a process async method."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor
        import inspect

        processor = ContentLogProcessor()
        self.assertTrue(inspect.iscoroutinefunction(processor.process))


class TestContentLogProcessorNoopWhenDisabled(unittest.TestCase):
    """Verify the processor is a no-op when content logging is disabled."""

    def test_noop_when_disabled_no_exception(self):
        """process() must not raise when content logging is disabled."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = False
            result = asyncio.run(processor.process(context))

        self.assertIs(result, context)

    def test_noop_when_disabled_no_log_calls(self):
        """process() must not call any content_logger functions when disabled."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = False
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_req:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_resp:
                    asyncio.run(processor.process(context))

        mock_req.assert_not_called()
        mock_resp.assert_not_called()

    def test_noop_preserves_response_when_disabled(self):
        """process() must leave response unchanged when disabled."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        original_response = _make_json_response()
        context.response = original_response
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = False
            asyncio.run(processor.process(context))

        self.assertIs(context.response, original_response)


class TestContentLogProcessorRequestId(unittest.TestCase):
    """Verify request_id generation and storage in metadata."""

    def test_request_id_stored_in_metadata(self):
        """process() must store a request_id in context.metadata['content_request_id']."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        self.assertIn("content_request_id", context.metadata)

    def test_request_id_is_8_char_hex(self):
        """The request_id must be an 8-character hex string."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        request_id = context.metadata["content_request_id"]
        self.assertEqual(len(request_id), 8)
        # Must be valid hex
        int(request_id, 16)

    def test_request_ids_are_unique(self):
        """Different requests must get different request IDs."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        ids = set()
        for _ in range(5):
            context = _make_context()
            context.response = _make_json_response()
            processor = ContentLogProcessor()

            with patch(
                "app.processors.claude_ai.content_log_processor.settings"
            ) as mock_settings:
                mock_settings.content_log_enabled = True
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_request_entry"
                ):
                    with patch(
                        "app.processors.claude_ai.content_log_processor.log_response_entry"
                    ):
                        asyncio.run(processor.process(context))

            ids.add(context.metadata["content_request_id"])

        self.assertEqual(len(ids), 5)


class TestInboundRequestLogging(unittest.TestCase):
    """Verify inbound request logging calls."""

    def test_inbound_request_logged_with_correct_direction(self):
        """Inbound request must be logged with '>>> INBOUND REQUEST' direction."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context(body=b'{"model": "claude-3"}')
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        # Find the inbound call
        inbound_calls = [
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        ]
        self.assertEqual(len(inbound_calls), 1)

    def test_inbound_request_includes_method_and_path(self):
        """Inbound request status_line must include HTTP method and URL path."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context(method="POST", path="/v1/messages")
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        inbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        )
        status_line = inbound_call[0][2]  # positional arg 2
        self.assertIn("POST", status_line)
        self.assertIn("/v1/messages", status_line)

    def test_inbound_request_includes_headers(self):
        """Inbound request must pass all headers to log_request_entry."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        inbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        )
        headers = inbound_call[0][3]  # positional arg 3
        self.assertIsInstance(headers, dict)
        # Must include authorization (no redaction)
        self.assertIn("authorization", headers)

    def test_inbound_request_includes_body(self):
        """Inbound request must pass the request body to log_request_entry."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        body = b'{"model": "claude-3", "messages": []}'
        context = _make_context(body=body)
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        inbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        )
        logged_body = inbound_call[0][4]  # positional arg 4
        self.assertIsNotNone(logged_body)

    def test_inbound_request_includes_request_id(self):
        """Inbound request must pass the generated request_id to log_request_entry."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        request_id = context.metadata["content_request_id"]
        inbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        )
        logged_request_id = inbound_call[0][1]  # positional arg 1
        self.assertEqual(logged_request_id, request_id)

    def test_no_header_redaction(self):
        """Authorization and cookie headers must be logged without redaction."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        req = _make_request()
        req.headers = {
            "authorization": "Bearer sk-ant-secret",
            "cookie": "session=abc123",
            "content-type": "application/json",
        }
        context = ClaudeAIContext(original_request=req)
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        inbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        )
        headers = inbound_call[0][3]
        self.assertIn("authorization", headers)
        self.assertIn("cookie", headers)
        self.assertEqual(headers["authorization"], "Bearer sk-ant-secret")
        self.assertEqual(headers["cookie"], "session=abc123")


class TestOutboundRequestLogging(unittest.TestCase):
    """Verify outbound request logging."""

    def test_outbound_request_logged_when_metadata_present(self):
        """Outbound request must be logged when context.metadata['outbound_request'] exists."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["outbound_request"] = {
            "method": "POST",
            "url": "https://api.anthropic.com/v1/messages",
            "headers": {
                "authorization": "Bearer tok",
                "content-type": "application/json",
            },
            "body": '{"model": "claude-3"}',
        }
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        outbound_calls = [
            c for c in mock_log.call_args_list if c[0][0] == ">>> OUTBOUND REQUEST"
        ]
        self.assertEqual(len(outbound_calls), 1)

    def test_outbound_request_status_line_includes_method_and_url(self):
        """Outbound request status_line must include method and URL."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["outbound_request"] = {
            "method": "POST",
            "url": "https://api.anthropic.com/v1/messages",
            "headers": {},
            "body": "{}",
        }
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        outbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> OUTBOUND REQUEST"
        )
        status_line = outbound_call[0][2]
        self.assertIn("POST", status_line)
        self.assertIn("api.anthropic.com", status_line)

    def test_outbound_request_skipped_gracefully_when_metadata_absent(self):
        """Outbound request logging must be skipped gracefully when metadata is absent."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        # No outbound_request in metadata
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    # Must not raise
                    asyncio.run(processor.process(context))

        outbound_calls = [
            c for c in mock_log.call_args_list if c[0][0] == ">>> OUTBOUND REQUEST"
        ]
        self.assertEqual(len(outbound_calls), 0)

    def test_outbound_request_still_logs_inbound_when_metadata_absent(self):
        """When outbound metadata is absent, inbound request must still be logged."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        # No outbound_request in metadata (early-failure scenario)
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_log:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        inbound_calls = [
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        ]
        self.assertEqual(len(inbound_calls), 1)

    def test_exceptions_in_logging_do_not_propagate(self):
        """Exceptions in logging must not propagate to the pipeline."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry",
                side_effect=Exception("unexpected failure"),
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    # Must not raise
                    result = asyncio.run(processor.process(context))

        self.assertIs(result, context)


# ---------------------------------------------------------------------------
# Task 4.2: Non-streaming (JSON) response logging
# ---------------------------------------------------------------------------


class TestJsonResponseLogging(unittest.TestCase):
    """Verify non-streaming JSON response logging."""

    def test_response_logged_for_json_response(self):
        """log_response_entry must be called for a JSONResponse."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response(
            body={"type": "message", "content": "Hi"}
        )
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_log:
                    asyncio.run(processor.process(context))

        mock_log.assert_called_once()

    def test_response_entry_includes_request_id(self):
        """The response log entry must include the request_id for correlation."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_log:
                    asyncio.run(processor.process(context))

        request_id = context.metadata["content_request_id"]
        logged_request_id = mock_log.call_args[0][0]
        self.assertEqual(logged_request_id, request_id)

    def test_response_entry_includes_status_line(self):
        """The response log entry must include the HTTP status line."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response(status_code=200)
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_log:
                    asyncio.run(processor.process(context))

        status_line = mock_log.call_args[0][1]
        self.assertIn("200", status_line)

    def test_response_entry_includes_response_body(self):
        """The response log entry must include the JSON response body."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        body = {"type": "message", "content": [{"type": "text", "text": "Hello!"}]}
        context = _make_context()
        context.response = _make_json_response(body=body)
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_log:
                    asyncio.run(processor.process(context))

        logged_body = mock_log.call_args[0][4]
        self.assertIsNotNone(logged_body)

    def test_response_entry_includes_both_header_sets(self):
        """The response log entry must include both outbound and inbound header dicts."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_log:
                    asyncio.run(processor.process(context))

        # log_response_entry(request_id, status_line, outbound_headers, inbound_headers, body)
        outbound_headers = mock_log.call_args[0][2]
        inbound_headers = mock_log.call_args[0][3]
        self.assertIsInstance(outbound_headers, dict)
        self.assertIsInstance(inbound_headers, dict)

    def test_json_response_not_modified(self):
        """The JSONResponse must be returned unmodified from the processor."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        original_response = _make_json_response()
        context.response = original_response
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        self.assertIs(context.response, original_response)

    def test_no_response_does_not_raise(self):
        """process() must not raise when context.response is None."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = None
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    result = asyncio.run(processor.process(context))

        self.assertIs(result, context)


# ---------------------------------------------------------------------------
# Task 4.3: Streaming response logging via body iterator wrapping
# ---------------------------------------------------------------------------


class TestExtractTextDelta(unittest.TestCase):
    """Verify the _extract_text_delta method."""

    def setUp(self):
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        self.processor = ContentLogProcessor()

    def test_extract_text_from_text_delta_event(self):
        """_extract_text_delta must return text from text_delta SSE events."""
        chunk = b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "Hello")

    def test_extract_returns_empty_for_non_text_delta(self):
        """_extract_text_delta must return empty string for non text_delta events."""
        chunk = b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "")

    def test_extract_returns_empty_for_message_start(self):
        """_extract_text_delta must return empty string for message_start events."""
        chunk = b'event: message_start\ndata: {"type": "message_start", "message": {"usage": {"input_tokens": 10}}}\n\n'
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "")

    def test_extract_returns_empty_for_empty_chunk(self):
        """_extract_text_delta must return empty string for empty chunk."""
        result = self.processor._extract_text_delta(b"")
        self.assertEqual(result, "")

    def test_extract_returns_empty_for_invalid_json(self):
        """_extract_text_delta must return empty string for invalid JSON."""
        chunk = b"data: not valid json\n\n"
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "")

    def test_extract_handles_string_chunks(self):
        """_extract_text_delta must handle str input as well as bytes."""
        chunk = 'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "World"}}\n\n'
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "World")

    def test_extract_multiple_text_deltas_in_chunk(self):
        """_extract_text_delta handles a chunk with one SSE event block."""
        chunk = b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}\n\n'
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "Hi")

    def test_extract_returns_empty_for_input_json_delta(self):
        """_extract_text_delta must return empty for input_json_delta events."""
        chunk = b'data: {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}}\n\n'
        result = self.processor._extract_text_delta(chunk)
        self.assertEqual(result, "")


class TestStreamingResponseLogging(unittest.TestCase):
    """Verify streaming response body iterator wrapping and text extraction."""

    def _run_streaming(self, chunks, context, enabled=True, outbound_request=None):
        """Run the processor on a streaming response and consume the iterator."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        if outbound_request:
            context.metadata["outbound_request"] = outbound_request

        streaming_resp = _make_streaming_response(chunks)
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = enabled
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ) as mock_req:
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_resp:
                    asyncio.run(processor.process(context))

                    # Consume the iterator to trigger the finally block
                    async def consume():
                        chunks_out = []
                        async for chunk in context.response.body_iterator:
                            chunks_out.append(chunk)
                        return chunks_out

                    consumed = asyncio.run(consume())

        return consumed, mock_req, mock_resp

    def test_streaming_response_body_iterator_wrapped(self):
        """For a streaming response, body_iterator must be replaced with wrapped generator."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        streaming_resp = _make_streaming_response()
        original_iterator = streaming_resp.body_iterator
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    asyncio.run(processor.process(context))

        # body_iterator must be replaced
        self.assertIsNot(context.response.body_iterator, original_iterator)

    def test_streaming_chunks_passed_through_byte_identical(self):
        """Every chunk yielded by the wrapped iterator must be byte-identical to original."""
        chunks = [
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}}\n\n',
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "B"}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
        context = _make_context()
        consumed, _, _ = self._run_streaming(chunks, context)

        # All chunks must be yielded unchanged
        self.assertEqual(consumed, chunks)

    def test_streaming_response_logged_after_stream_completes(self):
        """log_response_entry must be called after all chunks are yielded."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        chunks = [
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
        context = _make_context()
        streaming_resp = _make_streaming_response(chunks)
        context.response = streaming_resp
        processor = ContentLogProcessor()

        response_log_called_at = []

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_resp:
                    asyncio.run(processor.process(context))

                    async def consume():
                        async for chunk in context.response.body_iterator:
                            # response entry must NOT be called yet during iteration
                            if response_log_called_at:
                                pass
                        response_log_called_at.append("done")

                    asyncio.run(consume())

        mock_resp.assert_called_once()

    def test_streaming_accumulated_text_logged_as_response_body(self):
        """Accumulated text deltas must be logged as the response body."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        chunks = [
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n\n',
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
        context = _make_context()
        streaming_resp = _make_streaming_response(chunks)
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_resp:
                    asyncio.run(processor.process(context))

                    async def consume():
                        async for _ in context.response.body_iterator:
                            pass

                    asyncio.run(consume())

        logged_body = mock_resp.call_args[0][4]
        self.assertEqual(logged_body, "Hello world")

    def test_streaming_response_includes_both_header_sets(self):
        """Streaming response log entry must include both outbound and inbound header dicts."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        chunks = [b'data: {"type": "message_stop"}\n\n']
        context = _make_context()
        streaming_resp = _make_streaming_response(chunks)
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ) as mock_resp:
                    asyncio.run(processor.process(context))

                    async def consume():
                        async for _ in context.response.body_iterator:
                            pass

                    asyncio.run(consume())

        outbound_headers = mock_resp.call_args[0][2]
        inbound_headers = mock_resp.call_args[0][3]
        self.assertIsInstance(outbound_headers, dict)
        self.assertIsInstance(inbound_headers, dict)

    def test_streaming_no_response_logged_before_stream_ends(self):
        """log_response_entry must not be called until after all chunks are consumed."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        call_order = []
        chunks = [
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
        context = _make_context()
        streaming_resp = _make_streaming_response(chunks)
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=lambda *a, **kw: call_order.append("response_logged"),
                ):
                    asyncio.run(processor.process(context))

                    chunks_received = []

                    async def consume():
                        async for chunk in context.response.body_iterator:
                            call_order.append(f"chunk:{len(chunks_received)}")
                            chunks_received.append(chunk)

                    asyncio.run(consume())

        # response_logged must come after all chunks
        self.assertIn("response_logged", call_order)
        chunk_indices = [i for i, e in enumerate(call_order) if e.startswith("chunk:")]
        resp_index = call_order.index("response_logged")
        if chunk_indices:
            self.assertGreater(resp_index, max(chunk_indices))


class TestConcurrentStreamBufferIsolation(unittest.TestCase):
    """Verify per-request buffer isolation prevents interleaving."""

    def test_concurrent_streams_have_isolated_buffers(self):
        """Text accumulated for two concurrent streams must not intermix."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        chunks_a = [
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "AAA"}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
        chunks_b = [
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "BBB"}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]

        context_a = _make_context()
        context_a.response = _make_streaming_response(chunks_a)
        context_b = _make_context()
        context_b.response = _make_streaming_response(chunks_b)
        processor = ContentLogProcessor()

        logged_bodies = {}

        def capture_response(
            request_id, status_line, outbound_hdrs, inbound_hdrs, body, **kwargs
        ):
            logged_bodies[request_id] = body

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            mock_settings.content_log_include_body = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=capture_response,
                ):
                    asyncio.run(processor.process(context_a))
                    asyncio.run(processor.process(context_b))

                    async def consume_both():
                        async for _ in context_a.response.body_iterator:
                            pass
                        async for _ in context_b.response.body_iterator:
                            pass

                    asyncio.run(consume_both())

        id_a = context_a.metadata["content_request_id"]
        id_b = context_b.metadata["content_request_id"]
        self.assertNotEqual(id_a, id_b)
        self.assertEqual(logged_bodies.get(id_a), "AAA")
        self.assertEqual(logged_bodies.get(id_b), "BBB")


class TestStreamingErrorHandling(unittest.TestCase):
    """Verify streaming error handling - log whatever was accumulated."""

    def test_partial_text_logged_on_stream_error(self):
        """When the stream raises an error, accumulated text so far must still be logged."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        async def error_stream():
            yield b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "partial"}}\n\n'
            raise RuntimeError("stream broken")

        context = _make_context()
        resp = StreamingResponse(
            error_stream(),
            status_code=200,
            media_type="text/event-stream",
            headers={"content-type": "text/event-stream"},
        )
        context.response = resp
        processor = ContentLogProcessor()

        logged_body = None

        def capture_response(
            request_id, status_line, outbound_hdrs, inbound_hdrs, body, **kwargs
        ):
            nonlocal logged_body
            logged_body = body

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            mock_settings.content_log_include_body = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=capture_response,
                ):
                    asyncio.run(processor.process(context))

                    async def consume():
                        try:
                            async for _ in context.response.body_iterator:
                                pass
                        except RuntimeError:
                            pass

                    asyncio.run(consume())

        # Must have logged "partial" (whatever was accumulated before the error)
        self.assertEqual(logged_body, "partial")

    def test_stream_error_does_not_propagate_from_logging(self):
        """Logging errors during stream must not propagate to the pipeline."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        chunks = [b'data: {"type": "message_stop"}\n\n']
        context = _make_context()
        context.response = _make_streaming_response(chunks)
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry"
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=Exception("log write failed"),
                ):
                    asyncio.run(processor.process(context))

                    async def consume():
                        async for _ in context.response.body_iterator:
                            pass

                    # Must not raise
                    asyncio.run(consume())


class TestStreamingDisabled(unittest.TestCase):
    """Verify streaming wrapping is skipped when disabled."""

    def test_streaming_iterator_not_wrapped_when_disabled(self):
        """When disabled, body_iterator must not be replaced."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        streaming_resp = _make_streaming_response()
        original_iterator = streaming_resp.body_iterator
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = False
            asyncio.run(processor.process(context))

        self.assertIs(context.response.body_iterator, original_iterator)


if __name__ == "__main__":
    unittest.main()
