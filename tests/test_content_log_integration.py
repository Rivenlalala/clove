"""Integration tests for content logging feature.

Tasks 7.1, 7.2, 7.3 — covers gaps not addressed by individual unit tests:

Task 7.1: Content logger utility in isolation
  - Timestamp millisecond precision
  - No log file created on disk when disabled

Task 7.2: Processor integration (three-entry flow, both paths, partial logging, disabled)
  - All three entries produced in one complete non-streaming request pass
  - OAuth path vs Web path (different outbound URLs)
  - Partial logging when outbound leg absent (early failure)
  - No logging when disabled

Task 7.3: Non-interference with existing proxy behavior
  - Write failures produce logger.warning but do not interrupt request processing
  - Response body data is identical before and after processing
  - Streaming chunks pass through byte-identical
"""

import asyncio
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import JSONResponse, StreamingResponse

from app.processors.claude_ai.context import ClaudeAIContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(body: bytes = b'{"model": "claude-3"}', method: str = "POST", path: str = "/v1/messages"):
    req = MagicMock()
    req.method = method
    req.url.path = path
    req.headers = {"content-type": "application/json", "authorization": "Bearer sk-ant-test"}
    req.body = AsyncMock(return_value=body)
    return req


def _make_context(body: bytes = b'{"model": "claude-3"}'):
    return ClaudeAIContext(original_request=_make_request(body))


def _make_json_response(status_code: int = 200, body: dict = None):
    if body is None:
        body = {"type": "message", "content": [{"type": "text", "text": "Hello!"}]}
    return JSONResponse(content=body, status_code=status_code)


async def _iter_chunks(chunks):
    for chunk in chunks:
        yield chunk


def _make_streaming_response(chunks=None):
    if chunks is None:
        chunks = [
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}\n\n',
            b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]
    return StreamingResponse(
        _iter_chunks(chunks),
        status_code=200,
        media_type="text/event-stream",
        headers={"content-type": "text/event-stream"},
    )


# ---------------------------------------------------------------------------
# Task 7.1: Content logger utility in isolation
# ---------------------------------------------------------------------------

class TestTimestampMillisecondPrecision(unittest.TestCase):
    """Verify _timestamp() produces millisecond-precision output."""

    def test_timestamp_includes_milliseconds(self):
        """_timestamp() must include millisecond component (3 decimal digits on seconds)."""
        from app.utils.content_logger import _timestamp
        ts = _timestamp()
        # Format: "YYYY-MM-DD HH:MM:SS.mmm" — milliseconds are 3 digits after dot
        self.assertRegex(ts, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$")

    def test_timestamp_returned_as_string(self):
        """_timestamp() must return a string."""
        from app.utils.content_logger import _timestamp
        self.assertIsInstance(_timestamp(), str)


class TestNoFileCreatedWhenDisabled(unittest.TestCase):
    """Verify no content log file is created on disk when content logging is disabled."""

    def setUp(self):
        import app.utils.content_logger as mod
        mod.content_log = None

    def test_no_file_created_on_disk_when_disabled(self):
        """configure_content_logger() must not create a log file when disabled."""
        import app.utils.content_logger as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "content.log"

            settings = MagicMock()
            settings.content_log_enabled = False
            settings.content_log_file_path = str(log_path)

            with patch("app.utils.content_logger.settings", settings):
                mod.configure_content_logger()

            self.assertFalse(log_path.exists(), "Log file must not be created when disabled")

    def test_content_log_none_after_disabled_configure(self):
        """Module-level content_log must remain None after configure when disabled."""
        import app.utils.content_logger as mod

        settings = MagicMock()
        settings.content_log_enabled = False

        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger"):
                mod.configure_content_logger()

        self.assertIsNone(mod.content_log)


# ---------------------------------------------------------------------------
# Task 7.2: Processor integration — three-entry flow
# ---------------------------------------------------------------------------

class TestThreeEntriesNonStreamingRequest(unittest.TestCase):
    """Verify all three entries (inbound request, outbound request, response) are produced
    for a single complete non-streaming request pass."""

    def _run(self, outbound_url):
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["outbound_request"] = {
            "method": "POST",
            "url": outbound_url,
            "headers": {"authorization": "Bearer tok"},
            "body": '{"model": "claude-3"}',
        }
        processor = ContentLogProcessor()

        request_calls = []
        response_calls = []

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry",
                side_effect=lambda *a, **kw: request_calls.append(a[0]),
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=lambda *a, **kw: response_calls.append(a),
                ):
                    asyncio.run(processor.process(context))

        return request_calls, response_calls

    def test_three_entries_total_json_response(self):
        """Exactly two request entries and one response entry for a complete non-streaming pass."""
        request_calls, response_calls = self._run("https://api.anthropic.com/v1/messages")
        self.assertEqual(len(request_calls), 2, "Expected inbound + outbound request entries")
        self.assertEqual(len(response_calls), 1, "Expected exactly one response entry")

    def test_inbound_entry_present(self):
        """Inbound request entry must be among the request log calls."""
        request_calls, _ = self._run("https://api.anthropic.com/v1/messages")
        self.assertIn(">>> INBOUND REQUEST", request_calls)

    def test_outbound_entry_present(self):
        """Outbound request entry must be among the request log calls."""
        request_calls, _ = self._run("https://api.anthropic.com/v1/messages")
        self.assertIn(">>> OUTBOUND REQUEST", request_calls)

    def test_inbound_before_outbound(self):
        """Inbound entry must be logged before outbound entry."""
        request_calls, _ = self._run("https://api.anthropic.com/v1/messages")
        self.assertEqual(request_calls.index(">>> INBOUND REQUEST"), 0)
        self.assertEqual(request_calls.index(">>> OUTBOUND REQUEST"), 1)


class TestOAuthPathThreeEntries(unittest.TestCase):
    """Verify three entries are produced for OAuth API path (api.anthropic.com)."""

    def test_outbound_url_is_anthropic_api(self):
        """OAuth path simulation: outbound URL must point to api.anthropic.com."""
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
        outbound_status_lines = []

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry",
                side_effect=lambda direction, req_id, status_line, *a, **kw: (
                    outbound_status_lines.append(status_line)
                    if direction == ">>> OUTBOUND REQUEST" else None
                ),
            ):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    asyncio.run(processor.process(context))

        self.assertEqual(len(outbound_status_lines), 1)
        self.assertIn("api.anthropic.com", outbound_status_lines[0])


class TestWebPathThreeEntries(unittest.TestCase):
    """Verify three entries are produced for Claude Web path (claude.ai)."""

    def test_outbound_url_is_claude_ai(self):
        """Web path simulation: outbound URL must point to claude.ai."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["outbound_request"] = {
            "method": "POST",
            "url": "https://claude.ai/api/organizations/org123/chat_conversations",
            "headers": {},
            "body": "{}",
        }
        processor = ContentLogProcessor()
        outbound_status_lines = []

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry",
                side_effect=lambda direction, req_id, status_line, *a, **kw: (
                    outbound_status_lines.append(status_line)
                    if direction == ">>> OUTBOUND REQUEST" else None
                ),
            ):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    asyncio.run(processor.process(context))

        self.assertEqual(len(outbound_status_lines), 1)
        self.assertIn("claude.ai", outbound_status_lines[0])


class TestPartialLoggingEarlyFailure(unittest.TestCase):
    """Verify partial logging (inbound request + response only) when outbound leg is absent."""

    def test_only_inbound_and_response_when_no_outbound_metadata(self):
        """When outbound_request metadata is absent, only inbound request + response are logged."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        # No outbound_request in metadata — simulates early failure before Anthropic
        processor = ContentLogProcessor()

        request_calls = []
        response_calls = []

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry",
                side_effect=lambda *a, **kw: request_calls.append(a[0]),
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=lambda *a, **kw: response_calls.append(a),
                ):
                    asyncio.run(processor.process(context))

        self.assertEqual(len(request_calls), 1, "Only inbound request entry when no outbound metadata")
        self.assertEqual(request_calls[0], ">>> INBOUND REQUEST")
        self.assertEqual(len(response_calls), 1, "Response entry still produced")


class TestNoLoggingWhenDisabledIntegration(unittest.TestCase):
    """Verify no log entries are produced when content logging is disabled."""

    def test_no_entries_when_disabled_json_response(self):
        """When disabled, no log_request_entry or log_response_entry calls for JSON response."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = False
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry") as mock_req:
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry") as mock_resp:
                    asyncio.run(processor.process(context))

        mock_req.assert_not_called()
        mock_resp.assert_not_called()

    def test_no_entries_when_disabled_streaming_response(self):
        """When disabled, body_iterator is not wrapped and no log calls are made."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        streaming_resp = _make_streaming_response()
        original_iterator = streaming_resp.body_iterator
        context.response = streaming_resp
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = False
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry") as mock_req:
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry") as mock_resp:
                    asyncio.run(processor.process(context))

        mock_req.assert_not_called()
        mock_resp.assert_not_called()
        self.assertIs(context.response.body_iterator, original_iterator)


# ---------------------------------------------------------------------------
# Task 7.3: Non-interference with existing proxy behavior
# ---------------------------------------------------------------------------

class TestWriteFailureLogsWarning(unittest.TestCase):
    """Verify content log write failures emit a warning to the main log and never propagate."""

    def test_inbound_request_write_failure_logs_warning(self):
        """When log_request_entry raises during inbound logging, logger.warning must be called."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.processors.claude_ai.content_log_processor.log_request_entry",
                side_effect=RuntimeError("disk full"),
            ):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    with patch("app.processors.claude_ai.content_log_processor.logger") as mock_logger:
                        result = asyncio.run(processor.process(context))

        self.assertIs(result, context)
        mock_logger.warning.assert_called()

    def test_response_write_failure_logs_warning(self):
        """When log_response_entry raises, logger.warning must be called and context returned."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry"):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry",
                    side_effect=RuntimeError("permission denied"),
                ):
                    with patch("app.processors.claude_ai.content_log_processor.logger") as mock_logger:
                        result = asyncio.run(processor.process(context))

        self.assertIs(result, context)
        mock_logger.warning.assert_called()


class TestResponseDataUnmodified(unittest.TestCase):
    """Verify that content logging does not alter request or response data."""

    def test_json_response_body_identical_after_processing(self):
        """JSON response body must be byte-identical before and after ContentLogProcessor."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        original_body = {"type": "message", "content": [{"type": "text", "text": "Original content"}]}
        context = _make_context()
        response = _make_json_response(body=original_body)
        body_before = bytes(response.body)
        context.response = response
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry"):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    asyncio.run(processor.process(context))

        self.assertEqual(bytes(context.response.body), body_before)

    def test_streaming_chunks_pass_through_byte_identical(self):
        """Every streaming chunk must be yielded byte-identical to the original."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        original_chunks = [
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "B"}}\n\n',
            b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]
        context = _make_context()
        context.response = _make_streaming_response(original_chunks)
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry"):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    asyncio.run(processor.process(context))

                    async def consume():
                        received = []
                        async for chunk in context.response.body_iterator:
                            received.append(chunk)
                        return received

                    received = asyncio.run(consume())

        self.assertEqual(received, original_chunks)

    def test_request_context_unmodified_except_metadata(self):
        """ContentLogProcessor must not modify context.original_request."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        req = _make_request(body=b'{"model": "claude-3"}')
        context = ClaudeAIContext(original_request=req)
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry"):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    asyncio.run(processor.process(context))

        # original_request must be the same object
        self.assertIs(context.original_request, req)


class TestRequestLogProcessorCoexistence(unittest.TestCase):
    """Verify RequestLogProcessor continues to function when ContentLogProcessor is active."""

    def test_request_log_processor_emits_log_with_content_log_enabled(self):
        """RequestLogProcessor._emit_log must be called even when ContentLogProcessor is active."""
        from app.processors.claude_ai.request_log_processor import RequestLogProcessor
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()

        rlp = RequestLogProcessor()
        clp = ContentLogProcessor()

        emit_called = []

        with patch.object(rlp, "_emit_log", side_effect=lambda *a, **kw: emit_called.append(True)):
            asyncio.run(rlp.process(context))

        with patch("app.processors.claude_ai.content_log_processor.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.processors.claude_ai.content_log_processor.log_request_entry"):
                with patch("app.processors.claude_ai.content_log_processor.log_response_entry"):
                    asyncio.run(clp.process(context))

        self.assertEqual(len(emit_called), 1, "RequestLogProcessor._emit_log must be called once")
        # Response must still be unchanged
        self.assertIsInstance(context.response, JSONResponse)


if __name__ == "__main__":
    unittest.main()
