import asyncio
import json
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.responses import JSONResponse, StreamingResponse

from app.models.claude import InputMessage, Message, MessagesAPIRequest, Role, Usage
from app.processors.claude_ai.context import ClaudeAIContext
from app.processors.claude_ai.request_log_processor import RequestLogProcessor

PATCH_LOGGER = "app.processors.claude_ai.request_log_processor.logger"


def _make_request(model="claude-sonnet-4-20250514", stream=False):
    return MessagesAPIRequest(
        model=model,
        max_tokens=100,
        messages=[InputMessage(role=Role.USER, content="Hello")],
        stream=stream,
    )


def _make_context(**kwargs):
    ctx = ClaudeAIContext(original_request=MagicMock())
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


def _make_sse_bytes(event_type, data_dict):
    """Build raw SSE event bytes like the Anthropic API returns."""
    data_json = json.dumps(data_dict)
    return f"event: {event_type}\ndata: {data_json}\n\n".encode()


async def _collect_stream(async_iter):
    """Consume an async iterator and return all items."""
    chunks = []
    async for chunk in async_iter:
        chunks.append(chunk)
    return chunks


def _make_collected_message(input_tokens=100, output_tokens=50):
    msg = MagicMock(spec=Message)
    msg.usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


class TestEmitLogFormat(unittest.TestCase):
    def setUp(self):
        self.processor = RequestLogProcessor()

    def test_emit_log_correct_format_all_fields(self):
        start = time.monotonic() - 2.3
        context = _make_context(
            messages_api_request=_make_request(
                model="claude-sonnet-4-20250514", stream=True
            ),
            collected_message=_make_collected_message(
                input_tokens=1234, output_tokens=567
            ),
        )
        context.metadata["request_start_time"] = start
        context.metadata["cache_read"] = True
        context.metadata["cache_write"] = False
        context.metadata["account_id"] = "abc123"

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context)

        mock_logger.info.assert_called_once()
        log_line = mock_logger.info.call_args[0][0]

        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        self.assertEqual(pairs["model"], "claude-sonnet-4-20250514")
        self.assertEqual(pairs["input_tokens"], "1234")
        self.assertEqual(pairs["output_tokens"], "567")
        self.assertEqual(pairs["cache_read"], "true")
        self.assertEqual(pairs["cache_write"], "false")
        self.assertTrue(pairs["duration"].endswith("s"))
        self.assertEqual(pairs["account"], "abc123")
        self.assertEqual(pairs["stream"], "true")

    def test_emit_log_safe_defaults_missing_fields(self):
        context = _make_context()

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context)

        log_line = mock_logger.info.call_args[0][0]
        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        self.assertEqual(pairs["model"], "unknown")
        self.assertEqual(pairs["input_tokens"], "0")
        self.assertEqual(pairs["output_tokens"], "0")
        self.assertEqual(pairs["cache_read"], "false")
        self.assertEqual(pairs["cache_write"], "false")
        self.assertEqual(pairs["duration"], "0.0s")
        self.assertEqual(pairs["account"], "unknown")
        self.assertEqual(pairs["stream"], "false")

    def test_account_resolution_oauth_metadata_takes_priority(self):
        session = MagicMock()
        session.account.organization_uuid = "web-org-456"
        context = _make_context(claude_session=session)
        context.metadata["account_id"] = "oauth-org-123"

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context)

        log_line = mock_logger.info.call_args[0][0]
        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        self.assertEqual(pairs["account"], "oauth-org-123")

    def test_account_resolution_web_proxy_session_fallback(self):
        session = MagicMock()
        session.account.organization_uuid = "web-org-456"
        context = _make_context(claude_session=session)

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context)

        log_line = mock_logger.info.call_args[0][0]
        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        self.assertEqual(pairs["account"], "web-org-456")

    def test_account_resolution_fallback_unknown(self):
        context = _make_context(claude_session=None)

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context)

        log_line = mock_logger.info.call_args[0][0]
        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        self.assertEqual(pairs["account"], "unknown")

    def test_duration_computed_from_start_time(self):
        context = _make_context()
        context.metadata["request_start_time"] = time.monotonic() - 5.7

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context)

        log_line = mock_logger.info.call_args[0][0]
        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        duration_str = pairs["duration"]
        self.assertTrue(duration_str.endswith("s"))
        duration_val = float(duration_str[:-1])
        self.assertAlmostEqual(duration_val, 5.7, delta=0.5)

    def test_emit_log_oauth_cache_read_from_sse_usage(self):
        context = _make_context()
        oauth_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 0,
        }

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context, oauth_usage=oauth_usage)

        pairs = dict(
            pair.split("=", 1) for pair in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["cache_read"], "80")
        self.assertEqual(pairs["cache_write"], "0")

    def test_emit_log_oauth_cache_write_from_sse_usage(self):
        context = _make_context()
        oauth_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 200,
        }

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context, oauth_usage=oauth_usage)

        pairs = dict(
            pair.split("=", 1) for pair in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["cache_read"], "0")
        self.assertEqual(pairs["cache_write"], "200")

    def test_emit_log_oauth_no_cache_defaults_false(self):
        context = _make_context()
        oauth_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(context, oauth_usage=oauth_usage)

        pairs = dict(
            pair.split("=", 1) for pair in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["cache_read"], "0")
        self.assertEqual(pairs["cache_write"], "0")

    def test_emit_log_oauth_usage_overrides_collected_message(self):
        context = _make_context(
            collected_message=_make_collected_message(
                input_tokens=100, output_tokens=50
            ),
        )

        with patch(PATCH_LOGGER) as mock_logger:
            self.processor._emit_log(
                context, oauth_usage={"input_tokens": 999, "output_tokens": 888}
            )

        log_line = mock_logger.info.call_args[0][0]
        pairs = dict(pair.split("=", 1) for pair in log_line.split(" "))
        self.assertEqual(pairs["input_tokens"], "999")
        self.assertEqual(pairs["output_tokens"], "888")


class TestProcessMethod(unittest.TestCase):
    def setUp(self):
        self.processor = RequestLogProcessor()

    def test_process_non_streaming_json_response_logs_immediately(self):
        context = _make_context(
            messages_api_request=_make_request(stream=False),
            response=JSONResponse(content={"test": True}),
            collected_message=_make_collected_message(),
        )
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(self.processor.process(context))

        mock_logger.info.assert_called_once()

    def test_process_streaming_wraps_body_iterator(self):
        async def _orig_iter():
            yield b"data: test\n\n"

        orig_iter = _orig_iter()
        context = _make_context(
            messages_api_request=_make_request(stream=True),
            response=StreamingResponse(orig_iter),
        )
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(self.processor.process(context))

        # body_iterator replaced with wrapper; log not yet emitted
        self.assertIsNot(context.response.body_iterator, orig_iter)
        mock_logger.info.assert_not_called()

    def test_process_streaming_full_lifecycle_oauth(self):
        start = _make_sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            },
        )
        delta = _make_sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 100, "output_tokens": 40},
            },
        )

        async def _iter():
            yield start
            yield delta

        context = _make_context(
            messages_api_request=_make_request(stream=True),
            response=StreamingResponse(_iter()),
        )
        context.metadata["oauth_path"] = True
        context.metadata["request_start_time"] = time.monotonic()
        context.metadata["account_id"] = "test-org"

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(self.processor.process(context))
            asyncio.run(_collect_stream(context.response.body_iterator))

        mock_logger.info.assert_called_once()
        pairs = dict(
            p.split("=", 1) for p in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["input_tokens"], "100")
        self.assertEqual(pairs["output_tokens"], "40")

    def test_process_streaming_full_lifecycle_web_proxy(self):
        async def _iter():
            yield "event: ping\ndata: {}\n\n"

        context = _make_context(
            messages_api_request=_make_request(stream=True),
            response=StreamingResponse(_iter()),
            collected_message=_make_collected_message(
                input_tokens=500, output_tokens=200
            ),
        )
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(self.processor.process(context))
            asyncio.run(_collect_stream(context.response.body_iterator))

        mock_logger.info.assert_called_once()
        pairs = dict(
            p.split("=", 1) for p in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["input_tokens"], "500")
        self.assertEqual(pairs["output_tokens"], "200")

    def test_process_no_response_returns_context_unchanged(self):
        context = _make_context(messages_api_request=_make_request())
        context.response = None

        with patch(PATCH_LOGGER) as mock_logger:
            result = asyncio.run(self.processor.process(context))

        mock_logger.info.assert_not_called()
        self.assertIs(result, context)


class TestWrapStream(unittest.TestCase):
    def setUp(self):
        self.processor = RequestLogProcessor()

    def test_wrap_stream_yields_all_chunks_unchanged_oauth(self):
        original = [b"chunk1", b"chunk2", b"chunk3"]

        async def _iter():
            for c in original:
                yield c

        context = _make_context(messages_api_request=_make_request(stream=True))
        context.metadata["oauth_path"] = True
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER):
            result = asyncio.run(
                _collect_stream(self.processor._wrap_stream(_iter(), context))
            )

        self.assertEqual(result, original)

    def test_wrap_stream_yields_all_chunks_unchanged_web_proxy(self):
        original = ["event: ping\ndata: {}\n\n", "event: ping\ndata: {}\n\n"]

        async def _iter():
            for c in original:
                yield c

        context = _make_context(messages_api_request=_make_request(stream=True))
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER):
            result = asyncio.run(
                _collect_stream(self.processor._wrap_stream(_iter(), context))
            )

        self.assertEqual(result, original)

    def test_wrap_stream_emits_log_on_completion_oauth(self):
        start = _make_sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 200, "output_tokens": 1},
                },
            },
        )
        delta = _make_sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 200, "output_tokens": 75},
            },
        )

        async def _iter():
            yield start
            yield delta

        context = _make_context(messages_api_request=_make_request(stream=True))
        context.metadata["oauth_path"] = True
        context.metadata["request_start_time"] = time.monotonic()
        context.metadata["account_id"] = "test-account"

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(_collect_stream(self.processor._wrap_stream(_iter(), context)))

        mock_logger.info.assert_called_once()
        pairs = dict(
            p.split("=", 1) for p in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["input_tokens"], "200")
        self.assertEqual(pairs["output_tokens"], "75")

    def test_wrap_stream_emits_log_on_completion_web_proxy(self):
        async def _iter():
            yield "event: ping\ndata: {}\n\n"

        context = _make_context(
            messages_api_request=_make_request(stream=True),
            collected_message=_make_collected_message(
                input_tokens=300, output_tokens=120
            ),
        )
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(_collect_stream(self.processor._wrap_stream(_iter(), context)))

        mock_logger.info.assert_called_once()
        pairs = dict(
            p.split("=", 1) for p in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["input_tokens"], "300")
        self.assertEqual(pairs["output_tokens"], "120")

    def test_wrap_stream_emits_log_on_error(self):
        async def _error_iter():
            yield b"some data"
            raise RuntimeError("stream error")

        context = _make_context(messages_api_request=_make_request(stream=True))
        context.metadata["oauth_path"] = True
        context.metadata["request_start_time"] = time.monotonic()

        with patch(PATCH_LOGGER) as mock_logger:
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    _collect_stream(self.processor._wrap_stream(_error_iter(), context))
                )

        mock_logger.info.assert_called_once()

    def test_wrap_stream_extracts_usage_across_multiple_chunks(self):
        start = _make_sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 50, "output_tokens": 1},
                },
            },
        )
        filler = _make_sse_bytes(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        )
        delta = _make_sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 50, "output_tokens": 25},
            },
        )

        async def _iter():
            yield start
            yield filler
            yield delta

        context = _make_context(messages_api_request=_make_request(stream=True))
        context.metadata["oauth_path"] = True
        context.metadata["request_start_time"] = time.monotonic()
        context.metadata["account_id"] = "test"

        with patch(PATCH_LOGGER) as mock_logger:
            asyncio.run(_collect_stream(self.processor._wrap_stream(_iter(), context)))

        pairs = dict(
            p.split("=", 1) for p in mock_logger.info.call_args[0][0].split(" ")
        )
        self.assertEqual(pairs["input_tokens"], "50")
        self.assertEqual(pairs["output_tokens"], "25")


class TestExtractUsageFromSSE(unittest.TestCase):
    def setUp(self):
        self.processor = RequestLogProcessor()

    def test_extract_input_tokens_from_message_start(self):
        chunk = _make_sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 150, "output_tokens": 1},
                },
            },
        )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(chunk, usage)
        self.assertEqual(usage["input_tokens"], 150)

    def test_extract_cache_tokens_from_message_start(self):
        chunk = _make_sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {
                        "input_tokens": 150,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 100,
                        "cache_creation_input_tokens": 50,
                    },
                },
            },
        )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(chunk, usage)
        self.assertEqual(usage["cache_read_input_tokens"], 100)
        self.assertEqual(usage["cache_creation_input_tokens"], 50)

    def test_extract_output_tokens_from_message_delta(self):
        chunk = _make_sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 150, "output_tokens": 42},
            },
        )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(chunk, usage)
        self.assertEqual(usage["output_tokens"], 42)

    def test_extract_both_from_combined_chunk(self):
        start = _make_sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            },
        )
        delta = _make_sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 100, "output_tokens": 55},
            },
        )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(start + delta, usage)
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 55)

    def test_malformed_json_silently_ignored(self):
        chunk = b"event: message_start\ndata: {bad json}\n\n"
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(chunk, usage)  # must not raise
        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)

    def test_non_usage_events_ignored(self):
        chunk = _make_sse_bytes(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        )
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(chunk, usage)
        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)

    def test_empty_chunk_handled(self):
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.processor._extract_usage_from_sse(b"", usage)  # must not raise
        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
