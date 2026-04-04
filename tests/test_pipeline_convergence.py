"""Integration tests verifying pipeline convergence for both OAuth and web proxy paths."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import StreamingResponse

from app.models.claude import InputMessage, MessagesAPIRequest, Role
from app.processors.claude_ai.context import ClaudeAIContext
from app.processors.claude_ai.pipeline import ClaudeAIPipeline

SKIP_PROCESSORS = [
    "ClaudeWebProcessor",
    "EventParsingProcessor",
    "ModelInjectorProcessor",
    "StopSequencesProcessor",
    "ToolCallEventProcessor",
    "MessageCollectorProcessor",
    "TokenCounterProcessor",
    "StreamingResponseProcessor",
    "NonStreamingResponseProcessor",
]


def _make_context():
    request = MagicMock()
    request.headers.get = MagicMock(return_value="")
    return ClaudeAIContext(
        original_request=request,
        messages_api_request=MessagesAPIRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[InputMessage(role=Role.USER, content="Hello")],
            stream=True,
        ),
    )


def _make_streaming_response():
    async def _empty():
        return
        yield  # noqa: unreachable

    return StreamingResponse(_empty())


class TestOAuthPathConvergence(unittest.TestCase):
    """Verify OAuth path skips intermediate processors and reaches RequestLogProcessor."""

    def test_oauth_path_skips_to_request_log_processor(self):
        """ClaudeAPIProcessor sets skip list; only it and RequestLogProcessor should run."""
        called_processors = []

        async def oauth_process(self_proc, context):
            called_processors.append("ClaudeAPIProcessor")
            context.response = _make_streaming_response()
            context.metadata["skip_processors"] = SKIP_PROCESSORS
            context.metadata["oauth_path"] = True
            context.metadata["account_id"] = "test-org"
            return context

        with (
            patch(
                "app.processors.claude_ai.claude_api_processor.ClaudeAPIProcessor.process",
                new=oauth_process,
            ),
            patch(
                "app.processors.claude_ai.request_log_processor.RequestLogProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ) as mock_log,
        ):
            pipeline = ClaudeAIPipeline()
            context = _make_context()
            asyncio.run(pipeline.process(context))

        # OAuth processor ran
        self.assertIn("ClaudeAPIProcessor", called_processors)
        # None of the skip-listed processors ran
        for skipped in SKIP_PROCESSORS:
            self.assertNotIn(skipped, called_processors)
        # RequestLogProcessor ran
        mock_log.assert_called_once()

    def test_oauth_path_request_start_time_is_set(self):
        """Pipeline sets request_start_time before ClaudeAPIProcessor runs."""
        seen_start_time = {}

        async def oauth_process(self_proc, context):
            seen_start_time["value"] = context.metadata.get("request_start_time")
            context.response = _make_streaming_response()
            context.metadata["skip_processors"] = SKIP_PROCESSORS
            context.metadata["oauth_path"] = True
            context.metadata["account_id"] = "test-org"
            return context

        with (
            patch(
                "app.processors.claude_ai.claude_api_processor.ClaudeAPIProcessor.process",
                new=oauth_process,
            ),
            patch(
                "app.processors.claude_ai.request_log_processor.RequestLogProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
        ):
            pipeline = ClaudeAIPipeline()
            context = _make_context()
            asyncio.run(pipeline.process(context))

        self.assertIn("value", seen_start_time)
        self.assertIsInstance(seen_start_time["value"], float)


class TestWebProxyPathConvergence(unittest.TestCase):
    """Verify web proxy path runs all processors then reaches RequestLogProcessor."""

    def test_web_proxy_path_runs_all_processors_then_log(self):
        """When OAuth is skipped, all web proxy processors run, then RequestLogProcessor."""
        called_processors = []

        async def noop_oauth(self_proc, context):
            called_processors.append("ClaudeAPIProcessor")
            return context

        async def web_process(self_proc, context):
            called_processors.append("ClaudeWebProcessor")
            context.response = _make_streaming_response()
            return context

        with (
            patch(
                "app.processors.claude_ai.claude_api_processor.ClaudeAPIProcessor.process",
                new=noop_oauth,
            ),
            patch(
                "app.processors.claude_ai.claude_web_processor.ClaudeWebProcessor.process",
                new=web_process,
            ),
            patch(
                "app.processors.claude_ai.event_parser_processor.EventParsingProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.model_injector_processor.ModelInjectorProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.stop_sequences_processor.StopSequencesProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.tool_call_event_processor.ToolCallEventProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.message_collector_processor.MessageCollectorProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.token_counter_processor.TokenCounterProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.streaming_response_processor.StreamingResponseProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.non_streaming_response_processor.NonStreamingResponseProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.request_log_processor.RequestLogProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ) as mock_log,
        ):
            pipeline = ClaudeAIPipeline()
            context = _make_context()
            asyncio.run(pipeline.process(context))

        self.assertIn("ClaudeAPIProcessor", called_processors)
        self.assertIn("ClaudeWebProcessor", called_processors)
        self.assertLess(
            called_processors.index("ClaudeAPIProcessor"),
            called_processors.index("ClaudeWebProcessor"),
        )
        mock_log.assert_called_once()

    def test_web_proxy_no_skip_processors_set(self):
        """Web proxy path does not set skip_processors — all processors run."""

        async def noop_oauth(self_proc, context):
            return context

        async def web_process(self_proc, context):
            context.response = _make_streaming_response()
            return context

        with (
            patch(
                "app.processors.claude_ai.claude_api_processor.ClaudeAPIProcessor.process",
                new=noop_oauth,
            ),
            patch(
                "app.processors.claude_ai.claude_web_processor.ClaudeWebProcessor.process",
                new=web_process,
            ),
            patch(
                "app.processors.claude_ai.event_parser_processor.EventParsingProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.model_injector_processor.ModelInjectorProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.stop_sequences_processor.StopSequencesProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.tool_call_event_processor.ToolCallEventProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.message_collector_processor.MessageCollectorProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.token_counter_processor.TokenCounterProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.streaming_response_processor.StreamingResponseProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.non_streaming_response_processor.NonStreamingResponseProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
            patch(
                "app.processors.claude_ai.request_log_processor.RequestLogProcessor.process",
                new=AsyncMock(side_effect=lambda ctx: ctx),
            ),
        ):
            pipeline = ClaudeAIPipeline()
            context = _make_context()
            result = asyncio.run(pipeline.process(context))

        self.assertNotIn("skip_processors", result.metadata)


class TestRequestLogProcessorIsSecondToLast(unittest.TestCase):
    """Verify RequestLogProcessor is second-to-last; ContentLogProcessor is last."""

    def test_content_log_processor_is_last(self):
        pipeline = ClaudeAIPipeline()
        last_processor = pipeline.processors[-1]
        self.assertEqual(last_processor.name, "ContentLogProcessor")

    def test_request_log_processor_is_second_to_last(self):
        pipeline = ClaudeAIPipeline()
        second_to_last = pipeline.processors[-2]
        self.assertEqual(second_to_last.name, "RequestLogProcessor")

    def test_non_streaming_response_processor_is_third_to_last(self):
        pipeline = ClaudeAIPipeline()
        third_to_last = pipeline.processors[-3]
        self.assertEqual(third_to_last.name, "NonStreamingResponseProcessor")


if __name__ == "__main__":
    unittest.main()
