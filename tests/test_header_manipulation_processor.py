import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import Request

from app.core.config import settings
from app.processors.claude_ai.context import ClaudeAIContext


def _make_request(headers_dict):
    """Create a real Starlette Request with mutable ASGI scope."""
    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in headers_dict.items()],
    }
    return Request(scope)


def _make_context(headers_dict):
    """Create a ClaudeAIContext with the given request headers."""
    request = _make_request(headers_dict)
    return ClaudeAIContext(original_request=request)


class TestHeaderManipulationConfig(unittest.TestCase):
    """Test that header manipulation config fields exist with correct defaults."""

    def test_strip_headers_default_is_empty_list(self):
        self.assertEqual(settings.strip_headers, [])

    def test_add_headers_default_is_empty_dict(self):
        self.assertEqual(settings.add_headers, {})


class TestHeaderManipulationProcessor(unittest.TestCase):
    """Test header manipulation processor behavior."""

    def test_strip_exact_match_single_value_header(self):
        """Header removed when key:value matches exactly."""
        context = _make_context({"anthropic-beta": "context-1m-2025-08-07"})
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = ["anthropic-beta:context-1m-2025-08-07"]
            mock_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        headers_dict = {
            k.lower(): v for k, v in context.original_request.headers.items()
        }
        self.assertNotIn("anthropic-beta", headers_dict)

    def test_strip_one_value_from_comma_separated_header(self):
        """Only matching value removed, others preserved."""
        context = _make_context(
            {"anthropic-beta": "context-1m-2025-08-07,token-efficient-tools-2025-08-28"}
        )
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = ["anthropic-beta:context-1m-2025-08-07"]
            mock_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        self.assertEqual(
            context.original_request.headers.get("anthropic-beta"),
            "token-efficient-tools-2025-08-28",
        )

    def test_strip_all_values_from_comma_separated_header(self):
        """Header removed entirely when all values stripped."""
        context = _make_context({"anthropic-beta": "context-1m-2025-08-07"})
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = ["anthropic-beta:context-1m-2025-08-07"]
            mock_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        headers_dict = {
            k.lower(): v for k, v in context.original_request.headers.items()
        }
        self.assertNotIn("anthropic-beta", headers_dict)

    def test_add_headers_to_request(self):
        """New headers added from config."""
        context = _make_context({})
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = []
            mock_settings.add_headers = {"X-Custom": "value"}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        headers_dict = {
            k.lower(): v for k, v in context.original_request.headers.items()
        }
        self.assertEqual(headers_dict.get("x-custom"), "value")

    def test_strip_and_add_in_same_request(self):
        """Both operations applied in correct order."""
        context = _make_context({"anthropic-beta": "context-1m-2025-08-07"})
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = ["anthropic-beta:context-1m-2025-08-07"]
            mock_settings.add_headers = {"anthropic-beta": "my-beta-feature"}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        self.assertEqual(
            context.original_request.headers.get("anthropic-beta"),
            "my-beta-feature",
        )

    def test_case_insensitive_key_matching(self):
        """Anthropic-Beta matches anthropic-beta rule."""
        context = _make_context({"Anthropic-Beta": "context-1m-2025-08-07"})
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = ["anthropic-beta:context-1m-2025-08-07"]
            mock_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        headers_dict = {
            k.lower(): v for k, v in context.original_request.headers.items()
        }
        self.assertNotIn("anthropic-beta", headers_dict)

    def test_malformed_strip_rule_skipped(self):
        """Warning logged, rule skipped, request continues."""
        context = _make_context({"anthropic-beta": "context-1m-2025-08-07"})
        with (
            patch(
                "app.processors.claude_ai.header_manipulation_processor.settings"
            ) as mock_settings,
            patch(
                "app.processors.claude_ai.header_manipulation_processor.logger"
            ) as mock_logger,
        ):
            mock_settings.strip_headers = ["malformed-no-colon"]
            mock_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        mock_logger.warning.assert_called_once()
        self.assertEqual(
            context.original_request.headers.get("anthropic-beta"),
            "context-1m-2025-08-07",
        )

    def test_empty_config_no_op(self):
        """No-op, no headers modified."""
        original_headers = {
            "anthropic-beta": "context-1m-2025-08-07",
            "content-type": "application/json",
        }
        context = _make_context(original_headers.copy())
        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_settings:
            mock_settings.strip_headers = []
            mock_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            processor = HeaderManipulationProcessor()
            asyncio.run(processor.process(context))

        for key, value in original_headers.items():
            self.assertEqual(context.original_request.headers.get(key), value)


class TestPipelineIntegration(unittest.TestCase):
    """Test that HeaderManipulationProcessor is registered in the pipeline."""

    def test_header_processor_is_first_in_pipeline(self):
        """HeaderManipulationProcessor should be at position 0."""
        from app.processors.claude_ai.pipeline import ClaudeAIPipeline

        pipeline = ClaudeAIPipeline()
        first_processor = pipeline.processors[0]
        self.assertEqual(first_processor.name, "HeaderManipulationProcessor")


class TestHeaderManipulationEndToEnd(unittest.TestCase):
    """End-to-end test: headers modified by processor are read by ClaudeAPIProcessor."""

    def test_stripped_header_not_forwarded_to_api(self):
        """Verify that a stripped anthropic-beta value is not forwarded by ClaudeAPIProcessor."""
        from app.processors.claude_ai.claude_api_processor import ClaudeAPIProcessor
        from app.models.claude import InputMessage, MessagesAPIRequest, Role

        context = ClaudeAIContext(
            original_request=_make_request({"anthropic-beta": "context-1m-2025-08-07"}),
            messages_api_request=MessagesAPIRequest(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[InputMessage(role=Role.USER, content="Hello")],
            ),
        )

        with patch(
            "app.processors.claude_ai.header_manipulation_processor.settings"
        ) as mock_strip_settings:
            mock_strip_settings.strip_headers = ["anthropic-beta:context-1m-2025-08-07"]
            mock_strip_settings.add_headers = {}
            from app.processors.claude_ai.header_manipulation_processor import (
                HeaderManipulationProcessor,
            )

            asyncio.run(HeaderManipulationProcessor().process(context))

        processor = ClaudeAPIProcessor()
        headers = processor._prepare_headers(
            access_token="test-token",
            request=context.messages_api_request,
            original_request=context.original_request,
        )

        self.assertEqual(headers["anthropic-beta"], "oauth-2025-04-20")
        self.assertNotIn("context-1m-2025-08-07", headers["anthropic-beta"])


if __name__ == "__main__":
    unittest.main()
