"""Tests for the add-content-log redesign features.

Covers:
- ContentLogProcessor: pre-assigned request_id, content_inbound_logged flag
- ClaudeAPIProcessor: outbound_response_headers stashing
- ClaudeWebProcessor: outbound_response_headers stashing
- ContentLogHook: deduplication, error logging, failure handling
- Route handler: try/finally integration
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import JSONResponse

from app.processors.claude_ai.context import ClaudeAIContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    body: bytes = b'{"model": "claude-3"}',
    method: str = "POST",
    path: str = "/v1/messages",
):
    req = MagicMock()
    req.method = method
    req.url.path = path
    req.headers = {
        "content-type": "application/json",
        "authorization": "Bearer sk-ant-test",
    }
    req.body = AsyncMock(return_value=body)
    return req


def _make_context(body: bytes = b'{"model": "claude-3"}'):
    return ClaudeAIContext(original_request=_make_request(body))


def _make_json_response(status_code: int = 200, body: dict = None):
    if body is None:
        body = {"type": "message", "content": [{"type": "text", "text": "Hello!"}]}
    return JSONResponse(content=body, status_code=status_code)


# ---------------------------------------------------------------------------
# ContentLogProcessor: pre-assigned request_id
# ---------------------------------------------------------------------------


class TestContentLogProcessorPreAssignedRequestId(unittest.TestCase):
    """Verify ContentLogProcessor uses pre-assigned request_id from route handler."""

    def test_uses_pre_assigned_request_id(self):
        """When content_request_id is pre-assigned, processor must use it instead of generating."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["content_request_id"] = "preassigned1"
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

        # The request_id in log calls must be the pre-assigned one
        inbound_call = next(
            c for c in mock_log.call_args_list if c[0][0] == ">>> INBOUND REQUEST"
        )
        logged_request_id = inbound_call[0][1]
        self.assertEqual(logged_request_id, "preassigned1")

    def test_preserves_pre_assigned_request_id(self):
        """Pre-assigned request_id must not be overwritten."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["content_request_id"] = "preassigned2"
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

        self.assertEqual(context.metadata["content_request_id"], "preassigned2")

    def test_generates_request_id_when_not_pre_assigned(self):
        """When no pre-assigned ID exists, processor must generate one (backward compatibility)."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        # No content_request_id in metadata
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
        request_id = context.metadata["content_request_id"]
        self.assertEqual(len(request_id), 8)
        int(request_id, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# ContentLogProcessor: content_inbound_logged flag
# ---------------------------------------------------------------------------


class TestContentLogProcessorInboundLoggedFlag(unittest.TestCase):
    """Verify ContentLogProcessor sets content_inbound_logged flag after successful inbound logging."""

    def test_sets_content_inbound_logged_after_inbound_logging(self):
        """After logging inbound request, content_inbound_logged must be True."""
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

        self.assertTrue(context.metadata.get("content_inbound_logged"))

    def test_does_not_set_flag_when_inbound_logging_fails(self):
        """If inbound logging fails, content_inbound_logged must not be set."""
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
                side_effect=RuntimeError("disk full"),
            ):
                with patch(
                    "app.processors.claude_ai.content_log_processor.log_response_entry"
                ):
                    # Must not raise
                    asyncio.run(processor.process(context))

        # Flag must not be set when inbound logging failed
        self.assertNotIn("content_inbound_logged", context.metadata)

    def test_flag_absent_when_disabled(self):
        """When content logging is disabled, flag must not be set."""
        from app.processors.claude_ai.content_log_processor import ContentLogProcessor

        context = _make_context()
        context.response = _make_json_response()
        processor = ContentLogProcessor()

        with patch(
            "app.processors.claude_ai.content_log_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = False
            asyncio.run(processor.process(context))

        self.assertNotIn("content_inbound_logged", context.metadata)


# ---------------------------------------------------------------------------
# ClaudeAPIProcessor: outbound_response_headers stashing
# ---------------------------------------------------------------------------


class TestClaudeAPIProcessorOutboundResponseHeaders(unittest.TestCase):
    """Verify ClaudeAPIProcessor stashes outbound_response_headers."""

    def _run_processor_with_response_headers(self, response_headers=None):
        from app.models.claude import InputMessage, MessagesAPIRequest, Role
        from app.processors.claude_ai.claude_api_processor import ClaudeAPIProcessor

        if response_headers is None:
            response_headers = {
                "x-request-id": "req-abc",
                "content-type": "text/event-stream",
            }

        response = MagicMock()
        response.status_code = 200
        response.headers = response_headers

        async def aiter_bytes():
            yield b"data: test\n\n"

        response.aiter_bytes = aiter_bytes

        session = MagicMock()
        session.request = AsyncMock(return_value=response)
        session.close = AsyncMock()

        account = MagicMock()
        account.organization_uuid = "test-org-uuid"
        account.oauth_token.access_token = "fake-token"
        account.resets_at = None
        account.__enter__ = MagicMock(return_value=account)
        account.__exit__ = MagicMock(return_value=False)

        request = MagicMock()
        request.headers.get = MagicMock(return_value="")

        context = ClaudeAIContext(
            original_request=request,
            messages_api_request=MessagesAPIRequest(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[InputMessage(role=Role.USER, content="Hello")],
            ),
        )

        with (
            patch(
                "app.processors.claude_ai.claude_api_processor.cache_service"
            ) as mock_cache,
            patch(
                "app.processors.claude_ai.claude_api_processor.account_manager"
            ) as mock_am,
            patch(
                "app.processors.claude_ai.claude_api_processor.create_session",
                return_value=session,
            ),
            patch(
                "app.processors.claude_ai.claude_api_processor.settings"
            ) as mock_settings,
        ):
            mock_cache.process_messages.return_value = (None, [])
            mock_settings.max_models = []
            mock_settings.proxy_url = None
            mock_settings.request_timeout = 30
            mock_settings.content_log_enabled = True
            mock_settings.claude_api_baseurl.encoded_string.return_value = (
                "https://api.anthropic.com"
            )

            mock_am.get_account_by_id = AsyncMock(return_value=None)
            mock_am.get_account_for_oauth = AsyncMock(return_value=account)

            processor = ClaudeAPIProcessor()
            result = asyncio.run(processor.process(context))

        return result

    def test_outbound_response_headers_stashed(self):
        """ClaudeAPIProcessor must stash outbound_response_headers in metadata."""
        result = self._run_processor_with_response_headers()
        self.assertIn("outbound_response_headers", result.metadata)
        self.assertEqual(
            result.metadata["outbound_response_headers"]["x-request-id"], "req-abc"
        )

    def test_outbound_response_headers_stashed_when_content_log_disabled(self):
        """When content logging is disabled, headers must not be stashed."""
        from app.models.claude import InputMessage, MessagesAPIRequest, Role
        from app.processors.claude_ai.claude_api_processor import ClaudeAPIProcessor

        response = MagicMock()
        response.status_code = 200
        response.headers = {"x-request-id": "req-abc"}

        async def aiter_bytes():
            yield b"data: test\n\n"

        response.aiter_bytes = aiter_bytes

        session = MagicMock()
        session.request = AsyncMock(return_value=response)
        session.close = AsyncMock()

        account = MagicMock()
        account.organization_uuid = "test-org-uuid"
        account.oauth_token.access_token = "fake-token"
        account.resets_at = None
        account.__enter__ = MagicMock(return_value=account)
        account.__exit__ = MagicMock(return_value=False)

        request = MagicMock()
        request.headers.get = MagicMock(return_value="")

        context = ClaudeAIContext(
            original_request=request,
            messages_api_request=MessagesAPIRequest(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[InputMessage(role=Role.USER, content="Hello")],
            ),
        )

        with (
            patch(
                "app.processors.claude_ai.claude_api_processor.cache_service"
            ) as mock_cache,
            patch(
                "app.processors.claude_ai.claude_api_processor.account_manager"
            ) as mock_am,
            patch(
                "app.processors.claude_ai.claude_api_processor.create_session",
                return_value=session,
            ),
            patch(
                "app.processors.claude_ai.claude_api_processor.settings"
            ) as mock_settings,
        ):
            mock_cache.process_messages.return_value = (None, [])
            mock_settings.max_models = []
            mock_settings.proxy_url = None
            mock_settings.request_timeout = 30
            mock_settings.content_log_enabled = False
            mock_settings.claude_api_baseurl.encoded_string.return_value = (
                "https://api.anthropic.com"
            )

            mock_am.get_account_by_id = AsyncMock(return_value=None)
            mock_am.get_account_for_oauth = AsyncMock(return_value=account)

            processor = ClaudeAPIProcessor()
            result = asyncio.run(processor.process(context))

        self.assertNotIn("outbound_response_headers", result.metadata)


# ---------------------------------------------------------------------------
# ClaudeWebProcessor: outbound_response_headers stashing
# ---------------------------------------------------------------------------


class TestClaudeWebProcessorOutboundResponseHeaders(unittest.TestCase):
    """Verify ClaudeWebProcessor stashes outbound_response_headers (empty dict)."""

    def test_outbound_response_headers_stashed_as_empty_dict(self):
        """ClaudeWebProcessor must stash outbound_response_headers as empty dict."""
        from app.processors.claude_ai.claude_web_processor import ClaudeWebProcessor
        from app.models.internal import ClaudeWebRequest, Attachment
        from app.models.claude import MessagesAPIRequest, InputMessage, Role

        context = _make_context()
        context.original_stream = None
        context.messages_api_request = MessagesAPIRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[InputMessage(role=Role.USER, content="Hello")],
        )

        # Set up claude session
        mock_session = MagicMock()
        mock_session.completion_url = "https://claude.ai/api/organizations/org123/chat_conversations/conv456/completion"
        mock_session.account.is_pro = False
        mock_session.send_message = AsyncMock(return_value=AsyncMock())

        context.claude_session = mock_session
        context.claude_web_request = ClaudeWebRequest(
            max_tokens_to_sample=100,
            attachments=[Attachment.from_text("Hello")],
            files=[],
            model="claude-sonnet-4-20250514",
            rendering_mode="messages",
            prompt="",
            timezone="UTC",
            tools=[],
        )

        processor = ClaudeWebProcessor()

        with patch(
            "app.processors.claude_ai.claude_web_processor.settings"
        ) as mock_settings:
            mock_settings.content_log_enabled = True
            mock_settings.padtxt_length = 0
            mock_settings.pad_tokens = []
            asyncio.run(processor.process(context))

        self.assertIn("outbound_response_headers", context.metadata)
        self.assertEqual(context.metadata["outbound_response_headers"], {})


# ---------------------------------------------------------------------------
# ContentLogHook
# ---------------------------------------------------------------------------


class TestContentLogHookDeduplication(unittest.TestCase):
    """Verify ContentLogHook deduplication logic."""

    def test_skips_inbound_when_content_inbound_logged_is_true(self):
        """Hook must skip inbound logging when ContentLogProcessor already ran."""
        from app.utils.content_log_hook import content_log_hook

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["content_inbound_logged"] = True
        context.metadata["content_request_id"] = "req-123"

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry") as mock_req:
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123"))

        mock_req.assert_not_called()
        mock_err.assert_not_called()

    def test_logs_inbound_when_content_inbound_logged_is_absent(self):
        """Hook must log inbound request when ContentLogProcessor never ran."""
        from app.utils.content_log_hook import content_log_hook

        context = _make_context()
        context.response = _make_json_response()
        # No content_inbound_logged flag

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry") as mock_req:
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123"))

        mock_req.assert_called_once()
        mock_err.assert_not_called()

    def test_logs_error_when_response_is_none(self):
        """Hook must log error entry when context.response is None."""
        from app.utils.content_log_hook import content_log_hook

        context = _make_context()
        context.response = None
        # No content_inbound_logged flag

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry") as mock_req:
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123"))

        mock_req.assert_called_once()
        mock_err.assert_called_once()

    def test_logs_error_when_exception_provided(self):
        """Hook must log error entry when exception is passed."""
        from app.utils.content_log_hook import content_log_hook
        from app.core.exceptions import NoAccountsAvailableError

        context = _make_context()
        context.response = _make_json_response()
        context.metadata["content_inbound_logged"] = True
        error = NoAccountsAvailableError()

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry") as mock_req:
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123", error))

        mock_req.assert_not_called()  # inbound already logged
        mock_err.assert_called_once()

    def test_error_entry_uses_extract_error_details(self):
        """Hook must extract structured error details from AppError."""
        from app.utils.content_log_hook import content_log_hook
        from app.core.exceptions import NoAccountsAvailableError

        context = _make_context()
        context.response = None
        error = NoAccountsAvailableError()

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry"):
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123", error))

        # Verify error entry was called with correct keyword arguments
        _, kwargs = mock_err.call_args
        self.assertEqual(kwargs["request_id"], "req-123")
        self.assertEqual(kwargs["error_class"], "NoAccountsAvailableError")
        self.assertEqual(kwargs["error_code"], 503)
        self.assertEqual(kwargs["error_message"], "accountManager.noAccountsAvailable")

    def test_noop_when_content_logging_disabled(self):
        """Hook must be a no-op when content logging is disabled."""
        from app.utils.content_log_hook import content_log_hook

        context = _make_context()
        context.response = None

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = False
            with patch("app.utils.content_log_hook.log_request_entry") as mock_req:
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123"))

        mock_req.assert_not_called()
        mock_err.assert_not_called()

    def test_never_raises_exceptions(self):
        """Hook must never raise exceptions, even when logging fails."""
        from app.utils.content_log_hook import content_log_hook

        context = _make_context()
        context.response = None

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch(
                "app.utils.content_log_hook.log_request_entry",
                side_effect=RuntimeError("fail"),
            ):
                with patch(
                    "app.utils.content_log_hook.log_error_entry",
                    side_effect=RuntimeError("fail"),
                ):
                    # Must not raise
                    asyncio.run(content_log_hook(context, "req-123"))

    def test_logs_inbound_and_error_when_processor_never_ran_and_pipeline_failed(self):
        """When pipeline fails before ContentLogProcessor, hook logs both inbound and error."""
        from app.utils.content_log_hook import content_log_hook
        from app.core.exceptions import NoResponseError

        context = _make_context()
        context.response = None
        error = NoResponseError()

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry") as mock_req:
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123", error))

        self.assertEqual(mock_req.call_count, 1)
        self.assertEqual(mock_err.call_count, 1)


class TestContentLogHookGenericException(unittest.TestCase):
    """Verify ContentLogHook handles non-AppError exceptions."""

    def test_generic_exception_error_entry(self):
        """Hook must log error entry for non-AppError exceptions."""
        from app.utils.content_log_hook import content_log_hook

        context = _make_context()
        context.response = None
        error = RuntimeError("connection refused")

        with patch("app.utils.content_log_hook.settings") as mock_settings:
            mock_settings.content_log_enabled = True
            with patch("app.utils.content_log_hook.log_request_entry"):
                with patch("app.utils.content_log_hook.log_error_entry") as mock_err:
                    asyncio.run(content_log_hook(context, "req-123", error))

        _, kwargs = mock_err.call_args
        self.assertEqual(kwargs["error_class"], "RuntimeError")
        self.assertIsNone(kwargs["error_code"])  # error_code is None for non-AppError
        self.assertEqual(kwargs["error_message"], "connection refused")


if __name__ == "__main__":
    unittest.main()
