import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.claude import InputMessage, MessagesAPIRequest, Role
from app.processors.claude_ai.claude_api_processor import ClaudeAPIProcessor
from app.processors.claude_ai.context import ClaudeAIContext


def _make_account(org_uuid="test-org-uuid-1234"):
    account = MagicMock()
    account.organization_uuid = org_uuid
    account.oauth_token.access_token = "fake-token"
    account.resets_at = None
    account.__enter__ = MagicMock(return_value=account)
    account.__exit__ = MagicMock(return_value=False)
    return account


def _make_context():
    request = MagicMock()
    request.headers.get = MagicMock(return_value="")
    return ClaudeAIContext(
        original_request=request,
        messages_api_request=MessagesAPIRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[InputMessage(role=Role.USER, content="Hello")],
        ),
    )


def _make_response(status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}

    async def aiter_bytes():
        yield b"data: test\n\n"

    response.aiter_bytes = aiter_bytes
    return response


def _make_session(response):
    session = MagicMock()
    session.request = AsyncMock(return_value=response)
    session.close = AsyncMock()
    return session


PATCHES = {
    "cache_service": "app.processors.claude_ai.claude_api_processor.cache_service",
    "account_manager": "app.processors.claude_ai.claude_api_processor.account_manager",
    "create_session": "app.processors.claude_ai.claude_api_processor.create_session",
    "settings": "app.processors.claude_ai.claude_api_processor.settings",
}

SKIP_PROCESSORS_EXPECTED = [
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


class TestClaudeAPIProcessorMetadata(unittest.TestCase):
    def _run_processor(
        self,
        cached_account_id=None,
        checkpoints=None,
        account=None,
        raise_no_accounts=False,
    ):
        """Helper to run the processor with configurable mocks."""
        if checkpoints is None:
            checkpoints = []
        if account is None and not raise_no_accounts:
            account = _make_account()

        response = _make_response()
        session = _make_session(response)
        context = _make_context()

        with (
            patch(PATCHES["cache_service"]) as mock_cache,
            patch(PATCHES["account_manager"]) as mock_am,
            patch(PATCHES["create_session"], return_value=session),
            patch(PATCHES["settings"]) as mock_settings,
        ):
            mock_cache.process_messages.return_value = (cached_account_id, checkpoints)
            mock_settings.max_models = []
            mock_settings.proxy_url = None
            mock_settings.request_timeout = 30
            mock_settings.claude_api_baseurl.encoded_string.return_value = (
                "https://api.claude.ai"
            )

            if raise_no_accounts:
                from app.core.exceptions import NoAccountsAvailableError

                mock_am.get_account_by_id = AsyncMock(return_value=None)
                mock_am.get_account_for_oauth = AsyncMock(
                    side_effect=NoAccountsAvailableError()
                )
            elif cached_account_id and account:
                mock_am.get_account_by_id = AsyncMock(return_value=account)
            else:
                mock_am.get_account_by_id = AsyncMock(return_value=None)
                mock_am.get_account_for_oauth = AsyncMock(return_value=account)

            processor = ClaudeAPIProcessor()
            result = asyncio.run(processor.process(context))

        return result

    def test_no_cache_read_write_in_metadata(self):
        result = self._run_processor(
            cached_account_id="cached-id-123", checkpoints=["cp1"]
        )
        self.assertNotIn("cache_read", result.metadata)
        self.assertNotIn("cache_write", result.metadata)

    def test_skip_processors_set_instead_of_stop_pipeline(self):
        result = self._run_processor()
        self.assertNotIn("stop_pipeline", result.metadata)
        self.assertEqual(result.metadata["skip_processors"], SKIP_PROCESSORS_EXPECTED)

    def test_account_id_and_oauth_path_set(self):
        account = _make_account(org_uuid="my-org-uuid")
        result = self._run_processor(account=account)
        self.assertEqual(result.metadata["account_id"], "my-org-uuid")
        self.assertTrue(result.metadata["oauth_path"])

    def test_metadata_not_set_on_no_accounts_error(self):
        result = self._run_processor(raise_no_accounts=True)
        self.assertNotIn("skip_processors", result.metadata)
        self.assertNotIn("account_id", result.metadata)
        self.assertNotIn("oauth_path", result.metadata)
        self.assertNotIn("cache_read", result.metadata)
        self.assertNotIn("cache_write", result.metadata)


if __name__ == "__main__":
    unittest.main()
