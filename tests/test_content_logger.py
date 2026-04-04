"""Tests for the content logger utility module.

Task 2: Create the content logger utility module.
Subtasks: 2.1 (sink setup), 2.2 (formatting & writing), 2.3 (wiring into logger setup)
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1-3.7, 7.3, 7.4
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    enabled=True,
    file_path=None,
    rotation="50 MB",
    retention="7 days",
    compression="zip",
):
    """Return a mock Settings object with content_log_* fields set."""
    s = MagicMock()
    s.content_log_enabled = enabled
    s.content_log_file_path = file_path or "/tmp/test_content.log"
    s.content_log_file_rotation = rotation
    s.content_log_file_retention = retention
    s.content_log_file_compression = compression
    return s


# ---------------------------------------------------------------------------
# Task 2.1: Sink setup and module-level logger instance
# ---------------------------------------------------------------------------

class TestConfigureContentLoggerEnabled(unittest.TestCase):
    """Verify sink is added when content_log_enabled is True."""

    def setUp(self):
        # Reset module state before each test
        import app.utils.content_logger as mod
        mod.content_log = None

    def test_sink_added_when_enabled(self):
        """configure_content_logger() must add a loguru sink when enabled."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_sink.log")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mock_logger.bind.return_value = MagicMock()
                mod.configure_content_logger()
                mock_logger.add.assert_called_once()

    def test_sink_not_added_when_disabled(self):
        """configure_content_logger() must be a no-op when disabled."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=False)
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mod.configure_content_logger()
                mock_logger.add.assert_not_called()

    def test_content_log_bound_logger_set_when_enabled(self):
        """module-level content_log must be a bound logger after configure when enabled."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_bound.log")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                bound = MagicMock()
                mock_logger.bind.return_value = bound
                mod.configure_content_logger()
                self.assertIsNotNone(mod.content_log)
                mock_logger.bind.assert_called_once_with(content_log=True)

    def test_content_log_stays_none_when_disabled(self):
        """module-level content_log must remain None when disabled."""
        import app.utils.content_logger as mod

        mod.content_log = None
        settings = _make_settings(enabled=False)
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger"):
                mod.configure_content_logger()
                self.assertIsNone(mod.content_log)

    def test_sink_uses_enqueue_true(self):
        """The loguru sink must use enqueue=True for non-blocking writes."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_enqueue.log")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mock_logger.bind.return_value = MagicMock()
                mod.configure_content_logger()
                _, kwargs = mock_logger.add.call_args
                self.assertTrue(kwargs.get("enqueue", False))

    def test_sink_filter_accepts_content_log_records(self):
        """The sink filter must accept records with extra['content_log'] == True."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_filter.log")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mock_logger.bind.return_value = MagicMock()
                mod.configure_content_logger()
                _, kwargs = mock_logger.add.call_args
                filter_fn = kwargs.get("filter")
                self.assertIsNotNone(filter_fn)

                # A record with content_log=True must pass
                record_match = {"extra": {"content_log": True}}
                self.assertTrue(filter_fn(record_match))

                # A record without content_log must be rejected
                record_no_match = {"extra": {}}
                self.assertFalse(filter_fn(record_no_match))

                # A record with content_log=False must be rejected
                record_false = {"extra": {"content_log": False}}
                self.assertFalse(filter_fn(record_false))

    def test_sink_uses_rotation_from_settings(self):
        """The sink must use rotation from settings."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_rot.log", rotation="100 MB")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mock_logger.bind.return_value = MagicMock()
                mod.configure_content_logger()
                _, kwargs = mock_logger.add.call_args
                self.assertEqual(kwargs.get("rotation"), "100 MB")

    def test_sink_uses_retention_from_settings(self):
        """The sink must use retention from settings."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_ret.log", retention="14 days")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mock_logger.bind.return_value = MagicMock()
                mod.configure_content_logger()
                _, kwargs = mock_logger.add.call_args
                self.assertEqual(kwargs.get("retention"), "14 days")

    def test_sink_uses_compression_from_settings(self):
        """The sink must use compression from settings."""
        import app.utils.content_logger as mod

        settings = _make_settings(enabled=True, file_path="/tmp/cl_test_comp.log", compression="gz")
        with patch("app.utils.content_logger.settings", settings):
            with patch("app.utils.content_logger.logger") as mock_logger:
                mock_logger.bind.return_value = MagicMock()
                mod.configure_content_logger()
                _, kwargs = mock_logger.add.call_args
                self.assertEqual(kwargs.get("compression"), "gz")

    def test_parent_directories_created_when_enabled(self):
        """configure_content_logger() must ensure parent directories exist."""
        import app.utils.content_logger as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "nested", "dir", "content.log")
            settings = _make_settings(enabled=True, file_path=log_path)
            with patch("app.utils.content_logger.settings", settings):
                with patch("app.utils.content_logger.logger") as mock_logger:
                    mock_logger.bind.return_value = MagicMock()
                    mod.configure_content_logger()
                    # Parent directory must have been created
                    self.assertTrue(Path(log_path).parent.exists())


# ---------------------------------------------------------------------------
# Task 2.2: Log entry formatting and writing
# ---------------------------------------------------------------------------

class TestLogRequestEntry(unittest.TestCase):
    """Verify log_request_entry format and behaviour."""

    def setUp(self):
        import app.utils.content_logger as mod
        mod.content_log = None

    def _enable_content_log(self, mod):
        """Set module-level content_log to a mock that captures .info() calls."""
        mock_logger = MagicMock()
        mod.content_log = mock_logger
        return mock_logger

    def test_noop_when_disabled(self):
        """log_request_entry must be a no-op when content_log is None."""
        import app.utils.content_logger as mod
        mod.content_log = None

        # Must not raise
        mod.log_request_entry(">>> INBOUND REQUEST", "req-abc", "POST /v1/messages", {}, None)

    def test_calls_content_log_info_when_enabled(self):
        """log_request_entry must call content_log.info() when enabled."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_request_entry(
            ">>> INBOUND REQUEST",
            "req-abc123",
            "POST /v1/messages HTTP/1.1",
            {"content-type": "application/json"},
            '{"model": "claude-3"}',
        )
        mock_logger.info.assert_called_once()

    def test_entry_contains_direction_marker(self):
        """Log entry must contain the direction marker."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_request_entry(
            ">>> INBOUND REQUEST",
            "req-abc123",
            "POST /v1/messages HTTP/1.1",
            {},
            None,
        )
        entry = mock_logger.info.call_args[0][0]
        self.assertIn(">>> INBOUND REQUEST", entry)

    def test_entry_contains_request_id(self):
        """Log entry must contain the request identifier."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_request_entry(">>> OUTBOUND REQUEST", "req-xyz789", "POST /api", {}, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("req-xyz789", entry)

    def test_entry_contains_status_line(self):
        """Log entry must contain the status/request line."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages HTTP/1.1", {}, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("POST /v1/messages HTTP/1.1", entry)

    def test_entry_headers_formatted_as_key_value(self):
        """Headers must be formatted as key: value on separate lines."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_request_entry(
            ">>> INBOUND REQUEST",
            "req-1",
            "POST /v1/messages",
            {"content-type": "application/json", "authorization": "Bearer sk-ant"},
            None,
        )
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("content-type: application/json", entry)
        self.assertIn("authorization: Bearer sk-ant", entry)

    def test_entry_body_pretty_printed_for_valid_json(self):
        """Valid JSON bodies must be pretty-printed with indentation."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        body = '{"model": "claude-3", "messages": [{"role": "user", "content": "Hi"}]}'
        mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages", {}, body)
        entry = mock_logger.info.call_args[0][0]
        # Pretty-printed JSON has newlines and indentation
        self.assertIn('"model": "claude-3"', entry)
        # Should have indented structure (not single-line)
        self.assertIn("\n", entry)

    def test_entry_body_raw_for_non_json(self):
        """Non-JSON bodies must be included as raw text."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        body = "not valid json at all"
        mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages", {}, body)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("not valid json at all", entry)

    def test_entry_bytes_body_decoded_with_replacement(self):
        """Bytes bodies must be decoded with errors='replace'."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        # Include invalid UTF-8 bytes
        body = b"hello \xff\xfe world"
        mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages", {}, body)
        entry = mock_logger.info.call_args[0][0]
        # Must not raise and must contain "hello"
        self.assertIn("hello", entry)

    def test_entry_has_delimiter_lines(self):
        """Log entry must have delimiter lines of '=' characters."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages", {}, None)
        entry = mock_logger.info.call_args[0][0]
        # Must contain at least one row of '=' chars as delimiter
        self.assertIn("=" * 40, entry)

    def test_write_failure_logs_warning_not_raises(self):
        """Write failures must log a warning to main log, never raise."""
        import app.utils.content_logger as mod

        mock_logger = MagicMock()
        mock_logger.info.side_effect = Exception("disk full")
        mod.content_log = mock_logger

        with patch("app.utils.content_logger.logger") as mock_main_logger:
            # Must not raise
            mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages", {}, None)
            mock_main_logger.warning.assert_called_once()

    def test_entry_contains_no_body_section_when_body_none(self):
        """When body is None, the entry is still written but body section is empty/absent."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        # Should not raise; entry written with empty body
        mod.log_request_entry(">>> INBOUND REQUEST", "req-1", "POST /v1/messages", {}, None)
        mock_logger.info.assert_called_once()


class TestLogResponseEntry(unittest.TestCase):
    """Verify log_response_entry format and behaviour."""

    def setUp(self):
        import app.utils.content_logger as mod
        mod.content_log = None

    def _enable_content_log(self, mod):
        mock_logger = MagicMock()
        mod.content_log = mock_logger
        return mock_logger

    def test_noop_when_disabled(self):
        """log_response_entry must be a no-op when content_log is None."""
        import app.utils.content_logger as mod
        mod.content_log = None

        mod.log_response_entry("req-abc", "200 OK", {}, {}, None)

    def test_calls_content_log_info_when_enabled(self):
        """log_response_entry must call content_log.info() when enabled."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_response_entry("req-abc", "200 OK", {}, {}, "Hello world")
        mock_logger.info.assert_called_once()

    def test_entry_contains_response_direction_marker(self):
        """Response log entry must contain '<<< RESPONSE' marker."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_response_entry("req-abc", "200 OK", {}, {}, "body text")
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("<<< RESPONSE", entry)

    def test_entry_contains_request_id(self):
        """Response log entry must contain the request identifier."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_response_entry("req-xyz789", "200 OK", {}, {}, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("req-xyz789", entry)

    def test_entry_contains_status_line(self):
        """Response log entry must contain the status line."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_response_entry("req-1", "200 OK", {}, {}, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("200 OK", entry)

    def test_entry_contains_outbound_headers_section(self):
        """Response entry must include outbound response headers section."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        outbound = {"content-type": "text/event-stream", "x-request-id": "req_abc"}
        mod.log_response_entry("req-1", "200 OK", outbound, {}, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("Outbound Response Headers", entry)
        self.assertIn("content-type: text/event-stream", entry)
        self.assertIn("x-request-id: req_abc", entry)

    def test_entry_contains_inbound_headers_section(self):
        """Response entry must include inbound response headers section."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        inbound = {"transfer-encoding": "chunked", "content-type": "text/event-stream"}
        mod.log_response_entry("req-1", "200 OK", {}, inbound, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("Inbound Response Headers", entry)
        self.assertIn("transfer-encoding: chunked", entry)

    def test_entry_contains_body_section(self):
        """Response entry must include a body section."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_response_entry("req-1", "200 OK", {}, {}, "Hello world")
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("Body", entry)
        self.assertIn("Hello world", entry)

    def test_entry_has_delimiter_lines(self):
        """Response log entry must have delimiter lines."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        mod.log_response_entry("req-1", "200 OK", {}, {}, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("=" * 40, entry)

    def test_write_failure_logs_warning_not_raises(self):
        """Write failures must log a warning, never raise."""
        import app.utils.content_logger as mod

        mock_logger = MagicMock()
        mock_logger.info.side_effect = Exception("permission denied")
        mod.content_log = mock_logger

        with patch("app.utils.content_logger.logger") as mock_main_logger:
            mod.log_response_entry("req-1", "200 OK", {}, {}, "body")
            mock_main_logger.warning.assert_called_once()

    def test_both_outbound_and_inbound_headers_in_single_entry(self):
        """Both header sections must appear in a single response log entry."""
        import app.utils.content_logger as mod
        mock_logger = self._enable_content_log(mod)

        outbound = {"x-anthropic-id": "abc"}
        inbound = {"x-clove-id": "xyz"}
        mod.log_response_entry("req-1", "200 OK", outbound, inbound, None)
        entry = mock_logger.info.call_args[0][0]
        self.assertIn("x-anthropic-id: abc", entry)
        self.assertIn("x-clove-id: xyz", entry)


# ---------------------------------------------------------------------------
# Task 2.3: Wiring into configure_logger()
# ---------------------------------------------------------------------------

class TestConfigureLoggerCallsContentLogger(unittest.TestCase):
    """Verify configure_logger() calls configure_content_logger()."""

    def test_configure_logger_calls_configure_content_logger(self):
        """configure_logger() must call configure_content_logger() at the end."""
        import app.utils.logger as logger_mod
        import app.utils.content_logger as content_logger_mod

        with patch.object(content_logger_mod, "configure_content_logger") as mock_configure:
            with patch("app.utils.logger.logger"):
                with patch("app.utils.logger.settings") as mock_settings:
                    mock_settings.log_level = "INFO"
                    mock_settings.log_to_file = False
                    logger_mod.configure_logger()

        mock_configure.assert_called_once()

    def test_configure_content_logger_called_after_main_logger_setup(self):
        """configure_content_logger() must be called after the main logger is set up."""
        import app.utils.logger as logger_mod
        import app.utils.content_logger as content_logger_mod

        call_order = []

        def track_add(*args, **kwargs):
            call_order.append("logger.add")

        def track_content(*args, **kwargs):
            call_order.append("configure_content_logger")

        with patch.object(content_logger_mod, "configure_content_logger", side_effect=track_content):
            with patch("app.utils.logger.logger") as mock_logger:
                mock_logger.add.side_effect = track_add
                mock_logger.remove = MagicMock()
                with patch("app.utils.logger.settings") as mock_settings:
                    mock_settings.log_level = "INFO"
                    mock_settings.log_to_file = False
                    logger_mod.configure_logger()

        # configure_content_logger must come after at least one logger.add call
        self.assertIn("logger.add", call_order)
        self.assertIn("configure_content_logger", call_order)
        self.assertGreater(
            call_order.index("configure_content_logger"),
            call_order.index("logger.add"),
        )


if __name__ == "__main__":
    unittest.main()
