"""Tests for Task 6: Configuration documentation guidance for content logging.

Requirements: 6.2
"""

import unittest
from pathlib import Path


ENV_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / ".env.example"


class TestContentLogEnvDocs(unittest.TestCase):
    """Verify .env.example contains content logging documentation."""

    def _get_content(self):
        return ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    def test_env_example_has_content_log_enabled_setting(self):
        """CONTENT_LOG_ENABLED must be documented in .env.example."""
        content = self._get_content()
        self.assertIn("CONTENT_LOG_ENABLED", content)

    def test_env_example_has_content_log_file_path_setting(self):
        """CONTENT_LOG_FILE_PATH must be documented in .env.example."""
        content = self._get_content()
        self.assertIn("CONTENT_LOG_FILE_PATH", content)

    def test_env_example_mentions_sensitive_data(self):
        """Documentation must warn that content logging captures sensitive data."""
        content = self._get_content()
        # Accept common warning phrases
        lower = content.lower()
        self.assertTrue(
            "sensitive" in lower,
            "Expected a mention of 'sensitive' data in content logging docs",
        )

    def test_env_example_mentions_authorization_or_tokens(self):
        """Documentation must mention that authorization headers/tokens are captured."""
        content = self._get_content()
        lower = content.lower()
        self.assertTrue(
            "authorization" in lower or "token" in lower,
            "Expected mention of 'authorization' or 'token' in content logging docs",
        )

    def test_env_example_mentions_trusted_or_local_environment(self):
        """Documentation must warn about use only in trusted/local environments."""
        content = self._get_content()
        lower = content.lower()
        self.assertTrue(
            "trusted" in lower or "local" in lower,
            "Expected mention of 'trusted' or 'local' environment restriction",
        )

    def test_env_example_content_log_section_exists(self):
        """A dedicated Content Logging section must exist in .env.example."""
        content = self._get_content()
        self.assertIn("Content Log", content)


if __name__ == "__main__":
    unittest.main()
