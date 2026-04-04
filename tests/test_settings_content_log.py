"""Tests for content logging configuration fields on the Settings class.

Task 1: Add content logging configuration fields to the application settings.
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3, 5.4
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestContentLogSettingsDefaults(unittest.TestCase):
    """Verify default values for all content_log_* fields."""

    def _fresh_settings(self, env_overrides=None):
        """Import a fresh Settings instance with optional env overrides.

        Bypasses the module-level singleton by re-importing with a clean env.
        """
        import importlib
        import app.core.config as config_module

        env = {k: v for k, v in os.environ.items()}
        if env_overrides:
            env.update(env_overrides)

        # Suppress .env and config.json loading to get clean defaults
        with patch.dict(os.environ, env, clear=True):
            with patch.object(config_module.Settings, "_json_config_settings", return_value={}):
                return config_module.Settings()

    def test_content_log_enabled_defaults_false(self):
        s = self._fresh_settings()
        self.assertFalse(s.content_log_enabled)

    def test_content_log_file_path_defaults_to_logs_content_log(self):
        s = self._fresh_settings()
        self.assertEqual(s.content_log_file_path, "logs/content.log")

    def test_content_log_file_rotation_defaults_to_50_mb(self):
        s = self._fresh_settings()
        self.assertEqual(s.content_log_file_rotation, "50 MB")

    def test_content_log_file_retention_defaults_to_7_days(self):
        s = self._fresh_settings()
        self.assertEqual(s.content_log_file_retention, "7 days")

    def test_content_log_file_compression_defaults_to_zip(self):
        s = self._fresh_settings()
        self.assertEqual(s.content_log_file_compression, "zip")

    def test_content_log_fields_distinct_from_app_log_fields(self):
        """Content log fields must not shadow or interfere with app log fields."""
        s = self._fresh_settings()
        self.assertNotEqual(s.content_log_file_path, s.log_file_path)
        self.assertNotEqual(s.content_log_file_rotation, s.log_file_rotation)


class TestContentLogSettingsFromEnvVars(unittest.TestCase):
    """Verify content_log_* fields are loaded from environment variables."""

    def _fresh_settings(self, env_overrides):
        import app.core.config as config_module

        with patch.dict(os.environ, env_overrides, clear=True):
            with patch.object(config_module.Settings, "_json_config_settings", return_value={}):
                return config_module.Settings()

    def test_content_log_enabled_from_env_true(self):
        s = self._fresh_settings({"CONTENT_LOG_ENABLED": "true"})
        self.assertTrue(s.content_log_enabled)

    def test_content_log_enabled_from_env_false(self):
        s = self._fresh_settings({"CONTENT_LOG_ENABLED": "false"})
        self.assertFalse(s.content_log_enabled)

    def test_content_log_file_path_from_env(self):
        s = self._fresh_settings({"CONTENT_LOG_FILE_PATH": "/tmp/my-content.log"})
        self.assertEqual(s.content_log_file_path, "/tmp/my-content.log")

    def test_content_log_file_rotation_from_env(self):
        s = self._fresh_settings({"CONTENT_LOG_FILE_ROTATION": "100 MB"})
        self.assertEqual(s.content_log_file_rotation, "100 MB")

    def test_content_log_file_retention_from_env(self):
        s = self._fresh_settings({"CONTENT_LOG_FILE_RETENTION": "14 days"})
        self.assertEqual(s.content_log_file_retention, "14 days")

    def test_content_log_file_compression_from_env(self):
        s = self._fresh_settings({"CONTENT_LOG_FILE_COMPRESSION": "gz"})
        self.assertEqual(s.content_log_file_compression, "gz")


class TestContentLogSettingsFromDotEnv(unittest.TestCase):
    """Verify content_log_* fields can be loaded from a .env file."""

    def test_content_log_enabled_from_dotenv(self):
        """Settings must read CONTENT_LOG_ENABLED from a .env file."""
        import app.core.config as config_module

        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text("CONTENT_LOG_ENABLED=true\n", encoding="utf-8")

            # Patch model_config to point at our temp .env
            from pydantic_settings import SettingsConfigDict

            class TempSettings(config_module.Settings):
                model_config = SettingsConfigDict(
                    env_file=str(dotenv_path),
                    env_ignore_empty=True,
                    extra="ignore",
                )

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(config_module.Settings, "_json_config_settings", return_value={}):
                    with patch.object(TempSettings, "_json_config_settings", return_value={}):
                        s = TempSettings()

            self.assertTrue(s.content_log_enabled)

    def test_content_log_file_path_from_dotenv(self):
        """Settings must read CONTENT_LOG_FILE_PATH from a .env file."""
        import app.core.config as config_module

        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "CONTENT_LOG_FILE_PATH=/var/log/content.log\n", encoding="utf-8"
            )

            from pydantic_settings import SettingsConfigDict

            class TempSettings(config_module.Settings):
                model_config = SettingsConfigDict(
                    env_file=str(dotenv_path),
                    env_ignore_empty=True,
                    extra="ignore",
                )

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(TempSettings, "_json_config_settings", return_value={}):
                    s = TempSettings()

            self.assertEqual(s.content_log_file_path, "/var/log/content.log")


class TestContentLogSettingsFromJsonConfig(unittest.TestCase):
    """Verify content_log_* fields are loaded from config.json via the custom settings source."""

    def _fresh_settings_with_json(self, json_data):
        """Create Settings with a mocked config.json returning json_data."""
        import app.core.config as config_module

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                config_module.Settings, "_json_config_settings", return_value=json_data
            ):
                return config_module.Settings()

    def test_content_log_enabled_from_json_config(self):
        s = self._fresh_settings_with_json({"content_log_enabled": True})
        self.assertTrue(s.content_log_enabled)

    def test_content_log_file_path_from_json_config(self):
        s = self._fresh_settings_with_json({"content_log_file_path": "custom/path.log"})
        self.assertEqual(s.content_log_file_path, "custom/path.log")

    def test_content_log_file_rotation_from_json_config(self):
        s = self._fresh_settings_with_json({"content_log_file_rotation": "200 MB"})
        self.assertEqual(s.content_log_file_rotation, "200 MB")

    def test_content_log_file_retention_from_json_config(self):
        s = self._fresh_settings_with_json({"content_log_file_retention": "30 days"})
        self.assertEqual(s.content_log_file_retention, "30 days")

    def test_content_log_file_compression_from_json_config(self):
        s = self._fresh_settings_with_json({"content_log_file_compression": "bz2"})
        self.assertEqual(s.content_log_file_compression, "bz2")

    def test_json_config_takes_priority_over_env_var(self):
        """config.json has higher priority than environment variables (mirrors app log fields)."""
        import app.core.config as config_module

        with patch.dict(os.environ, {"CONTENT_LOG_FILE_PATH": "from_env.log"}, clear=True):
            with patch.object(
                config_module.Settings,
                "_json_config_settings",
                return_value={"content_log_file_path": "from_json.log"},
            ):
                s = config_module.Settings()

        self.assertEqual(s.content_log_file_path, "from_json.log")


class TestContentLogFieldNamingConvention(unittest.TestCase):
    """Verify field naming follows content_log_ prefix convention (mirrors log_ prefix)."""

    def setUp(self):
        import app.core.config as config_module

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(config_module.Settings, "_json_config_settings", return_value={}):
                self.settings = config_module.Settings()

    def test_all_content_log_fields_present(self):
        expected_fields = [
            "content_log_enabled",
            "content_log_file_path",
            "content_log_file_rotation",
            "content_log_file_retention",
            "content_log_file_compression",
        ]
        for field in expected_fields:
            self.assertTrue(
                hasattr(self.settings, field),
                f"Settings is missing field: {field}",
            )

    def test_content_log_enabled_is_bool(self):
        self.assertIsInstance(self.settings.content_log_enabled, bool)

    def test_content_log_file_path_is_str(self):
        self.assertIsInstance(self.settings.content_log_file_path, str)

    def test_content_log_file_rotation_is_str(self):
        self.assertIsInstance(self.settings.content_log_file_rotation, str)

    def test_content_log_file_retention_is_str(self):
        self.assertIsInstance(self.settings.content_log_file_retention, str)

    def test_content_log_file_compression_is_str(self):
        self.assertIsInstance(self.settings.content_log_file_compression, str)

    def test_existing_app_log_fields_unchanged(self):
        """Adding content_log_* fields must not alter existing log_* field defaults."""
        self.assertFalse(self.settings.log_to_file)
        self.assertEqual(self.settings.log_file_path, "logs/app.log")
        self.assertEqual(self.settings.log_file_rotation, "10 MB")
        self.assertEqual(self.settings.log_file_retention, "7 days")
        self.assertEqual(self.settings.log_file_compression, "zip")


if __name__ == "__main__":
    unittest.main()
