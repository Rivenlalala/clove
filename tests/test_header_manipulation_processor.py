import unittest
from app.core.config import settings


class TestHeaderManipulationConfig(unittest.TestCase):
    """Test that header manipulation config fields exist with correct defaults."""

    def test_strip_headers_default_is_empty_list(self):
        self.assertEqual(settings.strip_headers, [])

    def test_add_headers_default_is_empty_dict(self):
        self.assertEqual(settings.add_headers, {})


if __name__ == "__main__":
    unittest.main()
