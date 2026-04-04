"""Tests for Task 5: ContentLogProcessor registered in the pipeline.

Requirements: 7.2
"""

import unittest
from app.processors.claude_ai.pipeline import ClaudeAIPipeline
from app.processors.claude_ai.content_log_processor import ContentLogProcessor
from app.processors.claude_ai.request_log_processor import RequestLogProcessor


class TestContentLogProcessorRegistration(unittest.TestCase):
    """Verify ContentLogProcessor is registered after RequestLogProcessor in the pipeline."""

    def test_content_log_processor_is_last(self):
        """ContentLogProcessor must be the last processor in the default pipeline."""
        pipeline = ClaudeAIPipeline()
        last = pipeline.processors[-1]
        self.assertIsInstance(last, ContentLogProcessor)

    def test_request_log_processor_is_second_to_last(self):
        """RequestLogProcessor must immediately precede ContentLogProcessor."""
        pipeline = ClaudeAIPipeline()
        second_to_last = pipeline.processors[-2]
        self.assertIsInstance(second_to_last, RequestLogProcessor)

    def test_content_log_processor_after_request_log_processor(self):
        """ContentLogProcessor index must be greater than RequestLogProcessor index."""
        pipeline = ClaudeAIPipeline()
        processor_names = [p.name for p in pipeline.processors]
        self.assertIn("ContentLogProcessor", processor_names)
        self.assertIn("RequestLogProcessor", processor_names)
        rlp_index = processor_names.index("RequestLogProcessor")
        clp_index = processor_names.index("ContentLogProcessor")
        self.assertGreater(clp_index, rlp_index)

    def test_content_log_processor_registered_once(self):
        """ContentLogProcessor must appear exactly once in the pipeline."""
        pipeline = ClaudeAIPipeline()
        content_log_processors = [
            p for p in pipeline.processors if isinstance(p, ContentLogProcessor)
        ]
        self.assertEqual(len(content_log_processors), 1)

    def test_total_processor_count_includes_content_log(self):
        """Default pipeline must have 14 processors (13 original + ContentLogProcessor)."""
        pipeline = ClaudeAIPipeline()
        self.assertEqual(len(pipeline.processors), 14)


if __name__ == "__main__":
    unittest.main()
