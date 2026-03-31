import asyncio
import time
import unittest
from unittest.mock import MagicMock

from app.processors.base import BaseContext, BaseProcessor
from app.processors.pipeline import ProcessingPipeline


class SpyProcessor(BaseProcessor):
    """Records metadata snapshot when called."""

    def __init__(self):
        self.seen_metadata = None

    async def process(self, context):
        self.seen_metadata = dict(context.metadata)
        return context


class TestPipelineStartTime(unittest.TestCase):
    def _make_context(self):
        return BaseContext(original_request=MagicMock())

    def test_request_start_time_is_set_before_processors_run(self):
        spy = SpyProcessor()
        pipeline = ProcessingPipeline([spy])
        context = self._make_context()

        asyncio.run(pipeline.process(context))

        self.assertIn("request_start_time", spy.seen_metadata)
        self.assertIsInstance(spy.seen_metadata["request_start_time"], float)

    def test_request_start_time_is_monotonic(self):
        pipeline = ProcessingPipeline([])
        context = self._make_context()

        before = time.monotonic()
        result = asyncio.run(pipeline.process(context))
        after = time.monotonic()

        self.assertGreaterEqual(result.metadata["request_start_time"], before)
        self.assertLessEqual(result.metadata["request_start_time"], after)

    def test_request_start_time_set_with_empty_processor_list(self):
        pipeline = ProcessingPipeline([])
        context = self._make_context()

        result = asyncio.run(pipeline.process(context))

        self.assertIn("request_start_time", result.metadata)
        self.assertIsInstance(result.metadata["request_start_time"], float)


if __name__ == "__main__":
    unittest.main()
