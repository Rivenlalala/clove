from loguru import logger
from starlette.datastructures import MutableHeaders

from app.core.config import settings
from app.processors.base import BaseProcessor
from app.processors.claude_ai.context import ClaudeAIContext


class HeaderManipulationProcessor(BaseProcessor):
    """Strips and adds headers on inbound requests based on configuration."""

    async def process(self, context: ClaudeAIContext) -> ClaudeAIContext:
        headers: dict[str, str] = {}
        for key, value in context.original_request.headers.items():
            headers[key.lower()] = value

        for strip_rule in settings.strip_headers:
            if ":" not in strip_rule:
                logger.warning(f"Malformed strip rule (no colon): {strip_rule}")
                continue
            key, value = strip_rule.split(":", 1)
            key_lower = key.lower()
            if key_lower in headers:
                header_value = headers[key_lower]
                if "," in header_value:
                    values = [v.strip() for v in header_value.split(",")]
                    values = [v for v in values if v != value]
                    if values:
                        headers[key_lower] = ",".join(values)
                    else:
                        del headers[key_lower]
                elif header_value == value:
                    del headers[key_lower]

        headers.update(settings.add_headers)

        context.original_request.scope["headers"] = [
            (k.encode(), v.encode()) for k, v in headers.items()
        ]
        context.original_request._headers = MutableHeaders(
            raw=[(k.encode(), v.encode()) for k, v in headers.items()]
        )

        return context
