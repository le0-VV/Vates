"""OpenAI protocol validation and model-output parsing."""

from mlx_streaming.protocol.reasoning import (
    MalformedReasoningOutput,
    ReasoningDelta,
    ReasoningParser,
)
from mlx_streaming.protocol.tools import (
    MalformedToolCallOutput,
    ToolCall,
    ToolDefinition,
    ToolValidationError,
    parse_tool_calls,
    validate_tools,
)

__all__ = [
    "MalformedReasoningOutput",
    "MalformedToolCallOutput",
    "ReasoningDelta",
    "ReasoningParser",
    "ToolCall",
    "ToolDefinition",
    "ToolValidationError",
    "parse_tool_calls",
    "validate_tools",
]
