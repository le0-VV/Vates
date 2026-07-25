"""OpenAI protocol validation and model-output parsing."""

from mlx_streaming.protocol.images import (
    ImageLimits,
    ImageProtocolError,
    NormalisedContent,
    normalise_messages,
)
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
    "ImageLimits",
    "ImageProtocolError",
    "MalformedReasoningOutput",
    "MalformedToolCallOutput",
    "NormalisedContent",
    "ReasoningDelta",
    "ReasoningParser",
    "ToolCall",
    "ToolDefinition",
    "ToolValidationError",
    "parse_tool_calls",
    "normalise_messages",
    "validate_tools",
]
