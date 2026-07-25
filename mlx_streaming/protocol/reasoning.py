"""Strict streaming separation of Qwen thinking and final content."""

from __future__ import annotations

from dataclasses import dataclass


_OPEN = "<think>"
_CLOSE = "</think>"


class MalformedReasoningOutput(ValueError):
    """The model emitted an invalid thinking envelope."""


@dataclass(frozen=True)
class ReasoningDelta:
    reasoning_content: str = ""
    content: str = ""


def _split_safe_prefix(value: str, delimiter: str) -> tuple[str, str]:
    keep = 0
    limit = min(len(value), len(delimiter) - 1)
    for length in range(1, limit + 1):
        if value.endswith(delimiter[:length]):
            keep = length
    if keep:
        return value[:-keep], value[-keep:]
    return value, ""


class ReasoningParser:
    """Parse one optional Qwen thinking block without exposing delimiters."""

    def __init__(self, *, enable_thinking: bool = True):
        self.enable_thinking = enable_thinking
        self._state = "before" if enable_thinking else "content"
        self._buffer = ""

    def feed(self, value: str) -> ReasoningDelta:
        if not isinstance(value, str):
            raise TypeError("reasoning delta must be a string")
        self._buffer += value
        if not self.enable_thinking:
            content, self._buffer = _split_safe_prefix(self._buffer, _OPEN)
            return ReasoningDelta(content=content)
        if self._state == "before":
            if _OPEN.startswith(self._buffer):
                return ReasoningDelta()
            if not self._buffer.startswith(_OPEN):
                raise MalformedReasoningOutput(
                    "thinking output must start with <think>"
                )
            self._buffer = self._buffer[len(_OPEN) :]
            self._state = "reasoning"
        if self._state == "reasoning":
            close_at = self._buffer.find(_CLOSE)
            if close_at < 0:
                reasoning, self._buffer = _split_safe_prefix(self._buffer, _CLOSE)
                return ReasoningDelta(reasoning_content=reasoning)
            reasoning = self._buffer[:close_at]
            self._buffer = self._buffer[close_at + len(_CLOSE) :]
            self._state = "content"
            content = self._drain_content()
            return ReasoningDelta(reasoning_content=reasoning, content=content)
        return ReasoningDelta(content=self._drain_content())

    def _drain_content(self) -> str:
        if _OPEN in self._buffer:
            raise MalformedReasoningOutput("a second <think> block is forbidden")
        content, self._buffer = _split_safe_prefix(self._buffer, _OPEN)
        return content

    def finish(self) -> ReasoningDelta:
        if self.enable_thinking:
            if self._state == "before":
                raise MalformedReasoningOutput("missing <think> block")
            if self._state == "reasoning":
                raise MalformedReasoningOutput("unterminated reasoning block")
            if _OPEN in self._buffer:
                raise MalformedReasoningOutput("a second <think> block is forbidden")
        content = self._buffer
        self._buffer = ""
        return ReasoningDelta(content=content)
