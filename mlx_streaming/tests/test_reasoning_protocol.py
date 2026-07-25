from __future__ import annotations

import pytest

from mlx_streaming.protocol.reasoning import (
    MalformedReasoningOutput,
    ReasoningParser,
)


def _collect(parser, chunks):
    reasoning = ""
    content = ""
    for chunk in chunks:
        delta = parser.feed(chunk)
        reasoning += delta.reasoning_content
        content += delta.content
    final = parser.finish()
    return reasoning + final.reasoning_content, content + final.content


@pytest.mark.parametrize("split", range(1, len("<think>")))
def test_reasoning_parser_handles_every_open_delimiter_split(split):
    text = "<think>reason</think>answer"
    assert _collect(ReasoningParser(), [text[:split], text[split:]]) == (
        "reason",
        "answer",
    )


@pytest.mark.parametrize("split", range(1, len("</think>")))
def test_reasoning_parser_handles_every_close_delimiter_split(split):
    prefix = "<think>reason"
    close = "</think>"
    assert _collect(
        ReasoningParser(),
        [prefix + close[:split], close[split:] + "answer"],
    ) == ("reason", "answer")


def test_reasoning_parser_streams_multiple_final_chunks_without_delimiters():
    parser = ReasoningParser()
    first = parser.feed("<think>why</think>an")
    assert (first.reasoning_content, first.content) == ("why", "an")
    assert parser.feed("swer").content == "swer"
    assert parser.finish().content == ""


def test_reasoning_disabled_emits_only_content():
    assert _collect(ReasoningParser(enable_thinking=False), ["plain ", "answer"]) == (
        "",
        "plain answer",
    )


def test_reasoning_parser_rejects_second_think_block():
    parser = ReasoningParser()
    parser.feed("<think>one</think>answer")
    with pytest.raises(MalformedReasoningOutput, match="second"):
        parser.feed("<think>two</think>")


def test_reasoning_parser_rejects_unterminated_reasoning():
    parser = ReasoningParser()
    parser.feed("<think>unfinished")
    with pytest.raises(MalformedReasoningOutput, match="unterminated"):
        parser.finish()
