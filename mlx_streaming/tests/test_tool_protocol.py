from __future__ import annotations

import json
import re

import pytest

from mlx_streaming.protocol.tools import (
    MalformedToolCallOutput,
    ToolValidationError,
    parse_tool_calls,
    validate_tools,
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def _call(name="get_weather", parameters=(("city", "London"),)):
    rendered = [f"<tool_call>\n<function={name}>"]
    for parameter, value in parameters:
        rendered.extend(
            [
                f"<parameter={parameter}>",
                value,
                "</parameter>",
            ]
        )
    rendered.extend(["</function>", "</tool_call>"])
    return "\n".join(rendered)


def test_validate_tools_preserves_openai_schema_for_the_prompt_template():
    definition = validate_tools([WEATHER_TOOL])[0]
    assert definition.name == "get_weather"
    assert definition.as_openai_dict() == WEATHER_TOOL


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "command", "function": {}},
        {"type": "function", "function": {"name": "bad name", "parameters": {}}},
        {
            "type": "function",
            "function": {
                "name": "x",
                "parameters": {"type": "array"},
            },
        },
    ],
)
def test_validate_tools_rejects_nonstandard_function_schemas(tool):
    with pytest.raises(ToolValidationError):
        validate_tools([tool])


def test_parse_tool_call_returns_openai_function_arguments():
    definitions = validate_tools([WEATHER_TOOL])
    calls = parse_tool_calls(_call(), definitions)

    assert len(calls) == 1
    assert re.fullmatch(r"call_[0-9a-f]{24}", calls[0].id)
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "London"}


def test_tool_choice_none_rejects_a_generated_call():
    with pytest.raises(MalformedToolCallOutput, match="tool_choice"):
        parse_tool_calls(
            _call(),
            validate_tools([WEATHER_TOOL]),
            tool_choice="none",
        )


def test_tool_choice_required_rejects_missing_call():
    with pytest.raises(MalformedToolCallOutput, match="required"):
        parse_tool_calls(
            "",
            validate_tools([WEATHER_TOOL]),
            tool_choice="required",
        )


def test_forced_tool_choice_requires_the_named_function():
    choice = {"type": "function", "function": {"name": "get_weather"}}
    with pytest.raises(MalformedToolCallOutput, match="get_weather"):
        parse_tool_calls(
            _call(name="other"),
            validate_tools([WEATHER_TOOL]),
            tool_choice=choice,
        )


def test_parallel_tool_calls_gate_multiple_calls():
    text = _call() + "\n" + _call(parameters=(("city", "Paris"),))
    definitions = validate_tools([WEATHER_TOOL])

    assert len(parse_tool_calls(text, definitions, parallel_tool_calls=True)) == 2
    with pytest.raises(MalformedToolCallOutput, match="parallel"):
        parse_tool_calls(text, definitions, parallel_tool_calls=False)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<tool_call><function=get_weather>", "malformed"),
        (
            _call(parameters=(("city", "London"), ("city", "Paris"))),
            "duplicate",
        ),
        (_call(parameters=(("days", "2"),)), "required"),
        (_call(parameters=(("city", "London"), ("days", "many"))), "days"),
    ],
)
def test_parse_tool_calls_rejects_malformed_or_invalid_output(text, message):
    with pytest.raises(MalformedToolCallOutput, match=message):
        parse_tool_calls(text, validate_tools([WEATHER_TOOL]))
