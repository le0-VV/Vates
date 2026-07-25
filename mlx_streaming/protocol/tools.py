"""Strict OpenAI function-tool validation and Qwen XML parsing."""

from __future__ import annotations

import copy
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_CALL = re.compile(
    r"\s*<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_-]{0,63})>"
    r"(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER = re.compile(
    r"\s*<parameter=([A-Za-z_][A-Za-z0-9_-]{0,63})>"
    r"(.*?)</parameter>",
    re.DOTALL,
)


class ToolValidationError(ValueError):
    """A client supplied an invalid OpenAI tool definition."""


class MalformedToolCallOutput(ValueError):
    """The model emitted malformed or disallowed tool-call output."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str | None
    parameters: dict[str, Any]
    _source: dict[str, Any]

    def as_openai_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._source)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


def validate_tools(tools: list[dict]) -> tuple[ToolDefinition, ...]:
    if not isinstance(tools, list):
        raise ToolValidationError("tools must be an array")
    definitions = []
    names = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ToolValidationError(f"tools[{index}] must be a function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ToolValidationError(f"tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ToolValidationError(f"tools[{index}].function.name is invalid")
        if name in names:
            raise ToolValidationError(f"duplicate tool name {name!r}")
        names.add(name)
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolValidationError(
                f"tools[{index}].function.description must be a string"
            )
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ToolValidationError(
                f"tools[{index}].function.parameters must be an object schema"
            )
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, dict):
            raise ToolValidationError(
                f"tools[{index}].function.parameters.properties must be an object"
            )
        if (
            not isinstance(required, list)
            or any(not isinstance(value, str) for value in required)
            or any(value not in properties for value in required)
        ):
            raise ToolValidationError(
                f"tools[{index}].function.parameters.required is invalid"
            )
        definitions.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters=copy.deepcopy(parameters),
                _source=copy.deepcopy(tool),
            )
        )
    return tuple(definitions)


def _forced_name(
    tool_choice: str | dict | None,
    definitions: dict[str, ToolDefinition],
) -> str | None:
    if tool_choice is None or (
        isinstance(tool_choice, str)
        and tool_choice in {"none", "auto", "required"}
    ):
        return None
    if not isinstance(tool_choice, dict):
        raise ToolValidationError("tool_choice is invalid")
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if tool_choice.get("type") != "function" or name not in definitions:
        raise ToolValidationError("forced tool_choice names an unknown function")
    return name


def _decode_parameter(raw: str, schema: dict[str, Any], name: str) -> Any:
    value = raw.strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        if schema.get("type") == "string":
            decoded = value
        else:
            raise MalformedToolCallOutput(
                f"parameter {name!r} is not valid JSON for its schema"
            ) from exc
    expected = schema.get("type")
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    if expected in checks and not checks[expected](decoded):
        raise MalformedToolCallOutput(
            f"parameter {name!r} does not match type {expected!r}"
        )
    enum = schema.get("enum")
    if isinstance(enum, list) and decoded not in enum:
        raise MalformedToolCallOutput(f"parameter {name!r} is outside its enum")
    return decoded


def _parse_parameters(
    body: str,
    definition: ToolDefinition,
) -> dict[str, Any]:
    properties = definition.parameters.get("properties", {})
    additional = definition.parameters.get("additionalProperties", True)
    values = {}
    position = 0
    for match in _PARAMETER.finditer(body):
        if body[position : match.start()].strip():
            raise MalformedToolCallOutput("malformed tool parameter markup")
        name, raw = match.groups()
        if name in values:
            raise MalformedToolCallOutput(f"duplicate parameter {name!r}")
        schema = properties.get(name)
        if schema is None:
            if additional is False:
                raise MalformedToolCallOutput(f"unknown parameter {name!r}")
            schema = {}
        if not isinstance(schema, dict):
            raise ToolValidationError(
                f"schema for parameter {name!r} must be an object"
            )
        values[name] = _decode_parameter(raw, schema, name)
        position = match.end()
    if body[position:].strip():
        raise MalformedToolCallOutput("malformed tool parameter markup")
    missing = [
        name
        for name in definition.parameters.get("required", [])
        if name not in values
    ]
    if missing:
        raise MalformedToolCallOutput(
            f"required parameters are missing: {', '.join(missing)}"
        )
    return values


def parse_tool_calls(
    text: str,
    definitions: tuple[ToolDefinition, ...],
    *,
    tool_choice: str | dict | None = "auto",
    parallel_tool_calls: bool = True,
) -> tuple[ToolCall, ...]:
    by_name = {definition.name: definition for definition in definitions}
    forced = _forced_name(tool_choice, by_name)
    matches = list(_CALL.finditer(text))
    if not matches:
        if text.strip():
            raise MalformedToolCallOutput("malformed tool call output")
        if tool_choice == "required" or forced is not None:
            raise MalformedToolCallOutput("a tool call is required")
        return ()

    position = 0
    calls = []
    for match in matches:
        if text[position : match.start()].strip():
            raise MalformedToolCallOutput("malformed tool call output")
        name, body = match.groups()
        if forced is not None and name != forced:
            raise MalformedToolCallOutput(
                f"forced tool {forced!r} was not generated"
            )
        definition = by_name.get(name)
        if definition is None:
            raise MalformedToolCallOutput(f"unknown tool {name!r}")
        arguments = _parse_parameters(body, definition)
        calls.append(
            ToolCall(
                id=f"call_{secrets.token_hex(12)}",
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            )
        )
        position = match.end()
    if text[position:].strip():
        raise MalformedToolCallOutput("malformed tool call output")
    if tool_choice == "none":
        raise MalformedToolCallOutput("tool_choice forbids tool calls")
    if not parallel_tool_calls and len(calls) > 1:
        raise MalformedToolCallOutput("parallel tool calls are disabled")
    return tuple(calls)
