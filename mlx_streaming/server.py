"""Dependency-free OpenAI-compatible HTTP serving for one Vates backend."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from mlx_streaming.protocol.images import normalise_messages
from mlx_streaming.protocol.reasoning import ReasoningParser
from mlx_streaming.protocol.tools import (
    ToolCall,
    ToolDefinition,
    parse_tool_calls,
    validate_tools,
)
from mlx_streaming.tui.backend import ChatBackend, GenResult


LOG = logging.getLogger(__name__)
MAX_REQUEST_BODY_BYTES = 12 * 1024 * 1024
REQUEST_BODY_READ_TIMEOUT_SECONDS = 10.0
_ROLES = {"system", "user", "assistant", "tool"}
_UNSUPPORTED_REQUEST_FIELDS = (
    "functions",
    "function_call",
    "response_format",
)
_UNSUPPORTED_MESSAGE_FIELDS = ("function_call",)
_INFERENCE_LOCK = threading.Lock()


class RequestError(ValueError):
    """A client error that maps to OpenAI's invalid_request_error shape."""


class _RequestBodyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]
    stream: bool
    max_tokens: int
    enable_thinking: bool
    tools: tuple[ToolDefinition, ...]
    tool_choice: str | dict | None
    parallel_tool_calls: bool
    images: list[object]


def _clean_history(messages: list[object]) -> list[dict]:
    clean = []
    known_tool_calls = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError(f"messages[{index}] must be an object")
        for field in _UNSUPPORTED_MESSAGE_FIELDS:
            if field in message:
                raise RequestError(f"messages[{index}].{field} is not supported")
        role = message.get("role")
        if role not in _ROLES:
            raise RequestError(f"messages[{index}].role is not supported")
        content = message.get("content")
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise RequestError(f"messages[{index}].tool_call_id is required")
            if tool_call_id not in known_tool_calls:
                raise RequestError(
                    f"messages[{index}].tool_call_id must reference a known tool call"
                )
            if not isinstance(content, str):
                raise RequestError(f"messages[{index}].content must be a string")
            clean.append(
                {
                    "role": role,
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )
            continue
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = ""
        if not isinstance(content, (str, list)):
            raise RequestError(
                f"messages[{index}].content must be text or a non-empty array"
            )
        copied = {"role": role, "content": content}
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant" or not isinstance(reasoning, str):
                raise RequestError(
                    f"messages[{index}].reasoning_content is invalid"
                )
            copied["reasoning_content"] = reasoning
        calls = message.get("tool_calls")
        if calls is not None:
            if role != "assistant" or not isinstance(calls, list) or not calls:
                raise RequestError(f"messages[{index}].tool_calls is invalid")
            cleaned_calls = []
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    raise RequestError(
                        f"messages[{index}].tool_calls[{call_index}] is invalid"
                    )
                function = call.get("function")
                call_id = call.get("id")
                name = function.get("name") if isinstance(function, dict) else None
                arguments = (
                    function.get("arguments") if isinstance(function, dict) else None
                )
                if (
                    call.get("type") != "function"
                    or not isinstance(call_id, str)
                    or not call_id
                    or not isinstance(name, str)
                    or not isinstance(arguments, str)
                ):
                    raise RequestError(
                        f"messages[{index}].tool_calls[{call_index}] is invalid"
                    )
                try:
                    decoded_arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise RequestError(
                        f"messages[{index}].tool_calls[{call_index}]."
                        "function.arguments must be valid JSON"
                    ) from exc
                if not isinstance(decoded_arguments, dict):
                    raise RequestError(
                        f"messages[{index}].tool_calls[{call_index}]."
                        "function.arguments must decode to an object"
                    )
                known_tool_calls.add(call_id)
                cleaned_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": decoded_arguments,
                        },
                    }
                )
            copied["tool_calls"] = cleaned_calls
        clean.append(copied)
    return clean


def _validate_tool_choice(
    choice: object,
    tools: tuple[ToolDefinition, ...],
) -> str | dict | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice not in {"none", "auto", "required"}:
            raise RequestError("tool_choice must be none, auto or required")
        if choice == "required" and not tools:
            raise RequestError("tool_choice required needs at least one tool")
        return choice
    if not isinstance(choice, dict):
        raise RequestError("tool_choice must be a string or function object")
    function = choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if (
        choice.get("type") != "function"
        or not isinstance(name, str)
        or name not in {tool.name for tool in tools}
    ):
        raise RequestError("tool_choice names an unknown function")
    return {
        "type": "function",
        "function": {"name": name},
    }


def validate_request(
    payload: object, model_id: str, default_max_tokens: int
) -> ChatRequest:
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    if payload.get("model") != model_id:
        raise RequestError(f"unknown model {payload.get('model')!r}")
    for field in _UNSUPPORTED_REQUEST_FIELDS:
        if field in payload:
            raise RequestError(f"{field} is not supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    clean_messages = _clean_history(messages)
    try:
        normalised = normalise_messages(clean_messages)
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestError("stream must be a boolean")
    enable_thinking = payload.get("enable_thinking", True)
    if not isinstance(enable_thinking, bool):
        raise RequestError("enable_thinking must be a boolean")
    try:
        tools = validate_tools(payload.get("tools", []))
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    tool_choice = _validate_tool_choice(payload.get("tool_choice", "auto"), tools)
    parallel_tool_calls = payload.get("parallel_tool_calls", True)
    if not isinstance(parallel_tool_calls, bool):
        raise RequestError("parallel_tool_calls must be a boolean")
    token_fields = [
        field for field in ("max_tokens", "max_completion_tokens") if field in payload
    ]
    if len(token_fields) > 1:
        raise RequestError("provide only one of max_tokens and max_completion_tokens")
    max_tokens = payload[token_fields[0]] if token_fields else default_max_tokens
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise RequestError("maximum token count must be an integer")
    if not 1 <= max_tokens <= 4096:
        raise RequestError("maximum token count must be between 1 and 4096")
    return ChatRequest(
        messages=normalised.messages,
        stream=stream,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        images=normalised.images,
    )


def cumulative_delta(previous: str, current: str) -> str:
    return current[len(previous) :] if current.startswith(previous) else current


def _error(message: str, error_type: str = "invalid_request_error") -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def _chunk(completion_id: str, model_id: str, delta: dict, finish_reason=None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


@dataclass(frozen=True)
class _AssistantOutput:
    reasoning_content: str | None
    content: str | None
    tool_calls: tuple[ToolCall, ...]


def _parse_assistant_output(text: str, request: ChatRequest) -> _AssistantOutput:
    parser = ReasoningParser(enable_thinking=request.enable_thinking)
    first = parser.feed(text)
    final = parser.finish()
    reasoning = first.reasoning_content + final.reasoning_content
    content = first.content + final.content
    stripped = content.lstrip()
    forced_or_required = (
        request.tool_choice == "required"
        or isinstance(request.tool_choice, dict)
    )
    if stripped.startswith("<tool_call>") or forced_or_required:
        calls = parse_tool_calls(
            content,
            request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
        )
    else:
        calls = ()
    return _AssistantOutput(
        reasoning_content=reasoning or None,
        content=None if calls else content,
        tool_calls=calls,
    )


class VatesHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend, model_id, default_max_tokens):
        super().__init__(address, VatesRequestHandler)
        self.backend = backend
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens


class VatesRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> VatesHTTPServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format, *args):
        LOG.info("%s - %s", self.address_string(), format % args)

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": self.app.model_id})
        elif self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.app.model_id,
                            "object": "model",
                            "owned_by": "vates",
                        }
                    ],
                },
            )
        else:
            self._json(404, _error(f"unknown resource {self.path!r}"))

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, _error(f"unknown resource {self.path!r}"))
            return
        try:
            payload = json.loads(self._read_request_body())
            request = validate_request(
                payload, self.app.model_id, self.app.default_max_tokens
            )
        except _RequestBodyError as exc:
            self._json(exc.status, _error(str(exc)))
            return
        except (ValueError, json.JSONDecodeError, RequestError) as exc:
            self._json(400, _error(str(exc) or "malformed JSON"))
            return
        if request.stream:
            self._stream(request)
        else:
            self._complete(request)

    def _read_request_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise _RequestBodyError(411, "Content-Length header is required")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise _RequestBodyError(
                400,
                "Content-Length must be an integer",
            ) from exc
        if length <= 0:
            raise _RequestBodyError(
                400,
                "Content-Length must be greater than zero",
            )
        if length > MAX_REQUEST_BODY_BYTES:
            raise _RequestBodyError(
                413,
                f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
            )

        previous_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(REQUEST_BODY_READ_TIMEOUT_SECONDS)
            try:
                body = self.rfile.read(length)
            except TimeoutError as exc:
                raise _RequestBodyError(408, "request body read timed out") from exc
            except OSError as exc:
                raise _RequestBodyError(400, "request body was truncated") from exc
        finally:
            self.connection.settimeout(previous_timeout)
        if len(body) != length:
            raise _RequestBodyError(400, "request body was truncated")
        return body

    def _run(self, request: ChatRequest, on_text: Callable[[str, int], bool]) -> GenResult:
        with _INFERENCE_LOCK:
            args = getattr(self.app.backend, "args", None)
            old_limit = getattr(args, "max_tokens", None)
            if args is not None:
                args.max_tokens = request.max_tokens
            try:
                protocol_generate = getattr(
                    self.app.backend,
                    "generate_protocol",
                    None,
                )
                if callable(protocol_generate):
                    return protocol_generate(request, on_text)
                if request.images or request.tools or request.enable_thinking:
                    raise RuntimeError(
                        "backend does not support the requested protocol capabilities"
                    )
                return self.app.backend.generate(request.messages, on_text)
            finally:
                if args is not None:
                    args.max_tokens = old_limit

    def _complete(self, request: ChatRequest):
        try:
            result = self._run(request, lambda _text, _tokens: False)
            output = _parse_assistant_output(result.text, request)
        except Exception:
            LOG.exception("inference failed")
            self._json(500, _error("inference failed", "server_error"))
            return
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        message = {
            "role": "assistant",
            "content": output.content,
            "reasoning_content": output.reasoning_content,
        }
        finish_reason = "stop"
        if output.tool_calls:
            message["tool_calls"] = [
                call.as_openai_dict() for call in output.tool_calls
            ]
            finish_reason = "tool_calls"
        self._json(
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.app.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )

    def _write_sse(self, payload: dict | str) -> bool:
        data = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        try:
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _stream(self, request: ChatRequest):
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        if not self._write_sse(
            _chunk(completion_id, self.app.model_id, {"role": "assistant"})
        ):
            return
        previous = ""
        raw = ""
        parser = ReasoningParser(enable_thinking=request.enable_thinking)
        reasoning_sent = ""
        plain_stream = not request.enable_thinking and not request.tools

        def on_text(text: str, _tokens: int) -> bool:
            nonlocal previous, raw, reasoning_sent
            delta = cumulative_delta(previous, text)
            previous = text
            raw += delta
            if not delta:
                return False
            if plain_stream:
                return not self._write_sse(
                    _chunk(completion_id, self.app.model_id, {"content": delta})
                )
            parsed = parser.feed(delta)
            if parsed.reasoning_content:
                reasoning_sent += parsed.reasoning_content
                if not self._write_sse(
                    _chunk(
                        completion_id,
                        self.app.model_id,
                        {"reasoning_content": parsed.reasoning_content},
                    )
                ):
                    return True
            return False

        try:
            result = self._run(request, on_text)
            if not raw:
                raw = result.text
            output = _parse_assistant_output(raw, request)
            if not plain_stream:
                if output.reasoning_content:
                    remaining = output.reasoning_content[len(reasoning_sent) :]
                    if remaining and not self._write_sse(
                        _chunk(
                            completion_id,
                            self.app.model_id,
                            {"reasoning_content": remaining},
                        )
                    ):
                        return
                if output.tool_calls:
                    for index, call in enumerate(output.tool_calls):
                        if not self._write_sse(
                            _chunk(
                                completion_id,
                                self.app.model_id,
                                {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            **call.as_openai_dict(),
                                        }
                                    ]
                                },
                            )
                        ):
                            return
                elif output.content and not self._write_sse(
                    _chunk(
                        completion_id,
                        self.app.model_id,
                        {"content": output.content},
                    )
                ):
                    return
            finish_reason = "tool_calls" if output.tool_calls else "stop"
            if self._write_sse(
                _chunk(
                    completion_id,
                    self.app.model_id,
                    {},
                    finish_reason=finish_reason,
                )
            ):
                self._write_sse("[DONE]")
        except Exception:
            LOG.exception("streaming inference failed")
            self._write_sse(_error("inference failed", "server_error"))
        finally:
            self.close_connection = True


def make_server(address, backend, model_id, default_max_tokens):
    return VatesHTTPServer(address, backend, model_id, default_max_tokens)


def serve(backend, host, port, model_id, default_max_tokens):
    server = make_server((host, port), backend, model_id, default_max_tokens)
    try:
        server.serve_forever()
    finally:
        server.server_close()
