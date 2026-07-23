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

from mlx_streaming.tui.backend import ChatBackend, GenResult


LOG = logging.getLogger(__name__)
_ROLES = {"system", "user", "assistant"}
_UNSUPPORTED_TOOL_FIELDS = {"tools", "tool_choice"}


class RequestError(ValueError):
    """A client error that maps to OpenAI's invalid_request_error shape."""


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]
    stream: bool
    max_tokens: int


def validate_request(
    payload: object, model_id: str, default_max_tokens: int
) -> ChatRequest:
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    if payload.get("model") != model_id:
        raise RequestError(f"unknown model {payload.get('model')!r}")
    if any(field in payload for field in _UNSUPPORTED_TOOL_FIELDS):
        raise RequestError("tools are not supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    clean_messages = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in _ROLES:
            raise RequestError(f"messages[{index}].role is not supported")
        content = message.get("content")
        if not isinstance(content, str):
            raise RequestError(f"messages[{index}].content must be string content")
        clean_messages.append({"role": role, "content": content})
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestError("stream must be a boolean")
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
    return ChatRequest(clean_messages, stream, max_tokens)


def cumulative_delta(previous: str, current: str) -> str:
    return current[len(previous) :] if current.startswith(previous) else current


def _error(message: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
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


class VatesHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend, model_id, default_max_tokens):
        super().__init__(address, VatesRequestHandler)
        self.backend = backend
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens
        self.inference_lock = threading.Lock()


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
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            request = validate_request(
                payload, self.app.model_id, self.app.default_max_tokens
            )
        except (ValueError, json.JSONDecodeError, RequestError) as exc:
            self._json(400, _error(str(exc) or "malformed JSON"))
            return
        if request.stream:
            self._stream(request)
        else:
            self._complete(request)

    def _run(self, request: ChatRequest, on_text: Callable[[str, int], bool]) -> GenResult:
        with self.app.inference_lock:
            args = getattr(self.app.backend, "args", None)
            old_limit = getattr(args, "max_tokens", None)
            if args is not None:
                args.max_tokens = request.max_tokens
            try:
                return self.app.backend.generate(request.messages, on_text)
            finally:
                if args is not None:
                    args.max_tokens = old_limit

    def _complete(self, request: ChatRequest):
        try:
            result = self._run(request, lambda _text, _tokens: False)
        except Exception:
            LOG.exception("inference failed")
            self._json(500, _error("inference failed"))
            return
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
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
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": result.n_tokens,
                    "total_tokens": result.n_tokens,
                },
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

        def on_text(text: str, _tokens: int) -> bool:
            nonlocal previous
            delta = cumulative_delta(previous, text)
            previous = text
            return bool(delta) and not self._write_sse(
                _chunk(completion_id, self.app.model_id, {"content": delta})
            )

        try:
            self._run(request, on_text)
            if self._write_sse(
                _chunk(completion_id, self.app.model_id, {}, finish_reason="stop")
            ):
                self._write_sse("[DONE]")
        except Exception:
            LOG.exception("streaming inference failed")
            self._write_sse(_error("inference failed"))
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
