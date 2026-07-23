# Vates OpenAI-Compatible Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the persistent Qwen3-Next Vates backend as a Chatbox-compatible OpenAI v1 server on Leonard's closed LAN.

**Architecture:** A new dependency-free `mlx_streaming.server` module validates requests, formats OpenAI responses and serves them through `ThreadingHTTPServer`, while a process-wide lock serialises access to one loaded `MLXBackend`. The CLI gains a `serve` subcommand that loads and warms the model before binding, and the existing Mac mini launcher supplies the fixed 32/16/K=3 profile for both chat and server modes.

**Tech Stack:** Python 3.13 standard library (`http.server`, `threading`, `json`), MLX 0.31.2, mlx-lm 0.31.3, pytest, uv, macOS Apple Silicon, SSH.

## Global Constraints

- Bind the live server to `0.0.0.0:8000`; discover the Mac mini LAN address at deployment time.
- Advertise exactly `qwen3-next-80b-a3b-instruct-4bit`.
- Expose `GET /health`, `GET /v1/models` and `POST /v1/chat/completions` only.
- Do not require or validate an API key; the server is for Leonard's closed LAN and must not be internet-exposed.
- Load and warm one `MLXBackend` before opening the listening socket; retain it for the process lifetime.
- Serialise inference requests through one process-wide lock while allowing health and model discovery without that lock.
- Keep deterministic greedy generation; accept `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `user`, `seed` and `stream_options` as ignored compatibility metadata.
- Reject unknown model identifiers, tools, tool-choice directives and non-text message content with an OpenAI-shaped HTTP 400 error.
- Bound `max_tokens` or `max_completion_tokens` to 1–4096 and restore the configured backend limit after every request.
- Stream Chatbox-compatible SSE role, text-delta and stop chunks followed by `data: [DONE]`.
- Cancel generation at the next token callback after a streaming client disconnects.
- Return `Connection: close`; do not add a web-framework dependency.
- Keep `EXPERT_SLOTS=32`, `POOL_SPEC_SLOTS=16`, `K=3`, K4/V3 KV quantisation, prefill chunk 2, MTP adaptive threshold 0.3 and maximum depth 3.
- Keep only canonical original model source files on `/Volumes/Leonard's RAID/Vates`; keep the prepared MTP file and expert store below `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/`.
- The launcher checks only that `/Volumes/Leonard's RAID` is mounted; do not add a hard native-extension check.
- Write the persistent log to `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-openai-server.log` and the PID below the same internal root.
- Do not push directly to `main`; update the existing protected-main pull request from `agent/qwen3-next-mlx-raid-clean`.

## File map

- Create `mlx_streaming/server.py`: validation, response formatting, serial inference and HTTP protocol handling.
- Create `mlx_streaming/tests/test_server.py`: dependency-free unit and loopback integration tests using fake backends.
- Modify `mlx_streaming/cli.py`: add `serve`, its network arguments and load-before-bind wiring without changing `chat` defaults.
- Modify `mlx_streaming/tests/test_cli_chat_repl.py`: parser and serve wiring regression coverage.
- Modify `scripts/run_mac_mini_qwen3_next.py`: allow the fixed-profile launcher to select `serve` as an explicit subcommand.
- Modify `mlx_streaming/tests/test_mac_mini_launcher.py`: preserve override protection and verify the serve command.

---

### Task 1: Implement and test the OpenAI-compatible HTTP module

**Files:**
- Create: `mlx_streaming/server.py`
- Create: `mlx_streaming/tests/test_server.py`

**Interfaces:**
- Consumes: `ChatBackend.generate(messages: list[dict], on_text: Callable[[str, int], bool]) -> GenResult`.
- Produces: `validate_request(payload: object, model_id: str, default_max_tokens: int) -> ChatRequest`.
- Produces: `cumulative_delta(previous: str, current: str) -> str`.
- Produces: `make_server(address: tuple[str, int], backend: ChatBackend, model_id: str, default_max_tokens: int) -> ThreadingHTTPServer`.
- Produces: `serve(backend: ChatBackend, host: str, port: int, model_id: str, default_max_tokens: int) -> None`.

- [ ] **Step 1: Write failing validation and formatting tests**

Create `mlx_streaming/tests/test_server.py` with these imports, helpers and tests:

```python
import http.client
import json
import threading
import time
import types

import pytest

from mlx_streaming.server import (
    ChatRequest,
    RequestError,
    cumulative_delta,
    make_server,
    validate_request,
)
from mlx_streaming.tui.backend import FakeBackend, GenResult


MODEL_ID = "qwen3-next-80b-a3b-instruct-4bit"


def _valid_payload(**overrides):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return payload


def test_validate_request_accepts_chatbox_metadata():
    request = validate_request(
        _valid_payload(
            stream=True,
            max_completion_tokens=12,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0,
            frequency_penalty=0,
            user="chatbox",
            seed=7,
            stream_options={"include_usage": True},
        ),
        MODEL_ID,
        128,
    )
    assert request == ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        max_tokens=12,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": "other"}, "unknown model"),
        ({"messages": []}, "non-empty array"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "role"),
        ({"messages": [{"role": "user", "content": []}]}, "string content"),
        ({"tools": []}, "tools are not supported"),
        ({"tool_choice": "auto"}, "tools are not supported"),
        ({"max_tokens": 0}, "between 1 and 4096"),
        ({"max_tokens": 4097}, "between 1 and 4096"),
        ({"max_tokens": True}, "integer"),
        ({"stream": "yes"}, "boolean"),
    ],
)
def test_validate_request_rejects_invalid_input(change, message):
    with pytest.raises(RequestError, match=message):
        validate_request(_valid_payload(**change), MODEL_ID, 128)


def test_validate_request_rejects_two_token_limit_fields():
    with pytest.raises(RequestError, match="only one"):
        validate_request(
            _valid_payload(max_tokens=10, max_completion_tokens=11), MODEL_ID, 128
        )


def test_cumulative_delta_emits_only_new_suffix():
    assert cumulative_delta("hel", "hello") == "lo"
    assert cumulative_delta("old", "replacement") == "replacement"
```

- [ ] **Step 2: Run the tests to confirm the expected RED state**

Run:

```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_server.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'mlx_streaming.server'`.

- [ ] **Step 3: Add HTTP integration, disconnect and serialisation tests**

Append this exact code to `mlx_streaming/tests/test_server.py`:

```python
class _ServerContext:
    def __init__(self, backend, default_max_tokens=64):
        self.server = make_server(
            ("127.0.0.1", 0), backend, MODEL_ID, default_max_tokens
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.server.server_address

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _request(address, method, path, payload=None):
    connection = http.client.HTTPConnection(*address, timeout=5)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, response.headers, raw


def test_health_and_model_discovery():
    with _ServerContext(FakeBackend()) as address:
        status, headers, raw = _request(address, "GET", "/health")
        assert status == 200
        assert headers["Connection"] == "close"
        assert json.loads(raw) == {"status": "ok", "model": MODEL_ID}

        status, _, raw = _request(address, "GET", "/v1/models")
        assert status == 200
        assert json.loads(raw) == {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "vates"}],
        }


def test_non_streaming_completion_and_usage():
    backend = FakeBackend(reply="Hello back")
    with _ServerContext(backend) as address:
        status, _, raw = _request(
            address, "POST", "/v1/chat/completions", _valid_payload()
        )
    response = json.loads(raw)
    assert status == 200
    assert response["object"] == "chat.completion"
    assert response["model"] == MODEL_ID
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello back",
    }
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 10,
        "total_tokens": 10,
    }


def test_streaming_completion_has_role_deltas_stop_and_done():
    with _ServerContext(FakeBackend(reply="Hi")) as address:
        status, headers, raw = _request(
            address,
            "POST",
            "/v1/chat/completions",
            _valid_payload(stream=True),
        )
    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    records = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    assert records[-1] == "[DONE]"
    chunks = [json.loads(record) for record in records[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
    ) == "Hi"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_invalid_request_uses_openai_error_shape():
    with _ServerContext(FakeBackend()) as address:
        status, _, raw = _request(
            address, "POST", "/v1/chat/completions", _valid_payload(model="bad")
        )
    assert status == 400
    assert json.loads(raw)["error"] == {
        "message": "unknown model 'bad'",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }


class _ObservedBackend:
    def __init__(self):
        self.args = types.SimpleNamespace(max_tokens=77)
        self.active = 0
        self.peak_active = 0
        self.limits = []
        self.lock = threading.Lock()

    def generate(self, messages, on_text):
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.limits.append(self.args.max_tokens)
        time.sleep(0.05)
        on_text("ok", 1)
        with self.lock:
            self.active -= 1
        return GenResult("ok", 1, 1.0, stopped=False)


def test_inference_is_serial_and_token_limit_is_restored():
    backend = _ObservedBackend()
    with _ServerContext(backend) as address:
        threads = [
            threading.Thread(
                target=_request,
                args=(address, "POST", "/v1/chat/completions"),
                kwargs={"payload": _valid_payload(max_tokens=limit)},
            )
            for limit in (3, 5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert backend.peak_active == 1
    assert sorted(backend.limits) == [3, 5]
    assert backend.args.max_tokens == 77


class _DisconnectBackend:
    def __init__(self):
        self.cancelled = threading.Event()

    def generate(self, messages, on_text):
        for index in range(10_000):
            if on_text("x" * (index + 1), index + 1):
                self.cancelled.set()
                return GenResult("x" * (index + 1), index + 1, 0, stopped=True)
            time.sleep(0.001)
        return GenResult("x" * 10_000, 10_000, 0, stopped=False)


def test_stream_disconnect_requests_generation_cancellation():
    backend = _DisconnectBackend()
    with _ServerContext(backend) as address:
        connection = http.client.HTTPConnection(*address, timeout=5)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(_valid_payload(stream=True)),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read(1)
        connection.close()
        assert backend.cancelled.wait(2)
```

- [ ] **Step 4: Implement the server module**

Create `mlx_streaming/server.py`:

```python
"""Dependency-free OpenAI-compatible HTTP serving for one Vates backend."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
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
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
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

    def _run(self, request: ChatRequest, on_text: Callable[[str, int], bool]):
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
        data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
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
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_server.py -v
.venv/bin/python -m compileall -q mlx_streaming/server.py
git add mlx_streaming/server.py mlx_streaming/tests/test_server.py
git diff --cached --check
git commit -S -m "feat(server): add OpenAI-compatible HTTP API"
git log -1 --show-signature
```

Expected: all server tests pass, compilation exits zero and the commit signature is good.

---

### Task 2: Wire `vates serve` and the fixed Mac mini launcher

**Files:**
- Modify: `mlx_streaming/cli.py`
- Modify: `mlx_streaming/tests/test_cli_chat_repl.py`
- Modify: `scripts/run_mac_mini_qwen3_next.py`
- Modify: `mlx_streaming/tests/test_mac_mini_launcher.py`

**Interfaces:**
- Consumes: `mlx_streaming.server.serve(backend, host, port, model_id, default_max_tokens)`.
- Produces: `cmd_serve(args) -> None`.
- Produces: parser fields `host: str`, `port: int`, `model_id: str` for the `serve` subcommand.
- Preserves: no-subcommand and option-first invocations still select `chat`.
- Produces: `build_command(["serve", ...])` with `serve` before the fixed model arguments.

- [ ] **Step 1: Write failing CLI tests**

Append to `mlx_streaming/tests/test_cli_chat_repl.py`:

```python
def test_parser_preserves_chat_defaults_and_adds_serve():
    chat = cli_mod._build_parser().parse_args(["chat"])
    assert chat.func is cli_mod.cmd_chat
    assert chat.k == 3
    assert chat.max_tokens == 4096

    serve = cli_mod._build_parser().parse_args(["serve"])
    assert serve.func is cli_mod.cmd_serve
    assert serve.host == "127.0.0.1"
    assert serve.port == 8000
    assert serve.model_id == "qwen3-next-80b-a3b-instruct-4bit"


def test_main_dispatches_explicit_serve(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod, "cmd_serve", lambda args: seen.update(vars(args)))
    parser = cli_mod._build_parser()
    monkeypatch.setattr(cli_mod, "_build_parser", lambda: parser)
    cli_mod.main(["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 9000


def test_cmd_serve_loads_before_opening_socket(monkeypatch):
    events = []

    class Backend:
        def __init__(self, args):
            events.append(("construct", args))

        def load(self, on_status):
            events.append(("load", None))
            on_status("ready")

    monkeypatch.setattr("mlx_streaming.tui.backend.MLXBackend", Backend)
    monkeypatch.setattr(
        "mlx_streaming.server.serve",
        lambda **kwargs: events.append(("serve", kwargs)),
    )
    args = types.SimpleNamespace(
        host="0.0.0.0", port=8000, model_id=MODEL_ID, max_tokens=64
    )
    cli_mod.cmd_serve(args)
    assert [event[0] for event in events] == ["construct", "load", "serve"]
```

Add near the imports:

```python
MODEL_ID = "qwen3-next-80b-a3b-instruct-4bit"
```

- [ ] **Step 2: Write failing launcher test**

Append to `mlx_streaming/tests/test_mac_mini_launcher.py`:

```python
def test_command_accepts_explicit_serve_subcommand():
    launcher = _load_launcher()
    command = launcher.build_command(["serve", "--host", "0.0.0.0", "--port", "8000"])
    assert command[:2] == [
        "/Users/leonardw/Projects/Vates/.venv/bin/vates",
        "serve",
    ]
    assert command[command.index("--expert-slots") + 1] == "32"
    assert command[command.index("--spec-slots") + 1] == "16"
    assert command[command.index("-k") + 1] == "3"
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Run focused tests to confirm the RED state**

Run:

```bash
.venv/bin/python -m pytest \
  mlx_streaming/tests/test_cli_chat_repl.py \
  mlx_streaming/tests/test_mac_mini_launcher.py -v
```

Expected: failures show that `serve`, `cmd_serve` and launcher subcommand selection do not exist.

- [ ] **Step 4: Add the serve command to `mlx_streaming/cli.py`**

Insert before `_build_parser`:

```python
def cmd_serve(args):
    """Load and warm one backend, then expose it through the OpenAI v1 API."""
    import logging

    from mlx_streaming.server import serve
    from mlx_streaming.tui.backend import MLXBackend

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backend = MLXBackend(args)
    backend.load(lambda message: print(message, file=sys.stderr, flush=True))
    print(
        f"Vates OpenAI server ready on http://{args.host}:{args.port}/v1",
        file=sys.stderr,
        flush=True,
    )
    serve(
        backend=backend,
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        default_max_tokens=args.max_tokens,
    )


def _add_serve_args(parser):
    _add_chat_args(parser)
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP listen port")
    parser.add_argument(
        "--model-id",
        default="qwen3-next-80b-a3b-instruct-4bit",
        help="model identifier advertised through the OpenAI API",
    )
    parser.set_defaults(func=cmd_serve)
```

Replace `_build_parser` with:

```python
def _build_parser():
    parser = argparse.ArgumentParser(
        prog="vates",
        description="vates:Apple Silicon 上的流式 MoE + Qwen3-Next MTP 自投机推理",
    )
    sub = parser.add_subparsers(dest="cmd")
    chat = sub.add_parser("chat", help="进入交互式多轮对话(MTP 自投机快路径)")
    _add_chat_args(chat)
    serve_parser = sub.add_parser("serve", help="启动 OpenAI v1 兼容 HTTP 服务")
    _add_serve_args(serve_parser)
    return parser
```

In `main`, replace the `subcmds` assignment with:

```python
    subcmds = {"chat", "serve"}
```

- [ ] **Step 5: Allow the deployment launcher to select `serve`**

Replace `build_command` in `scripts/run_mac_mini_qwen3_next.py` with:

```python
def build_command(extra_args: list[str]) -> list[str]:
    extra_args = list(extra_args)
    subcommand = "chat"
    if extra_args and extra_args[0] in {"chat", "serve"}:
        subcommand = extra_args.pop(0)
    _ensure_no_profile_overrides(extra_args)
    return [
        str(VATES_BIN),
        subcommand,
        "--model",
        str(MODEL_DIR),
        "--expert-dir",
        str(EXPERT_DIR),
        "--mtp-out",
        str(MTP_PATH),
        "--qn-config",
        str(MODEL_DIR / "config.json"),
        "--expert-slots",
        "32",
        "--spec-slots",
        "16",
        "-k",
        "3",
        *extra_args,
    ]
```

- [ ] **Step 6: Run focused and full tests, then commit**

Run:

```bash
.venv/bin/python -m pytest \
  mlx_streaming/tests/test_server.py \
  mlx_streaming/tests/test_cli_chat_repl.py \
  mlx_streaming/tests/test_mac_mini_launcher.py -v
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q mlx_streaming scripts
git add \
  mlx_streaming/cli.py \
  mlx_streaming/tests/test_cli_chat_repl.py \
  scripts/run_mac_mini_qwen3_next.py \
  mlx_streaming/tests/test_mac_mini_launcher.py
git diff --cached --check
git commit -S -m "feat(server): wire persistent Mac mini service"
git log -1 --show-signature
```

Expected: all tests pass, compilation exits zero and the commit signature is good.

---

### Task 3: Review, publish, deploy and verify the live service

**Files:**
- Modify: `.agents/TODO.md`
- No production source changes unless a failed acceptance check is first reproduced by a focused test.

**Interfaces:**
- Consumes: `scripts/run_mac_mini_qwen3_next.py serve --host 0.0.0.0 --port 8000`.
- Produces: PID file `~/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/openai-server.pid`.
- Produces: log `~/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-openai-server.log`.
- Produces: a Chatbox base URL made from the LAN address discovered with `ipconfig`, model `qwen3-next-80b-a3b-instruct-4bit`, with no API key.

- [ ] **Step 1: Independently review the implementation against the design**

Run:

```bash
git diff 9cab3d106df0e7c7d8481d3de19819e1b755309c -- \
  mlx_streaming/server.py \
  mlx_streaming/cli.py \
  scripts/run_mac_mini_qwen3_next.py \
  mlx_streaming/tests/test_server.py \
  mlx_streaming/tests/test_cli_chat_repl.py \
  mlx_streaming/tests/test_mac_mini_launcher.py
rg -n "TODO|TBD|pass$|NotImplemented|api.key|Authorization" \
  mlx_streaming/server.py mlx_streaming/cli.py scripts/run_mac_mini_qwen3_next.py
```

Expected: the diff implements every endpoint and failure rule in the approved design; the scan finds no placeholder or authentication requirement.

- [ ] **Step 2: Verify the complete branch and signatures**

Run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q mlx_streaming scripts
git diff --check
git status --short
git log --show-signature origin/agent/qwen3-next-mlx-raid-clean..HEAD
```

Expected: tests and compilation pass, no whitespace errors exist, only the deliberate TODO state may remain untracked, and every outgoing commit has a good signature.

- [ ] **Step 3: Push the task branch and verify the protected-main pull request**

Run outside the sandbox:

```bash
git push origin agent/qwen3-next-mlx-raid-clean
/opt/homebrew/bin/gh pr view 1 --json url,headRefName,baseRefName,state,statusCheckRollup
/opt/homebrew/bin/gh pr checks 1 --watch
```

Expected: PR 1 remains open from `agent/qwen3-next-mlx-raid-clean` to protected `main`, and all required checks pass.

- [ ] **Step 4: Preflight the live Mac mini without stopping unrelated processes**

Run:

```bash
ssh leonardw@leonards-mac-mini 'test -d "/Volumes/Leonard'"'"'s RAID" && mount | grep -F "/Volumes/Leonard'"'"'s RAID" && test -d "/Volumes/Leonard'"'"'s RAID/Vates/models/qwen3_next_80b_4bit" && test -f "$HOME/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/mtp/qn_mtp_weights.safetensors" && test -d "$HOME/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts" && ! lsof -nP -iTCP:8000 -sTCP:LISTEN'
```

Expected: the RAID and original model checks succeed, the internal MTP and expert store checks succeed, and TCP port 8000 has no listener.

- [ ] **Step 5: Start the service with a stable PID and internal log**

Run:

```bash
ssh leonardw@leonards-mac-mini '
  set -e
  internal_root="$HOME/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit"
  mkdir -p "$internal_root/logs"
  cd /Users/leonardw/Projects/Vates
  nohup .venv/bin/python scripts/run_mac_mini_qwen3_next.py \
    serve --host 0.0.0.0 --port 8000 \
    > "$internal_root/logs/qwen3-next-openai-server.log" 2>&1 < /dev/null &
  server_pid=$!
  printf "%s\n" "$server_pid" > "$internal_root/openai-server.pid"
  printf "%s\n" "$server_pid"
'
```

Expected: the command prints one PID and returns while the process continues loading and warming the model.

- [ ] **Step 6: Wait for readiness and discover the live LAN address**

Poll for up to 20 minutes, without restarting the process:

```bash
ssh leonardw@leonards-mac-mini 'pid=$(cat "$HOME/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/openai-server.pid") && kill -0 "$pid" && curl -fsS http://127.0.0.1:8000/health && ipconfig getifaddr en0'
```

Expected after warm-up: health returns `{"status":"ok","model":"qwen3-next-80b-a3b-instruct-4bit"}` and `ipconfig` prints the current LAN IPv4 address. If `en0` has no address, inspect `route get default` and query the named interface with `ipconfig getifaddr`.

- [ ] **Step 7: Verify model discovery, non-streaming and streaming from the controlling Mac**

Discover the address again into a shell variable, then run:

```bash
VATES_LAN_IP=$(ssh leonardw@leonards-mac-mini 'ipconfig getifaddr en0')
curl -fsS "http://${VATES_LAN_IP}:8000/v1/models"
curl -fsS "http://${VATES_LAN_IP}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-next-80b-a3b-instruct-4bit","messages":[{"role":"user","content":"Reply with exactly: VATES READY"}],"max_tokens":32}'
curl -fsSN "http://${VATES_LAN_IP}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-next-80b-a3b-instruct-4bit","messages":[{"role":"user","content":"Count from one to three."}],"stream":true,"max_tokens":64}'
```

Expected: model discovery lists the advertised identifier, the non-streaming response contains a non-empty assistant message, and streaming yields role/text/stop records followed by `data: [DONE]`.

- [ ] **Step 8: Check persistence, logs and memory pressure**

Run:

```bash
ssh leonardw@leonards-mac-mini 'internal_root="$HOME/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit" && pid=$(cat "$internal_root/openai-server.pid") && ps -p "$pid" -o pid=,etime=,command= && lsof -nP -a -p "$pid" -iTCP:8000 -sTCP:LISTEN && ! grep -E "Traceback|allocation error|capacity warning" "$internal_root/logs/qwen3-next-openai-server.log" && memory_pressure | sed -n "1,20p"'
```

Expected: the recorded process is alive and listening on `*:8000`, the log has no fatal markers, and memory pressure is non-critical.

- [ ] **Step 9: Complete the task ledger and report operations**

Tick every completed item in `.agents/TODO.md`. Report:

```text
Chatbox provider: OpenAI-compatible
Base URL: http://${VATES_LAN_IP}:8000/v1
Model: qwen3-next-80b-a3b-instruct-4bit
API key: none (use any placeholder only if Chatbox requires a non-empty field)
Log: ~/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-openai-server.log
PID: ~/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/openai-server.pid
Stop: kill "$(cat "$HOME/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/openai-server.pid")"
```

Expected: the user can connect Chatbox immediately and has an exact, PID-targeted shutdown command.
