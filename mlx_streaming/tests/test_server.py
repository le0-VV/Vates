import http.client
import json
import socket
import threading
import time
import types

import pytest

import mlx_streaming.server as server_mod
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
        "enable_thinking": False,
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
        enable_thinking=False,
        tools=(),
        tool_choice="auto",
        parallel_tool_calls=True,
        images=[],
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": "other"}, "unknown model"),
        ({"messages": []}, "non-empty array"),
        (
            {"messages": [{"role": "tool", "content": "x"}]},
            "tool_call_id",
        ),
        ({"messages": [{"role": "user", "content": []}]}, "non-empty array"),
        ({"max_tokens": 0}, "between 1 and 4096"),
        ({"max_tokens": 4097}, "between 1 and 4096"),
        ({"max_tokens": True}, "integer"),
        ({"stream": "yes"}, "boolean"),
    ],
)
def test_validate_request_rejects_invalid_input(change, message):
    with pytest.raises(RequestError, match=message):
        validate_request(_valid_payload(**change), MODEL_ID, 128)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("functions", []),
        ("function_call", "auto"),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_validate_request_rejects_unsupported_top_level_fields(field, value):
    with pytest.raises(RequestError, match="not supported"):
        validate_request(_valid_payload(**{field: value}), MODEL_ID, 128)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("function_call", {"name": "lookup"}),
    ],
)
def test_validate_request_rejects_unsupported_message_fields(field, value):
    message = {"role": "assistant", "content": "", field: value}
    with pytest.raises(RequestError, match=rf"messages\[0\]\.{field}.*not supported"):
        validate_request(_valid_payload(messages=[message]), MODEL_ID, 128)


def test_validate_request_rejects_two_token_limit_fields():
    with pytest.raises(RequestError, match="only one"):
        validate_request(
            _valid_payload(max_tokens=10, max_completion_tokens=11), MODEL_ID, 128
        )


def test_validate_request_accepts_tools_history_and_decodes_arguments():
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    request = validate_request(
        _valid_payload(
            enable_thinking=True,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            parallel_tool_calls=False,
            messages=[
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should use the tool.",
                    "tool_calls": [
                        {
                            "id": "call_0123456789abcdef01234567",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"London"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_0123456789abcdef01234567",
                    "content": '{"temperature":18}',
                },
            ],
        ),
        MODEL_ID,
        128,
    )

    assert request.enable_thinking is True
    assert request.parallel_tool_calls is False
    assert request.tools[0].name == "get_weather"
    assert request.messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "city": "London"
    }


def test_validate_request_rejects_unknown_tool_result_id():
    with pytest.raises(RequestError, match="known tool call"):
        validate_request(
            _valid_payload(
                messages=[
                    {
                        "role": "tool",
                        "tool_call_id": "call_unknown",
                        "content": "result",
                    }
                ]
            ),
            MODEL_ID,
            128,
        )


def test_cumulative_delta_emits_only_new_suffix():
    assert cumulative_delta("hel", "hello") == "lo"
    assert cumulative_delta("old", "replacement") == "replacement"


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


def _raw_request(address, request, *, shutdown_write=True, timeout=1.0):
    with socket.create_connection(address, timeout=timeout) as client:
        client.settimeout(timeout)
        client.sendall(request)
        if shutdown_write:
            client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while True:
            chunk = client.recv(64 * 1024)
            if not chunk:
                return bytes(response)
            response.extend(chunk)


def _parse_raw_response(raw):
    raw_headers, body = raw.split(b"\r\n\r\n", 1)
    lines = raw_headers.decode("iso-8859-1").splitlines()
    status = int(lines[0].split()[1])
    headers = dict(line.split(": ", 1) for line in lines[1:])
    return status, headers, body


def _assert_openai_body_error(raw, expected_status):
    status, headers, body = _parse_raw_response(raw)
    assert status == expected_status
    assert headers["Connection"] == "close"
    error = json.loads(body)["error"]
    assert isinstance(error["message"], str) and error["message"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] is None
    assert error["code"] is None


def _raw_post_headers(content_length=None):
    headers = [
        b"POST /v1/chat/completions HTTP/1.1",
        b"Host: localhost",
        b"Content-Type: application/json",
        b"Connection: close",
    ]
    if content_length is not None:
        headers.append(f"Content-Length: {content_length}".encode())
    return b"\r\n".join(headers) + b"\r\n\r\n"


def test_request_body_limits_are_explicit_and_bounded():
    assert server_mod.MAX_REQUEST_BODY_BYTES == 12 * 1024 * 1024
    assert 0 < server_mod.REQUEST_BODY_READ_TIMEOUT_SECONDS <= 30


@pytest.mark.parametrize(
    ("content_length", "expected_status"),
    [
        (None, 411),
        ("not-an-integer", 400),
        ("0", 400),
        ("-1", 400),
    ],
)
def test_invalid_request_body_lengths_use_openai_errors(
    content_length,
    expected_status,
):
    with _ServerContext(FakeBackend()) as address:
        raw = _raw_request(address, _raw_post_headers(content_length))
    _assert_openai_body_error(raw, expected_status)


def test_oversized_request_body_is_rejected_before_reading():
    with _ServerContext(FakeBackend()) as address:
        raw = _raw_request(
            address,
            _raw_post_headers(server_mod.MAX_REQUEST_BODY_BYTES + 1),
        )
    _assert_openai_body_error(raw, 413)


def test_truncated_request_body_is_rejected_even_when_received_prefix_is_json():
    body = json.dumps(_valid_payload()).encode()
    request = _raw_post_headers(len(body) + 10) + body
    with _ServerContext(FakeBackend()) as address:
        raw = _raw_request(address, request)
    _assert_openai_body_error(raw, 400)


def test_stalled_request_body_times_out_without_waiting_for_client_close(monkeypatch):
    monkeypatch.setattr(server_mod, "REQUEST_BODY_READ_TIMEOUT_SECONDS", 0.05)
    request = _raw_post_headers(10) + b"{"
    with _ServerContext(FakeBackend()) as address:
        raw = _raw_request(
            address,
            request,
            shutdown_write=False,
            timeout=0.5,
        )
    _assert_openai_body_error(raw, 408)


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


def test_non_streaming_completion_omits_unavailable_usage():
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
        "reasoning_content": None,
    }
    assert response["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in response


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


class _ProtocolBackend:
    def __init__(self, reply):
        self.reply = reply
        self.args = types.SimpleNamespace(max_tokens=77)
        self.requests = []

    def generate_protocol(self, request, on_text):
        self.requests.append(request)
        accumulated = ""
        for character in self.reply:
            accumulated += character
            if on_text(accumulated, len(accumulated)):
                return GenResult(
                    accumulated,
                    len(accumulated),
                    1.0,
                    stopped=True,
                )
        return GenResult(self.reply, len(self.reply), 1.0, stopped=False)


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def _weather_xml():
    return (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "London\n</parameter>\n</function>\n</tool_call>"
    )


def test_non_streaming_separates_reasoning_from_content():
    backend = _ProtocolBackend("<think>Because.</think>Answer")
    with _ServerContext(backend) as address:
        status, _, raw = _request(
            address,
            "POST",
            "/v1/chat/completions",
            _valid_payload(enable_thinking=True),
        )

    message = json.loads(raw)["choices"][0]["message"]
    assert status == 200
    assert message == {
        "role": "assistant",
        "content": "Answer",
        "reasoning_content": "Because.",
    }
    assert backend.requests[0].enable_thinking is True


def test_non_streaming_returns_structured_tool_calls():
    backend = _ProtocolBackend("<think>Use tool.</think>" + _weather_xml())
    with _ServerContext(backend) as address:
        status, _, raw = _request(
            address,
            "POST",
            "/v1/chat/completions",
            _valid_payload(
                enable_thinking=True,
                tools=[_weather_tool()],
                tool_choice="required",
            ),
        )

    choice = json.loads(raw)["choices"][0]
    assert status == 200
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == "Use tool."
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "London"}


def test_streaming_orders_reasoning_content_and_tool_call_deltas():
    backend = _ProtocolBackend("<think>Use tool.</think>" + _weather_xml())
    with _ServerContext(backend) as address:
        status, _, raw = _request(
            address,
            "POST",
            "/v1/chat/completions",
            _valid_payload(
                stream=True,
                enable_thinking=True,
                tools=[_weather_tool()],
            ),
        )

    records = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    chunks = [json.loads(record) for record in records[:-1]]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    reasoning_indexes = [
        index for index, delta in enumerate(deltas) if "reasoning_content" in delta
    ]
    tool_indexes = [index for index, delta in enumerate(deltas) if "tool_calls" in delta]
    assert status == 200
    assert reasoning_indexes
    assert tool_indexes
    assert max(reasoning_indexes) < min(tool_indexes)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


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


class _FailingBackend:
    def __init__(self):
        self.args = types.SimpleNamespace(max_tokens=77)
        self.seen_limit = None

    def generate(self, _messages, _on_text):
        self.seen_limit = self.args.max_tokens
        raise RuntimeError("generation failed")


def test_inference_failure_uses_server_error_type():
    with _ServerContext(_FailingBackend()) as address:
        status, _, raw = _request(
            address,
            "POST",
            "/v1/chat/completions",
            _valid_payload(max_tokens=3),
        )
    assert status == 500
    assert json.loads(raw)["error"]["type"] == "server_error"


def test_token_limit_is_restored_after_generation_raises():
    backend = _FailingBackend()
    with _ServerContext(backend) as address:
        status, _, _ = _request(
            address,
            "POST",
            "/v1/chat/completions",
            _valid_payload(max_tokens=3),
        )
    assert status == 500
    assert backend.seen_limit == 3
    assert backend.args.max_tokens == 77


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


def test_inference_is_serial_across_server_instances():
    backend = _ObservedBackend()
    with (
        _ServerContext(backend) as first_address,
        _ServerContext(backend) as second_address,
    ):
        start = threading.Barrier(3)

        def send(address, limit):
            start.wait()
            _request(
                address,
                "POST",
                "/v1/chat/completions",
                _valid_payload(max_tokens=limit),
            )

        threads = [
            threading.Thread(target=send, args=(first_address, 3)),
            threading.Thread(target=send, args=(second_address, 5)),
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()
    assert backend.peak_active == 1


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
        response.close()
        assert backend.cancelled.wait(2)
