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
        response.close()
        assert backend.cancelled.wait(2)
