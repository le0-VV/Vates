"""Portable CLI coverage for the OpenAI-compatible server command."""

import types

import pytest

import mlx_streaming.cli as cli_mod
import mlx_streaming.tui.backend as backend_mod


MODEL_ID = "qwen3-next-80b-a3b-instruct-4bit"


def _serve_args():
    return types.SimpleNamespace(
        host="0.0.0.0",
        port=8000,
        model_id=MODEL_ID,
        max_tokens=64,
    )


def test_parser_preserves_chat_defaults_and_adds_serve():
    chat = cli_mod._build_parser().parse_args(["chat"])
    assert chat.func is cli_mod.cmd_chat
    assert chat.k == 3
    assert chat.max_tokens == 4096

    serve = cli_mod._build_parser().parse_args(["serve"])
    assert serve.func is cli_mod.cmd_serve
    assert serve.host == "127.0.0.1"
    assert serve.port == 8000
    assert serve.model_id == MODEL_ID


def test_main_dispatches_explicit_serve(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod, "cmd_serve", lambda args: seen.update(vars(args)))
    parser = cli_mod._build_parser()
    monkeypatch.setattr(cli_mod, "_build_parser", lambda: parser)
    cli_mod.main(["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 9000


def test_cmd_serve_requests_strict_warmup_before_opening_socket(monkeypatch):
    events = []

    class Backend:
        def __init__(self, args):
            events.append(("construct", args))

        def load(self, on_status, *, strict_warmup=False):
            events.append(("load", strict_warmup))
            on_status("ready")

    monkeypatch.setattr(backend_mod, "MLXBackend", Backend)
    monkeypatch.setattr(
        "mlx_streaming.server.serve",
        lambda **kwargs: events.append(("serve", kwargs)),
    )

    cli_mod.cmd_serve(_serve_args())

    assert [event[0] for event in events] == ["construct", "load", "serve"]
    assert events[1] == ("load", True)


def test_cmd_serve_does_not_open_socket_after_strict_load_failure(monkeypatch):
    strict_requests = []
    serve_called = False

    class Backend:
        def __init__(self, _args):
            pass

        def load(self, _on_status, *, strict_warmup=False):
            strict_requests.append(strict_warmup)
            raise RuntimeError("kernel compile failed")

    def record_serve(**_kwargs):
        nonlocal serve_called
        serve_called = True

    monkeypatch.setattr(backend_mod, "MLXBackend", Backend)
    monkeypatch.setattr("mlx_streaming.server.serve", record_serve)

    with pytest.raises(RuntimeError, match="kernel compile failed"):
        cli_mod.cmd_serve(_serve_args())

    assert strict_requests == [True]
    assert serve_called is False


def test_mlx_backend_load_forwards_strict_warmup(monkeypatch):
    strict_requests = []
    engine = (object(), object(), object())

    monkeypatch.setattr(cli_mod, "_build_engine", lambda args, on_status: engine)

    def record_warmup(model, tok, drafter, args, *, strict=False):
        assert (model, tok, drafter) == engine
        strict_requests.append(strict)

    monkeypatch.setattr(cli_mod, "_warmup", record_warmup)
    backend = backend_mod.MLXBackend(types.SimpleNamespace())

    backend.load(lambda _message: None, strict_warmup=True)

    assert strict_requests == [True]
