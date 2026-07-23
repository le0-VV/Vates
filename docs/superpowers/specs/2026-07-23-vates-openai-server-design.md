# Vates OpenAI-Compatible Server Design

**Date:** 23 July 2026

**Target:** `leonardw@leonards-mac-mini`

**Model:** `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` at revision `d8a069bfa8ae87d3d468412e1034acae19b5892b`

## Objective

Expose Vates' existing persistent Qwen3-Next MLX backend through the OpenAI v1 HTTP shape expected by desktop clients such as Chatbox. The server must load and warm the model once, retain the validated 32/16/K=3 profile, support streamed and non-streamed chat completions, and remain running after the controlling SSH session disconnects.

The server is intended for Leonard's closed LAN. It binds to all interfaces on TCP port 8000 without authentication, as explicitly requested. It is not suitable for direct internet exposure.

## Scope

The first server surface is deliberately narrow:

- `GET /health` reports readiness.
- `GET /v1/models` exposes one fixed model identifier.
- `POST /v1/chat/completions` accepts OpenAI-style message arrays and returns either one JSON completion or server-sent event chunks terminated by `[DONE]`.
- Requests are processed one inference at a time because the shared MLX model, caches and expert pools are not safe or useful to execute concurrently on a 16 GB target.
- Common client metadata and sampling fields are accepted for compatibility, but the current Vates path remains deterministic greedy generation.

Embeddings, images, audio, tool execution, structured-output guarantees, batch inference and multiple models are out of scope. Unknown `/v1/*` resources return a normal OpenAI-style error response rather than silently doing something else.

## Architecture

Create `mlx_streaming/server.py` with three isolated responsibilities:

1. Request validation converts OpenAI-shaped JSON into Vates' existing `list[dict]` chat-message representation and selects a bounded output-token limit.
2. Completion formatting converts `GenResult` and cumulative `MLXBackend` callbacks into OpenAI completion objects or incremental SSE delta objects.
3. An HTTP handler owns protocol details and delegates inference through a process-wide lock to one already-loaded backend.

Add a `serve` subcommand to `mlx_streaming.cli`. It reuses the existing model, expert, MTP and pool arguments; adds `--host`, `--port` and `--model-id`; constructs `MLXBackend`, loads and warms it before opening the listening socket, then serves until terminated. The existing `chat` command and TUI behaviour remain unchanged.

No web-framework dependency is required. Python's `ThreadingHTTPServer` handles health and model-discovery requests concurrently, while an inference lock serialises calls into MLX. This keeps installation reproducible on the deployed locked environment and avoids adding a second asynchronous runtime around blocking Metal inference.

## API contract

The advertised model identifier is `qwen3-next-80b-a3b-instruct-4bit`.

`GET /health` returns HTTP 200 after model warm-up:

```json
{"status":"ok","model":"qwen3-next-80b-a3b-instruct-4bit"}
```

`GET /v1/models` returns an OpenAI-style list containing that identifier.

`POST /v1/chat/completions` requires:

- `model` equal to the advertised identifier;
- a non-empty `messages` array;
- each message to have role `system`, `user` or `assistant` and string `content`;
- optional boolean `stream`;
- optional integer `max_tokens` or `max_completion_tokens`, bounded to 1–4096 and defaulting to the server's configured maximum.

Chatbox may send fields such as `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `user`, `seed` and `stream_options`. They are accepted but do not alter the current greedy Vates decoder. A request containing tools, tool-choice directives, non-text content or an unknown model receives HTTP 400 with an OpenAI-shaped error object.

Non-streaming responses use `object: "chat.completion"`, one assistant choice, `finish_reason: "stop"`, and prompt/completion/total token usage when available. Streaming responses use `object: "chat.completion.chunk"`, first emit the assistant role, then text deltas, then a final stop chunk and `data: [DONE]`.

`MLXBackend` callbacks contain cumulative text. The server tracks the previously sent text and emits only its suffix. If the client disconnects, the callback requests cancellation so generation stops at the next token boundary.

## State and concurrency

The backend persists for the lifetime of the process. Chatbox sends the complete message history on each request. `MLXBackend` reuses its cache only when the new encoded prompt strictly extends the cached prompt; unrelated or edited conversations automatically rebuild safely.

Only one request holds the inference lock. Additional completion requests wait until the lock is available rather than racing Metal allocations; no strict fairness guarantee is exposed. Health and model-list requests do not require that lock. The response includes `Connection: close`, avoiding idle HTTP connection state in the first implementation.

The per-request token limit is applied only while the inference lock is held and restored afterwards, so one request cannot change later requests. Server configuration still fixes model paths, real/side expert capacities and speculative width at process start.

## Failure behaviour

Malformed JSON, invalid messages, unsupported features and unknown models return HTTP 400 with:

```json
{"error":{"message":"...","type":"invalid_request_error","param":null,"code":null}}
```

Unexpected inference failures return HTTP 500, log the traceback on the server, and do not fabricate a partial successful response. If an exception occurs after SSE headers are sent, the server emits an error event when the connection remains writable, then terminates the stream.

Startup remains fail-fast for required data and allocation failures. The model is loaded and warmed before the port opens; missing RAID assets, invalid expert stores or allocation failures cause the process to exit rather than advertise a ready endpoint. The launcher adds no hard native-extension check and retains Vates' existing fallback behaviour. The Mac mini launch procedure checks only that Leonard's RAID is mounted; deployment automation separately confirms that port 8000 is free before starting.

## Deployment

The service runs the reviewed branch at the fixed profile:

```text
EXPERT_SLOTS=32
POOL_SPEC_SLOTS=16
K=3
KV_QUANT=1
KV_K_BITS=4
KV_V_BITS=3
PREFILL_CHUNK=2
MTP_ADAPTIVE_DEPTH=1
MTP_CONF_TAU=0.3
MTP_DEPTH_MAX=3
```

The generic command is `vates serve --host 0.0.0.0 --port 8000` plus the validated absolute model, expert, MTP and configuration paths. It is launched with `nohup` so it survives SSH disconnects. Standard output and error go to `/Volumes/Leonard's RAID/Vates/logs/qwen3-next-openai-server.log`; the process ID is recorded below `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/` for precise shutdown.

The LAN IP is discovered from the live target at launch time rather than hard-coded. Chatbox uses `http://<mac-mini-lan-ip>:8000/v1` as its OpenAI base URL. No API key is required; if Chatbox requires a non-empty key field, an arbitrary placeholder is accepted because the server ignores `Authorization`.

## Testing and acceptance

Unit and integration tests use `FakeBackend`; they do not load MLX. Coverage includes:

- health and model discovery;
- non-streaming completion shape and usage;
- streaming role/text/final chunks and `[DONE]`;
- cumulative-text-to-delta conversion;
- message, model and token-limit validation;
- unsupported tool/non-text inputs;
- client-disconnect cancellation;
- serial inference locking;
- CLI parsing without changing `chat` defaults.

Live acceptance on the Mac mini requires:

1. RAID mounted, reviewed checkout clean, fixed 32/16/K=3 values, and port 8000 free.
2. Process survives SSH disconnection and produces a stable PID/log.
3. `/health` and `/v1/models` succeed from another machine on the LAN.
4. A non-streaming completion returns a non-empty assistant message.
5. A streaming completion emits multiple valid SSE records and `[DONE]`.
6. The log contains no traceback, allocation error or capacity warning, and system memory pressure remains non-critical.

The server is ready for Chatbox only after all six checks pass.
