# Qwen3.5 General-Purpose MoE Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pinned `mlx-community/Qwen3.5-35B-A3B-4bit` checkpoint through a general-purpose Vates streaming-MoE runtime with OpenAI-compatible thinking, tools and image input, qualify the exact 131,072-token boundary, establish standardised intelligence baselines, and only then optimise it.

**Architecture:** A registered model adapter owns model-specific loading, language-model discovery, expert metadata, hybrid-cache accounting, prompt processing and capability parsing. The shared runtime owns file-backed experts, serial inference, bounded image transport, OpenAI response formatting, instrumentation and process policy. Qwen3.5 initially runs greedy non-MTP generation; performance features remain gated until the exact-131k and intelligence baselines are preserved.

**Tech Stack:** Python 3.13, MLX 0.31+, `mlx-lm` 0.31+, `mlx-vlm` 0.3.12, Pillow 10.3+, Requests 2.31+, pytest, safetensors, the existing Vates native MoE extension, Hugging Face Hub, macOS launch tooling.

## Global Constraints

- The model is `mlx-community/Qwen3.5-35B-A3B-4bit` at revision `1e20fd8d42056f870933bf98ca6211024744f7ec`.
- The verified source repository size is `20,411,668,782` bytes.
- The production/default context is exactly 131,072 tokens; short contexts are setup and screening only; no 262k attempt is allowed.
- Thinking is enabled by default, can be disabled per request, and is returned separately as `reasoning_content`.
- Tool use is standard OpenAI protocol only; Vates validates and emits tool calls but never executes a tool.
- Users attach or paste images through standard OpenAI `image_url` content parts; embedded data URLs and bounded public HTTPS URLs are supported.
- Only one model process may run. Failed Qwen3.5 experiments do not restore or start another model.
- Existing model files remain untouched. Canonical source files stay on RAID and derived runtime assets stay on the internal SSD.
- Deletion of existing model files or RAID-derived artefacts requires separate explicit user approval.
- MTP, expert-residency tuning, prediction-width tuning and KV quantisation are disabled until a clean exact-131k run and the initial intelligence baseline are complete.
- All tracked changes remain on `agent/qwen35-general-moe-runtime`, use signed focused commits, pass CI, and reach `main` only through a protected pull request.

---

## Boundary 1: General adapter and streaming text

### Task 1: Pin the multimodal runtime and define adapter-owned model metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `mlx_streaming/models/base.py`
- Create: `mlx_streaming/models/registry.py`
- Modify: `mlx_streaming/models/__init__.py`
- Create: `mlx_streaming/tests/test_model_registry.py`

**Interfaces:**
- Produces: `ModelDimensions`, `ExpertLayerSpec`, `HybridCacheState`, `LoadedModel` and the `ModelAdapter` protocol.
- Produces: `register_adapter(adapter)`, `adapter_for_config(config)` and `adapter_for_path(path)`.
- Consumes: no new runtime interfaces.

- [ ] **Step 1: Write registry tests that require fail-closed architecture selection**

```python
def test_registry_selects_exact_architecture(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    assert adapter_for_path(tmp_path).architecture == "qwen3_5_moe"


def test_registry_rejects_unknown_architecture(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"unknown_moe"}')
    with pytest.raises(UnsupportedArchitecture, match="unknown_moe"):
        adapter_for_path(tmp_path)
```

- [ ] **Step 2: Run the focused tests and verify the missing interface fails**

Run: `uv run pytest mlx_streaming/tests/test_model_registry.py -q`

Expected: FAIL because `mlx_streaming.models.base` and `registry` do not exist.

- [ ] **Step 3: Add exact immutable metadata and adapter protocols**

```python
@dataclass(frozen=True)
class ModelDimensions:
    architecture: str
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    shared_expert_intermediate_size: int | None
    quant_mode: str
    quant_bits: int
    quant_group_size: int
    max_context: int


@dataclass(frozen=True)
class ExpertLayerSpec:
    layer_index: int
    block: object
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    normalise_topk: bool


@dataclass
class HybridCacheState:
    entries: list[object]
    kinds: tuple[str, ...]
    logical_offset: int = 0


@dataclass(frozen=True)
class LoadedModel:
    model: object
    processor: object
    config: Mapping[str, object]


class ModelAdapter(Protocol):
    architecture: str
    def validate_config(self, config: Mapping[str, object]) -> ModelDimensions: ...
    def load(self, path: str, *, revision: str | None, lazy: bool) -> LoadedModel: ...
    def language_model(self, model: object) -> object: ...
    def layers(self, model: object) -> Sequence[object]: ...
    def expert_layers(self, model: object) -> tuple[ExpertLayerSpec, ...]: ...
    def make_cache(self, model: object) -> HybridCacheState: ...
    def prepare_inputs(
        self,
        model: object,
        processor: object,
        messages: list[dict],
        images: list[object],
        *,
        tools: list[dict] | None,
        enable_thinking: bool,
    ) -> tuple[object, dict[str, object]]: ...
    def forward(
        self,
        model: object,
        input_ids: object,
        cache: HybridCacheState,
        **kwargs: object,
    ) -> object: ...
    def cache_offsets(self, cache: HybridCacheState) -> tuple[int, ...]: ...
```

- [ ] **Step 4: Register adapters by exact `model_type` and reject duplicates and unknown values**

`adapter_for_path()` must parse `config.json` without loading weights, require a string `model_type`, and raise `UnsupportedArchitecture` rather than falling back to Qwen3-Next behaviour.

- [ ] **Step 5: Pin the dependency floor used by the checkpoint**

Set `mlx-vlm==0.3.12`, `Pillow>=10.3.0` and `requests>=2.31.0` in `pyproject.toml`, then run `uv lock`. Keep the existing MLX and `mlx-lm` floors.

- [ ] **Step 6: Run registry tests and portable CI**

Run: `uv run pytest mlx_streaming/tests/test_model_registry.py -q`

Run: `uv run pytest mlx_streaming/tests/test_ci_portable.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the adapter foundation**

```bash
git add pyproject.toml uv.lock mlx_streaming/models mlx_streaming/tests/test_model_registry.py
git commit -S -m "feat(models): add fail-closed model adapter registry"
```

### Task 2: Implement the Qwen3.5 adapter and hybrid-cache accounting

**Files:**
- Create: `mlx_streaming/models/qwen35.py`
- Modify: `mlx_streaming/models/__init__.py`
- Create: `mlx_streaming/tests/test_qwen35_adapter.py`
- Create: `mlx_streaming/tests/fakes/qwen35.py`

**Interfaces:**
- Consumes: `ModelAdapter`, `ModelDimensions`, `ExpertLayerSpec`, `HybridCacheState`, `LoadedModel`.
- Produces: `Qwen35MoeAdapter`, `forward(model, input_ids, cache, **kwargs)` and `cache_offsets(cache)`.

- [ ] **Step 1: Add a fake Qwen3.5 object tree and validation tests**

The fake tree must match the real `mlx-vlm 0.3.12` paths:

```text
outer model
└── language_model
    └── model
        └── layers[40]
            └── mlp
                ├── gate
                ├── switch_mlp
                ├── shared_expert
                └── shared_expert_gate
```

Assert the adapter reports 40 layers, 256 experts, top-k 8, hidden size 2,048, expert intermediate size 512, shared intermediate size 512, affine 4-bit/group-64 quantisation and native context 262,144.

- [ ] **Step 2: Add negative tests for every published invariant**

Parameterise mutations of `num_hidden_layers`, `num_experts`, `num_experts_per_tok`, `hidden_size`, `moe_intermediate_size`, `shared_expert_intermediate_size`, quantisation mode/bits/group size, `max_position_embeddings`, `vision_config`, and `mtp_num_hidden_layers`. Each must raise `ModelConfigurationError` containing the field name.

- [ ] **Step 3: Run the tests and verify they fail**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_adapter.py -q`

Expected: FAIL because `Qwen35MoeAdapter` is missing.

- [ ] **Step 4: Implement exact load and object-path behaviour**

```python
class Qwen35MoeAdapter:
    architecture = "qwen3_5_moe"

    def load(self, path, *, revision=None, lazy=True):
        model, processor = mlx_vlm.load(
            path, revision=revision, lazy=lazy
        )
        return LoadedModel(model=model, processor=processor, config=load_config(path))

    def language_model(self, model):
        return model.language_model

    def layers(self, model):
        return model.language_model.model.layers
```

`expert_layers()` must derive dimensions from each block’s real `switch_mlp.gate_proj` and verify that all 40 layers agree with the validated configuration. It must set `normalise_topk=True`, because `mlx-vlm` Qwen3.5 always renormalises selected router probabilities.

- [ ] **Step 5: Implement hybrid cache creation and checked advancement**

Use `model.language_model.make_cache()`, classify `ArraysCache` entries as `linear` and `KVCache` entries as `full_attention`, and require the exact repeating sequence `linear, linear, linear, full_attention` ten times.

```python
def forward(self, model, input_ids, cache, **kwargs):
    count = int(input_ids.shape[1])
    output = model.language_model(
        input_ids, cache=cache.entries, **kwargs
    )
    expected = cache.logical_offset + count
    for index, (kind, entry) in enumerate(zip(cache.kinds, cache.entries)):
        if kind == "full_attention" and int(entry.offset) != expected:
            raise CacheOffsetError(index, expected, int(entry.offset))
    cache.logical_offset = expected
    return output


def cache_offsets(self, cache):
    return tuple(
        int(entry.offset) if kind == "full_attention" else cache.logical_offset
        for kind, entry in zip(cache.kinds, cache.entries)
    )
```

- [ ] **Step 6: Test text and image embedding delegation**

Assert text-only `prepare_inputs()` delegates to `model.get_input_embeddings(input_ids, None, mask=mask)` and image input delegates with `pixel_values` and `image_grid_thw` produced by the processor. No server code may reach into `vision_tower` directly.

- [ ] **Step 7: Run the adapter tests**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_adapter.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the Qwen3.5 adapter**

```bash
git add mlx_streaming/models mlx_streaming/tests/test_qwen35_adapter.py mlx_streaming/tests/fakes/qwen35.py
git commit -S -m "feat(qwen35): add model and hybrid-cache adapter"
```

### Task 3: Make expert preparation and patching model-derived

**Files:**
- Create: `mlx_streaming/prep/expert_manifest.py`
- Modify: `mlx_streaming/prep/split_experts.py`
- Modify: `mlx_streaming/prep/pack_blob_from_experts.py`
- Modify: `mlx_streaming/core/prefetch/patch.py`
- Modify: `mlx_streaming/core/moe/block.py`
- Modify: `mlx_streaming/model_builder.py`
- Create: `mlx_streaming/tests/test_expert_manifest.py`
- Create: `mlx_streaming/tests/test_qwen35_file_streaming.py`
- Modify: `mlx_streaming/tests/test_file_streaming.py`

**Interfaces:**
- Consumes: `ModelAdapter.expert_layers()`, `ModelDimensions`.
- Produces: `ExpertStoreManifest`, `write_manifest()`, `read_manifest()`, and adapter-driven `patch_model_filebacked(model, adapter, store, manifest)`.

- [ ] **Step 1: Write manifest round-trip and rejection tests**

Require schema version 1 and these exact fields: source repository, source revision, architecture, layer indices, expert count, top-k, hidden size, expert intermediate size, projection names, per-projection bits, group size, quantisation mode, file naming and per-file byte hashes.

Assert any mismatch between manifest and adapter dimensions raises `ExpertManifestError` before a model layer is patched.

- [ ] **Step 2: Write a tiny fake-Qwen3.5 streaming equivalence test**

Construct two small MoE layers with a shared expert. Compare the resident reference against the file-backed block, then assert:

```python
assert mx.allclose(reference, streamed, atol=1e-4).item()
assert streamed_block.shared_expert is reference_block.shared_expert
assert streamed_block.switch_mlp is not present
```

- [ ] **Step 3: Run the focused tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_expert_manifest.py mlx_streaming/tests/test_qwen35_file_streaming.py -q`

Expected: FAIL on missing manifest and adapter-driven patching.

- [ ] **Step 4: Replace loader-specific layer scans with adapter metadata**

`split_model()` must receive an adapter, call `adapter.load(..., lazy=True)`, iterate `adapter.expert_layers()`, extract only `switch_mlp` routed projections, and preserve `shared_expert` in the main model. It must write each file atomically through a same-directory temporary name and record its SHA-256.

- [ ] **Step 5: Validate the manifest in both per-expert and blob paths**

`pack_blob_from_experts.py` must use the manifest’s actual 256-expert count and 40 layer indices. `_make_blob_source()` and `_attach_blob_source()` must receive the count from `ExpertStoreManifest`; remove the legacy 512-expert fallback from production assembly.

- [ ] **Step 6: Generalise file-backed patching**

Patch the object returned by `adapter.layers(model)`. Construct each `FileStreamingMoeBlock` from its corresponding `ExpertLayerSpec`, including shared expert references. Require `patched == manifest.num_layers`; a partial patch is a startup error.

- [ ] **Step 7: Preserve Qwen3-Next compatibility tests**

Keep the existing tiny Qwen3-MoE tests passing through a small compatibility adapter used only by those tests. Do not add architecture branches to `FileStreamingMoeBlock`.

- [ ] **Step 8: Run all expert and patch tests**

Run: `uv run pytest mlx_streaming/tests/test_expert_manifest.py mlx_streaming/tests/test_qwen35_file_streaming.py mlx_streaming/tests/test_file_streaming.py mlx_streaming/tests/test_patch_model.py mlx_streaming/tests/test_stream_blob_equiv.py -q`

Expected: PASS.

- [ ] **Step 9: Commit model-derived expert streaming**

```bash
git add mlx_streaming/prep mlx_streaming/core/prefetch/patch.py mlx_streaming/core/moe/block.py mlx_streaming/model_builder.py mlx_streaming/tests
git commit -S -m "feat(moe): derive expert streaming from model adapters"
```

### Task 4: Add a non-speculative general generation engine

**Files:**
- Create: `mlx_streaming/runtime/engine.py`
- Create: `mlx_streaming/runtime/generate.py`
- Modify: `mlx_streaming/tui/backend.py`
- Modify: `mlx_streaming/cli.py`
- Create: `mlx_streaming/tests/test_general_engine.py`
- Modify: `mlx_streaming/tests/test_tui_backend.py`

**Interfaces:**
- Consumes: `LoadedModel`, `ModelAdapter.forward()`, `HybridCacheState`.
- Produces: `GenerationRequest`, `GenerationDelta`, `GenerationResult`, `GeneralModelEngine.generate()`.

- [ ] **Step 1: Write greedy generation and serial-state tests with a fake adapter**

Assert chunked prefill advances the logical cache by the exact number of consumed tokens, decode emits deterministic token IDs, EOS is excluded, callbacks receive incremental decoded text, and request state cannot leak into the next request.

- [ ] **Step 2: Run the engine tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_general_engine.py -q`

Expected: FAIL because the general engine is missing.

- [ ] **Step 3: Implement non-MTP chunked prefill and decode**

`GeneralModelEngine` must:

- format messages through the adapter;
- prepare embeddings once for text or image input;
- process prompt chunks without retaining logits except the last required row;
- use greedy `mx.argmax` decoding for the correctness baseline;
- call `adapter.forward()` for every state mutation;
- expose prompt tokens, generated tokens, prefill seconds, decode seconds, peak MLX bytes and final cache offsets; and
- clear only request-owned state on failure.

Use a configurable setup `prefill_chunk_size`; the initial Qwen3.5 screening value is 64 tokens and the first 131k candidate is chosen from correctness-preserving short screens.

- [ ] **Step 4: Route CLI and HTTP backends through an engine protocol**

Replace assumptions that every engine has a Qwen3-Next MTP drafter. Keep the old MTP engine selectable for its existing adapter, but set Qwen3.5 to `GeneralModelEngine` with MTP disabled.

- [ ] **Step 5: Run engine, backend and portable tests**

Run: `uv run pytest mlx_streaming/tests/test_general_engine.py mlx_streaming/tests/test_tui_backend.py mlx_streaming/tests/test_ci_portable.py -q`

Expected: PASS.

- [ ] **Step 6: Commit general text generation**

```bash
git add mlx_streaming/runtime mlx_streaming/tui/backend.py mlx_streaming/cli.py mlx_streaming/tests
git commit -S -m "feat(runtime): add non-speculative adapter generation"
```

## Boundary 2: OpenAI reasoning, tools and images

### Task 5: Parse thinking and Qwen tool calls as explicit protocol events

**Files:**
- Create: `mlx_streaming/protocol/__init__.py`
- Create: `mlx_streaming/protocol/reasoning.py`
- Create: `mlx_streaming/protocol/tools.py`
- Create: `mlx_streaming/tests/test_reasoning_protocol.py`
- Create: `mlx_streaming/tests/test_tool_protocol.py`

**Interfaces:**
- Produces: `ReasoningParser.feed(delta)`, `ReasoningParser.finish()`, `ReasoningDelta`.
- Produces: `ToolDefinition`, `ToolCall`, `validate_tools()`, `parse_tool_calls(text)`.
- Consumes: decoded text deltas from `GeneralModelEngine`.

- [ ] **Step 1: Write reasoning parser tests across every delimiter split**

For every split position in `<think>` and `</think>`, feed the fragments independently and require identical reasoning/final output. Test empty reasoning for `enable_thinking=False`, multiple final chunks, forbidden second `<think>`, and unterminated reasoning.

- [ ] **Step 2: Write tool schema and parser tests**

Cover standard OpenAI function tools, `tool_choice` values `none`, `auto`, `required`, and a forced function object, multiple calls when `parallel_tool_calls=True`, call IDs shaped `call_<24 hex characters>`, malformed XML, duplicate parameters, missing required parameters and invalid parameter JSON values.

The parser input format is the checkpoint’s pinned template:

```text
<tool_call>
<function=get_weather>
<parameter=city>
London
</parameter>
</function>
</tool_call>
```

- [ ] **Step 3: Run focused tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_reasoning_protocol.py mlx_streaming/tests/test_tool_protocol.py -q`

Expected: FAIL because the protocol parsers are missing.

- [ ] **Step 4: Implement a streaming reasoning state machine**

The parser must retain at most the longest delimiter minus one character between calls, never expose delimiter text, emit `reasoning_content` before `content`, and raise `MalformedReasoningOutput` from `finish()` if the model ends inside reasoning or re-enters reasoning.

- [ ] **Step 5: Implement strict tool validation and parsing**

Vates must not import or call tool implementations. Preserve JSON schema dictionaries for `processor.apply_chat_template(..., tools=tools)` and turn only syntactically valid generated calls into `ToolCall` values. Invalid model output becomes `MalformedToolCallOutput`, not a guessed plain-text answer.

- [ ] **Step 6: Run the protocol tests**

Run: `uv run pytest mlx_streaming/tests/test_reasoning_protocol.py mlx_streaming/tests/test_tool_protocol.py -q`

Expected: PASS.

- [ ] **Step 7: Commit reasoning and tool protocol primitives**

```bash
git add mlx_streaming/protocol mlx_streaming/tests/test_reasoning_protocol.py mlx_streaming/tests/test_tool_protocol.py
git commit -S -m "feat(protocol): parse reasoning and tool calls"
```

### Task 6: Normalise bounded OpenAI image attachments

**Files:**
- Create: `mlx_streaming/protocol/images.py`
- Create: `mlx_streaming/tests/test_image_protocol.py`

**Interfaces:**
- Produces: `ImageLimits`, `NormalisedContent`, `normalise_messages(messages, limits, fetcher)`.
- Consumes: OpenAI message content strings or arrays.

- [ ] **Step 1: Write data-URL, HTTPS and content-order tests**

Accept `data:image/png;base64,...`, `data:image/jpeg;base64,...` and an injected HTTPS fetcher. Preserve the order of text and image parts passed to the official Qwen processor. Reject images in system messages, malformed base64, unsupported MIME types, empty content arrays and unknown part types.

The accepted OpenAI shape is
`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`;
plain text parts use `{"type":"text","text":"..."}`.

- [ ] **Step 2: Write security and resource-bound tests**

Use these initial transport limits:

```python
ImageLimits(
    max_images=4,
    max_encoded_bytes=11 * 1024 * 1024,
    max_decoded_bytes=8 * 1024 * 1024,
    max_pixels=16_777_216,
    connect_timeout_seconds=5.0,
    read_timeout_seconds=10.0,
    max_redirects=3,
)
```

Reject `http`, `file`, credentials in URLs, fragments, localhost, loopback, private, link-local, multicast, reserved and non-global resolved addresses. Revalidate every redirect target and final target. Reject decompression bombs and images whose decoded dimensions exceed the pixel cap.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_image_protocol.py -q`

Expected: FAIL because image normalisation is missing.

- [ ] **Step 4: Implement bounded decoding and public-HTTPS fetching**

Decode data URLs with `base64.b64decode(..., validate=True)`. Fetch HTTPS with streaming enabled, explicit connect/read timeouts, a byte counter that aborts above the limit, redirect handling disabled in the underlying request, and explicit revalidation before each followed redirect. Verify content with Pillow, call `image.load()`, then return an RGB `PIL.Image.Image`.

- [ ] **Step 5: Run image protocol tests**

Run: `uv run pytest mlx_streaming/tests/test_image_protocol.py -q`

Expected: PASS.

- [ ] **Step 6: Commit bounded image attachments**

```bash
git add mlx_streaming/protocol/images.py mlx_streaming/tests/test_image_protocol.py
git commit -S -m "feat(protocol): accept bounded OpenAI image parts"
```

### Task 7: Expose complete OpenAI chat request and response semantics

**Files:**
- Modify: `mlx_streaming/server.py`
- Modify: `mlx_streaming/runtime/engine.py`
- Modify: `mlx_streaming/models/qwen35.py`
- Modify: `mlx_streaming/tui/backend.py`
- Modify: `mlx_streaming/tests/test_server.py`
- Create: `mlx_streaming/tests/test_qwen35_prompt_protocol.py`

**Interfaces:**
- Consumes: reasoning, tool and image protocol values.
- Produces: OpenAI chat responses with `content`, `reasoning_content` and `tool_calls`.

- [ ] **Step 1: Replace rejection tests with complete request-validation tests**

`ChatRequest` must contain:

```python
@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]
    stream: bool
    max_tokens: int
    enable_thinking: bool
    tools: tuple[ToolDefinition, ...]
    tool_choice: str | dict | None
    parallel_tool_calls: bool
```

Thinking defaults to `True`. An explicit boolean `enable_thinking` disables it. Tool result messages require `role="tool"`, a known `tool_call_id`, and string content. Assistant history may contain `reasoning_content` and `tool_calls`.
Before applying the Qwen template, decode each historical OpenAI
`tool_calls[].function.arguments` JSON string into the mapping expected by the
pinned template; reject invalid JSON instead of passing a string to Jinja.

- [ ] **Step 2: Add prompt-template spy tests**

Require the adapter to call:

```python
processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    tools=tools or None,
    enable_thinking=enable_thinking,
)
```

Pass normalised image objects to the processor and never flatten multimodal arrays into strings.

- [ ] **Step 3: Add non-streaming and streaming response tests**

Non-streaming assistant messages include `reasoning_content` separately and either final `content` or `tool_calls`. Streaming emits role, reasoning deltas, content deltas and indexed tool-call deltas in that order. Use finish reason `tool_calls` when calls are present and `stop` otherwise.

- [ ] **Step 4: Run tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_server.py mlx_streaming/tests/test_qwen35_prompt_protocol.py -q`

Expected: FAIL under the current text-only server.

- [ ] **Step 5: Implement request validation and increase only the bounded body cap**

Set `MAX_REQUEST_BODY_BYTES = 12 * 1024 * 1024` so one 8 MiB data image plus JSON/base64 overhead fits. Keep the 10-second body-read timeout. Return deterministic OpenAI-shaped 400/408/413 errors for transport, schema and protocol failures.

- [ ] **Step 6: Connect protocol events to the engine and response writer**

Do not derive streaming deltas from cumulative mixed text. Feed decoded deltas through `ReasoningParser`, buffer possible tool XML only after reasoning closes, and serialise structured calls after validation. Preserve the global one-inference lock.

- [ ] **Step 7: Run server and complete portable tests**

Run: `uv run pytest mlx_streaming/tests/test_server.py mlx_streaming/tests/test_qwen35_prompt_protocol.py -q`

Run: `uv run pytest mlx_streaming/tests/test_ci_portable.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the OpenAI capability surface**

```bash
git add mlx_streaming/server.py mlx_streaming/runtime/engine.py mlx_streaming/models/qwen35.py mlx_streaming/tui/backend.py mlx_streaming/tests
git commit -S -m "feat(server): support thinking tools and images"
```

## Boundary 3: Mac mini preparation and exact-131k qualification

### Task 8: Add the Qwen3.5 launcher and one-process storage guards

**Files:**
- Create: `scripts/run_mac_mini_qwen35.py`
- Create: `mlx_streaming/tests/test_mac_mini_qwen35_launcher.py`
- Modify: `mlx_streaming/cli.py`

**Interfaces:**
- Consumes: the general engine and Qwen3.5 adapter.
- Produces: a fail-fast Mac mini launcher for the pinned model and internal derived assets.

- [ ] **Step 1: Write launcher-policy tests**

Require:

```text
source:
/Volumes/Leonard's RAID/Vates/models/Qwen3.5-35B-A3B-4bit

runtime:
/Users/leonardw/Library/Application Support/Vates/qwen3.5-35b-a3b-4bit

model id:
qwen3.5-35b-a3b-4bit

context:
131072
```

Assert the launcher rejects an unmounted RAID, missing revision marker, missing manifest, revision mismatch, occupied port 8000 and another recognised Vates model PID. Assert MTP and KV quantisation are absent from the initial environment.

- [ ] **Step 2: Run launcher tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_mac_mini_qwen35_launcher.py -q`

Expected: FAIL because the launcher is missing.

- [ ] **Step 3: Implement explicit source, runtime and PID checks**

The launcher must verify the source revision file contains the exact 40-character revision, validate the expert manifest before `execve`, write one PID file only after acquiring an exclusive lock, and remove it on normal or signal-driven shutdown. It must never terminate another process automatically.

- [ ] **Step 4: Add general-engine CLI options**

Add `--adapter auto`, `--context-length 131072`, `--prefill-chunk-size`, `--thinking-default`, and `--engine general`. Keep legacy Qwen3-Next options isolated behind its adapter and do not require `--mtp-out` for Qwen3.5.

- [ ] **Step 5: Run launcher and CLI tests**

Run: `uv run pytest mlx_streaming/tests/test_mac_mini_qwen35_launcher.py mlx_streaming/tests/test_cli_server_portable.py -q`

Expected: PASS.

- [ ] **Step 6: Commit launcher policy**

```bash
git add scripts/run_mac_mini_qwen35.py mlx_streaming/cli.py mlx_streaming/tests
git commit -S -m "feat(qwen35): add guarded Mac mini launcher"
```

### Task 9: Acquire, verify and prepare the pinned model

**Files:**
- Create: `scripts/prepare_mac_mini_qwen35.py`
- Create: `mlx_streaming/tests/test_prepare_qwen35.py`
- Create after execution: `benchmarks/results/qwen35-small-reference-2026-07-25.json`

**Interfaces:**
- Consumes: `split_model()`, `pack_blob_from_experts.py`, pinned Hugging Face revision.
- Produces: canonical RAID source, internal expert store and preserved small-context reference evidence.

- [ ] **Step 1: Test an isolated preparation state machine**

Test free-space calculation, revision pinning, resumable Hugging Face download, source byte accounting, SHA-256 verification against the repository LFS metadata, atomic completion markers and refusal to delete or overwrite unrelated files.

- [ ] **Step 2: Run the preparation tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_prepare_qwen35.py -q`

Expected: FAIL because the preparation script is missing.

- [ ] **Step 3: Implement preparation with explicit phases**

The phases are `inspect`, `download`, `verify-source`, `split-experts`, `pack-blobs`, and `verify-runtime`. Every phase writes a resumable JSON state containing revision, input hashes, completed layer indices and output hashes. `inspect` must stop with the exact deficit if either volume lacks space.

- [ ] **Step 4: Run preparation tests**

Run: `uv run pytest mlx_streaming/tests/test_prepare_qwen35.py -q`

Expected: PASS.

- [ ] **Step 5: Inspect Mac mini storage and process state over SSH**

Run read-only checks for mounted volumes, free bytes, port 8000, Vates PIDs and target directories. Confirm no model process is running before download or reference load.

- [ ] **Step 6: Download and verify the pinned source**

Run the preparation script with `--phase download`, then `--phase verify-source`. Require repository revision `1e20fd8d42056f870933bf98ca6211024744f7ec` and total source bytes `20,411,668,782`. Preserve resumable Hugging Face cache data.

- [ ] **Step 7: Establish one-process direct MLX references**

With Vates stopped, use `mlx-vlm 0.3.12` directly at a small context and greedy decoding to record:

- plain text;
- thinking enabled;
- thinking disabled;
- one local attached image;
- one forced tool call;
- one automatic tool call and returned tool result.

Store exact requests, token IDs, outputs, logit hashes, timings, MLX peak and RSS peak in `benchmarks/results/qwen35-small-reference-2026-07-25.json`.

- [ ] **Step 8: Prepare and byte-verify all 40 routed expert layers**

Run `split-experts`, `pack-blobs`, and `verify-runtime`. Require 256 experts per layer, three routed projections, affine 4-bit/group-64 metadata, all manifest hashes and no shared-expert files in the routed store.

- [ ] **Step 9: Commit preparation code and reference evidence**

```bash
git add scripts/prepare_mac_mini_qwen35.py mlx_streaming/tests/test_prepare_qwen35.py benchmarks/results/qwen35-small-reference-2026-07-25.json
git commit -S -m "test(qwen35): preserve small-context reference"
```

### Task 10: Qualify streaming capabilities at small context

**Files:**
- Create: `scripts/qualify_qwen35_capabilities.py`
- Create: `mlx_streaming/tests/test_qwen35_capability_harness.py`
- Create after execution: `benchmarks/results/qwen35-vates-capabilities-2026-07-25.json`

**Interfaces:**
- Consumes: live OpenAI server and direct reference evidence.
- Produces: reproducible text, thinking, image and tool capability qualification.

- [ ] **Step 1: Write harness tests with a fake OpenAI endpoint**

Require exact request preservation, resumable case state, content/reasoning separation, image attachment as a data URL, forced/automatic tool calls, client-side tool execution simulation, tool-result continuation and failure on malformed or missing fields.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_capability_harness.py -q`

Expected: FAIL because the harness is missing.

- [ ] **Step 3: Implement reference comparison**

For greedy text cases require exact token equality and final-logit SHA-256 equality. For vision require the same normalised answer under a fixed prompt plus a correct deterministic visual fact. For tools require exact function name and JSON-equivalent arguments.

- [ ] **Step 4: Run tests**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_capability_harness.py -q`

Expected: PASS.

- [ ] **Step 5: Start only Qwen3.5 with a small setup context and run all cases**

Verify `/health`, `/v1/models`, plain text, thinking on/off, attached image, forced tool, automatic tool and tool result. Preserve server logs and the result JSON. Any failure is fixed and rerun; another model is not started.

- [ ] **Step 6: Commit capability evidence**

```bash
git add scripts/qualify_qwen35_capabilities.py mlx_streaming/tests/test_qwen35_capability_harness.py benchmarks/results/qwen35-vates-capabilities-2026-07-25.json
git commit -S -m "test(qwen35): qualify protocol capabilities"
```

### Task 11: Complete the exact 131,072-token boundary

**Files:**
- Create: `benchmarks/qwen35_context_131k.py`
- Create: `mlx_streaming/tests/test_qwen35_context_harness.py`
- Create after execution: `benchmarks/results/qwen35-context-131072-2026-07-25.jsonl`
- Create after execution: `benchmarks/reports/mac-mini-qwen35-131k-context-2026-07-25.md`

**Interfaces:**
- Consumes: Qwen3.5 adapter, general engine and prepared expert store.
- Produces: progressive exact-context correctness and resource evidence.

- [ ] **Step 1: Write deterministic harness and resume tests**

Generate one deterministic prompt token sequence, prefill to exactly 131,071 tokens, then decode once. Tests must assert stage planning, resume-file validation, prompt identity hashing, final-logit hashing and failure when any cache offset disagrees.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_context_harness.py -q`

Expected: FAIL because the harness is missing.

- [ ] **Step 3: Implement progressive checkpoints and hard gates**

Record every 2,048 tokens and explicit stages at 32,768, 65,536, 98,304 and 131,071 prompt tokens. At each checkpoint record all 40 adapter offsets, cumulative/stage throughput, demand loads, resident hits, fast/fallback counts, expert bytes, cache bytes, MLX allocation, RSS, free-memory percentage, swap, pressure level and disk bytes.

After one decode require:

```python
assert cache.logical_offset == 131_072
assert set(adapter.cache_offsets(cache)) == {131_072}
```

Critical pressure, monotonically unbounded swap, allocation failure, byte mismatch, offset disagreement, non-finite logits or unhandled exceptions fail qualification and preserve all logs.

- [ ] **Step 4: Run harness tests**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_context_harness.py -q`

Expected: PASS.

- [ ] **Step 5: Select the first conservative configuration from short correctness screens**

Screen only prefill chunk size and the minimum safe routed-residency capacity needed to avoid protected-union overflow. MTP, prediction, speculative slots and KV quantisation remain disabled. Pick the lowest-pressure configuration that is bit-equivalent to the direct reference.

- [ ] **Step 6: Run the detached exact-131k supervisor**

Do not interrupt a healthy run. Preserve supervisor, JSONL, resource and final-logit files. On failure, identify the first invariant, make a focused correction, rerun small screening, then rerun the exact boundary.

- [ ] **Step 7: Independently verify the live server**

After qualification start the reviewed 131,072-token Qwen3.5 server and independently verify `/health`, `/v1/models`, a short exact text response, thinking separation, one image and one tool call.

- [ ] **Step 8: Write and commit the qualification report**

```bash
git add benchmarks/qwen35_context_131k.py mlx_streaming/tests/test_qwen35_context_harness.py benchmarks/results/qwen35-context-131072-2026-07-25.jsonl benchmarks/reports/mac-mini-qwen35-131k-context-2026-07-25.md
git commit -S -m "test(qwen35): qualify exact 131k context"
```

## Boundary 4: Standardised intelligence baseline

### Task 12: Build pinned, resumable benchmark adapters

**Files:**
- Create: `benchmarks/intelligence/manifest.py`
- Create: `benchmarks/intelligence/runner.py`
- Create: `benchmarks/intelligence/scorers.py`
- Create: `benchmarks/intelligence/tasks.py`
- Create: `benchmarks/intelligence/pin_datasets.py`
- Create: `mlx_streaming/tests/test_intelligence_runner.py`
- Create after execution: `benchmarks/intelligence/dataset-lock.json`

**Interfaces:**
- Produces: `BenchmarkSpec`, `DatasetPin`, `ItemResult`, `BenchmarkRunner.run()`.
- Consumes: the live OpenAI endpoint; never imports model internals.

- [ ] **Step 1: Write manifest, resume and scoring tests**

Require a commit SHA for every Hugging Face dataset, a package version/hash for every evaluator, exact prompt template hashes, sampling parameters, thinking setting, output-token limit, item ID, raw response, latency and score. Refuse mutable revisions such as `main`.

- [ ] **Step 2: Write task-specific golden scoring tests**

Cover answer extraction for MMLU-Pro and GPQA Diamond, symbolic/numeric equivalence for MATH-500, official IFEval instruction checks, sandboxed HumanEval+ results, Berkeley function-name/argument scoring, MMMU option extraction, OCRBench normalisation and RULER exact-match scoring.

- [ ] **Step 3: Run tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_intelligence_runner.py -q`

Expected: FAIL because the benchmark package is missing.

- [ ] **Step 4: Implement dataset pinning**

`pin_datasets.py` must resolve each official repository’s immutable SHA through the Hugging Face API, store it in `dataset-lock.json`, and download only that revision. It must pin:

- MMLU-Pro;
- GPQA Diamond;
- MATH-500;
- IFEval;
- HumanEval+;
- Berkeley Function Calling Leaderboard;
- MMMU validation;
- OCRBench; and
- RULER configured for 131,072 tokens.

- [ ] **Step 5: Implement serial, per-item resumable execution**

Write one atomic JSON result per item and an aggregate only after all selected items validate. Primary runs use thinking enabled and task-appropriate deterministic decoding. A stable stratified subset also runs thinking disabled with identical prompts and limits.

- [ ] **Step 6: Run benchmark harness tests**

Run: `uv run pytest mlx_streaming/tests/test_intelligence_runner.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the benchmark harness and immutable lock**

```bash
git add benchmarks/intelligence mlx_streaming/tests/test_intelligence_runner.py
git commit -S -m "feat(benchmarks): add pinned intelligence suite"
```

### Task 13: Run and publish the intelligence baseline

**Files:**
- Create after execution: `benchmarks/results/qwen35-intelligence-baseline-2026-07-25/`
- Create after execution: `benchmarks/reports/mac-mini-qwen35-intelligence-2026-07-25.md`

**Interfaces:**
- Consumes: the pinned benchmark harness and qualified live server.
- Produces: raw per-item records and aggregate text, reasoning, code, tool, vision and long-context scores.

- [ ] **Step 1: Verify the server and benchmark lock before execution**

Require Qwen3.5 model identity, exact 131,072 context, one process, the qualified code commit and immutable dataset/evaluator hashes.

- [ ] **Step 2: Run all primary thinking-enabled suites serially**

Run MMLU-Pro, GPQA Diamond, MATH-500, IFEval, HumanEval+, BFCL, MMMU validation, OCRBench and RULER 131k. Resume from validated per-item outputs after interruptions.

- [ ] **Step 3: Run the matched thinking-disabled subset**

Use the same selected item IDs, prompts, seeds, temperature and output-token limits. Report score, time-to-first-token, output tokens and total latency deltas.

- [ ] **Step 4: Audit raw outputs and recompute aggregates**

Re-run all scorers from the preserved raw records in an offline mode. The recomputed aggregates must exactly match the published report.

- [ ] **Step 5: Commit benchmark results**

```bash
git add benchmarks/results/qwen35-intelligence-baseline-2026-07-25 benchmarks/reports/mac-mini-qwen35-intelligence-2026-07-25.md
git commit -S -m "docs(benchmarks): publish Qwen3.5 intelligence baseline"
```

## Boundary 5: Post-qualification optimisation

### Task 14: Optimise one variable at a time without weakening gates

**Files:**
- Create: `benchmarks/qwen35_optimisation_matrix.py`
- Create: `mlx_streaming/tests/test_qwen35_optimisation_matrix.py`
- Create after execution: `benchmarks/results/qwen35-optimisation-2026-07-25.jsonl`
- Create after execution: `benchmarks/reports/mac-mini-qwen35-optimisation-2026-07-25.md`
- Modify only after evidence: `scripts/run_mac_mini_qwen35.py`

**Interfaces:**
- Consumes: qualified output hashes, capability cases, pressure gates and intelligence baseline.
- Produces: ranked, reproducible configuration candidates and one qualified default.

- [ ] **Step 1: Write matrix isolation and acceptance tests**

Require every case to differ from the qualified baseline in exactly one declared variable. Reject a result lacking reference token/hash comparison, capability checks, peak resources or recovered post-process pressure.

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest mlx_streaming/tests/test_qwen35_optimisation_matrix.py -q`

Expected: FAIL because the matrix harness is missing.

- [ ] **Step 3: Implement short screens in this order**

Screen routed expert residency, speculative staging capacity, cross-layer prediction width/timing, demand-loader concurrency, attention-cache quantisation/growth, Qwen3.5 MTP depth/verification, and mixed text/image vision-encoder residency. Each screen starts from the qualified configuration and changes one value only.

- [ ] **Step 4: Gate every candidate**

Require exact greedy token equality and final-logit equality for non-lossy changes. For explicitly lossy KV quantisation require the predeclared tolerance in the qualification report plus no capability or benchmark regression. Reject critical pressure, unbounded swap, byte errors, cache disagreement or slower end-to-end throughput.

- [ ] **Step 5: Revalidate survivors at exact 131k**

Run the complete boundary harness and combined capability smoke for each surviving candidate. Rerun affected intelligence suites for changes to decoding, quantisation, prompting, vision processing or MTP.

- [ ] **Step 6: Publish ranked options before changing defaults**

Report prefill, decode, generation, peak pressure, swap, disk I/O, capability and intelligence trade-offs. Change `scripts/run_mac_mini_qwen35.py` only to the highest-ranked candidate that passes every predeclared gate; ask the user only if the evidence leaves a material quality/performance trade-off without a dominant candidate.

- [ ] **Step 7: Commit the optimisation harness and approved profile**

```bash
git add benchmarks/qwen35_optimisation_matrix.py mlx_streaming/tests/test_qwen35_optimisation_matrix.py benchmarks/results/qwen35-optimisation-2026-07-25.jsonl benchmarks/reports/mac-mini-qwen35-optimisation-2026-07-25.md scripts/run_mac_mini_qwen35.py
git commit -S -m "perf(qwen35): promote qualified runtime profile"
```

## Final integration and hand-off

### Task 15: Validate, record durable memory, push and verify CI

**Files:**
- Modify: `.agents/TODO.md`
- Modify through Brick only: `.agents/memory/`
- Modify through Brick only: `.agents/brick/index/brick.sqlite3`

**Interfaces:**
- Consumes: all completed boundaries.
- Produces: verified signed branch, pull request, CI evidence and durable operational memory.

- [ ] **Step 1: Run the complete portable suite**

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 2: Run static package and launcher checks**

Run: `uv run python -m compileall -q mlx_streaming scripts benchmarks`

Run: `uv run python scripts/run_mac_mini_qwen35.py --help`

Expected: PASS without starting a model.

- [ ] **Step 3: Independently verify live Qwen3.5**

Verify one process, `/health`, `/v1/models`, text, separate reasoning, reasoning disabled, attached image, automatic tool call, returned tool result and a short follow-up. Confirm the advertised context is 131,072 and no old Vates model process is live.

- [ ] **Step 4: Record Brick memory through the CLI**

Add the pinned model/storage policy, adapter/runtime architecture, exact-131k result, capability evidence, intelligence scores and approved optimisation profile using `brick memory add`. Run `brick memory validate` and `brick rebuild`; never edit memory files directly.

- [ ] **Step 5: Verify commit signatures and branch scope**

Run: `git log --show-signature origin/main..HEAD`

Run: `git diff --check origin/main...HEAD`

Run: `git status --short`

Expected: every task commit has a good signature; only intended tracked files differ; generated `.agents/brick/` state is either part of the dedicated Brick commit or remains unstaged.

- [ ] **Step 6: Push and verify the protected pull request**

Run: `git push -u origin agent/qwen35-general-moe-runtime`

Use `/opt/homebrew/bin/gh` outside the sandbox to create or update the pull request and wait for all required checks. Do not merge until CI passes and the boundary review is complete.

- [ ] **Step 7: Mark the ledger complete and remove the continuation automation**

Tick every completed Qwen3.5 item in `.agents/TODO.md`. Delete the heartbeat only after the live server, committed evidence, pushed branch and PR CI are all independently verified.
