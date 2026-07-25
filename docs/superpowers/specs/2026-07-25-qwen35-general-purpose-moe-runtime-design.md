# Qwen3.5 General-Purpose MoE Runtime Design

**Date:** 25 July 2026

**Target:** `leonardw@leonards-mac-mini`

**Initial model:** `mlx-community/Qwen3.5-35B-A3B-4bit` at revision
`1e20fd8d42056f870933bf98ca6211024744f7ec`

## Objective

Evolve Vates from a model-specific text runtime into a general-purpose
streaming mixture-of-experts runtime, then use Qwen3.5-35B-A3B as the first new
architecture qualification. The Qwen3.5 endpoint must support thinking,
protocol-only tool calling and image attachments through an OpenAI-compatible
chat API while retaining an exact 131,072-token production context on a
16 GB Apple M4 Mac mini.

This is an experimental environment. A failed load, capability probe or
context run is evidence that narrows the next experiment; it is not a reason
to restore another running model. Existing model files remain untouched, but
only one model process may run and the Mac mini is dedicated to Qwen3.5 work
for the duration of this migration.

## Success criteria

The first complete Qwen3.5 runtime must:

- load the pinned 4-bit source through Vates' bounded-memory path;
- generate ordinary text with thinking enabled by default;
- allow thinking to be disabled per request;
- expose thinking separately from final content;
- accept a normal image attachment from Chatbox or another OpenAI-compatible
  agent application without client-specific preparation;
- accept OpenAI tool definitions, emit structured tool calls and consume tool
  results without executing tools inside Vates;
- complete exactly 131,071 prompt tokens and one boundary decode with every
  relevant cache offset at exactly 131,072;
- remain free of byte-integrity errors, incorrect expert routing, Metal
  allocation failures and critical or unbounded memory pressure;
- publish reproducible standardised intelligence results after the clean 131k
  run; and
- preserve all source and result evidence needed to reproduce the outcome.

Short contexts are permitted for implementation, correctness screening and
capability testing. They are not production qualification. A 262k attempt is
explicitly out of scope.

## Delivery order

Correctness and completeness precede performance work:

1. Establish a direct small-context MLX reference for the pinned checkpoint.
2. Add the general model and MoE architecture boundaries to Vates.
3. Add Qwen3.5 expert streaming and hybrid-cache support.
4. Add multimodal, reasoning and tool-protocol API support.
5. Complete one clean exact-131k run with no other problem.
6. Run the standardised intelligence suite.
7. Tune configuration and then perform deeper code/runtime optimisation.

MTP, prediction-width tuning, expert-residency tuning and other speculative
performance work must not delay or contaminate the first non-speculative 131k
qualification. Once optimisation begins, each material change must preserve
the qualified output and capability gates.

## Implementation boundaries

The migration is delivered as sequential, independently reviewed changes
rather than one unbounded pull request:

1. General model-adapter and model-derived expert-store interfaces, including
   the Qwen3.5 small-context reference and streaming text path.
2. OpenAI multimodal, reasoning and tool-protocol support.
3. Mac mini asset preparation, progressive context harness and exact-131k
   qualification.
4. Reproducible standardised benchmark harnesses and baseline results.
5. Post-qualification configuration and runtime optimisation.

Each boundary must leave portable tests passing and expose the interfaces
needed by the next boundary. Operational experiments may continue across
those changes, but tracked-file commits and pull requests remain focused.

## Runtime architecture

### Model boundary

Introduce an explicit model-adapter boundary instead of branching throughout
the server and cache implementation. An adapter owns:

- architecture detection and configuration validation;
- language-model and processor construction;
- MoE layer discovery and expert metadata;
- cache construction and cache-offset reporting;
- chat-template invocation;
- multimodal input preparation;
- reasoning-output parsing; and
- optional MTP integration.

The shared runtime owns process lifecycle, request serialisation, resource
limits, expert-store I/O, pool policies, OpenAI response formatting and
instrumentation. The first new adapter targets `qwen3_5_moe`; later adapters
must be addable without changing the HTTP protocol or duplicating the expert
store.

The Qwen3.5 adapter validates the published architecture:

- 35B total parameters and approximately 3B active parameters;
- 40 language-model layers;
- 256 routed experts per MoE layer;
- eight routed experts plus one shared expert per token;
- hidden size 2,048 and expert intermediate size 512;
- affine 4-bit MLX quantisation metadata;
- a vision encoder and matching image/video processor metadata;
- native context length 262,144, although Vates qualifies 131,072; and
- trained multi-token prediction weights, which remain disabled until the
  correctness baseline is established.

Unknown, incomplete or inconsistent model metadata fails at startup. The
runtime must not silently reinterpret an unsupported architecture as an
existing adapter.

### Expert streaming

Canonical checkpoint shards remain file-backed. Preparation extracts routed
expert projections into a validated per-layer store and records all
architecture-dependent dimensions in a manifest. Shared experts and dense
weights follow the adapter's declared residency policy; they are never
mistaken for routed slots.

The common expert store accepts the number of layers, number of experts,
top-k, projection layout, bits and group size from the adapter manifest. No
production path may retain hard-coded values such as 48 layers or 512
experts. Pool-capacity checks derive the protected expert union from the
actual batch and decoding mode and fail fast rather than truncate routing.

The first exact-131k run uses a conservative, non-MTP configuration selected
from short correctness screens. It records demand loads, resident hits,
fast/fallback selections, expert bytes, cache bytes and all capacity events.
Only after that run qualifies may the existing residency, prefetch and MTP
surfaces be adapted and tuned.

### Hybrid context state

Qwen3.5 combines Gated DeltaNet and gated-attention layers. The adapter must
identify each cache type, construct the correct state and report a unified
logical token offset without assuming every layer is a full-attention KV
cache.

The boundary harness incrementally prefills deterministic content to 131,071
tokens, checks every adapter-reported offset, decodes once, then requires a
logical offset of exactly 131,072. Quantisation of attention KV state is an
optimisation and is enabled only after an unquantised or reference-equivalent
path establishes the model's expected output.

## OpenAI-compatible capability surface

### Thinking

Thinking is enabled by default through the official Qwen3.5 chat-template
argument. A request may disable it explicitly. The decoder separates the
model's `<think>...</think>` region from the final answer without exposing
delimiter tokens.

Non-streaming responses place final output in `message.content` and thinking
in `message.reasoning_content`. Streaming responses use corresponding
incremental fields and never merge an earlier reasoning delta into final
content. Malformed or unterminated reasoning output is reported explicitly
and retained in diagnostics; it is not silently relabelled as a valid final
answer.

### Protocol-only tool calling

The endpoint accepts standard OpenAI `tools`, `tool_choice`,
`parallel_tool_calls`, assistant `tool_calls` and `tool` result messages. The
adapter passes schemas through the official tool-aware chat template and a
Qwen3.5-compatible parser maps model output to stable call IDs, function names
and JSON arguments.

Vates never executes a tool. Chatbox or another agent client remains
responsible for permissions, execution and returning tool results. Invalid
schemas, unknown forced tool names, malformed argument JSON and unsupported
message sequences receive deterministic OpenAI-shaped errors.

### Image attachments

Users interact with vision by attaching or pasting an image in Chatbox or
another compatible application. The API accepts the standard OpenAI message
content array containing text and `image_url` parts. The transport layer
normalises both embedded data URLs and HTTPS image URLs into bytes for the
processor, so callers do not need model-specific encoding.

Image handling applies explicit limits for body size, decoded pixels, image
count, MIME type and fetch duration. HTTPS fetching rejects credentials,
non-HTTPS schemes, loopback, link-local, private and otherwise non-public
destinations before and after redirects. Decode or fetch failures are client
errors and never fall back to text-only interpretation. Video and audio input
remain out of scope for the first qualification.

## Process and deployment policy

Before Qwen3.5 model loading begins, stop every current Vates model process
and verify port 8000 is free. Do not run reference and streaming models
concurrently. Failed Qwen3.5 experiments leave the machine available for the
next corrected Qwen3.5 run rather than automatically starting another model.

The 20,411,668,782-byte source repository is pinned at revision
`1e20fd8d42056f870933bf98ca6211024744f7ec` and stored below the canonical RAID
model-source tree. Derived expert blobs, manifests, prepared MTP assets and
logs live below:

```text
/Users/leonardw/Library/Application Support/Vates/
└── qwen3.5-35b-a3b-4bit/
    ├── experts/
    ├── mtp/
    ├── logs/
    └── benchmarks/
```

Existing model files are not deleted. Before downloading or preparing data,
verify free space on both volumes. If the approved layout cannot fit without
deletion, stop and report the exact deficit; do not remove RAID-derived or
other model artefacts without explicit authorisation.

The eventual server advertises `qwen3.5-35b-a3b-4bit`, binds to the currently
approved LAN interface and port, and processes one inference request at a time
on the shared 16 GB machine.

## Failure and experiment behaviour

Preparation and startup are fail-fast for:

- model-revision or manifest mismatch;
- unsupported quantisation or projection layout;
- missing vision processor or chat template;
- expert-store byte mismatch;
- protected expert-union overflow;
- inconsistent cache offsets;
- invalid reasoning or tool protocol state; and
- allocation failure or critical memory pressure.

Every experiment records its model revision, code commit, configuration,
prompt identity, generated tokens, output hash, elapsed time and resource
summary. A failure report identifies the first violated invariant and
preserves the logs and resumable source data required for the next run. It
does not change production defaults, lower correctness checks or delete
evidence to make the next attempt pass.

## Verification before optimisation

Portable tests use fake model and processor adapters. They cover:

- adapter registration and fail-closed architecture detection;
- model-derived expert dimensions and shared-expert separation;
- multimodal content validation and bounded image fetching;
- thinking and non-thinking request templating;
- split reasoning/final streaming deltas;
- tool definition, forced-tool, tool-call and tool-result round trips;
- malformed reasoning and malformed tool-call behaviour;
- serial inference and process lifecycle; and
- launcher storage, model identity and one-process invariants.

Mac mini qualification proceeds from cheap to expensive:

1. Direct MLX small-context reference generation.
2. Streaming small-context text equivalence.
3. Thinking-on and thinking-off responses.
4. A real attached-image description and visual question.
5. One forced and one automatic tool call, followed by a tool-result answer.
6. Progressive context checkpoints through exactly 131,071 prompt tokens.
7. One decoded token with every logical cache offset at exactly 131,072.
8. Independent `/health`, `/v1/models` and combined capability smoke tests.

The exact-131k run must report cumulative and per-stage prefill throughput,
boundary decode latency, expert demand and hit counts, fast/fallback counts,
cache growth and final bytes, peak MLX allocation, peak RSS, minimum free
memory, maximum swap, memory-pressure levels and storage read/write rates.
Critical pressure, unbounded swap, incorrect output, cache disagreement or an
unhandled exception prevents qualification but informs the next experiment.

## Standardised intelligence evaluation

After the clean 131k run and before performance tuning, run pinned,
reproducible evaluations:

- MMLU-Pro for broad knowledge and reasoning;
- GPQA Diamond for difficult scientific reasoning;
- MATH-500 for mathematical reasoning;
- IFEval for instruction following;
- HumanEval+ for executable code correctness;
- Berkeley Function Calling Leaderboard for tool selection and arguments;
- MMMU validation for multimodal reasoning;
- OCRBench for visual text understanding; and
- RULER at 131k for long-context retrieval and manipulation.

Use official task splits and evaluators. Record dataset revisions, prompt
templates, thinking setting, sampling parameters, output-token limits,
scoring versions, raw outputs and aggregate scores. The primary scoreboard
uses thinking enabled. A documented matched subset also runs with thinking
disabled to quantify latency and quality trade-offs. Benchmark execution is
serial and retains resumable per-item results so a stopped run does not lose
completed evidence.

## Performance phase

Performance work begins only after the exact-131k qualification and
standardised baseline are preserved. Screen one variable at a time at short
context, then revalidate survivors at 131k. Candidate surfaces include:

- real and speculative expert residency;
- cross-layer prediction width and prefetch timing;
- demand-loader concurrency and I/O placement;
- attention-cache quantisation and growth policy;
- Qwen3.5 MTP depth and verification; and
- image-encoder residency when serving mixed text and image traffic.

Optimisation acceptance requires output equivalence or an explicitly defined
numerical tolerance, no capability regression, bounded pressure and a
reproducible end-to-end gain. Intelligence benchmarks affected by a material
decoder, quantisation or prompt-path change are rerun before that change may
become the default.

## Out of scope

- Running multiple model processes concurrently.
- Automatic rollback to another model.
- Deleting existing model or RAID-derived artefacts without approval.
- A 262k context qualification.
- Server-side tool execution.
- Audio or video input in the first Qwen3.5 qualification.
- Performance tuning before the first clean exact-131k run.
