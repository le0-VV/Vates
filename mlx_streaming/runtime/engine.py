"""Architecture-neutral, non-speculative generation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mlx_streaming.core.mem import reset_peak, snapshot
from mlx_streaming.models.base import ModelAdapter


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict]
    max_tokens: int
    images: list[object] = field(default_factory=list)
    tools: list[dict] | None = None
    enable_thinking: bool = True


@dataclass(frozen=True)
class GenerationDelta:
    text: str
    token_id: int
    generated_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    peak_mlx_bytes: int
    cache_offsets: tuple[int, ...]
    stopped: bool


def _eos_token_ids(processor: object) -> frozenset[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    values = getattr(tokenizer, "eos_token_ids", None)
    eos = {int(value) for value in values} if values else set()
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        eos.add(int(value))
    return frozenset(eos)


def _slice_prompt_kwargs(
    kwargs: dict[str, object],
    start: int,
    end: int,
) -> dict[str, object]:
    chunk = dict(kwargs)
    inputs_embeds = chunk.get("inputs_embeds")
    if inputs_embeds is not None:
        chunk["inputs_embeds"] = inputs_embeds[:, start:end]
    mask = chunk.get("mask")
    if mask is not None:
        chunk["mask"] = mask[..., start:end]
    return chunk


class GeneralModelEngine:
    """Greedy baseline engine with a fresh hybrid cache for every request."""

    def __init__(
        self,
        *,
        model: object,
        processor: object,
        adapter: ModelAdapter,
        prefill_chunk_size: int = 64,
    ):
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        self.model = model
        self.processor = processor
        self.adapter = adapter
        self.prefill_chunk_size = prefill_chunk_size

    def generate(
        self,
        request: GenerationRequest,
        on_delta: Callable[[GenerationDelta], bool] | None = None,
    ) -> GenerationResult:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        reset_peak()
        input_ids, prompt_kwargs = self.adapter.prepare_inputs(
            self.model,
            self.processor,
            request.messages,
            request.images,
            tools=request.tools,
            enable_thinking=request.enable_thinking,
        )
        if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("general generation requires one non-empty prompt")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens == 0:
            raise ValueError("general generation requires one non-empty prompt")

        cache = self.adapter.make_cache(self.model)
        last_logits: Any = None
        prefill_started = time.perf_counter()
        for start in range(0, prompt_tokens, self.prefill_chunk_size):
            end = min(start + self.prefill_chunk_size, prompt_tokens)
            output = self.adapter.forward(
                self.model,
                input_ids[:, start:end],
                cache,
                **_slice_prompt_kwargs(prompt_kwargs, start, end),
            )
            last_logits = output.logits[:, -1, :]
            mx.eval(last_logits)
        prefill_seconds = time.perf_counter() - prefill_started

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        eos = _eos_token_ids(self.processor)
        produced: list[int] = []
        text = ""
        stopped = False
        decode_seconds = 0.0

        while len(produced) < request.max_tokens:
            token_id = int(mx.argmax(last_logits, axis=-1).item())
            if token_id in eos:
                break

            produced.append(token_id)
            decoded = tokenizer.decode(produced)
            delta_text = decoded[len(text) :] if decoded.startswith(text) else decoded
            text = decoded
            delta = GenerationDelta(
                text=delta_text,
                token_id=token_id,
                generated_tokens=len(produced),
            )
            if on_delta is not None and on_delta(delta):
                stopped = True
                break
            if len(produced) >= request.max_tokens:
                break

            decode_started = time.perf_counter()
            current = mx.array([[token_id]], dtype=input_ids.dtype)
            output = self.adapter.forward(self.model, current, cache)
            last_logits = output.logits[:, -1, :]
            mx.eval(last_logits)
            decode_seconds += time.perf_counter() - decode_started

        return GenerationResult(
            text=text,
            token_ids=tuple(produced),
            prompt_tokens=prompt_tokens,
            generated_tokens=len(produced),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            peak_mlx_bytes=snapshot().mlx_peak_bytes,
            cache_offsets=self.adapter.cache_offsets(cache),
            stopped=stopped,
        )
