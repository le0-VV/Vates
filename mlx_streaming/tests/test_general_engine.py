from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mlx_streaming.models.base import HybridCacheState
from mlx_streaming.runtime.engine import (
    GeneralModelEngine,
    GenerationRequest,
)


class _Tokenizer:
    eos_token_ids = {9}

    def decode(self, token_ids):
        return "".join({4: "A", 5: "B", 6: "C"}.get(token, "") for token in token_ids)


class _Processor:
    tokenizer = _Tokenizer()


class _Adapter:
    def __init__(self, next_tokens=(4, 5, 6)):
        self.next_tokens = tuple(next_tokens)
        self.forward_lengths = []
        self.forward_kwargs = []
        self.caches = []

    def prepare_inputs(
        self,
        model,
        processor,
        messages,
        images,
        *,
        tools,
        enable_thinking,
    ):
        assert messages == [{"role": "user", "content": "test"}]
        assert images == ["image"]
        assert tools == [{"type": "function", "function": {"name": "lookup"}}]
        assert enable_thinking is False
        ids = mx.array([[10, 11, 12, 13, 14, 15, 16]])
        embeds = mx.arange(7 * 2).reshape(1, 7, 2)
        return ids, {
            "inputs_embeds": embeds,
            "mask": mx.ones((1, 7)),
            "image_grid_thw": mx.array([[1, 2, 2]]),
        }

    def make_cache(self, model):
        cache = HybridCacheState(entries=[object()], kinds=("linear",))
        self.caches.append(cache)
        return cache

    def forward(self, model, input_ids, cache, **kwargs):
        length = int(input_ids.shape[1])
        self.forward_lengths.append(length)
        self.forward_kwargs.append(kwargs)
        cache.logical_offset += length
        call = len(self.forward_lengths) - 1
        token = self.next_tokens[min(call, len(self.next_tokens) - 1)]
        logits = mx.full((1, length, 10), -1000.0)
        logits[:, -1, token] = 1000.0
        return SimpleNamespace(logits=logits)

    def cache_offsets(self, cache):
        return (cache.logical_offset,)


def _request(max_tokens=3):
    return GenerationRequest(
        messages=[{"role": "user", "content": "test"}],
        images=["image"],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        enable_thinking=False,
        max_tokens=max_tokens,
    )


def test_general_engine_chunks_prefill_and_tracks_pending_decode_token():
    adapter = _Adapter(next_tokens=(4, 4, 4, 5, 6))
    deltas = []
    engine = GeneralModelEngine(
        model=object(),
        processor=_Processor(),
        adapter=adapter,
        prefill_chunk_size=3,
    )

    result = engine.generate(_request(), lambda delta: deltas.append(delta) or False)

    assert adapter.forward_lengths == [3, 3, 1, 1, 1]
    assert [delta.text for delta in deltas] == ["A", "B", "C"]
    assert [delta.token_id for delta in deltas] == [4, 5, 6]
    assert result.text == "ABC"
    assert result.token_ids == (4, 5, 6)
    assert result.prompt_tokens == 7
    assert result.generated_tokens == 3
    assert result.cache_offsets == (9,)
    assert result.stopped is False
    assert result.prefill_seconds >= 0
    assert result.decode_seconds >= 0


def test_general_engine_passes_prompt_state_only_during_prefill():
    adapter = _Adapter(next_tokens=(4, 4, 4, 5))
    engine = GeneralModelEngine(
        model=object(),
        processor=_Processor(),
        adapter=adapter,
        prefill_chunk_size=3,
    )

    engine.generate(_request(max_tokens=2), lambda _delta: False)

    assert [set(kwargs) for kwargs in adapter.forward_kwargs[:3]] == [
        {"inputs_embeds", "mask", "image_grid_thw"},
        {"inputs_embeds", "mask", "image_grid_thw"},
        {"inputs_embeds", "mask", "image_grid_thw"},
    ]
    assert adapter.forward_kwargs[3] == {}


def test_general_engine_excludes_eos_and_starts_each_request_with_fresh_cache():
    adapter = _Adapter(next_tokens=(4, 4, 4, 9))
    engine = GeneralModelEngine(
        model=object(),
        processor=_Processor(),
        adapter=adapter,
        prefill_chunk_size=3,
    )

    first = engine.generate(_request(max_tokens=3), lambda _delta: False)
    second = engine.generate(_request(max_tokens=1), lambda _delta: False)

    assert first.text == "A"
    assert first.token_ids == (4,)
    assert first.cache_offsets == (8,)
    assert len(adapter.caches) == 2
    assert adapter.caches[0] is not adapter.caches[1]
    assert second.cache_offsets == (7,)


def test_general_engine_stops_when_callback_requests_cancellation():
    adapter = _Adapter(next_tokens=(4, 4, 4, 4))
    engine = GeneralModelEngine(
        model=object(),
        processor=_Processor(),
        adapter=adapter,
        prefill_chunk_size=3,
    )

    result = engine.generate(_request(max_tokens=3), lambda _delta: True)

    assert result.text == "A"
    assert result.stopped is True
    assert result.cache_offsets == (7,)
