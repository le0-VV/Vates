"""Architecture-neutral contracts used by the streaming runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


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

    def validate_config(
        self, config: Mapping[str, object]
    ) -> ModelDimensions: ...

    def load(
        self,
        path: str,
        *,
        revision: str | None,
        lazy: bool,
    ) -> LoadedModel: ...

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
