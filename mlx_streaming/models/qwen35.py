"""Qwen3.5 MoE adapter for mlx-vlm 0.3.12."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_streaming.models.base import (
    ExpertLayerSpec,
    HybridCacheState,
    LoadedModel,
    ModelDimensions,
)
from mlx_streaming.models.registry import register_adapter


class ModelConfigurationError(ValueError):
    """Pinned Qwen3.5 metadata is missing or inconsistent."""


class CacheOffsetError(RuntimeError):
    """A full-attention cache did not advance with the logical context."""

    def __init__(self, layer: int, expected: int, actual: int):
        super().__init__(
            f"cache offset mismatch at layer {layer}: expected {expected}, got {actual}"
        )
        self.layer = layer
        self.expected = expected
        self.actual = actual


_MISSING = object()


def _value_at(config: Mapping[str, object], path: str) -> object:
    value: object = config
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return _MISSING
        value = value.get(part, _MISSING)
    return value


def _require(config: Mapping[str, object], path: str, expected: object) -> None:
    actual = _value_at(config, path)
    if actual != expected:
        rendered = "<missing>" if actual is _MISSING else repr(actual)
        raise ModelConfigurationError(
            f"{path} must be {expected!r}, got {rendered}"
        )


def _integer_offset(value: object) -> int:
    if isinstance(value, int):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return int(item())
    raise TypeError(f"cache offset must be integer-like, got {type(value).__name__}")


class Qwen35MoeAdapter:
    architecture = "qwen3_5_moe"

    def __init__(self, input_preparer: Callable[..., Mapping[str, object]] | None = None):
        self._input_preparer = input_preparer

    def validate_config(
        self, config: Mapping[str, object]
    ) -> ModelDimensions:
        expected = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config.model_type": "qwen3_5_moe_text",
            "text_config.hidden_size": 2048,
            "text_config.num_hidden_layers": 40,
            "text_config.num_experts": 256,
            "text_config.num_experts_per_tok": 8,
            "text_config.moe_intermediate_size": 512,
            "text_config.shared_expert_intermediate_size": 512,
            "text_config.full_attention_interval": 4,
            "text_config.max_position_embeddings": 262144,
            "text_config.mtp_num_hidden_layers": 1,
            "quantization.mode": "affine",
            "quantization.bits": 4,
            "quantization.group_size": 64,
            "vision_config.model_type": "qwen3_5_moe",
            "vision_config.depth": 27,
            "vision_config.hidden_size": 1152,
            "vision_config.out_hidden_size": 2048,
            "vision_config.patch_size": 16,
            "vision_config.spatial_merge_size": 2,
            "vision_config.temporal_patch_size": 2,
            "vision_config.deepstack_visual_indexes": [],
        }
        for path, value in expected.items():
            _require(config, path, value)
        return ModelDimensions(
            architecture=self.architecture,
            hidden_size=2048,
            num_layers=40,
            num_experts=256,
            top_k=8,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=512,
            quant_mode="affine",
            quant_bits=4,
            quant_group_size=64,
            max_context=262144,
        )

    def load(
        self,
        path: str,
        *,
        revision: str | None = None,
        lazy: bool = True,
    ) -> LoadedModel:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, processor = load(path, revision=revision, lazy=lazy)
        config = load_config(Path(path))
        self.validate_config(config)
        return LoadedModel(model=model, processor=processor, config=config)

    def language_model(self, model: object) -> object:
        return getattr(model, "language_model")

    def layers(self, model: object) -> Sequence[object]:
        language_model = self.language_model(model)
        return getattr(getattr(language_model, "model"), "layers")

    def expert_layers(self, model: object) -> tuple[ExpertLayerSpec, ...]:
        specs = []
        for index, layer in enumerate(self.layers(model)):
            block = getattr(layer, "mlp", None)
            switch_mlp = getattr(block, "switch_mlp", None)
            gate_proj = getattr(switch_mlp, "gate_proj", None)
            values = {
                "hidden_size": getattr(gate_proj, "input_dims", None),
                "intermediate_size": getattr(gate_proj, "output_dims", None),
                "num_experts": getattr(gate_proj, "num_experts", None),
                "top_k": getattr(block, "top_k", None),
                "group_size": getattr(gate_proj, "group_size", None),
                "bits": getattr(gate_proj, "bits", None),
            }
            expected = {
                "hidden_size": 2048,
                "intermediate_size": 512,
                "num_experts": 256,
                "top_k": 8,
                "group_size": 64,
                "bits": 4,
            }
            for field, wanted in expected.items():
                if values[field] != wanted:
                    raise ModelConfigurationError(
                        f"layer {index} {field} must be {wanted}, "
                        f"got {values[field]!r}"
                    )
            if getattr(block, "shared_expert", None) is None:
                raise ModelConfigurationError(
                    f"layer {index} shared_expert is missing"
                )
            if getattr(block, "shared_expert_gate", None) is None:
                raise ModelConfigurationError(
                    f"layer {index} shared_expert_gate is missing"
                )
            specs.append(
                ExpertLayerSpec(
                    layer_index=index,
                    block=block,
                    hidden_size=2048,
                    intermediate_size=512,
                    num_experts=256,
                    top_k=8,
                    normalise_topk=True,
                )
            )
        if len(specs) != 40:
            raise ModelConfigurationError(
                f"num_hidden_layers must be 40, got {len(specs)}"
            )
        return tuple(specs)

    def make_cache(self, model: object) -> HybridCacheState:
        from mlx_lm.models.cache import ArraysCache, KVCache

        entries = list(self.language_model(model).make_cache())
        kinds = []
        for index, entry in enumerate(entries):
            if isinstance(entry, ArraysCache):
                kinds.append("linear")
            elif isinstance(entry, KVCache):
                kinds.append("full_attention")
            else:
                raise ModelConfigurationError(
                    f"unsupported cache type at layer {index}: "
                    f"{type(entry).__name__}"
                )
        expected = ("linear", "linear", "linear", "full_attention") * 10
        if tuple(kinds) != expected:
            raise ModelConfigurationError(
                f"hybrid cache layout must be {expected!r}, got {tuple(kinds)!r}"
            )
        return HybridCacheState(entries=entries, kinds=tuple(kinds))

    def prepare_inputs(
        self,
        model: object,
        processor: object,
        messages: list[dict],
        images: list[object],
        *,
        tools: list[dict] | None,
        enable_thinking: bool,
    ) -> tuple[object, dict[str, object]]:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
            enable_thinking=enable_thinking,
        )
        input_preparer = self._input_preparer
        if input_preparer is None:
            from mlx_vlm.utils import prepare_inputs

            input_preparer = prepare_inputs
        model_inputs = dict(
            input_preparer(
                processor,
                images=images or None,
                prompts=rendered,
                add_special_tokens=True,
                padding=True,
            )
        )
        input_ids = model_inputs.pop("input_ids")
        mask = model_inputs.pop("attention_mask", None)
        pixel_values = model_inputs.pop("pixel_values", None)
        embedding_output = model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=mask,
            **model_inputs,
        )
        embedding_values = {
            key: value
            for key, value in embedding_output.to_dict().items()
            if value is not None
        }
        kwargs = dict(model_inputs)
        kwargs.update(embedding_values)
        if mask is not None:
            kwargs["mask"] = mask
        return input_ids, kwargs

    def forward(
        self,
        model: object,
        input_ids: object,
        cache: HybridCacheState,
        **kwargs: Any,
    ) -> object:
        count = int(input_ids.shape[1])
        output = self.language_model(model)(
            input_ids,
            cache=cache.entries,
            **kwargs,
        )
        expected = cache.logical_offset + count
        for index, (kind, entry) in enumerate(zip(cache.kinds, cache.entries)):
            if kind != "full_attention":
                continue
            actual = _integer_offset(getattr(entry, "offset"))
            if actual != expected:
                raise CacheOffsetError(index, expected, actual)
        cache.logical_offset = expected
        return output

    def cache_offsets(self, cache: HybridCacheState) -> tuple[int, ...]:
        return tuple(
            _integer_offset(getattr(entry, "offset"))
            if kind == "full_attention"
            else cache.logical_offset
            for kind, entry in zip(cache.kinds, cache.entries)
        )


QWEN35_ADAPTER = Qwen35MoeAdapter()
register_adapter(QWEN35_ADAPTER)
