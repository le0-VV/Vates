from __future__ import annotations

from copy import deepcopy

import mlx.core as mx
import pytest

from mlx_streaming.models.qwen35 import (
    CacheOffsetError,
    ModelConfigurationError,
    Qwen35MoeAdapter,
)
from mlx_streaming.tests.fakes.qwen35 import FakeQwen35Model, qwen35_config


def _replace(config: dict, path: str, value: object) -> dict:
    changed = deepcopy(config)
    target = changed
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    return changed


def test_qwen35_adapter_validates_pinned_architecture_dimensions():
    dimensions = Qwen35MoeAdapter().validate_config(qwen35_config())

    assert dimensions.architecture == "qwen3_5_moe"
    assert dimensions.hidden_size == 2048
    assert dimensions.num_layers == 40
    assert dimensions.num_experts == 256
    assert dimensions.top_k == 8
    assert dimensions.expert_intermediate_size == 512
    assert dimensions.shared_expert_intermediate_size == 512
    assert dimensions.quant_mode == "affine"
    assert dimensions.quant_bits == 4
    assert dimensions.quant_group_size == 64
    assert dimensions.max_context == 262144


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model_type", "qwen3_moe", "model_type"),
        ("architectures", ["OtherModel"], "architectures"),
        ("text_config.num_hidden_layers", 39, "num_hidden_layers"),
        ("text_config.num_experts", 255, "num_experts"),
        ("text_config.num_experts_per_tok", 4, "num_experts_per_tok"),
        ("text_config.hidden_size", 4096, "hidden_size"),
        ("text_config.moe_intermediate_size", 1024, "moe_intermediate_size"),
        (
            "text_config.shared_expert_intermediate_size",
            1024,
            "shared_expert_intermediate_size",
        ),
        ("quantization.mode", "mxfp4", "quantization.mode"),
        ("quantization.bits", 8, "quantization.bits"),
        ("quantization.group_size", 128, "quantization.group_size"),
        ("text_config.max_position_embeddings", 131072, "max_position_embeddings"),
        ("text_config.mtp_num_hidden_layers", 0, "mtp_num_hidden_layers"),
        ("vision_config.model_type", "qwen3_vl", "vision_config.model_type"),
        ("vision_config.out_hidden_size", 4096, "vision_config.out_hidden_size"),
    ],
)
def test_qwen35_adapter_rejects_inconsistent_metadata(path, value, message):
    config = _replace(qwen35_config(), path, value)

    with pytest.raises(ModelConfigurationError, match=message):
        Qwen35MoeAdapter().validate_config(config)


def test_qwen35_adapter_discovers_real_language_and_expert_paths():
    model = FakeQwen35Model()
    adapter = Qwen35MoeAdapter()

    assert adapter.language_model(model) is model.language_model
    assert tuple(adapter.layers(model)) == tuple(model.language_model.model.layers)
    specs = adapter.expert_layers(model)
    assert len(specs) == 40
    assert specs[0].layer_index == 0
    assert specs[0].block is model.language_model.model.layers[0].mlp
    assert specs[0].hidden_size == 2048
    assert specs[0].intermediate_size == 512
    assert specs[0].num_experts == 256
    assert specs[0].top_k == 8
    assert specs[0].normalise_topk is True


def test_qwen35_adapter_tracks_hybrid_cache_offsets_across_forward():
    model = FakeQwen35Model()
    adapter = Qwen35MoeAdapter()
    cache = adapter.make_cache(model)

    assert cache.kinds == ("linear", "linear", "linear", "full_attention") * 10
    assert adapter.cache_offsets(cache) == (0,) * 40

    output = adapter.forward(model, mx.array([[1, 2, 3]]), cache)

    assert output.logits.shape == (1, 3, 16)
    assert cache.logical_offset == 3
    assert adapter.cache_offsets(cache) == (3,) * 40


def test_qwen35_adapter_rejects_full_attention_offset_disagreement(monkeypatch):
    model = FakeQwen35Model()
    adapter = Qwen35MoeAdapter()
    cache = adapter.make_cache(model)

    monkeypatch.setattr(
        type(model.language_model),
        "__call__",
        lambda self, input_ids, *, cache, **kwargs: type(
            "Output", (), {"logits": mx.zeros((1, 1, 16))}
        )(),
    )

    with pytest.raises(CacheOffsetError, match=r"layer 3.*expected 1.*got 0"):
        adapter.forward(model, mx.array([[1]]), cache)


class _Processor:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "rendered prompt"


def test_qwen35_adapter_prepares_text_and_image_embeddings():
    prepared = {
        "input_ids": mx.array([[11, 12]]),
        "attention_mask": mx.array([[1, 1]]),
        "pixel_values": mx.array([[3.0]]),
        "image_grid_thw": mx.array([[1, 2, 2]]),
    }
    preparation_calls = []

    def input_preparer(processor, **kwargs):
        preparation_calls.append((processor, kwargs))
        return dict(prepared)

    model = FakeQwen35Model()
    processor = _Processor()
    adapter = Qwen35MoeAdapter(input_preparer=input_preparer)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Look"}]}]
    image = object()

    input_ids, kwargs = adapter.prepare_inputs(
        model,
        processor,
        messages,
        [image],
        tools=[{"type": "function", "function": {"name": "inspect"}}],
        enable_thinking=False,
    )

    assert mx.array_equal(input_ids, prepared["input_ids"]).item()
    assert processor.template_calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": [{"type": "function", "function": {"name": "inspect"}}],
                "enable_thinking": False,
            },
        )
    ]
    assert preparation_calls == [
        (
            processor,
            {
                "images": [image],
                "prompts": "rendered prompt",
                "add_special_tokens": True,
                "padding": True,
            },
        )
    ]
    assert model.embedding_calls == [
        {
            "input_ids": prepared["input_ids"],
            "pixel_values": prepared["pixel_values"],
            "mask": prepared["attention_mask"],
            "kwargs": {"image_grid_thw": prepared["image_grid_thw"]},
        }
    ]
    assert kwargs == {
        "inputs_embeds": "prepared-embeddings",
        "mask": prepared["attention_mask"],
        "image_grid_thw": prepared["image_grid_thw"],
    }
