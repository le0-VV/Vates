from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


QWEN35_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "shared_expert_intermediate_size": 512,
        "moe_intermediate_size": 512,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "vocab_size": 248320,
    },
    "vision_config": {
        "model_type": "qwen3_5_moe",
        "depth": 27,
        "hidden_size": 1152,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [],
    },
}


def qwen35_config() -> dict:
    return deepcopy(QWEN35_CONFIG)


class FakeLanguageModel:
    def __init__(self, layers):
        self.model = SimpleNamespace(layers=layers)
        self.calls = []

    def make_cache(self):
        return [
            ArraysCache(size=2) if (index + 1) % 4 else KVCache()
            for index in range(len(self.model.layers))
        ]

    def __call__(self, input_ids, *, cache, **kwargs):
        count = int(input_ids.shape[1])
        for index, entry in enumerate(cache):
            if (index + 1) % 4 == 0:
                entry.offset += count
        self.calls.append((input_ids, cache, kwargs))
        return SimpleNamespace(logits=mx.zeros((1, count, 16)))


class FakeQwen35Model:
    def __init__(self):
        layers = []
        for _ in range(40):
            gate_proj = SimpleNamespace(
                input_dims=2048,
                output_dims=512,
                num_experts=256,
                group_size=64,
                bits=4,
            )
            switch_mlp = SimpleNamespace(gate_proj=gate_proj)
            block = SimpleNamespace(
                gate=object(),
                switch_mlp=switch_mlp,
                shared_expert=object(),
                shared_expert_gate=object(),
                top_k=8,
                num_experts=256,
            )
            layers.append(SimpleNamespace(mlp=block))
        self.language_model = FakeLanguageModel(layers)
        self.embedding_calls = []

    def get_input_embeddings(
        self,
        input_ids,
        pixel_values=None,
        *,
        mask=None,
        **kwargs,
    ):
        self.embedding_calls.append(
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "mask": mask,
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(
            inputs_embeds="prepared-embeddings",
            to_dict=lambda: {"inputs_embeds": "prepared-embeddings"},
        )
