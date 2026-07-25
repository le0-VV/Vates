from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import SwitchGLU

from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.core.prefetch.patch import patch_model_filebacked
from mlx_streaming import model_builder
from mlx_streaming.models.base import ExpertLayerSpec, LoadedModel
from mlx_streaming.prep.expert_manifest import (
    ExpertFileRecord,
    ExpertStoreManifest,
    read_manifest,
)
from mlx_streaming.prep.split_experts import split_model, split_switch_glu


class _SharedExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class _MoeBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, intermediate, experts)
        self.shared_expert = _SharedExpert(hidden, intermediate)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)
        self.top_k = top_k

    def __call__(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return routed + shared


class _Adapter:
    architecture = "tiny_qwen3_5_moe"

    def __init__(self, model, hidden, intermediate, experts, top_k):
        self.model = model
        self.hidden = hidden
        self.intermediate = intermediate
        self.experts = experts
        self.top_k = top_k

    def layers(self, model):
        return model.layers

    def language_model(self, model):
        return SimpleNamespace(model=model)

    def load(self, path, *, revision, lazy):
        return LoadedModel(model=self.model, processor=object(), config={})

    def expert_layers(self, model):
        return tuple(
            ExpertLayerSpec(
                layer_index=index,
                block=layer.mlp,
                hidden_size=self.hidden,
                intermediate_size=self.intermediate,
                num_experts=self.experts,
                top_k=self.top_k,
                normalise_topk=True,
            )
            for index, layer in enumerate(model.layers)
        )


def _file_record(path, root):
    data = path.read_bytes()
    return ExpertFileRecord(
        path=str(path.relative_to(root)),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_adapter_patch_streams_routed_experts_and_keeps_shared_expert(tmp_path):
    mx.random.seed(7)
    hidden, intermediate, experts, top_k = 64, 128, 8, 2
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                mlp=_MoeBlock(hidden, intermediate, experts, top_k)
            )
            for _ in range(2)
        ]
    )
    mx.eval(
        [
            parameter
            for layer in model.layers
            for parameter in layer.mlp.parameters().values()
        ]
    )
    for layer in model.layers:
        nn.quantize(layer.mlp.switch_mlp, group_size=64, bits=4)
    mx.eval(
        [
            parameter
            for layer in model.layers
            for parameter in layer.mlp.parameters().values()
        ]
    )
    x = mx.random.normal((1, 4, hidden))
    references = [layer.mlp(x) for layer in model.layers]
    mx.eval(references)
    shared = [layer.mlp.shared_expert for layer in model.layers]

    for index, layer in enumerate(model.layers):
        split_switch_glu(layer.mlp.switch_mlp, str(tmp_path), index)
    files = tuple(
        _file_record(path, tmp_path)
        for path in sorted(tmp_path.glob("layer*_expert*.safetensors"))
    )
    manifest = ExpertStoreManifest(
        schema_version=1,
        source_repository="test/tiny-qwen35",
        source_revision="1" * 40,
        architecture="tiny_qwen3_5_moe",
        layer_indices=(0, 1),
        num_experts=experts,
        top_k=top_k,
        hidden_size=hidden,
        expert_intermediate_size=intermediate,
        projection_names=("gate_proj", "up_proj", "down_proj"),
        projection_bits={"gate_proj": 4, "up_proj": 4, "down_proj": 4},
        group_size=64,
        quant_mode="affine",
        file_pattern="layer{layer:02d}_expert{expert:03d}.safetensors",
        files=files,
    )
    store = FileExpertStore(str(tmp_path), capacity=experts)
    adapter = _Adapter(model, hidden, intermediate, experts, top_k)

    patched = patch_model_filebacked(
        model,
        store,
        adapter=adapter,
        manifest=manifest,
    )

    assert patched == 2
    for index, layer in enumerate(model.layers):
        assert isinstance(layer.mlp, FileStreamingMoeBlock)
        assert layer.mlp.shared_expert is shared[index]
        assert not hasattr(layer.mlp, "switch_mlp")
        output = layer.mlp(x)
        mx.eval(output)
        assert mx.allclose(output, references[index], atol=1e-4).item()


def test_split_model_uses_adapter_and_writes_complete_hashed_manifest(tmp_path):
    hidden, intermediate, experts, top_k = 64, 128, 4, 2
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                mlp=_MoeBlock(hidden, intermediate, experts, top_k)
            )
        ]
    )
    mx.eval(model.layers[0].mlp.parameters())
    nn.quantize(model.layers[0].mlp.switch_mlp, group_size=64, bits=4)
    mx.eval(model.layers[0].mlp.parameters())
    adapter = _Adapter(model, hidden, intermediate, experts, top_k)

    result = split_model(
        "unused-local-model",
        str(tmp_path),
        adapter=adapter,
        source_repository="test/tiny-qwen35",
        source_revision="2" * 40,
    )

    assert result["moe_layers"] == [0]
    assert result["dims"] == {
        "hidden": hidden,
        "moe_intermediate": intermediate,
        "num_experts": experts,
        "group_size": 64,
        "bits": 4,
    }
    manifest = read_manifest(tmp_path / "expert_manifest.json")
    assert manifest.architecture == "tiny_qwen3_5_moe"
    assert len(manifest.files) == experts
    manifest.verify_files(tmp_path)
    assert all("shared_expert" not in record.path for record in manifest.files)


def test_model_builder_assembles_adapter_model_from_manifest(tmp_path, monkeypatch):
    hidden, intermediate, experts, top_k = 64, 128, 4, 2
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                mlp=_MoeBlock(hidden, intermediate, experts, top_k)
            )
        ]
    )
    mx.eval(model.layers[0].mlp.parameters())
    nn.quantize(model.layers[0].mlp.switch_mlp, group_size=64, bits=4)
    mx.eval(model.layers[0].mlp.parameters())
    adapter = _Adapter(model, hidden, intermediate, experts, top_k)
    split_model(
        "unused-local-model",
        str(tmp_path),
        adapter=adapter,
        source_repository="test/tiny-qwen35",
        source_revision="3" * 40,
    )
    monkeypatch.setattr(model_builder, "MODEL", "unused-local-model")
    monkeypatch.setattr(model_builder, "EXPERT_DIR", str(tmp_path))
    monkeypatch.setattr(model_builder, "EXPERT_SLOTS", experts)

    built, processor, store = model_builder.build_streaming_model(adapter=adapter)

    assert built is model
    assert processor is not None
    assert store.root == str(tmp_path)
    assert isinstance(model.layers[0].mlp, FileStreamingMoeBlock)


def test_blob_expert_count_is_required_and_model_derived(tmp_path):
    with pytest.raises(ValueError, match="blob_index.json"):
        model_builder._blob_expert_count(str(tmp_path), expected=256)

    (tmp_path / "blob_index.json").write_text(
        json.dumps({"num_experts": 256})
    )
    assert model_builder._blob_expert_count(str(tmp_path), expected=256) == 256

    (tmp_path / "blob_index.json").write_text(
        json.dumps({"num_experts": 512})
    )
    with pytest.raises(ValueError, match="expected 256"):
        model_builder._blob_expert_count(str(tmp_path), expected=256)
