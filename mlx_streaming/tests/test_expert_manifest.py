import hashlib
import json

import pytest

from mlx_streaming.models.base import ModelDimensions
from mlx_streaming.prep.expert_manifest import (
    ExpertFileRecord,
    ExpertManifestError,
    ExpertStoreManifest,
    read_manifest,
    write_manifest,
)
from mlx_streaming.prep import pack_blob_from_experts
from mlx_streaming.prep.blob_layout import BLOB_V1_AFFINE


def _dimensions(**overrides):
    values = {
        "architecture": "qwen3_5_moe",
        "hidden_size": 2048,
        "num_layers": 40,
        "num_experts": 256,
        "top_k": 8,
        "expert_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "quant_mode": "affine",
        "quant_bits": 4,
        "quant_group_size": 64,
        "max_context": 262144,
    }
    values.update(overrides)
    return ModelDimensions(**values)


def _manifest(**overrides):
    values = {
        "schema_version": 1,
        "source_repository": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "source_revision": "1e20fd8d42056f870933bf98ca6211024744f7ec",
        "architecture": "qwen3_5_moe",
        "layer_indices": tuple(range(40)),
        "num_experts": 256,
        "top_k": 8,
        "hidden_size": 2048,
        "expert_intermediate_size": 512,
        "projection_names": ("gate_proj", "up_proj", "down_proj"),
        "projection_bits": {
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        },
        "group_size": 64,
        "quant_mode": "affine",
        "file_pattern": "layer{layer:02d}_expert{expert:03d}.safetensors",
        "files": (
            ExpertFileRecord(
                path="layer00_expert000.safetensors",
                size=123,
                sha256="a" * 64,
            ),
        ),
    }
    values.update(overrides)
    return ExpertStoreManifest(**values)


def test_expert_manifest_round_trips_without_losing_types(tmp_path):
    path = tmp_path / "expert_manifest.json"
    expected = _manifest()

    write_manifest(path, expected)

    assert read_manifest(path) == expected
    assert not list(tmp_path.glob("*.tmp"))
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 1
    assert raw["layer_indices"] == list(range(40))
    assert raw["files"][0]["sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("architecture", "other", "architecture"),
        ("layer_indices", tuple(range(39)), "layer_indices"),
        ("num_experts", 512, "num_experts"),
        ("top_k", 4, "top_k"),
        ("hidden_size", 4096, "hidden_size"),
        ("expert_intermediate_size", 1024, "expert_intermediate_size"),
        ("projection_bits", {"gate_proj": 8, "up_proj": 4, "down_proj": 4}, "projection_bits"),
        ("group_size", 128, "group_size"),
        ("quant_mode", "mxfp4", "quant_mode"),
    ],
)
def test_expert_manifest_rejects_adapter_mismatch(field, value, message):
    manifest = _manifest(**{field: value})

    with pytest.raises(ExpertManifestError, match=message):
        manifest.validate_against(_dimensions())


def test_expert_manifest_rejects_malformed_file_records(tmp_path):
    path = tmp_path / "expert_manifest.json"
    path.write_text(
        json.dumps(
            {
                **_manifest().to_dict(),
                "files": [
                    {
                        "path": "../outside.safetensors",
                        "size": 0,
                        "sha256": "not-a-hash",
                    }
                ],
            }
        )
    )

    with pytest.raises(ExpertManifestError, match="files"):
        read_manifest(path)


def test_expert_manifest_verifies_file_bytes(tmp_path):
    expert = tmp_path / "layer00_expert000.safetensors"
    expert.write_bytes(b"verified expert bytes")

    record = ExpertFileRecord(
        path=expert.name,
        size=expert.stat().st_size,
        sha256=hashlib.sha256(expert.read_bytes()).hexdigest(),
    )
    manifest = _manifest(files=(record,))
    manifest.verify_files(tmp_path)

    expert.write_bytes(b"modified expert bytes")
    with pytest.raises(ExpertManifestError, match="sha256"):
        manifest.verify_files(tmp_path)


def test_blob_packer_uses_expert_manifest_dimensions(tmp_path, monkeypatch):
    manifest = _manifest(
        layer_indices=(0,),
        num_experts=4,
        top_k=2,
        hidden_size=64,
        expert_intermediate_size=128,
        projection_bits={"gate_proj": 4, "up_proj": 4, "down_proj": 4},
        group_size=64,
        files=(),
    )
    write_manifest(tmp_path / "expert_manifest.json", manifest)
    monkeypatch.setattr(pack_blob_from_experts, "EXPERT_DIR", str(tmp_path))

    assert pack_blob_from_experts._meta() == (
        4,
        64,
        128,
        BLOB_V1_AFFINE,
        "affine",
    )
