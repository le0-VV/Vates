import json

import pytest

from mlx_streaming.models.registry import (
    DuplicateArchitecture,
    UnsupportedArchitecture,
    adapter_for_config,
    adapter_for_path,
    register_adapter,
)


class _Adapter:
    def __init__(self, architecture: str):
        self.architecture = architecture


def test_registry_selects_adapter_from_exact_architecture(tmp_path):
    adapter = _Adapter("test_exact_architecture")
    register_adapter(adapter)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "test_exact_architecture"})
    )

    assert adapter_for_path(tmp_path) is adapter
    assert adapter_for_config({"model_type": "test_exact_architecture"}) is adapter


def test_registry_rejects_unknown_architecture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "unknown_moe"})
    )

    with pytest.raises(UnsupportedArchitecture, match="unknown_moe"):
        adapter_for_path(tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": None},
        {"model_type": 35},
        {"model_type": ""},
    ],
)
def test_registry_rejects_missing_or_non_string_architecture(config):
    with pytest.raises(UnsupportedArchitecture, match="model_type"):
        adapter_for_config(config)


def test_registry_rejects_duplicate_architecture():
    architecture = "test_duplicate_architecture"
    register_adapter(_Adapter(architecture))

    with pytest.raises(DuplicateArchitecture, match=architecture):
        register_adapter(_Adapter(architecture))
