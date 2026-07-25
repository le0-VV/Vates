"""Fail-closed model-adapter registration and configuration lookup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from mlx_streaming.models.base import ModelAdapter


class UnsupportedArchitecture(ValueError):
    """The model configuration does not name a registered architecture."""


class DuplicateArchitecture(ValueError):
    """Two adapters attempted to claim the same architecture."""


_ADAPTERS: dict[str, ModelAdapter] = {}


def register_adapter(adapter: ModelAdapter) -> None:
    architecture = getattr(adapter, "architecture", None)
    if not isinstance(architecture, str) or not architecture:
        raise UnsupportedArchitecture("adapter architecture must be a non-empty string")
    if architecture in _ADAPTERS:
        raise DuplicateArchitecture(
            f"adapter already registered for architecture {architecture!r}"
        )
    _ADAPTERS[architecture] = adapter


def adapter_for_config(config: Mapping[str, object]) -> ModelAdapter:
    architecture = config.get("model_type")
    if not isinstance(architecture, str) or not architecture:
        raise UnsupportedArchitecture(
            "model config model_type must be a non-empty string"
        )
    try:
        return _ADAPTERS[architecture]
    except KeyError as exc:
        raise UnsupportedArchitecture(
            f"unsupported model architecture {architecture!r}"
        ) from exc


def adapter_for_path(path: str | Path) -> ModelAdapter:
    config_path = Path(path) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise UnsupportedArchitecture("model config must be a JSON object")
    return adapter_for_config(config)
