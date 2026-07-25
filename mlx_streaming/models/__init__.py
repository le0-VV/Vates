"""Model adapter contracts and fail-closed architecture lookup."""

from mlx_streaming.models.base import (
    ExpertLayerSpec,
    HybridCacheState,
    LoadedModel,
    ModelAdapter,
    ModelDimensions,
)
from mlx_streaming.models.registry import (
    DuplicateArchitecture,
    UnsupportedArchitecture,
    adapter_for_config,
    adapter_for_path,
    register_adapter,
)

__all__ = [
    "DuplicateArchitecture",
    "ExpertLayerSpec",
    "HybridCacheState",
    "LoadedModel",
    "ModelAdapter",
    "ModelDimensions",
    "UnsupportedArchitecture",
    "adapter_for_config",
    "adapter_for_path",
    "register_adapter",
]
