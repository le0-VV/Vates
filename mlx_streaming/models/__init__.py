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
from mlx_streaming.models.qwen35 import QWEN35_ADAPTER, Qwen35MoeAdapter

__all__ = [
    "DuplicateArchitecture",
    "ExpertLayerSpec",
    "HybridCacheState",
    "LoadedModel",
    "ModelAdapter",
    "ModelDimensions",
    "QWEN35_ADAPTER",
    "Qwen35MoeAdapter",
    "UnsupportedArchitecture",
    "adapter_for_config",
    "adapter_for_path",
    "register_adapter",
]
