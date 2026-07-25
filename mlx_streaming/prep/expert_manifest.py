"""Versioned metadata for a byte-verifiable routed-expert store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from mlx_streaming.models.base import ModelDimensions


class ExpertManifestError(ValueError):
    """An expert manifest is malformed or incompatible with its model."""


@dataclass(frozen=True)
class ExpertFileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ExpertStoreManifest:
    schema_version: int
    source_repository: str
    source_revision: str
    architecture: str
    layer_indices: tuple[int, ...]
    num_experts: int
    top_k: int
    hidden_size: int
    expert_intermediate_size: int
    projection_names: tuple[str, ...]
    projection_bits: dict[str, int]
    group_size: int
    quant_mode: str
    file_pattern: str
    files: tuple[ExpertFileRecord, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["layer_indices"] = list(self.layer_indices)
        value["projection_names"] = list(self.projection_names)
        value["files"] = [asdict(record) for record in self.files]
        return value

    def validate_against(
        self,
        dimensions: ModelDimensions,
        *,
        require_complete: bool = False,
    ) -> None:
        expected = {
            "schema_version": 1,
            "architecture": dimensions.architecture,
            "layer_indices": tuple(range(dimensions.num_layers)),
            "num_experts": dimensions.num_experts,
            "top_k": dimensions.top_k,
            "hidden_size": dimensions.hidden_size,
            "expert_intermediate_size": dimensions.expert_intermediate_size,
            "projection_bits": {
                name: dimensions.quant_bits for name in self.projection_names
            },
            "group_size": dimensions.quant_group_size,
            "quant_mode": dimensions.quant_mode,
        }
        for field, wanted in expected.items():
            actual = getattr(self, field)
            if actual != wanted:
                raise ExpertManifestError(
                    f"{field} must be {wanted!r}, got {actual!r}"
                )
        if self.projection_names != ("gate_proj", "up_proj", "down_proj"):
            raise ExpertManifestError(
                "projection_names must be "
                "('gate_proj', 'up_proj', 'down_proj')"
            )
        _validate_file_records(self.files)
        if require_complete:
            expected_count = len(self.layer_indices) * self.num_experts
            if len(self.files) != expected_count:
                raise ExpertManifestError(
                    f"files must contain {expected_count} records, "
                    f"got {len(self.files)}"
                )

    def verify_files(self, root: str | Path) -> None:
        directory = Path(root)
        for record in self.files:
            path = directory / record.path
            if not path.is_file():
                raise ExpertManifestError(f"files entry is missing: {record.path}")
            actual_size = path.stat().st_size
            if actual_size != record.size:
                raise ExpertManifestError(
                    f"size mismatch for {record.path}: "
                    f"expected {record.size}, got {actual_size}"
                )
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != record.sha256:
                raise ExpertManifestError(
                    f"sha256 mismatch for {record.path}: "
                    f"expected {record.sha256}, got {actual_hash}"
                )


def _validate_file_records(records: tuple[ExpertFileRecord, ...]) -> None:
    seen = set()
    for record in records:
        relative = PurePosixPath(record.path)
        valid_path = (
            bool(record.path)
            and not relative.is_absolute()
            and ".." not in relative.parts
            and relative.suffix == ".safetensors"
        )
        valid_hash = bool(re.fullmatch(r"[0-9a-f]{64}", record.sha256))
        if (
            not valid_path
            or record.size <= 0
            or not valid_hash
            or record.path in seen
        ):
            raise ExpertManifestError(
                f"files contains invalid record {record!r}"
            )
        seen.add(record.path)


def _manifest_from_dict(value: object) -> ExpertStoreManifest:
    if not isinstance(value, dict):
        raise ExpertManifestError("manifest must be a JSON object")
    try:
        manifest = ExpertStoreManifest(
            schema_version=value["schema_version"],
            source_repository=value["source_repository"],
            source_revision=value["source_revision"],
            architecture=value["architecture"],
            layer_indices=tuple(value["layer_indices"]),
            num_experts=value["num_experts"],
            top_k=value["top_k"],
            hidden_size=value["hidden_size"],
            expert_intermediate_size=value["expert_intermediate_size"],
            projection_names=tuple(value["projection_names"]),
            projection_bits=dict(value["projection_bits"]),
            group_size=value["group_size"],
            quant_mode=value["quant_mode"],
            file_pattern=value["file_pattern"],
            files=tuple(ExpertFileRecord(**record) for record in value["files"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExpertManifestError(f"manifest schema is invalid: {exc}") from exc
    if manifest.schema_version != 1:
        raise ExpertManifestError(
            f"schema_version must be 1, got {manifest.schema_version!r}"
        )
    if not manifest.source_repository:
        raise ExpertManifestError("source_repository must be non-empty")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.source_revision):
        raise ExpertManifestError("source_revision must be a 40-character SHA")
    _validate_file_records(manifest.files)
    return manifest


def read_manifest(path: str | Path) -> ExpertStoreManifest:
    with Path(path).open(encoding="utf-8") as handle:
        return _manifest_from_dict(json.load(handle))


def write_manifest(path: str | Path, manifest: ExpertStoreManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _manifest_from_dict(manifest.to_dict())
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
