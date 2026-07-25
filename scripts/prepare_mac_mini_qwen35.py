#!/usr/bin/env python3
"""Resumable acquisition and preparation for the pinned Qwen3.5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple


REPOSITORY = "mlx-community/Qwen3.5-35B-A3B-4bit"
REVISION = "1e20fd8d42056f870933bf98ca6211024744f7ec"
SOURCE_BYTES = 20_411_668_782
SOURCE_REQUIRED_BYTES = 22_000_000_000
RUNTIME_REQUIRED_BYTES = 38_000_000_000

SOURCE_DIR = Path(
    "/Volumes/Leonard's RAID/Vates/models/Qwen3.5-35B-A3B-4bit"
)
RUNTIME_DIR = Path(
    "/Users/leonardw/Library/Application Support/Vates/"
    "qwen3.5-35b-a3b-4bit"
)
EXPERT_DIR = RUNTIME_DIR / "experts"
BLOB_DIR = RUNTIME_DIR / "blobs"
STATE_FILE = RUNTIME_DIR / "preparation-state.json"
REVISION_FILE_NAME = ".vates-source-revision"
PHASES = (
    "inspect",
    "download",
    "verify-source",
    "split-experts",
    "pack-blobs",
    "verify-runtime",
)


class SourceFile(NamedTuple):
    path: str
    size: int
    sha256: str | None


def space_deficits(*, source_free: int, runtime_free: int) -> dict[str, int]:
    deficits = {}
    if source_free < SOURCE_REQUIRED_BYTES:
        deficits["source"] = SOURCE_REQUIRED_BYTES - source_free
    if runtime_free < RUNTIME_REQUIRED_BYTES:
        deficits["runtime"] = RUNTIME_REQUIRED_BYTES - runtime_free
    return deficits


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"no existing parent for {path}")
        current = current.parent
    return current


def inspect_space(source: Path, runtime: Path) -> dict[str, int]:
    source_free = shutil.disk_usage(_nearest_existing(source)).free
    runtime_free = shutil.disk_usage(_nearest_existing(runtime)).free
    deficits = space_deficits(
        source_free=source_free,
        runtime_free=runtime_free,
    )
    if deficits:
        details = ", ".join(
            f"{name} deficit={amount} bytes"
            for name, amount in deficits.items()
        )
        raise RuntimeError(f"insufficient preparation space: {details}")
    return {
        "source_free_bytes": source_free,
        "runtime_free_bytes": runtime_free,
        "source_required_bytes": SOURCE_REQUIRED_BYTES,
        "runtime_required_bytes": RUNTIME_REQUIRED_BYTES,
    }


def repository_records(
    info,
    *,
    expected_total: int = SOURCE_BYTES,
) -> tuple[SourceFile, ...]:
    actual_revision = getattr(info, "sha", None)
    if actual_revision != REVISION:
        raise RuntimeError(
            f"repository revision must be {REVISION}, got {actual_revision!r}"
        )
    records = []
    for sibling in getattr(info, "siblings", ()) or ():
        path = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if not isinstance(path, str) or not isinstance(size, int) or size < 0:
            raise RuntimeError("repository file metadata is incomplete")
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            sha256 = lfs.get("sha256")
        else:
            sha256 = getattr(lfs, "sha256", None)
        if sha256 is not None and (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RuntimeError(f"repository LFS hash is invalid for {path}")
        records.append(SourceFile(path, size, sha256))
    records.sort(key=lambda record: record.path)
    total = sum(
        record.size
        for record in records
        if record.sha256 is not None
    )
    if total != expected_total:
        raise RuntimeError(
            f"repository source bytes must be {expected_total}, got {total}"
        )
    return tuple(records)


def fetch_repository_records() -> tuple[SourceFile, ...]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(
        REPOSITORY,
        revision=REVISION,
        files_metadata=True,
    )
    return repository_records(info)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _allowed_source_files(records: tuple[SourceFile, ...] | list[SourceFile]) -> set[str]:
    return {record.path for record in records}


def _refuse_unrelated_files(
    source: Path,
    records: tuple[SourceFile, ...] | list[SourceFile],
) -> None:
    if not source.exists():
        return
    allowed = _allowed_source_files(records)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if (
            relative == REVISION_FILE_NAME
            or relative.startswith(".cache/huggingface/")
            or relative in allowed
        ):
            continue
        raise RuntimeError(f"refusing to overwrite unrelated source file {relative}")


def download_source(
    source: Path,
    records: tuple[SourceFile, ...] | list[SourceFile],
    *,
    snapshot_download_fn=None,
) -> dict[str, object]:
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    _refuse_unrelated_files(source, records)
    source.mkdir(parents=True, exist_ok=True)
    result = snapshot_download_fn(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=str(source),
        force_download=False,
    )
    if Path(result).resolve() != source.resolve():
        raise RuntimeError(
            f"snapshot download returned {result}, expected {source}"
        )
    _atomic_write_text(source / REVISION_FILE_NAME, REVISION + "\n")
    return {
        "source": str(source),
        "revision": REVISION,
        "expected_files": len(records),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(
    source: Path,
    records: tuple[SourceFile, ...] | list[SourceFile],
    *,
    expected_total: int = SOURCE_BYTES,
) -> dict[str, int]:
    repository_total = 0
    source_total = 0
    for record in records:
        path = source / record.path
        if not path.is_file():
            raise RuntimeError(f"source file is missing: {record.path}")
        size = path.stat().st_size
        if size != record.size:
            raise RuntimeError(
                f"source size mismatch for {record.path}: "
                f"expected {record.size}, got {size}"
            )
        repository_total += size
        if record.sha256 is not None:
            source_total += size
            actual_hash = _sha256(path)
            if actual_hash != record.sha256:
                raise RuntimeError(
                    f"source sha256 mismatch for {record.path}: "
                    f"expected {record.sha256}, got {actual_hash}"
                )
    if source_total != expected_total:
        raise RuntimeError(
            f"verified source bytes must be {expected_total}, got {source_total}"
        )
    _atomic_write_text(source / REVISION_FILE_NAME, REVISION + "\n")
    return {
        "source_bytes": source_total,
        "repository_bytes": repository_total,
        "verified_files": len(records),
    }


def record_phase(
    state_path: Path,
    phase: str,
    details: dict[str, object],
) -> None:
    if phase not in PHASES:
        raise ValueError(f"unknown preparation phase {phase!r}")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("revision") != REVISION:
            raise RuntimeError("preparation state belongs to another revision")
    else:
        state = {
            "schema_version": 1,
            "repository": REPOSITORY,
            "revision": REVISION,
            "completed_phases": [],
            "phases": {},
        }
    if phase not in state["completed_phases"]:
        state["completed_phases"].append(phase)
    state["phases"][phase] = details
    _atomic_write_text(
        state_path,
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )


def split_experts(source: Path, expert_dir: Path) -> dict:
    from mlx_streaming.models import adapter_for_path
    from mlx_streaming.prep.split_experts import split_model

    adapter = adapter_for_path(source)
    return split_model(
        str(source),
        str(expert_dir),
        adapter=adapter,
        source_repository=REPOSITORY,
        source_revision=REVISION,
    )


def pack_blobs(expert_dir: Path, blob_dir: Path) -> dict[str, object]:
    expert_bytes = sum(
        path.stat().st_size
        for path in expert_dir.glob("layer*_expert*.safetensors")
    )
    free = shutil.disk_usage(_nearest_existing(blob_dir)).free
    required = expert_bytes + 2_000_000_000
    if free < required:
        raise RuntimeError(
            f"insufficient blob space: deficit={required - free} bytes"
        )
    environment = dict(os.environ)
    environment.update(
        {
            "EXPERT_DIR": str(expert_dir),
            "BLOB_DIR": str(blob_dir),
            "LAYERS": "all",
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_streaming.prep.pack_blob_from_experts",
        ],
        check=True,
        env=environment,
    )
    return {
        "expert_bytes": expert_bytes,
        "blob_dir": str(blob_dir),
    }


def verify_blob_store(blob_dir: Path) -> dict[str, int]:
    index_path = blob_dir / "blob_index.json"
    if not index_path.is_file():
        raise RuntimeError(f"blob index is missing at {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("num_experts") != 256:
        raise RuntimeError("blob index must contain 256 experts")
    if index.get("layers") != list(range(40)):
        raise RuntimeError("blob index must contain layers 0 through 39")
    stride = index.get("stride")
    if not isinstance(stride, int) or stride <= 0:
        raise RuntimeError("blob index stride must be positive")
    expected_size = stride * 256
    total = 0
    for layer in range(40):
        path = blob_dir / f"layer{layer:02d}.blob"
        if not path.is_file():
            raise RuntimeError(f"blob layer{layer:02d} is missing")
        size = path.stat().st_size
        if size != expected_size:
            raise RuntimeError(
                f"blob layer{layer:02d} size must be {expected_size}, got {size}"
            )
        total += size
    return {"blob_bytes": total, "blob_layers": 40}


def verify_runtime(source: Path, expert_dir: Path, blob_dir: Path) -> dict[str, int]:
    from mlx_streaming.models import adapter_for_path
    from mlx_streaming.prep.expert_manifest import read_manifest

    adapter = adapter_for_path(source)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dimensions = adapter.validate_config(config)
    manifest = read_manifest(expert_dir / "expert_manifest.json")
    manifest.validate_against(dimensions, require_complete=True)
    manifest.verify_files(expert_dir)
    blob = verify_blob_store(blob_dir)
    return {
        "expert_files": len(manifest.files),
        "expert_bytes": sum(record.size for record in manifest.files),
        **blob,
    }


def run_phase(
    phase: str,
    *,
    source: Path = SOURCE_DIR,
    runtime: Path = RUNTIME_DIR,
) -> dict[str, object]:
    state_path = runtime / STATE_FILE.name
    if phase == "inspect":
        result = inspect_space(source, runtime)
    elif phase == "download":
        records = fetch_repository_records()
        result = download_source(source, records)
    elif phase == "verify-source":
        records = fetch_repository_records()
        result = verify_source(source, records)
    elif phase == "split-experts":
        result = split_experts(source, runtime / "experts")
    elif phase == "pack-blobs":
        result = pack_blobs(runtime / "experts", runtime / "blobs")
    elif phase == "verify-runtime":
        result = verify_runtime(
            source,
            runtime / "experts",
            runtime / "blobs",
        )
    else:
        raise ValueError(f"unknown preparation phase {phase!r}")
    record_phase(state_path, phase, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--runtime", type=Path, default=RUNTIME_DIR)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    result = run_phase(
        args.phase,
        source=args.source,
        runtime=args.runtime,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
