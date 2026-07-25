from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_mac_mini_qwen35.py"
REVISION = "1e20fd8d42056f870933bf98ca6211024744f7ec"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_mac_mini_qwen35", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_space_check_reports_exact_deficits():
    script = _load_script()
    assert script.space_deficits(
        source_free=script.SOURCE_REQUIRED_BYTES - 7,
        runtime_free=script.RUNTIME_REQUIRED_BYTES - 11,
    ) == {"source": 7, "runtime": 11}
    assert script.space_deficits(
        source_free=script.SOURCE_REQUIRED_BYTES,
        runtime_free=script.RUNTIME_REQUIRED_BYTES,
    ) == {}


def test_repository_metadata_requires_pinned_revision_and_exact_total():
    script = _load_script()
    sibling = SimpleNamespace(
        rfilename="model.safetensors",
        size=5,
        lfs=SimpleNamespace(sha256="a" * 64),
    )
    metadata = SimpleNamespace(
        rfilename="config.json",
        size=7,
        lfs=None,
    )
    with pytest.raises(RuntimeError, match=REVISION):
        script.repository_records(
            SimpleNamespace(sha="0" * 40, siblings=[sibling]),
            expected_total=5,
        )
    with pytest.raises(RuntimeError, match="source bytes"):
        script.repository_records(
            SimpleNamespace(sha=REVISION, siblings=[sibling]),
            expected_total=6,
        )
    records = script.repository_records(
        SimpleNamespace(sha=REVISION, siblings=[sibling, metadata]),
        expected_total=5,
    )
    assert len(records) == 2
    assert records[1].sha256 == "a" * 64


def test_download_is_revision_pinned_resumable_and_writes_marker(tmp_path):
    script = _load_script()
    calls = []
    source = tmp_path / "source"
    source.mkdir()
    records = [script.SourceFile("config.json", 2, None)]

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        (source / "config.json").write_bytes(b"{}")
        return str(source)

    script.download_source(source, records, snapshot_download_fn=snapshot_download)

    assert calls == [
        {
            "repo_id": script.REPOSITORY,
            "revision": REVISION,
            "local_dir": str(source),
            "force_download": False,
        }
    ]
    assert (source / ".vates-source-revision").read_text().strip() == REVISION


def test_download_refuses_unrelated_existing_files(tmp_path):
    script = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "unrelated.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="unrelated"):
        script.download_source(
            source,
            [script.SourceFile("config.json", 2, None)],
            snapshot_download_fn=lambda **_kwargs: pytest.fail("must not download"),
        )


def test_verify_source_checks_size_sha_and_total(tmp_path):
    script = _load_script()
    source = tmp_path / "source"
    source.mkdir()
    payload = b"verified"
    (source / "model.safetensors").write_bytes(payload)
    (source / "config.json").write_bytes(b"{}")
    record = script.SourceFile(
        "model.safetensors",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    metadata = script.SourceFile("config.json", 2, None)

    result = script.verify_source(
        source,
        [record, metadata],
        expected_total=len(payload),
    )
    assert result["source_bytes"] == len(payload)
    assert result["repository_bytes"] == len(payload) + 2
    assert result["verified_files"] == 2

    (source / "model.safetensors").write_bytes(b"altered!")
    with pytest.raises(RuntimeError, match="sha256"):
        script.verify_source(
            source,
            [record, metadata],
            expected_total=len(payload),
        )


def test_phase_state_is_atomic_and_revision_scoped(tmp_path):
    script = _load_script()
    state_path = tmp_path / "state.json"
    script.record_phase(
        state_path,
        "download",
        {"source_bytes": 10},
    )
    state = json.loads(state_path.read_text())
    assert state["revision"] == REVISION
    assert state["completed_phases"] == ["download"]
    assert state["phases"]["download"]["source_bytes"] == 10
    assert list(tmp_path.glob(".*.tmp")) == []


def test_verify_runtime_requires_all_40_blob_layers(tmp_path):
    script = _load_script()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "blob_index.json").write_text(
        json.dumps(
            {
                "num_experts": 256,
                "layers": list(range(40)),
                "stride": 4,
            }
        )
    )
    for layer in range(39):
        (blobs / f"layer{layer:02d}.blob").write_bytes(b"x" * (4 * 256))
    with pytest.raises(RuntimeError, match="layer39"):
        script.verify_blob_store(blobs)
