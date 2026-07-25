from __future__ import annotations

import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_mac_mini_qwen35.py"
REVISION = "1e20fd8d42056f870933bf98ca6211024744f7ec"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("run_mac_mini_qwen35", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifest(path, *, revision=REVISION):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_repository": "mlx-community/Qwen3.5-35B-A3B-4bit",
                "source_revision": revision,
                "architecture": "qwen3_5_moe",
                "layer_indices": list(range(40)),
                "num_experts": 256,
                "top_k": 8,
                "hidden_size": 2048,
                "expert_intermediate_size": 512,
                "projection_names": ["gate_proj", "up_proj", "down_proj"],
                "projection_bits": {
                    "gate_proj": 4,
                    "up_proj": 4,
                    "down_proj": 4,
                },
                "group_size": 64,
                "quant_mode": "affine",
                "file_pattern": "layer_{layer}/expert_{expert}.safetensors",
                "files": [],
            }
        )
    )


def test_launcher_paths_and_command_are_fixed_for_qwen35_protocol_server():
    launcher = _load_launcher()
    command = launcher.build_command(["--host", "0.0.0.0"])
    joined = "\n".join(command)

    assert str(launcher.MODEL_DIR) == (
        "/Volumes/Leonard's RAID/Vates/models/Qwen3.5-35B-A3B-4bit"
    )
    assert str(launcher.RUNTIME_DIR) == (
        "/Users/leonardw/Library/Application Support/Vates/"
        "qwen3.5-35b-a3b-4bit"
    )
    assert command[:2] == [
        "/Users/leonardw/Projects/Vates/.venv/bin/vates",
        "serve",
    ]
    assert command[command.index("--engine") + 1] == "general"
    assert command[command.index("--adapter") + 1] == "auto"
    assert command[command.index("--context-length") + 1] == "131072"
    assert command[command.index("--model-id") + 1] == "qwen3.5-35b-a3b-4bit"
    assert "--mtp-out" not in command
    assert "KV_QUANT" not in joined
    assert command[-2:] == ["--host", "0.0.0.0"]


def test_launcher_environment_removes_mtp_and_kv_tuning():
    launcher = _load_launcher()
    environment = launcher.runtime_environment(
        {
            "LANG": "en_GB.UTF-8",
            "KV_QUANT": "1",
            "MTP_ADAPTIVE_DEPTH": "1",
            "CROSS_LAYER_PREDICT_WIDTH": "16",
        }
    )
    assert environment["LANG"] == "en_GB.UTF-8"
    assert "KV_QUANT" not in environment
    assert "MTP_ADAPTIVE_DEPTH" not in environment
    assert "CROSS_LAYER_PREDICT_WIDTH" not in environment


def test_source_revision_marker_is_required_and_exact(tmp_path):
    launcher = _load_launcher()
    marker = tmp_path / ".vates-source-revision"
    with pytest.raises(RuntimeError, match="revision marker"):
        launcher.ensure_source_revision(marker)
    marker.write_text("wrong\n")
    with pytest.raises(RuntimeError, match=REVISION):
        launcher.ensure_source_revision(marker)
    marker.write_text(REVISION + "\n")
    launcher.ensure_source_revision(marker)


def test_runtime_manifest_must_match_pinned_model(tmp_path):
    launcher = _load_launcher()
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    manifest = expert_dir / "expert_manifest.json"
    _write_manifest(manifest, revision="0" * 40)
    with pytest.raises(RuntimeError, match="revision"):
        launcher.ensure_runtime_manifest(expert_dir)
    _write_manifest(manifest)
    launcher.ensure_runtime_manifest(expert_dir)


def test_occupied_port_is_rejected():
    launcher = _load_launcher()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="occupied"):
            launcher.ensure_port_available("127.0.0.1", port)


def test_recognised_vates_process_is_rejected_but_unrelated_python_is_allowed():
    launcher = _load_launcher()
    launcher.ensure_no_vates_process(
        "101 /usr/bin/python unrelated.py\n",
        current_pid=999,
    )
    with pytest.raises(RuntimeError, match="already running"):
        launcher.ensure_no_vates_process(
            "101 /Users/leonardw/Projects/Vates/.venv/bin/vates serve\n",
            current_pid=999,
        )


def test_pid_lock_is_exclusive_and_pid_file_is_removed(tmp_path):
    launcher = _load_launcher()
    pid_path = tmp_path / "qwen35.pid"
    with launcher.pid_lock(pid_path):
        assert pid_path.read_text().strip() == str(os.getpid())
        with pytest.raises(RuntimeError, match="already held"):
            with launcher.pid_lock(pid_path):
                pass
        assert pid_path.read_text().strip() == str(os.getpid())
    assert not pid_path.exists()


def test_help_does_not_inspect_or_start_the_model(monkeypatch, capsys):
    launcher = _load_launcher()
    monkeypatch.setattr(
        launcher,
        "ensure_raid_mounted",
        lambda: pytest.fail("help must not inspect runtime state"),
    )
    assert launcher.main(["--help"]) == 0
    assert "Qwen3.5" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "/tmp/model"],
        ["--engine", "mtp"],
        ["--context-length=4096"],
        ["--model-id", "other"],
        ["--expert-slots", "1"],
        ["--port", "9000"],
    ],
)
def test_fixed_profile_cannot_be_overridden(extra):
    launcher = _load_launcher()
    with pytest.raises(ValueError, match="fixed Qwen3.5 profile"):
        launcher.build_command(extra)
