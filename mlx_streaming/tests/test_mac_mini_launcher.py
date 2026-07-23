import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_mac_mini_qwen3_next.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("run_mac_mini_qwen3_next", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_raid_is_rejected(monkeypatch, tmp_path):
    launcher = _load_launcher()
    monkeypatch.setattr(os.path, "ismount", lambda path: False)
    with pytest.raises(RuntimeError, match="Leonard's RAID is not mounted"):
        launcher.ensure_raid_mounted(tmp_path)


def test_mounted_raid_is_accepted(monkeypatch, tmp_path):
    launcher = _load_launcher()
    monkeypatch.setattr(os.path, "ismount", lambda path: Path(path) == tmp_path)
    launcher.ensure_raid_mounted(tmp_path)


def test_command_uses_hot_internal_experts_and_cold_raid_assets():
    launcher = _load_launcher()
    command = launcher.build_command(["--plain", "--stats", "-n", "32"])
    joined = "\n".join(command)
    assert command[:2] == [
        "/Users/leonardw/Projects/Vates/.venv/bin/vates",
        "chat",
    ]
    assert "/Volumes/Leonard's RAID/Vates/models/qwen3_next_80b_4bit" in joined
    assert "/Volumes/Leonard's RAID/Vates/models/qn_mtp_weights.safetensors" in joined
    assert "/Users/leonardw/Library/Application Support/Vates/" in joined
    assert command[command.index("--expert-slots") + 1] == "32"
    assert command[command.index("--spec-slots") + 1] == "16"
    assert command[command.index("-k") + 1] == "3"
    assert command[-4:] == ["--plain", "--stats", "-n", "32"]


def test_command_accepts_explicit_serve_subcommand():
    launcher = _load_launcher()
    command = launcher.build_command(["serve", "--host", "0.0.0.0", "--port", "8000"])
    assert command[:2] == [
        "/Users/leonardw/Projects/Vates/.venv/bin/vates",
        "serve",
    ]
    assert command[command.index("--expert-slots") + 1] == "32"
    assert command[command.index("--spec-slots") + 1] == "16"
    assert command[command.index("-k") + 1] == "3"
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["--model", "/tmp/model"], id="model-separated"),
        pytest.param(["--model=/tmp/model"], id="model-equals"),
        pytest.param(["--mod", "/tmp/model"], id="model-abbreviation"),
        pytest.param(["--expert-dir", "/tmp/experts"], id="expert-dir-separated"),
        pytest.param(["--expert-dir=/tmp/experts"], id="expert-dir-equals"),
        pytest.param(["--expert-d", "/tmp/experts"], id="expert-dir-abbreviation"),
        pytest.param(["--mtp-out", "/tmp/mtp"], id="mtp-out-separated"),
        pytest.param(["--mtp-out=/tmp/mtp"], id="mtp-out-equals"),
        pytest.param(["--mtp", "/tmp/mtp"], id="mtp-out-abbreviation"),
        pytest.param(["--qn-config", "/tmp/config"], id="qn-config-separated"),
        pytest.param(["--qn-config=/tmp/config"], id="qn-config-equals"),
        pytest.param(["--qn", "/tmp/config"], id="qn-config-abbreviation"),
        pytest.param(["--expert-slots", "1"], id="expert-slots-separated"),
        pytest.param(["--expert-slots=1"], id="expert-slots-equals"),
        pytest.param(["--expert-s", "1"], id="expert-slots-abbreviation"),
        pytest.param(["--spec-slots", "1"], id="spec-slots-separated"),
        pytest.param(["--spec-slots=1"], id="spec-slots-equals"),
        pytest.param(["--spec", "1"], id="spec-slots-abbreviation"),
        pytest.param(["-k", "4"], id="k-short-separated"),
        pytest.param(["-k4"], id="k-short-attached"),
        pytest.param(["-k=4"], id="k-short-attached-equals"),
        pytest.param(["--k", "4"], id="k-long-separated"),
        pytest.param(["--k=4"], id="k-long-equals"),
    ],
)
def test_command_rejects_fixed_profile_overrides(extra_args):
    launcher = _load_launcher()
    with pytest.raises(ValueError, match="fixed Qwen profile"):
        launcher.build_command(extra_args)


def test_runtime_environment_is_fixed_and_preserves_unrelated_values():
    launcher = _load_launcher()
    environment = launcher.runtime_environment({"LANG": "en_GB.UTF-8", "EXPERT_SLOTS": "99"})
    assert environment["LANG"] == "en_GB.UTF-8"
    assert environment["EXPERT_SLOTS"] == "32"
    assert environment["POOL_SPEC_SLOTS"] == "16"
    assert environment["KV_QUANT"] == "1"
    assert environment["KV_K_BITS"] == "4"
    assert environment["KV_V_BITS"] == "3"
    assert environment["PREFILL_CHUNK"] == "2"
    assert environment["MTP_ADAPTIVE_DEPTH"] == "1"
    assert environment["MTP_DEPTH_MAX"] == "3"


def test_main_reports_missing_raid_without_exec(monkeypatch, capsys):
    launcher = _load_launcher()
    monkeypatch.setattr(os.path, "ismount", lambda path: False)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("execve must not run"))
    assert launcher.main([]) == 2
    assert "Leonard's RAID is not mounted" in capsys.readouterr().err


def test_main_reports_profile_override_without_exec(monkeypatch, capsys):
    launcher = _load_launcher()
    monkeypatch.setattr(os.path, "ismount", lambda path: True)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("execve must not run"))
    assert launcher.main(["--expert-s=1"]) == 2
    assert "fixed Qwen profile" in capsys.readouterr().err
