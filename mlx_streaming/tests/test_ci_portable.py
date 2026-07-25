"""Regression coverage for the dependency-free hosted CI surface."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def test_hosted_ci_runs_server_cli_and_launcher_tests():
    workflow = WORKFLOW.read_text()
    for test_path in (
        "mlx_streaming/tests/test_server.py",
        "mlx_streaming/tests/test_cli_server_portable.py",
        "mlx_streaming/tests/test_ci_portable.py",
        "mlx_streaming/tests/test_mac_mini_launcher.py",
    ):
        assert test_path in workflow
    assert "mlx_streaming/tests/test_cli_chat_repl.py" not in workflow


def test_portable_server_imports_do_not_load_mlx():
    check = """
import sys
import mlx_streaming.cli
import mlx_streaming.models.registry
import mlx_streaming.server
import mlx_streaming.tui.backend

loaded = sorted(
    name for name in sys.modules if name == "mlx" or name.startswith("mlx.")
)
if loaded:
    raise SystemExit(f"portable imports loaded MLX: {loaded}")
"""
    result = subprocess.run(
        [sys.executable, "-c", check],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
