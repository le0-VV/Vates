#!/usr/bin/env python3
"""Launch the pinned Qwen3.5 protocol server under one-process guards."""

from __future__ import annotations

import errno
import fcntl
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from mlx_streaming.prep.expert_manifest import read_manifest


REVISION = "1e20fd8d42056f870933bf98ca6211024744f7ec"
SOURCE_REPOSITORY = "mlx-community/Qwen3.5-35B-A3B-4bit"
ARCHITECTURE = "qwen3_5_moe"
MODEL_ID = "qwen3.5-35b-a3b-4bit"
CONTEXT_LENGTH = 131072
PORT = 8000

RAID_MOUNT = Path("/Volumes/Leonard's RAID")
MODEL_DIR = RAID_MOUNT / "Vates/models/Qwen3.5-35B-A3B-4bit"
SOURCE_REVISION_FILE = MODEL_DIR / ".vates-source-revision"
PROJECT_ROOT = Path("/Users/leonardw/Projects/Vates")
VATES_BIN = PROJECT_ROOT / ".venv/bin/vates"
RUNTIME_DIR = Path(
    "/Users/leonardw/Library/Application Support/Vates/"
    "qwen3.5-35b-a3b-4bit"
)
EXPERT_DIR = RUNTIME_DIR / "experts"
PID_FILE = RUNTIME_DIR / "qwen35-server.pid"

_TUNING_ENV = {
    "KV_QUANT",
    "KV_K_BITS",
    "KV_V_BITS",
    "KV_GROUP_SIZE",
    "KV_ROTATE",
    "MTP_ADAPTIVE_DEPTH",
    "MTP_CONF_TAU",
    "MTP_DEPTH_MAX",
    "CROSS_LAYER_PREDICT_WIDTH",
    "NATIVE_FUSED_PREFETCH",
    "ZEROCOPY_DUAL_SOURCE",
    "SIDEREGION_LFU",
    "POOL_SPEC_SLOTS",
}
_FIXED_OPTIONS = (
    "--model",
    "--expert-dir",
    "--expert-slots",
    "--engine",
    "--adapter",
    "--context-length",
    "--prefill-chunk-size",
    "--thinking-default",
    "--no-thinking-default",
    "--model-id",
    "--port",
    "--mtp-out",
    "--k",
)


def ensure_raid_mounted(mount: Path = RAID_MOUNT) -> None:
    if not os.path.ismount(mount):
        raise RuntimeError(f"Leonard's RAID is not mounted at {mount}")


def ensure_source_revision(marker: Path = SOURCE_REVISION_FILE) -> None:
    if not marker.is_file():
        raise RuntimeError(f"Qwen3.5 revision marker is missing at {marker}")
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != REVISION:
        raise RuntimeError(
            f"Qwen3.5 revision must be {REVISION}, got {actual!r}"
        )


def ensure_runtime_manifest(expert_dir: Path = EXPERT_DIR) -> None:
    path = expert_dir / "expert_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Qwen3.5 expert manifest is missing at {path}")
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Qwen3.5 expert manifest is invalid: {exc}") from exc
    expected = {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": REVISION,
        "architecture": ARCHITECTURE,
        "layer_indices": tuple(range(40)),
        "num_experts": 256,
        "top_k": 8,
    }
    for field, wanted in expected.items():
        actual = getattr(manifest, field)
        if actual != wanted:
            raise RuntimeError(
                f"Qwen3.5 expert manifest {field} must be {wanted!r}, "
                f"got {actual!r}"
            )


def ensure_port_available(host: str = "127.0.0.1", port: int = PORT) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError(f"port {port} is occupied") from exc


def ensure_no_vates_process(process_table: str, *, current_pid: int) -> None:
    recognised = (
        "/Projects/Vates/.venv/bin/vates",
        "mlx_streaming.cli",
        "run_mac_mini_qwen3_next.py",
        "run_mac_mini_qwen35.py",
    )
    for line in process_table.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        command = fields[1]
        if pid != current_pid and any(marker in command for marker in recognised):
            raise RuntimeError(
                f"a Vates model process is already running: pid {pid}"
            )


def process_table() -> str:
    return subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _ensure_no_profile_overrides(arguments: list[str]) -> None:
    for argument in arguments:
        if not argument.startswith("--") or argument == "--":
            continue
        option = argument.partition("=")[0]
        matches = [fixed for fixed in _FIXED_OPTIONS if fixed.startswith(option)]
        if matches:
            raise ValueError(
                f"argument {argument!r} cannot override the fixed Qwen3.5 "
                f"profile option(s): {', '.join(matches)}"
            )


def build_command(extra_args: list[str]) -> list[str]:
    extra_args = list(extra_args)
    if extra_args and extra_args[0] in {"chat", "serve"}:
        if extra_args.pop(0) != "serve":
            raise ValueError("Qwen3.5 is exposed through the protocol server only")
    _ensure_no_profile_overrides(extra_args)
    return [
        str(VATES_BIN),
        "serve",
        "--engine",
        "general",
        "--adapter",
        "auto",
        "--model",
        str(MODEL_DIR),
        "--expert-dir",
        str(EXPERT_DIR),
        "--expert-slots",
        "40",
        "--context-length",
        str(CONTEXT_LENGTH),
        "--prefill-chunk-size",
        "64",
        "--thinking-default",
        "--model-id",
        MODEL_ID,
        "--port",
        str(PORT),
        *extra_args,
    ]


def runtime_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    for name in _TUNING_ENV:
        environment.pop(name, None)
    return environment


@contextmanager
def pid_lock(path: Path = PID_FILE) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(f"Qwen3.5 PID lock is already held at {path}") from exc
            raise
        locked = True
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            if locked:
                path.unlink(missing_ok=True)


def run_server(command: list[str], environment: dict[str, str]) -> int:
    child = subprocess.Popen(command, env=environment)
    previous = {}

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        return child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if child.poll() is None:
            child.terminate()
            child.wait()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["-h"], ["--help"]):
        print(
            "Launch the pinned Qwen3.5 OpenAI protocol server at "
            "131,072-token context.\n"
            "Additional non-profile vates serve options may follow."
        )
        return 0
    try:
        ensure_raid_mounted()
        ensure_source_revision()
        ensure_runtime_manifest()
        ensure_port_available()
        ensure_no_vates_process(process_table(), current_pid=os.getpid())
        command = build_command(arguments)
        with pid_lock():
            return run_server(command, runtime_environment())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
