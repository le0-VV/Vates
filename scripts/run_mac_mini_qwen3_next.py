#!/usr/bin/env python3
"""Launch the approved Qwen3-Next profile on Leonard's Mac mini."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


RAID_MOUNT = Path("/Volumes/Leonard's RAID")
RAID_VATES = RAID_MOUNT / "Vates"
PROJECT_ROOT = Path("/Users/leonardw/Projects/Vates")
VATES_BIN = PROJECT_ROOT / ".venv/bin/vates"
MODEL_DIR = RAID_VATES / "models/qwen3_next_80b_4bit"
MTP_PATH = RAID_VATES / "models/qn_mtp_weights.safetensors"
EXPERT_DIR = Path(
    "/Users/leonardw/Library/Application Support/Vates/"
    "qwen3-next-80b-a3b-instruct-4bit/experts"
)

RUNTIME_ENV = {
    "EXPERT_SLOTS": "32",
    "POOL_SPEC_SLOTS": "16",
    "STREAM_BLOB_LOADER": "1",
    "ZEROCOPY_DUAL_SOURCE": "1",
    "NATIVE_FUSED_PREFETCH": "1",
    "SIDEREGION_LFU": "1",
    "KV_QUANT": "1",
    "KV_K_BITS": "4",
    "KV_V_BITS": "3",
    "KV_GROUP_SIZE": "64",
    "KV_ROTATE": "1",
    "PREFILL_CHUNK": "2",
    "MTP_ADAPTIVE_DEPTH": "1",
    "MTP_CONF_TAU": "0.3",
    "MTP_DEPTH_MAX": "3",
    "MLX_CACHE_LIMIT_GB": "1",
}

_FIXED_LONG_OPTIONS = (
    "--model",
    "--expert-dir",
    "--mtp-out",
    "--qn-config",
    "--expert-slots",
    "--spec-slots",
    "--k",
)


def ensure_raid_mounted(mount: Path = RAID_MOUNT) -> None:
    if not os.path.ismount(mount):
        raise RuntimeError(f"Leonard's RAID is not mounted at {mount}")


def _ensure_no_profile_overrides(extra_args: list[str]) -> None:
    for argument in extra_args:
        fixed_options: tuple[str, ...] = ()
        if argument.startswith("-k"):
            fixed_options = ("-k",)
        elif argument.startswith("--") and argument != "--":
            option = argument.partition("=")[0]
            fixed_options = tuple(
                fixed for fixed in _FIXED_LONG_OPTIONS if fixed.startswith(option)
            )
        if fixed_options:
            options = ", ".join(fixed_options)
            raise ValueError(
                f"argument {argument!r} cannot override the fixed Qwen profile "
                f"option(s): {options}"
            )


def build_command(extra_args: list[str]) -> list[str]:
    _ensure_no_profile_overrides(extra_args)
    return [
        str(VATES_BIN),
        "chat",
        "--model",
        str(MODEL_DIR),
        "--expert-dir",
        str(EXPERT_DIR),
        "--mtp-out",
        str(MTP_PATH),
        "--qn-config",
        str(MODEL_DIR / "config.json"),
        "--expert-slots",
        "32",
        "--spec-slots",
        "16",
        "-k",
        "3",
        *extra_args,
    ]


def runtime_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(RUNTIME_ENV)
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    try:
        ensure_raid_mounted()
        command = build_command(list(sys.argv[1:] if argv is None else argv))
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.execve(command[0], command, runtime_environment())
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
