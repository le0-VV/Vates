# Mac mini Qwen3-Next MLX RAID Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pinned Qwen3-Next-80B-A3B-Instruct 4-bit MLX model through Vates on Leonard's 16 GB M4 Mac mini, with only canonical original model source files remaining on Leonard's RAID and all derived runtime assets on the internal SSD.

**Architecture:** Add one tested device-specific launcher that checks only the RAID mount for RAID availability, supplies exact model/runtime paths, fails fast when internal MTP or expert assets are absent, and fixes the measured 32/16/K=3 runtime profile. Use RAID capacity temporarily for preparation, atomically publish the validated prepared MTP and packed experts to internal storage, qualify the target, then remove derived RAID caches, intermediates, archives and logs. The initial acceptance used 32/8; a later controlled A/B/A/B benchmark changed only the side-region capacity and established 32/16 as the current launcher profile.

**Tech Stack:** Python 3.13, MLX 0.31.2, mlx-lm 0.31.3, uv, pytest, CMake/nanobind, GitHub Actions, Hugging Face Hub, macOS Apple Silicon, SSH.

## Global Constraints

- Target host: `leonardw@leonards-mac-mini`, Apple M4, 16 GB unified memory, macOS 26.5.2 arm64.
- Pin `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` to revision `d8a069bfa8ae87d3d468412e1034acae19b5892b`.
- In the verified steady state, only canonical original model source files live below `/Volumes/Leonard's RAID/Vates`.
- The prepared MTP file, expert runtime store and logs live below `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/`.
- Keep the Vates checkout and virtual environment at `/Users/leonardw/Projects/Vates` on the internal SSD.
- Preserve at least 30 GB of internal free space after copying the 43,486,543,872-byte expert blob store; require at least 75 GB free before copying.
- Runtime profile: `EXPERT_SLOTS=32`, `POOL_SPEC_SLOTS=16`, `K=3`, K4/V3 KV quantisation, prefill chunk 2, MTP adaptive depth at 0.3 and maximum depth 3.
- Never lower `EXPERT_SLOTS` below 32 or raise K/depth above 3 to force a fit.
- The launcher checks only that `/Volumes/Leonard's RAID` is mounted; it does not introduce a hard native-extension check.
- RAID-backed download caches, preparation intermediates, packed archives and logs are temporary and must be removed after verified internal publication.
- Retry failed internet operations through `http://127.0.0.1:1087` only after the direct attempt fails.
- Do not push directly to `main`; use the protected `agent/qwen3-next-mlx-raid` branch and a pull request.

## File map

- Create `.github/workflows/ci.yml`: hosted portable launcher and source checks required by repository policy; full MLX/native verification runs on the actual M4 target.
- Create `scripts/run_mac_mini_qwen3_next.py`: mount-aware launcher with exact model, expert, MTP, and runtime settings.
- Create `mlx_streaming/tests/test_mac_mini_launcher.py`: unit coverage for mount handling, command construction, and environment construction.
- Maintain one active Brick routine through `./brick memory` tooling: durable hot/cold paths, launch command, and the measured 32/16 profile. Retire stale active guidance through Brick rather than editing generated memory files.
- Do not change core inference code unless a live, reproducible incompatibility is found and separately diagnosed.

---

### Task 1: Add the tested Mac mini launcher

**Files:**
- Create: `scripts/run_mac_mini_qwen3_next.py`
- Create: `mlx_streaming/tests/test_mac_mini_launcher.py`

**Interfaces:**
- Produces: `ensure_raid_mounted(mount: Path = RAID_MOUNT) -> None`.
- Produces: `ensure_internal_runtime_assets(mtp_path: Path = MTP_PATH, expert_dir: Path = EXPERT_DIR) -> None`.
- Produces: `build_command(extra_args: list[str]) -> list[str]`.
- Produces: `runtime_environment(base: Mapping[str, str] | None = None) -> dict[str, str]`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.
- Consumes: the installed `.venv/bin/vates` command and paths fixed in the approved design.

- [ ] **Step 1: Write the failing launcher tests**

Create `mlx_streaming/tests/test_mac_mini_launcher.py`:

```python
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


def test_missing_internal_mtp_is_rejected(tmp_path):
    launcher = _load_launcher()
    experts = tmp_path / "experts"
    experts.mkdir()
    with pytest.raises(RuntimeError, match="Internal MTP weights are missing"):
        launcher.ensure_internal_runtime_assets(tmp_path / "mtp.safetensors", experts)


def test_missing_internal_expert_store_is_rejected(tmp_path):
    launcher = _load_launcher()
    mtp_path = tmp_path / "mtp.safetensors"
    mtp_path.touch()
    with pytest.raises(RuntimeError, match="Internal expert store is missing"):
        launcher.ensure_internal_runtime_assets(mtp_path, tmp_path / "experts")


def test_command_uses_raid_only_for_original_model_assets():
    launcher = _load_launcher()
    command = launcher.build_command(["--plain", "--stats", "-n", "32"])
    joined = "\n".join(command)
    assert command[:2] == [
        "/Users/leonardw/Projects/Vates/.venv/bin/vates",
        "chat",
    ]
    assert "/Volumes/Leonard's RAID/Vates/models/qwen3_next_80b_4bit" in joined
    assert command[command.index("--mtp-out") + 1] == (
        "/Users/leonardw/Library/Application Support/Vates/"
        "qwen3-next-80b-a3b-instruct-4bit/mtp/qn_mtp_weights.safetensors"
    )
    assert command[command.index("--expert-dir") + 1] == (
        "/Users/leonardw/Library/Application Support/Vates/"
        "qwen3-next-80b-a3b-instruct-4bit/experts"
    )
    assert "/Volumes/Leonard's RAID/Vates/models/qn_mtp_weights.safetensors" not in joined
    assert command[command.index("--expert-slots") + 1] == "32"
    assert command[command.index("--spec-slots") + 1] == "16"
    assert command[command.index("-k") + 1] == "3"
    assert command[-4:] == ["--plain", "--stats", "-n", "32"]


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


def test_main_reports_missing_internal_assets_without_exec(monkeypatch, capsys):
    launcher = _load_launcher()
    monkeypatch.setattr(os.path, "ismount", lambda path: True)

    def reject_missing_assets():
        raise RuntimeError("Internal MTP weights are missing")

    monkeypatch.setattr(launcher, "ensure_internal_runtime_assets", reject_missing_assets)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("execve must not run"))
    assert launcher.main([]) == 2
    assert "Internal MTP weights are missing" in capsys.readouterr().err
```

- [ ] **Step 2: Run the focused test and confirm the expected red state**

Run:

```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_mac_mini_launcher.py -v
```

Expected: collection fails because `scripts/run_mac_mini_qwen3_next.py` does not exist.

- [ ] **Step 3: Implement the launcher**

Create `scripts/run_mac_mini_qwen3_next.py`:

```python
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
INTERNAL_RUNTIME_DIR = Path(
    "/Users/leonardw/Library/Application Support/Vates/"
    "qwen3-next-80b-a3b-instruct-4bit"
)
MTP_PATH = INTERNAL_RUNTIME_DIR / "mtp/qn_mtp_weights.safetensors"
EXPERT_DIR = INTERNAL_RUNTIME_DIR / "experts"

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


def ensure_raid_mounted(mount: Path = RAID_MOUNT) -> None:
    if not os.path.ismount(mount):
        raise RuntimeError(f"Leonard's RAID is not mounted at {mount}")


def ensure_internal_runtime_assets(
    mtp_path: Path = MTP_PATH,
    expert_dir: Path = EXPERT_DIR,
) -> None:
    if not mtp_path.is_file():
        raise RuntimeError(f"Internal MTP weights are missing at {mtp_path}")
    if not expert_dir.is_dir():
        raise RuntimeError(f"Internal expert store is missing at {expert_dir}")


def build_command(extra_args: list[str]) -> list[str]:
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
        ensure_internal_runtime_assets()
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.execve(command[0], command, runtime_environment())
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
.venv/bin/python -m pytest \
  mlx_streaming/tests/test_mac_mini_launcher.py \
  mlx_streaming/tests/test_cli_chat_repl.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Make the launcher executable and commit the logical unit**

Run:

```bash
chmod +x scripts/run_mac_mini_qwen3_next.py
git add scripts/run_mac_mini_qwen3_next.py mlx_streaming/tests/test_mac_mini_launcher.py
git diff --cached --check
git commit -m "feat(deploy): add Mac mini Qwen launcher"
git log -1 --show-signature
```

Expected: a signed `feat(deploy)` commit containing only the launcher and its tests.

---

### Task 2: Add hosted CI and publish the task branch

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `uv.lock`, `pyproject.toml`, `native/ext/Makefile`, and the Python test suite.
- Produces: a GitHub Actions check named `portable` on pull requests and task-branch pushes.

- [ ] **Step 1: Add the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches:
      - main
      - "agent/**"

permissions:
  contents: read

jobs:
  portable:
    runs-on: macos-15
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Install portable test dependency
        run: python -m pip install pytest==8.4.1
      - name: Compile Python sources
        run: python -m compileall -q mlx_streaming scripts
      - name: Test the device launcher
        run: python -m pytest mlx_streaming/tests/test_mac_mini_launcher.py -v
```

- [ ] **Step 2: Verify the workflow and full local suite**

Run:

```bash
git diff --check
.venv/bin/python -m pytest
PY_SITE="$PWD/.venv/lib/python3.13/site-packages"
make -C native/ext PYTHON="$PWD/.venv/bin/python" PY_SITE="$PY_SITE" native_moe_ext
.venv/bin/python -c 'import mlx_streaming.native_moe_ext'
```

Expected: the full suite passes locally, the native build exits zero, and the import exits zero. Hosted CI intentionally tests the portable launcher and source compilation because MLX/native execution requires the real arm64 target qualified in Task 3.

- [ ] **Step 3: Commit and push the workflow**

Run:

```bash
git add .github/workflows/ci.yml
git diff --cached --check
git commit -m "CI(test): add macOS test workflow"
git log -1 --show-signature
git push -u origin agent/qwen3-next-mlx-raid
```

Expected: a signed `CI(test)` commit and a new remote task branch; `main` is unchanged.

- [ ] **Step 4: Verify the hosted run**

Run outside the sandbox with the required absolute GitHub CLI:

```bash
/opt/homebrew/bin/gh run list \
  --repo le0-VV/Vates \
  --branch agent/qwen3-next-mlx-raid \
  --workflow CI \
  --limit 1
```

Then watch the newest run returned by the same exact filters:

```bash
/opt/homebrew/bin/gh run watch "$(/opt/homebrew/bin/gh run list \
  --repo le0-VV/Vates \
  --branch agent/qwen3-next-mlx-raid \
  --workflow CI \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')" \
  --repo le0-VV/Vates --exit-status
```

Expected: the task-branch `CI` run completes successfully.

---

### Task 3: Bootstrap and verify Vates on the Mac mini

**Files:**
- Remote create: `/Users/leonardw/Projects/Vates/`
- Remote create: `/Users/leonardw/Projects/Vates/.venv/`
- Remote create: `/Users/leonardw/Projects/Vates/mlx_streaming/native_moe_ext*.so`

**Interfaces:**
- Consumes: remote task branch `agent/qwen3-next-mlx-raid` and locked dependencies.
- Produces: a tested arm64 Vates checkout with a loadable native extension.

- [ ] **Step 1: Reconfirm the non-destructive target preflight**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  test "$(uname -m)" = arm64
  test "$(sysctl -n hw.memsize)" = 17179869184
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  test "$(diskutil info "$RAID_MOUNT" | awk -F ": *" "/Volume UUID/ {print \\$2}")" = A4FF636C-BBCB-4541-9E48-7B8078EDFA0F
  mount | grep -F "on $RAID_MOUNT "
  df -k "$RAID_MOUNT" /
  test ! -e /Users/leonardw/Projects/Vates
'
```

Expected: arm64 and 16 GB checks pass, the exact RAID mount is printed, both free-space rows are printed, and the absent checkout assertion passes.

- [ ] **Step 2: Install uv directly, retrying through the configured proxy only on network failure**

Run the direct attempt:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini \
  'curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/Users/leonardw/.local/bin sh'
```

If and only if that fails because of network access, retry:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  export http_proxy=http://127.0.0.1:1087
  export https_proxy=http://127.0.0.1:1087
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/Users/leonardw/.local/bin sh
'
```

Verify:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini \
  '/Users/leonardw/.local/bin/uv --version'
```

Expected: an installed uv version is printed.

- [ ] **Step 3: Clone the task branch and install the locked environment**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  mkdir -p /Users/leonardw/Projects
  git clone --branch agent/qwen3-next-mlx-raid --single-branch \
    https://github.com/le0-VV/Vates.git /Users/leonardw/Projects/Vates
  /Users/leonardw/.local/bin/uv sync \
    --directory /Users/leonardw/Projects/Vates \
    --frozen --all-groups --python 3.13
'
```

Expected: the branch clones without modifying `main`, uv creates `.venv`, and dependency installation exits zero.

- [ ] **Step 4: Build and import the native extension**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  cd /Users/leonardw/Projects/Vates
  make -C native/ext \
    PYTHON=/Users/leonardw/Projects/Vates/.venv/bin/python \
    PY_SITE=/Users/leonardw/Projects/Vates/.venv/lib/python3.13/site-packages \
    native_moe_ext
  /Users/leonardw/Projects/Vates/.venv/bin/python -c \
    "import mlx_streaming.native_moe_ext; print(\"native extension loaded\")"
'
```

Expected: CMake completes and `native extension loaded` is printed.

- [ ] **Step 5: Run the complete target test suite**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  cd /Users/leonardw/Projects/Vates
  .venv/bin/python -m pytest
'
```

Expected: pytest exits zero with no failed tests.

---

### Task 4: Download the canonical model and prepare temporary RAID artefacts

**Files:**
- Remote temporary create: `/Volumes/Leonard's RAID/Vates/cache/huggingface/`
- Remote create: `/Volumes/Leonard's RAID/Vates/models/qwen3_next_80b_4bit/`
- Remote create: `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/mtp/qn_mtp_weights.safetensors`
- Remote temporary create: `/Volumes/Leonard's RAID/Vates/preparation/qwen3_next_expert_files_4bit_g64/`
- Remote temporary create: `/Volumes/Leonard's RAID/Vates/preparation/mtp-source/`
- Remote temporary create: `/Volumes/Leonard's RAID/Vates/expert-archive/qwen3_next_experts_4bit_g64/`

**Interfaces:**
- Consumes: pinned Hugging Face model, original Qwen MTP shard, and existing Vates preparation modules.
- Produces: the canonical main model on RAID, the prepared MTP file internally, and temporary split/packed expert artefacts for internal publication.

- [ ] **Step 1: Create temporary preparation paths and reconfirm capacity**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  test "$(diskutil info "$RAID_MOUNT" | awk -F ": *" "/Volume UUID/ {print \\$2}")" = A4FF636C-BBCB-4541-9E48-7B8078EDFA0F
  export VATES_RAID_ROOT="$RAID_MOUNT/Vates"
  test "$(df -Pk "$RAID_MOUNT" | awk "NR==2 {print \\$4}")" -ge 146800640
  mkdir -p \
    "$VATES_RAID_ROOT/cache/huggingface" \
    "$VATES_RAID_ROOT/models" \
    "$VATES_RAID_ROOT/preparation/mtp-source" \
    "$VATES_RAID_ROOT/expert-archive/qwen3_next_experts_4bit_g64/blobs"
  mkdir -p "/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/mtp"
'
```

Expected: the RAID has at least 140 GiB free, temporary preparation paths are created on that mount, and the internal MTP directory exists.

- [ ] **Step 2: Download the pinned 4-bit MLX model directly to the RAID**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  export HF_HOME="$RAID_MOUNT/Vates/cache/huggingface"
  /Users/leonardw/Projects/Vates/.venv/bin/hf download \
    mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
    --revision d8a069bfa8ae87d3d468412e1034acae19b5892b \
    --local-dir "$RAID_MOUNT/Vates/models/qwen3_next_80b_4bit"
'
```

If the direct network request fails, rerun the same command after exporting the configured `http_proxy` and `https_proxy` values.

Expected: nine safetensor shards and all model/tokenizer metadata are present at the final RAID path; interrupted execution is resumable.

- [ ] **Step 3: Validate the pinned model manifest and configuration**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  export MODEL_DIR="$RAID_MOUNT/Vates/models/qwen3_next_80b_4bit"
  MODEL_BYTES=$(find "$MODEL_DIR" -maxdepth 1 -name "model-*.safetensors" \
    -exec stat -f "%z" {} \; | awk "{sum += \\$1} END {print sum + 0}")
  test "$MODEL_BYTES" -eq 44844286500
  /Users/leonardw/Projects/Vates/.venv/bin/python -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
c = json.loads((p / \"config.json\").read_text())
q = c[\"quantization\"]
assert c[\"model_type\"] == \"qwen3_next\"
assert c[\"num_hidden_layers\"] == 48
assert c[\"num_experts\"] == 512
assert c[\"num_experts_per_tok\"] == 10
assert q[\"bits\"] == 4 and q[\"group_size\"] == 64 and q[\"mode\"] == \"affine\"
print(\"model manifest valid\")
" "$MODEL_DIR"
'
```

Expected: `model manifest valid` is printed and the shard total is exactly 44,844,286,500 bytes.

- [ ] **Step 4: Split experts on the RAID**

Run in a persistent exec session so progress can be monitored:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  cd /Users/leonardw/Projects/Vates
  .venv/bin/python -m mlx_streaming.prep.split_experts \
    "$RAID_MOUNT/Vates/models/qwen3_next_80b_4bit" \
    "$RAID_MOUNT/Vates/preparation/qwen3_next_expert_files_4bit_g64"
'
```

Expected: `_split_meta.json` reports 48 MoE layers, 512 experts, hidden size 2,048, MoE intermediate size 512, four bits, and group size 64; 24,576 expert files are created.

- [ ] **Step 5: Pack the temporary expert archive on the RAID**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  cd /Users/leonardw/Projects/Vates
  export EXPERT_DIR="$RAID_MOUNT/Vates/preparation/qwen3_next_expert_files_4bit_g64"
  export BLOB_DIR="$RAID_MOUNT/Vates/expert-archive/qwen3_next_experts_4bit_g64/blobs"
  export BITS=4
  export GROUP=64
  export LAYERS=all
  .venv/bin/python -m mlx_streaming.prep.pack_blob_from_experts
  cp "$EXPERT_DIR/_split_meta.json" \
    "$RAID_MOUNT/Vates/expert-archive/qwen3_next_experts_4bit_g64/_split_meta.json"
'
```

Expected: all 48 layer blobs are packed and `blob_index.json` reports stride 1,769,472, 512 experts, 48 layers, four bits, and group size 64.

- [ ] **Step 6: Extract the prepared MTP file to internal storage**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  cd /Users/leonardw/Projects/Vates
  export QN_CONFIG="$RAID_MOUNT/Vates/models/qwen3_next_80b_4bit/config.json"
  export MTP_SHARD_DIR="$RAID_MOUNT/Vates/preparation/mtp-source"
  export MTP_OUT="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/mtp/qn_mtp_weights.safetensors"
  .venv/bin/python -m mlx_streaming.prep.extract_mtp
'
```

If ModelScope fails directly, rerun after exporting the configured proxy values.

Expected: the source shard is exactly 3,301,131,296 bytes and the prepared MTP file is written to the internal `mtp/` directory.

- [ ] **Step 7: Validate the temporary expert archive**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  /Users/leonardw/Projects/Vates/.venv/bin/python -c "
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
blobs = sorted((root / \"blobs\").glob(\"layer*.blob\"))
assert len(blobs) == 48, len(blobs)
assert all(p.stat().st_size == 905_969_664 for p in blobs)
assert sum(p.stat().st_size for p in blobs) == 43_486_543_872
index = json.loads((root / \"blobs/blob_index.json\").read_text())
assert index[\"layers\"] == list(range(48))
assert index[\"stride\"] == 1_769_472
assert index[\"num_experts\"] == 512
assert index[\"bits\"] == 4 and index[\"group_size\"] == 64
assert (root / \"_split_meta.json\").is_file()
print(\"temporary expert archive valid\")
" "$RAID_MOUNT/Vates/expert-archive/qwen3_next_experts_4bit_g64"
'
```

Expected: `temporary expert archive valid` is printed.

---

### Task 5: Copy the hot expert store to internal storage atomically

**Files:**
- Remote create: `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts/`

**Interfaces:**
- Consumes: the validated temporary expert archive from Task 4.
- Produces: the internal `EXPERT_DIR` consumed by the launcher.

- [ ] **Step 1: Enforce internal capacity before copying**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  INTERNAL_FREE_KIB=$(df -Pk / | awk "NR==2 {print \\$4}")
  test "$INTERNAL_FREE_KIB" -ge 78643200
  test ! -e "/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts"
  printf "internal_free_kib=%s\n" "$INTERNAL_FREE_KIB"
'
```

Expected: at least 75 GiB is free before the copy and the final target does not already exist.

- [ ] **Step 2: Copy into a unique incomplete directory**

Run in a persistent exec session:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  export INTERNAL_PARENT="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit"
  export ARCHIVE_DIR="$RAID_MOUNT/Vates/expert-archive/qwen3_next_experts_4bit_g64"
  mkdir -p "$INTERNAL_PARENT"
  STAGE_DIR=$(mktemp -d "$INTERNAL_PARENT/experts.incomplete.XXXXXX")
  printf "%s\n" "$STAGE_DIR"
  rsync -a --checksum "$ARCHIVE_DIR/" "$STAGE_DIR/"
  test -z "$(rsync -ani --checksum "$ARCHIVE_DIR/" "$STAGE_DIR/")"
  mv "$STAGE_DIR" "$INTERNAL_PARENT/experts"
'
```

Expected: the second rsync emits no differences and the completed store becomes visible at the final path only after validation.

- [ ] **Step 3: Validate the internal runtime copy and remaining headroom**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  /Users/leonardw/Projects/Vates/.venv/bin/python -c "
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
blobs = sorted((root / \"blobs\").glob(\"layer*.blob\"))
assert len(blobs) == 48
assert all(p.stat().st_size == 905_969_664 for p in blobs)
assert sum(p.stat().st_size for p in blobs) == 43_486_543_872
index = json.loads((root / \"blobs/blob_index.json\").read_text())
assert index[\"layers\"] == list(range(48))
assert (root / \"_split_meta.json\").is_file()
print(\"internal expert store valid\")
" "/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts"
  test "$(df -Pk / | awk "NR==2 {print \\$4}")" -ge 31457280
'
```

Expected: `internal expert store valid` is printed and at least 30 GiB remains free.

---

### Task 6: Historical initial qualification of the 32/8/K=3 profile

This task records the 22 July deployment acceptance exactly as run. Its 32/8 commands and log names are historical byte-truth, real-response and 256-token stability evidence; they do not describe the current fixed launcher profile.

**Files:**
- Remote create: `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-32-8-smoke.log`
- Remote create: `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-32-8-soak.log`
- Remote create: `/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-32-8-memory-samples.log`

**Interfaces:**
- Consumes: the launcher, canonical RAID paths, internal expert store, and native extension.
- Produces: byte-verification, correctness, throughput, MLX allocation, RSS, swap, and real-response evidence.

- [ ] **Step 1: Capture the pre-run system baseline**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  memory_pressure | sed -n "1,40p"
  sysctl vm.swapusage
  df -H / "$RAID_MOUNT"
'
```

Expected: memory pressure, swap use, and both filesystem capacities are recorded before loading the model.

- [ ] **Step 2: Run the short byte-truth and capacity smoke**

Run in a persistent exec session:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  cd /Users/leonardw/Projects/Vates
  export MODEL="$RAID_MOUNT/Vates/models/qwen3_next_80b_4bit"
  export QN_CONFIG="$MODEL/config.json"
  export INTERNAL_ROOT="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit"
  export MTP_OUT="$INTERNAL_ROOT/mtp/qn_mtp_weights.safetensors"
  export EXPERT_DIR="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts"
  export EXPERT_SLOTS=32 POOL_SPEC_SLOTS=8 K=3
  export STREAM_BLOB_LOADER=1 ZEROCOPY_DUAL_SOURCE=1 NATIVE_FUSED_PREFETCH=1 SIDEREGION_LFU=1
  export KV_QUANT=1 KV_K_BITS=4 KV_V_BITS=3 KV_GROUP_SIZE=64 KV_ROTATE=1 PREFILL_CHUNK=2
  export MTP_ADAPTIVE_DEPTH=1 MTP_CONF_TAU=0.3 MTP_DEPTH_MAX=3 MLX_CACHE_LIMIT_GB=1
  export STG_VERIFY=1 UNION_PROF=1 MAXTOK=16 WARMUP_TOK=0 REPEAT=1
  mkdir -p "$INTERNAL_ROOT/logs"
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | \
    tee "$INTERNAL_ROOT/logs/qwen3-next-32-8-smoke.log"
'
```

Expected: `VERIFY_SUMMARY` reports virtual verifier calls greater than zero and `bad: 0`; result JSON reports `expert_slots: 32`, `fallback_replays: 0`, finite positive throughput, and no union above 30 or capacity warning.

- [ ] **Step 3: Generate a real answer through the launcher**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  cd /Users/leonardw/Projects/Vates
  printf "Explain in two sentences why mixture-of-experts models can be efficient.\nNow summarise that in five words.\n/exit\n" | \
    .venv/bin/python scripts/run_mac_mini_qwen3_next.py --plain --stats -n 32
'
```

Expected: Vates prints two non-empty assistant responses, token counts, positive tokens per second, and exits cleanly after `/exit`; the second turn exercises the multi-turn cache path.

- [ ] **Step 4: Run the 256-token stability soak**

Run in a persistent exec session:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  cd /Users/leonardw/Projects/Vates
  export MODEL="$RAID_MOUNT/Vates/models/qwen3_next_80b_4bit"
  export QN_CONFIG="$MODEL/config.json"
  export INTERNAL_ROOT="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit"
  export MTP_OUT="$INTERNAL_ROOT/mtp/qn_mtp_weights.safetensors"
  export EXPERT_DIR="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts"
  export EXPERT_SLOTS=32 POOL_SPEC_SLOTS=8 K=3
  export STREAM_BLOB_LOADER=1 ZEROCOPY_DUAL_SOURCE=1 NATIVE_FUSED_PREFETCH=1 SIDEREGION_LFU=1
  export KV_QUANT=1 KV_K_BITS=4 KV_V_BITS=3 KV_GROUP_SIZE=64 KV_ROTATE=1 PREFILL_CHUNK=2
  export MTP_ADAPTIVE_DEPTH=1 MTP_CONF_TAU=0.3 MTP_DEPTH_MAX=3 MLX_CACHE_LIMIT_GB=1
  export STG_VERIFY=1 UNION_PROF=1 MAXTOK=256 WARMUP_TOK=32 REPEAT=1
  set -o pipefail
  mkdir -p "$INTERNAL_ROOT/logs"
  SOAK_LOG="$INTERNAL_ROOT/logs/qwen3-next-32-8-soak.log"
  MEMORY_LOG="$INTERNAL_ROOT/logs/qwen3-next-32-8-memory-samples.log"
  (.venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee "$SOAK_LOG") &
  SOAK_PID=$!
  (
    while kill -0 "$SOAK_PID" 2>/dev/null; do
      date -u "+%Y-%m-%dT%H:%M:%SZ"
      sysctl vm.swapusage
      memory_pressure | awk "/System-wide memory free percentage/ {print}"
      sleep 15
    done
  ) > "$MEMORY_LOG" &
  MONITOR_PID=$!
  wait "$SOAK_PID"
  SOAK_STATUS=$?
  wait "$MONITOR_PID" || true
  exit "$SOAK_STATUS"
'
```

Expected: the run completes without Metal allocation failure or repeated latency cliffs; verifier `bad` remains zero, `fallback_replays` is zero, `rss_gb` and `mlx_peak_gb` are finite, throughput remains positive, and 15-second memory/swap samples are written for the entire run.

- [ ] **Step 5: Capture post-run memory pressure and enforce the acceptance decision**

Run:

```bash
ssh -o BatchMode=yes leonardw@leonards-mac-mini '
  set -- /Volumes/Leonard*RAID
  test "$#" -eq 1
  RAID_MOUNT=$1
  memory_pressure | sed -n "1,40p"
  sysctl vm.swapusage
  INTERNAL_ROOT="/Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit"
  tail -n 120 "$INTERNAL_ROOT/logs/qwen3-next-32-8-memory-samples.log"
  tail -n 120 "$INTERNAL_ROOT/logs/qwen3-next-32-8-soak.log"
'
```

Expected: system memory pressure is not critical, swap does not grow continuously across the run, and the log contains no allocation error, byte-verification failure, or capacity warning. If any criterion fails, stop and diagnose; do not silently lower `EXPERT_SLOTS`.

- [ ] **Step 6: Remove temporary derived artefacts from the RAID after runtime qualification**

After the internal expert store and prepared MTP file have passed integrity checks and Task 6 has qualified the runtime, remove the RAID-backed Hugging Face cache, split-expert directory, packed expert archive and preparation logs. The original model directory remains. An original MTP source shard may remain only when explicitly classified as a canonical source file.

Expected: the steady-state RAID tree contains canonical original model source files, not derived runtime assets, caches, split experts, packed archives or logs.

---

### Post-deployment tuning: adopt the measured 32/16/K=3 profile

On 23 July, four verifier-off end-to-end processes ran in A/B/A/B order with every variable fixed except `POOL_SPEC_SLOTS`. Both 32/16 processes reproduced the improvement over their neighbouring 32/8 controls. Across four speculative repeats per capacity, the median increased from 4.85 to 5.095 tok/s (approximately 5.05%), demand loads fell by approximately 31.5%, and MLX peak allocation rose from 6.00 to 6.68 GB. All eight speculative repeats exactly matched greedy output with `n_mismatch=0`, `fallback_replays=0` and identical dumped token sequences across capacities. System pressure was non-critical and recovered after the four processes.

The launcher and operational routine therefore use `EXPERT_SLOTS=32`, `POOL_SPEC_SLOTS=16` and `K=3`. K4/V3 KV quantisation, prefill chunk 2, adaptive threshold 0.3, maximum depth 3, the mount-only RAID check and immutable protected arguments remain unchanged. The tracked benchmark report is `benchmarks/reports/mac-mini-qwen3-next-32-16-2026-07-23.md`.

The historical 32/32 regression report is not a 10 tok/s serving claim. It recorded 9.6–10.77 speculative decode tok/s for a 48-token run on a 32 GB machine with nearly full swap and explicitly marked absolute throughput untrustworthy. Its host chip, storage, exact command and raw logs were not retained, so it is neither directly comparable with Leonard's 16 GB Mac mini nor evidence that this target should approach 10 tok/s.

---

### Task 7: Record and maintain the validated operational routine in Brick

**Files:**
- Create through Brick: one superseding Markdown routine under `.agents/memory/routine/`; use the exact path returned by Brick for staging and verification.
- Update through Brick: retire the earlier 32/8 routine so it cannot remain active launch guidance.

**Interfaces:**
- Consumes: successful Tasks 3–6 and their exact paths/profile.
- Produces: one active 32/16 repository routine retrievable by later agents, with stale storage guidance retained only as superseded or redacted history.

- [ ] **Step 1: Prepare the exact Brick candidate and add the routine through Brick**

Use `apply_patch` to create `/private/tmp/vates-mac-mini-memory.json` with this exact candidate after successful live acceptance:

```json
{
  "title": "Run Qwen3-Next on Leonard's Mac mini",
  "type": "routine",
  "status": "active",
  "tags": [
    "vates",
    "qwen3-next",
    "mac-mini",
    "mlx"
  ],
  "body": "Use /Users/leonardw/Projects/Vates with only the canonical original model source files under /Volumes/Leonard's RAID/Vates/models/qwen3_next_80b_4bit. The prepared MTP file is /Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/mtp/qn_mtp_weights.safetensors, the expert store is in the sibling experts directory, and the server log is /Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/logs/qwen3-next-openai-server.log. Launch scripts/run_mac_mini_qwen3_next.py with EXPERT_SLOTS=32, POOL_SPEC_SLOTS=16, and K=3. The launcher requires Leonard's RAID to be mounted. The pinned model revision is d8a069bfa8ae87d3d468412e1034acae19b5892b. Do not lower EXPERT_SLOTS below 32 or increase K/depth above 3.",
  "source": {
    "kind": "deployment",
    "ref": "Mac mini Qwen3-Next 32/16 tuning on 2026-07-23"
  },
  "evidence": [
    {
      "kind": "verification",
      "text": "The initial 32/8 target run completed the native build/import, full test suite, byte-truth smoke, real Vates response, and 256-token stability soak."
    },
    {
      "kind": "benchmark",
      "text": "A verifier-off A/B/A/B comparison changed only POOL_SPEC_SLOTS and measured a 5.05% median gain at 32/16, with exact greedy/speculative output, zero fallback replays, non-critical pressure, and recovered memory after four processes."
    }
  ],
  "fields": {
    "prerequisites": [
      "Leonard's RAID is mounted at /Volumes/Leonard's RAID.",
      "The internal prepared MTP file exists below the runtime root.",
      "The internal expert store contains all 48 validated layer blobs.",
      "Server logs are written below the internal runtime root."
    ],
    "steps": [
      "Connect to leonardw@leonards-mac-mini.",
      "Change directory to /Users/leonardw/Projects/Vates.",
      "Run .venv/bin/python scripts/run_mac_mini_qwen3_next.py with any desired Vates chat arguments."
    ],
    "verify": "Vates loads the pinned model, prints a non-empty response, and reports positive throughput without critical memory pressure."
  },
  "supersedes": ["01KY6543A2VFB06PD9GAJDYZKB"],
  "confirm_public": true
}
```

Run from the primary checkout:

```bash
./brick memory add --pretty < /private/tmp/vates-mac-mini-memory.json
```

Expected: Brick reports `status: ok` and prints the generated superseding routine path.

- [ ] **Step 2: Confirm the stale active storage guidance is superseded through Brick**

Confirm that adding the candidate with `supersedes: ["01KY6543A2VFB06PD9GAJDYZKB"]` marks that active routine as superseded. The older redacted 32/8 routine remains historical. Do not edit any generated memory file directly.

Expected: Brick reports `status: ok`; routine `01KY6543A2VFB06PD9GAJDYZKB` becomes superseded, and only the new routine remains active.

- [ ] **Step 3: Validate and rebuild retrieval state**

Run outside the sandbox because the embedding endpoint is host-local:

```bash
./brick memory validate --pretty
./brick rebuild
./brick memory search "Leonard Mac mini hot expert store" --pretty
```

Expected: validation succeeds, rebuild indexes both records, and the default active-only search returns only the new 32/16 routine.

- [ ] **Step 4: Commit the coherent memory transition**

Run:

```bash
git add .agents/memory/routine
git diff --cached --check
git diff --cached --name-only
git commit -m "docs(memory): record Mac mini Qwen deployment"
git log -1 --show-signature
```

Expected: the staged-name check lists only the Brick-generated superseding routine and the Brick-updated routine it supersedes. Commit these memory files separately from all non-memory changes with the required `docs(memory): ...` subject.

---

### Task 8: Final verification and protected-main pull request

**Files:**
- Verify all files changed by Tasks 1, 2, and 7.
- No direct changes to `main`.

**Interfaces:**
- Consumes: all local commits, remote acceptance evidence, and the hosted CI workflow.
- Produces: a focused pull request ready for review without bypassing branch protection.

- [ ] **Step 1: Run fresh local verification**

Run:

```bash
.venv/bin/python -m pytest
PY_SITE="$PWD/.venv/lib/python3.13/site-packages"
make -C native/ext PYTHON="$PWD/.venv/bin/python" PY_SITE="$PY_SITE" native_moe_ext
.venv/bin/python -c 'import mlx_streaming.native_moe_ext'
git diff origin/main...HEAD --check
git status --short
```

Expected: all tests pass, native build/import succeed, diff check is clean, and the worktree has no unexpected changes.

- [ ] **Step 2: Verify outgoing signatures and scope**

Run:

```bash
git log --show-signature origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
/opt/homebrew/bin/gh api repos/le0-VV/Vates/branches/main/protection \
  --jq '{enforce_admins: .enforce_admins.enabled, pull_requests: (.required_pull_request_reviews != null), force_pushes: .allow_force_pushes.enabled, deletions: .allow_deletions.enabled}'
```

Expected: every outgoing commit has a good signature; changed files are limited to the approved design/plan, benchmark report, launcher/test, authorised native/CMake repairs, CI workflow, the redacted historical routine and one active 32/16 routine; protection reports enforced administrators, required pull requests, force pushes disabled, and deletions disabled.

- [ ] **Step 3: Push the final branch and verify hosted CI**

Run:

```bash
git push origin agent/qwen3-next-mlx-raid
/opt/homebrew/bin/gh run list \
  --repo le0-VV/Vates \
  --branch agent/qwen3-next-mlx-raid \
  --workflow CI \
  --limit 1
```

Watch the newest run returned by the same exact filters:

```bash
/opt/homebrew/bin/gh run watch "$(/opt/homebrew/bin/gh run list \
  --repo le0-VV/Vates \
  --branch agent/qwen3-next-mlx-raid \
  --workflow CI \
  --limit 1 \
  --json databaseId \
  --jq '.[0].databaseId')" \
  --repo le0-VV/Vates --exit-status
```

Expected: final branch push succeeds and current hosted CI completes successfully.

- [ ] **Step 4: Create the pull request**

Run:

```bash
/opt/homebrew/bin/gh pr create \
  --repo le0-VV/Vates \
  --base main \
  --head agent/qwen3-next-mlx-raid \
  --title "feat(deploy): run Qwen3-Next on 16 GB Mac mini" \
  --body "Adds a tested Mac mini launcher and hosted CI, documents the approved storage design, and records the validated operational routine. Only canonical original model source files remain on Leonard's RAID; prepared MTP weights, expert blobs and logs live on the internal SSD. Verification includes the full suite, native extension build/import, byte-truth smoke, a real Vates response, and a 256-token memory/swap soak."
```

Expected: GitHub returns a pull-request URL; `main` remains protected and unmodified pending review.
