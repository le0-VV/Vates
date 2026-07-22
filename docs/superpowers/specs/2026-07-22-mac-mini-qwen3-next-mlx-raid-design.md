# Mac mini Qwen3-Next MLX RAID Deployment Design

**Date:** 22 July 2026

**Target:** `leonardw@leonards-mac-mini`

**Model:** `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` at revision `d8a069bfa8ae87d3d468412e1034acae19b5892b`

## Objective

Run Qwen3-Next-80B-A3B-Instruct 4-bit MLX through Vates on Leonard's M4 Mac mini with 16 GB unified memory. Keep the canonical model and all preparation artefacts on Leonard's RAID, while keeping only the expert blobs that Vates repeatedly streams during generation on the internal SSD.

Success means that the target Mac can load the prepared model through Vates, generate a real response, and complete a multi-turn stability check without sustained swap growth, a Metal allocation failure, byte-integrity failures, or expert-capacity warnings.

## Verified target constraints

- Hardware: Apple M4, 10 logical CPUs, 16 GB unified memory.
- Operating system: macOS 26.5.2, arm64.
- Internal APFS container: approximately 88 GB available before deployment.
- RAID mount: `/Volumes/Leonard's RAID`.
- RAID: online AppleRAID stripe across two external SSDs, approximately 798 GB available.
- Observed RAID sequential read rate: approximately 343 MB/s from a 2 GiB read.
- Toolchain: Apple Command Line Tools and Apple clang 21 are installed; `uv`, Homebrew, and a suitable Python 3.11+ are not currently installed.
- The GitHub fork's `main` branch is protected and requires pull requests.

## Storage layout

The canonical asset tree will live under:

```text
/Volumes/Leonard's RAID/Vates/
├── cache/
│   └── huggingface/                         # Download cache; never falls back to internal storage
├── models/
│   ├── qwen3_next_80b_4bit/                 # Pinned 4-bit MLX main model
│   └── qn_mtp_weights.safetensors           # Prepared MTP runtime weights
├── preparation/
│   ├── qwen3_next_expert_files_4bit_g64/    # Split per-expert intermediates
│   └── mtp-source/                          # Original 3.30 GB MTP shard
└── expert-archive/
    └── qwen3_next_experts_4bit_g64/
        ├── _split_meta.json
        └── blobs/                           # Canonical packed expert blobs
```

The repeatedly read runtime expert store will live on the internal SSD:

```text
/Users/leonardw/Library/Application Support/Vates/
└── qwen3-next-80b-a3b-instruct-4bit/
    └── experts/
        ├── _split_meta.json
        └── blobs/                           # Validated runtime copy, approximately 43.49 GB
```

The Vates checkout and virtual environment will live at `/Users/leonardw/Projects/Vates`. The internal runtime copy is a derived hot cache, not the canonical model store. Canonical copies and preparation inputs remain on Leonard's RAID.

The internal copy is limited to the expert blobs because these are the files Vates repeatedly reads with `pread` while replacing entries in its fixed-size expert pool. The split expert files and MTP source shard are preparation-only. The main model and prepared MTP file are read during model construction and remain backed by their RAID paths; they are not continually cycled through the expert pool.

The deployment must verify at least 75 GB of internal free space before copying the runtime expert store, preserving at least 30 GB of headroom after the approximately 43.49 GB copy. The currently observed 88 GB leaves approximately 44 GB after the copy. It must reserve at least 140 GB on the RAID for preparation; the currently observed free space is ample.

## Model and data preparation

The main model will be downloaded directly to its final RAID path using the pinned Hugging Face revision. `HF_HOME` and related download caches will explicitly point to the RAID. The model manifest contains 44,844,286,500 bytes of safetensor shards, or approximately 44.84 GB (41.76 GiB).

Vates will prepare the expert store with its existing tools:

1. `mlx_streaming.prep.split_experts` writes per-expert files to the RAID preparation directory.
2. `mlx_streaming.prep.pack_blob_from_experts` writes the canonical packed blobs to the RAID archive.
3. The packed archive is validated before it is copied to the internal runtime directory.
4. `mlx_streaming.prep.extract_mtp` downloads its 3,301,131,296-byte source shard to the RAID by setting `MTP_SHARD_DIR`, and writes the prepared MTP file to the RAID by setting `MTP_OUT`.

For affine 4-bit, group-size-64 weights with hidden size 2,048 and MoE intermediate size 512, each packed expert occupies 1,769,472 bytes. Each of the 48 layer blobs must therefore contain 512 experts and be exactly 905,969,664 bytes. The complete blob store must contain 43,486,543,872 blob bytes, excluding its small JSON indexes.

No preparation intermediates will be deleted. Deletion can be considered separately after successful deployment, but requires explicit user authorisation.

## Runtime configuration

The initial 16 GB profile is:

```text
EXPERT_SLOTS=32
POOL_SPEC_SLOTS=8
K=3
STREAM_BLOB_LOADER=1
ZEROCOPY_DUAL_SOURCE=1
NATIVE_FUSED_PREFETCH=1
SIDEREGION_LFU=1
KV_QUANT=1
KV_K_BITS=4
KV_V_BITS=3
KV_GROUP_SIZE=64
KV_ROTATE=1
PREFILL_CHUNK=2
MTP_ADAPTIVE_DEPTH=1
MTP_CONF_TAU=0.3
MTP_DEPTH_MAX=3
MLX_CACHE_LIMIT_GB=1
```

`EXPERT_SLOTS=32` is the validated real-region capacity floor. A K=3 verification pass can route to as many as 30 distinct experts in a layer, so lowering this value risks incorrect output. `POOL_SPEC_SLOTS=8` is an additional per-layer side region for predicted experts; reducing it from 32 to 8 saves approximately 2.04 GB of expert-pool storage at the cost of a lower cache hit rate. `K=3` permits the MTP drafter to propose up to three tokens before verification. K=4 requires at least 40 real expert slots and was slower in the repository's measurements.

The later MTP benchmark reports measured approximately 8.23–8.27 decimal GB of MLX peak allocation for the documented 32/32 profile. The repository's earlier side-region study measured 5.07 GB peak for a 32/8 non-MTP run. A current-path 32/8 peak near 6.2 GB is a layout-based estimate, not a measurement. Total process RSS and system pressure cannot be inferred from MLX allocation alone, so the live target run is authoritative.

The launcher will pass absolute paths for the RAID-backed main model, MTP output, and Qwen configuration, plus the internal expert directory. It will check only that `/Volumes/Leonard's RAID` is mounted before invoking Vates. The deployment process will build and test the native extension because it materially affects performance, but the launcher will not introduce a new hard failure if that extension is unavailable; Vates retains its existing fallback behaviour.

## Data flow

```text
Pinned Hugging Face model ───────────────┐
                                        ├─ RAID canonical assets
Original MTP shard ──> prepared MTP ─────┤
                                        │
Main model ──> split experts ──> packed expert archive
                                      │
                                      └─ validated copy
                                             │
                                             v
                                  Internal runtime expert blobs
                                             │ repeated pread
                                             v
                                  Fixed Vates expert pool in memory
                                             │
                    main model + MTP ────────┴─> generated tokens
```

macOS may cache ordinary file-backed pages, but the expert loader's production configuration uses no-cache reads. Vates does not load every model artefact into unified memory. Only the currently resident model state, fixed expert pool, active tensors, caches, and MTP runtime state consume the working set; the split expert files and MTP source shard are never used during inference.

## Fail-fast and recovery behaviour

- Before download or preparation, verify the RAID resolves to the exact mounted volume and has sufficient free space.
- Before copying the hot expert store, verify internal free space and validate the canonical archive.
- Pin the source model revision and retain resumable download state on the RAID.
- Validate model configuration fields: `qwen3_next`, 48 layers, 512 experts, top-k 10, affine 4-bit weights, and group size 64.
- Validate every layer blob and its indexes before using or copying it.
- Copy the internal runtime store through a staging directory and expose it at the final path only after validation, so interrupted copies cannot masquerade as complete stores.
- At launch, check only that Leonard's RAID is mounted. Missing or corrupt files then fail through the existing Vates and MLX error paths.
- Preserve download and preparation artefacts after failure so the operation can resume without repeating completed large transfers.
- Do not fall back to an unpinned remote model identifier or a different local volume.

## Verification

Deployment acceptance requires fresh evidence from the Mac mini:

1. Confirm the RAID mount, free space, pinned model revision, complete model manifest, and configuration values.
2. Validate all 48 layer blobs, per-layer indexes, total blob bytes, and `_split_meta.json` before and after the internal copy.
3. Build the native extension against the deployed virtual environment and confirm that it imports.
4. Run the complete Python test suite on the target.
5. Run a short K=3, 32/8 generation with byte-truth verification enabled. Require zero bad expert reads, no expert-union capacity warning, and no unexpected fallback replay.
6. Generate a real answer through Vates using the intended absolute paths.
7. Run a multi-turn and at least 256-token soak while recording MLX active and peak allocation, process RSS, memory pressure, swap before and after, token throughput, and latency outliers.
8. Reject the deployment profile if byte validation fails, generation fails, swap grows continuously, memory pressure becomes critical, or repeated multi-second latency cliffs occur.

The documented 32/32 profile is not an automatic fallback because its higher memory requirement may be unsafe on 16 GB. If 32/8 fails a correctness gate, investigate the cause before changing capacity. If it is correct but memory pressure is unacceptable, stop and report the measured limit rather than silently lowering `EXPERT_SLOTS` below its validated floor.

## Expected outcome and remaining uncertainty

The hot expert blobs will use the internal SSD, avoiding the RAID's approximately 343 MB/s streaming ceiling during generation. The remaining RAID-backed assets are not continuously streamed through the expert pool. This should materially improve throughput over an all-RAID runtime, but no token-per-second claim is made until the real run.

The primary uncertainty is total system working-set behaviour on a 16 GB M4. Repository evidence makes the 32/8 profile plausible, but current-path 32/8 MTP memory, process RSS, swap behaviour, and sustained throughput have not yet been measured on this machine. The acceptance run resolves those unknowns.
