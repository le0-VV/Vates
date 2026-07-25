---
id: "01KY6M9RG28BD9CRKYWQZ8A6ZH"
title: "Run Qwen3-Next on Leonard's Mac mini"
type: "routine"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "mac-mini"
  - "mlx"
  - "storage-policy"
created_at: "2026-07-23T04:38:49Z"
updated_at: "2026-07-23T04:38:49Z"
source:
  kind: "user_instruction"
  ref: "Storage placement clarification and live audit on 2026-07-23"
evidence:
  -
    kind: "user_instruction"
    text: "The user clarified that only original model files should remain on Leonard's RAID and that derived runtime assets belong on the internal SSD."
  -
    kind: "verification"
    text: "The internal prepared MTP copy is exactly 3,300,945,134 bytes and matched SHA-256 9d45df2194bf932c7f1fd9cd270e2faee39c87c72c118fb1efd276677e78e395; the internal expert store was previously byte-qualified."
confirm_public: true
supersedes:
  - "01KY6543A2VFB06PD9GAJDYZKB"
prerequisites:
  - "Leonard's RAID is mounted at /Volumes/Leonard's RAID."
  - "The internal prepared MTP file passes its expected size and SHA-256 checks."
  - "The internal expert store contains all 48 validated layer blobs."
steps:
  - "Connect to leonardw@leonards-mac-mini."
  - "Change directory to /Users/leonardw/Projects/Vates."
  - "Confirm the launcher uses the RAID only for the original model and uses internal MTP and expert paths."
  - "Run scripts/run_mac_mini_qwen3_next.py with the desired chat or serve arguments."
  - "Write persistent service logs below the internal runtime logs directory."
verify: "The process command uses the RAID original-model path, internal MTP and expert paths, fixed 32/16/K=3, and produces a non-empty response without critical memory pressure."
content_hash: "sha256:f75bb205ff95bcd2e82fb72b5d5539a1078899d779090cb0c4d071adb1e71984"
---
Use /Users/leonardw/Projects/Vates with only canonical original model source files retained under /Volumes/Leonard's RAID/Vates. The pinned 4-bit model directory remains at /Volumes/Leonard's RAID/Vates/models/qwen3_next_80b_4bit. The prepared MTP file must be /Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/mtp/qn_mtp_weights.safetensors, the expert store must be the sibling experts directory, and service logs must be under the sibling logs directory on the internal SSD. Temporary derived RAID preparation artefacts must be removed after verified internal publication; do not silently fall back to RAID runtime paths. Launch scripts/run_mac_mini_qwen3_next.py with EXPERT_SLOTS=32, POOL_SPEC_SLOTS=16, and K=3. The launcher requires Leonard's RAID to be mounted for the original model. The pinned model revision is d8a069bfa8ae87d3d468412e1034acae19b5892b. Do not lower EXPERT_SLOTS below 32 or increase K/depth above 3.
