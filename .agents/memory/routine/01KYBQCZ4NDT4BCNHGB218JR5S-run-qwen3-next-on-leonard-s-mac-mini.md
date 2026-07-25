---
id: "01KYBQCZ4NDT4BCNHGB218JR5S"
title: "Run Qwen3-Next on Leonard's Mac mini"
type: "routine"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "mac-mini"
  - "mlx"
  - "long-context"
  - "performance"
  - "deployment"
  - "storage-policy"
created_at: "2026-07-25T04:09:12Z"
updated_at: "2026-07-25T04:09:12Z"
source:
  kind: "user_instruction"
  ref: "Persistent Qwen profile approval and live restart on 2026-07-25"
evidence:
  -
    kind: "user_instruction"
    text: "The user explicitly approved w16/40r/24s/k3 as the default persistent profile and requested a server restart."
  -
    kind: "benchmark"
    text: "Two exact 131,072-token validations measured 7.2446 and 6.9507 prompt tok/s, 0.2085 and 0.2061 second boundary decode, exact matching tokens and the same final-logit SHA-256."
  -
    kind: "resource_measurement"
    text: "At 131k the profile peaked at 8.685 GB MLX with 21–23% minimum sampled free memory; pressure remained bounded and recovered."
  -
    kind: "operational_verification"
    text: "The restarted live process reported EXPERT_SLOTS=40, POOL_SPEC_SLOTS=24, CROSS_LAYER_PREDICT_WIDTH=16, PREFILL_CHUNK=2 and MTP_DEPTH_MAX=3; health, model enumeration and exact VATES_OK inference passed."
confirm_public: true
supersedes:
  - "01KY6M9RG28BD9CRKYWQZ8A6ZH"
related:
  - "01KYA0K7CFXZV15HJJEH4N2H6K"
  - "01KYANBCFCBWMC3GSCVX8Q6X7W"
prerequisites:
  - "Leonard's RAID is mounted at /Volumes/Leonard's RAID."
  - "The internal MTP file and 48-layer expert store are present under the internal runtime directory."
steps:
  - "Connect to leonardw@leonards-mac-mini."
  - "Change directory to /Users/leonardw/Projects/Vates."
  - "Run scripts/run_mac_mini_qwen3_next.py with the desired chat or serve arguments."
  - "Keep persistent service logs in the internal runtime logs directory."
verify: "The live command uses 40 real slots, 24 speculative slots and K=3; the environment sets prediction width 16; health, model enumeration and an exact VATES_OK completion pass."
content_hash: "sha256:29221ceb39cd784cc45d1f182b5efc87eefffb8d19a206f0d6cf9dc6b0a112e0"
---
Use /Users/leonardw/Projects/Vates with only the canonical original model source files under /Volumes/Leonard's RAID/Vates/models/qwen3_next_80b_4bit. Keep the prepared MTP file, expert store and service logs under the internal runtime directory. The approved persistent server profile is w16/40r/24s/k3: CROSS_LAYER_PREDICT_WIDTH=16, EXPERT_SLOTS=40, POOL_SPEC_SLOTS=24 and MTP K=3. Keep PREFILL_CHUNK=2, quantised-KV step 256, K4/V3 KV settings and maximum MTP depth 3. Launch scripts/run_mac_mini_qwen3_next.py; its protected command and environment defaults enforce the approved profile. Do not change the profile again without explicit user approval.
