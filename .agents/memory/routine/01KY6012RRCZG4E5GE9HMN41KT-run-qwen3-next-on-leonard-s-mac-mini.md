---
id: "01KY6012RRCZG4E5GE9HMN41KT"
title: "Run Qwen3-Next on Leonard's Mac mini"
type: "routine"
status: "redacted"
tags:
  - "vates"
  - "qwen3-next"
  - "mac-mini"
  - "mlx"
created_at: "2026-07-22T22:44:33Z"
updated_at: "2026-07-23T00:16:08Z"
source:
  kind: "deployment"
  ref: "Mac mini Qwen3-Next acceptance run on 2026-07-22"
evidence:
  -
    kind: "verification"
    text: "The target completed the native build/import, full test suite, byte-truth smoke, real Vates response, and 256-token stability soak."
  -
    kind: "redaction"
    text: "This operating routine was superseded by the measured 32/16 profile on 2026-07-23; retain the historical qualification in the tracked design and benchmark report, not as active launch guidance."
confirm_public: true
prerequisites:
  - "Leonard's RAID is mounted at /Volumes/Leonard's RAID."
  - "The internal expert store contains all 48 validated layer blobs."
steps:
  - "Connect to leonardw@leonards-mac-mini."
  - "Change directory to /Users/leonardw/Projects/Vates."
  - "Run .venv/bin/python scripts/run_mac_mini_qwen3_next.py with any desired Vates chat arguments."
verify: "Vates loads the pinned model, prints a non-empty response, and reports positive throughput without critical memory pressure."
content_hash: "sha256:dcfc94a779b7fee5d84d665a9b2874f08ee41b188373fac43e42a33bb079ff9a"
---
Use /Users/leonardw/Projects/Vates with canonical model and MTP assets under /Volumes/Leonard's RAID/Vates, and the hot expert store under /Users/leonardw/Library/Application Support/Vates/qwen3-next-80b-a3b-instruct-4bit/experts. Launch scripts/run_mac_mini_qwen3_next.py with EXPERT_SLOTS=32, [REDACTED], and K=3. The launcher requires Leonard's RAID to be mounted. The pinned model revision is d8a069bfa8ae87d3d468412e1034acae19b5892b. Do not lower EXPERT_SLOTS below 32 or increase K/depth above 3.
