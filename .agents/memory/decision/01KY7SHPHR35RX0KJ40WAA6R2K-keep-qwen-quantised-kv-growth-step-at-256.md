---
id: "01KY7SHPHR35RX0KJ40WAA6R2K"
title: "Keep Qwen quantised-KV growth step at 256"
type: "decision"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "kv-cache"
created_at: "2026-07-23T15:29:47Z"
updated_at: "2026-07-23T15:29:47Z"
source:
  kind: "benchmark_analysis"
  ref: "Mac mini Qwen3-Next order-controlled 8,192-token KV-growth A/B/A screen on 2026-07-23"
evidence:
  -
    kind: "benchmark"
    text: "Step256-a, step8192 and step256-b measured 8.530, 8.864 and 8.751 tok/s respectively with identical final-logit arrays."
  -
    kind: "correctness"
    text: "All three final-logit arrays shared SHA-256 1b4aaf4ed078908e3783076cb23dea6c5925e5d26c0954bb52fb5a6ba6949c98 and cache offsets reached 8,193."
  -
    kind: "resource_measurement"
    text: "Step8192 ended at 100,663,296 KV bytes and 16,384-token capacity with 2,459 MiB peak swap; step256 ended at 51,904,512 bytes and 8,448-token capacity."
  -
    kind: "operational_verification"
    text: "The reviewed 32/16/K=3 server restored and passed independent health, model enumeration and real VATES_OK inference checks."
confirm_public: true
related:
  - "01KY7P3P2G75ZMMDX66GNCNQKH"
  - "01KY7FBVPX06MJRPJBJG1MK2QD"
  - "01KY6S0AP139NTVXJE3S59FHZ8"
content_hash: "sha256:b505a2e1da3bf0cd3a775112c8151e6bd777bde07fe0da3cb20303a3f2f7d516"
---
Keep AsymmetricQuantizedKVCache.step at 256 for the exact 131,072-token candidate validation. In an order-controlled 8,192-prompt-token A/B/A screen, step 8192 reached 8.864 tok/s versus step-256 references of 8.530 and 8.751 tok/s: only 2.6% above their median and 1.3% above the faster reference. All final logits matched exactly, but the 8,193rd boundary token doubled step-8192 capacity from 8,192 to 16,384, increasing final KV bytes from 51,904,512 to 100,663,296. Swap peaked at 2,459 MiB versus 1,593 MiB in the first reference. The candidate fails the equal-final-bytes gate and the small speed lead does not justify a boundary-aligned follow-up. A future bounded or geometric policy may revisit cache-copy overhead without whole-step boundary over-allocation.
