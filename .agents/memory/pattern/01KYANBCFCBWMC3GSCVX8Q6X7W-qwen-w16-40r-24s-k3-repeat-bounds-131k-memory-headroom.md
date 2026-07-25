---
id: "01KYANBCFCBWMC3GSCVX8Q6X7W"
title: "Qwen w16/40r/24s/k3 repeat bounds 131k memory headroom"
type: "pattern"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "prediction"
  - "expert-residency"
  - "memory-pressure"
  - "mlx"
created_at: "2026-07-24T18:14:09Z"
updated_at: "2026-07-24T18:14:09Z"
source:
  kind: "benchmark_analysis"
  ref: "Mac mini Qwen3-Next exact 131k w16/40r/24s/k3 repeat on 2026-07-24"
evidence:
  -
    kind: "benchmark"
    text: "The repeat processed 131,071 prompt tokens in 18,857.28 seconds at 6.9507 tok/s; stage throughput was 9.052, 7.660, 6.449 and 5.573 tok/s, and the one-token boundary decode took 0.2061 seconds."
  -
    kind: "correctness"
    text: "Stage tokens were 88, 13, 220 and 79; feeding token 79 advanced every cache to exactly 131,072, produced following token 220 and reproduced final-logit SHA-256 7817f7d722692fa5cd50e11447bf217d2ee038828100e863697fa98b483709c4."
  -
    kind: "resource_measurement"
    text: "Peak MLX/RSS were 8.685/5.681 GB. Minimum sampled free memory was 21% near 81,920 tokens; swap peaked at 1,872.75 MiB around 90k–92k tokens and fell to 1,824.75 MiB by the final sample."
  -
    kind: "profiling"
    text: "The repeat recorded 4,966,756 demand loads, 57,805,648 resident hits, 1,458,516 GPU fast selections, 1,687,260 fallbacks and 805,306,368 KV bytes."
  -
    kind: "operational_verification"
    text: "After the repeat, the supervisor restored the original RAID-backed model with internal MTP and experts at fixed 32/16/K=3; independent health, model enumeration and exact VATES_OK checks passed."
confirm_public: true
related:
  - "01KYA0K7CFXZV15HJJEH4N2H6K"
  - "01KY7P3P2G75ZMMDX66GNCNQKH"
  - "01KY7FBVPX06MJRPJBJG1MK2QD"
  - "01KY7SHPHR35RX0KJ40WAA6R2K"
content_hash: "sha256:866db5b1910f90f38112e829ea41d4439ac9cdf85749c9b9313e588a3cd50d04"
---
A second fresh-process exact 131,072-token validation of transient w16/40r/24s/k3 reproduced correctness and bounded pressure. It prefills 131,071 prompt tokens at 6.9507 tok/s, 4.1% below the 7.2446 tok/s qualification but still 20.1% above the reviewed 5.786 tok/s baseline. Boundary decode was 0.2061 seconds. Peak MLX was unchanged at 8.685 GB, peak RSS was 5.681 GB, minimum sampled free memory was 21% near the 81,920-token checkpoint and swap peaked at 1,872.75 MiB around 90k–92k tokens before declining. The repeat shows w16/40r/24s/k3 has bounded but not excessive headroom on the 16 GB system. If more residency is tested, use w16/48r/24s/k3 only as a gated one-variable short screen; the projected 0.68 GB extra MLX allocation requires direct measurement. The reviewed persistent server remains 32/16/K=3 pending explicit user approval.
