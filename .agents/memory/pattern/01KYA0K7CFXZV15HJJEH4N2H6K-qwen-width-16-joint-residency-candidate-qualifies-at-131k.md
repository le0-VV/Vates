---
id: "01KYA0K7CFXZV15HJJEH4N2H6K"
title: "Qwen width-16 joint residency candidate qualifies at 131k"
type: "pattern"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "prediction"
  - "expert-residency"
  - "mlx"
created_at: "2026-07-24T12:11:25Z"
updated_at: "2026-07-24T12:11:25Z"
source:
  kind: "benchmark_analysis"
  ref: "Mac mini Qwen3-Next exact 131k candidate qualification and exact 262k stretch on 2026-07-23/24"
evidence:
  -
    kind: "benchmark"
    text: "At exactly 131,071 prompt tokens, width16/real40/spec24 measured 7.2446 tok/s versus the reviewed 5.786 tok/s baseline and reduced one-token boundary decode from 0.2598 to 0.2085 seconds."
  -
    kind: "correctness"
    text: "The 131k stage tokens were 88, 13, 220 and 79; one decode advanced all cache offsets to 131,072 and produced following token 220. Final-logit SHA-256 was 7817f7d722692fa5cd50e11447bf217d2ee038828100e863697fa98b483709c4."
  -
    kind: "stretch_validation"
    text: "The same transient profile completed 262,143 prompt tokens in 49,567.44 seconds, decoded once to exact cache offset 262,144 in 0.4609 seconds, and recorded final-logit SHA-256 6db97f1105eb2e9005bc9c9fc9e80ab2fc9172c0fcf09101a734820c1bcc14ea."
  -
    kind: "resource_measurement"
    text: "Peak MLX/RSS and minimum free memory were 8.685/6.208 GB and 23% at 131k, then 9.567/6.289 GB and 18% at 262k; pressure stayed bounded and recovered after exit."
  -
    kind: "operational_verification"
    text: "After each validation the reviewed internal-MTP/internal-expert 32/16/K=3 server restored; final independent health, model enumeration and exact VATES_OK inference checks passed."
confirm_public: true
supersedes:
  - "01KY7P3P2G75ZMMDX66GNCNQKH"
related:
  - "01KY7FBVPX06MJRPJBJG1MK2QD"
  - "01KY7SHPHR35RX0KJ40WAA6R2K"
  - "01KY6S0AP139NTVXJE3S59FHZ8"
  - "01KY6SC6DNS876DREPT4PK4RVS"
content_hash: "sha256:07ac38ff5ddc9781a6a46dbb647fee9ec8cd1dc723ef23a625179e493b88cd72"
---
The transient Qwen3-Next width16/real40/spec24 candidate is long-context-qualified at exactly 131,072 tokens. It prefills 131,071 prompt tokens at 7.2446 tok/s versus the reviewed 32/16/K=3 baseline's 5.786 tok/s (+25.2%) and reduces one-token boundary-decode latency from 0.2598 to 0.2085 seconds (-19.8%). Stage tokens 88, 13, 220 and 79 matched the preserved baseline; feeding token 79 advanced every cache to exactly 131,072 and produced following token 220. Peak MLX was 8.685 GB, peak RSS 6.208 GB, minimum sampled free memory 23% and maximum swap 2,668 MiB. The same transient profile also completed the exact 262,144-token stretch at 5.2886 tok/s overall with a 0.4609-second boundary decode, exact cache offset, 9.567 GB peak MLX, 6.289 GB peak RSS, 18% minimum free memory and 3,084 MiB peak swap. This proves 262k support and stability, not a 262k speed-up, because no matching 262k reviewed-profile baseline exists. The reviewed persistent server remains 32/16/K=3 pending explicit user approval.
