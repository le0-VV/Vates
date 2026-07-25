---
id: "01KY7P3P2G75ZMMDX66GNCNQKH"
title: "Qwen width-16 joint residency combination lead"
type: "pattern"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "prediction"
  - "expert-residency"
created_at: "2026-07-23T14:29:42Z"
updated_at: "2026-07-23T14:29:42Z"
source:
  kind: "benchmark_analysis"
  ref: "Mac mini Qwen3-Next order-controlled 4,096-token prediction-width/residency combination screen on 2026-07-23"
evidence:
  -
    kind: "benchmark"
    text: "Repeated baseline median was 6.337 tok/s, repeated width-16 median was 8.667 tok/s and the joint width16/real40/spec24 case was 9.542 tok/s."
  -
    kind: "correctness"
    text: "All seven final-logit arrays shared SHA-256 e2f88986314f3b3051cd270d25ce4685ef7be90829e384cf4cc6adb99a818d48 and produced matching next and decoded tokens."
  -
    kind: "resource_measurement"
    text: "The joint case peaked at 6.904 GB MLX, retained 38% minimum sampled free memory and swap fell from 1,601 to 1,593 MiB."
  -
    kind: "operational_verification"
    text: "The reviewed 32/16/K=3 server restored and passed independent health, model enumeration and real VATES_OK inference checks."
confirm_public: true
related:
  - "01KY7J9323TMF07Q5XP0NMNB45"
  - "01KY7FBVPX06MJRPJBJG1MK2QD"
  - "01KY6S0AP139NTVXJE3S59FHZ8"
content_hash: "sha256:f8c60d7dc07277e15f3c0bf982b62a6a547d2ed642a9322afd1432d5d663872a"
---
An order-controlled fresh-process 4,096-token screen reproduced width 16 at 8.654 and 8.679 tok/s versus repeated width-24 baselines of 6.352 and 6.323 tok/s. Combining width 16 separately with real-40 and speculative-24 produced 9.123 and 9.119 tok/s. After both individual cases passed exact-logit and memory gates, the joint width16/real40/spec24 case reached 9.542 tok/s: 50.6% above the repeated baseline median and 10.1% above the width-16 median. Demand loads fell to 168,519, 27.9% below the repeated baseline. The joint case used 6.904 GB peak MLX, retained 38% minimum sampled free memory, and swap declined during the run. All seven final-logit arrays had the same SHA-256, and next-token and boundary-decode outputs matched. This is the leading short-context candidate only; it requires exact 131,072-token validation before any persistent recommendation.
