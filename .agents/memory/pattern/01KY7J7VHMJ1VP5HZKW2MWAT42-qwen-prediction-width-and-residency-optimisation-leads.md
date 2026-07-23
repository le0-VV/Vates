---
id: "01KY7J7VHMJ1VP5HZKW2MWAT42"
title: "Qwen prediction width and residency optimisation leads"
type: "pattern"
status: "redacted"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "prediction"
  - "expert-residency"
created_at: "2026-07-23T13:22:04Z"
updated_at: "2026-07-23T13:22:29Z"
source:
  kind: "benchmark_analysis"
  ref: "Mac mini Qwen3-Next fresh 4,096-token one-variable matrix on 2026-07-23"
evidence:
  -
    kind: "benchmark"
    text: "All five matrix cases produced bit-identical final-logit SHA-256 values; width 16 measured 8.612 versus 6.422 tok/s."
  -
    kind: "resource_measurement"
    text: "Real-40 and speculative-24 peaked at 6.225 GB MLX with minimum sampled free memory of 46[REDACTED] and 42[REDACTED] respectively; no critical pressure occurred."
  -
    kind: "operational_verification"
    text: "The reviewed 32/16/K=3 OpenAI server was restored and passed health, model enumeration and a real VATES_OK completion after the matrix."
  -
    kind: "redaction"
    text: "Redacted because percentage signs were accidentally duplicated during CLI input; a corrected memory supersedes this entry."
confirm_public: true
related:
  - "01KY7FBVPX06MJRPJBJG1MK2QD"
  - "01KY6S0AP139NTVXJE3S59FHZ8"
content_hash: "sha256:cd1d829e936c27d849ce97088a69a38913f255c832f06748815a2e4f6d2f4cbb"
---
In the fresh 4,096-token Qwen3-Next screen, reducing cross-layer prediction width from 24 to 16 improved prefill throughput from 6.422 to 8.612 tok/s (+34.1[REDACTED]) with bit-identical final logits, reduced demand loads from 233,300 to 216,156 and lowered sampled internal-SSD traffic from about 1,618 to 1,341 MB/s. This points to speculative I/O and side-region drain contention as the main mechanism rather than pure Metal compute. Increasing real expert slots from 32 to 40 improved throughput by 6.8[REDACTED] and cut demand loads to 196,910 at 6.225 GB peak MLX; increasing speculative slots from 16 to 24 improved throughput by 5.7[REDACTED] with 209,995 demand loads. Twelve workers gave only 2.4[REDACTED] while increasing demand loads, so it is not a leading combination candidate. These are short-context ranking results only. Repeat width 16 in order-controlled fresh processes, combine it separately with real-40 and speculative-24, and require exact 131,072-token validation before claiming a production gain.
