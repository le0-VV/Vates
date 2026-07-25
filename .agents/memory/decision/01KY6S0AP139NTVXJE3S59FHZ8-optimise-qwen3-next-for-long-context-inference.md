---
id: "01KY6S0AP139NTVXJE3S59FHZ8"
title: "Optimise Qwen3-Next for long-context inference"
type: "decision"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "mlx"
created_at: "2026-07-23T06:01:03Z"
updated_at: "2026-07-23T06:01:03Z"
source:
  kind: "user_instruction"
  ref: "Long-context Qwen inference optimisation direction on 2026-07-23"
evidence:
  -
    kind: "user_instruction"
    text: "The user said the ideal result is an 80B MoE runtime at 131k or 256k context at reasonable speed, and that smaller contexts may be used during testing but do not adequately validate improvements."
confirm_public: true
related:
  - "01KY6M9RG28BD9CRKYWQZ8A6ZH"
content_hash: "sha256:b73da49a7bc39668e7621dd357ac67fae7bfed620681470ef4505deb20bbe48b"
---
Treat smaller-context benchmarks only as fast iteration probes. Final performance claims must reproduce at an exact 131,072-token context, and 262,144 tokens is the stretch target when the pinned model and runtime configuration support it. The objective is a practical Qwen3-Next-80B-A3B runtime that improves long-context prefill and decode through evidence-backed memory residency, expert streaming, storage scheduling, synchronisation and Metal utilisation changes without sacrificing correctness or stability. Do not accept a speed-up that exists only at short context.
