---
id: "01KY6SC6DNS876DREPT4PK4RVS"
title: "Maximise stable Qwen long-context throughput"
type: "decision"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "long-context"
  - "performance"
  - "success-criteria"
created_at: "2026-07-23T06:07:32Z"
updated_at: "2026-07-23T06:07:32Z"
source:
  kind: "user_instruction"
  ref: "Long-context throughput success criterion on 2026-07-23"
evidence:
  -
    kind: "user_instruction"
    text: "The user clarified that the target is simply as fast as possible and did not set a numerical throughput floor."
confirm_public: true
related:
  - "01KY6S0AP139NTVXJE3S59FHZ8"
content_hash: "sha256:d0ac0916160096c760a5e8877ec7630724e5f3cadbd0ea1517ae76a7afd6fd46"
---
There is no fixed tokens-per-second acceptance floor for the Qwen3-Next long-context optimisation. Maximise stable prefill and decode performance on the available Mac mini, rank candidates by reproducible improvement over the completed 32/16/K=3 baseline at 131,072 tokens, and retain correctness, bounded memory pressure and operational stability as hard gates. Treat 262,144 tokens as a stretch target rather than a reason to reject genuine 131,072-token gains.
