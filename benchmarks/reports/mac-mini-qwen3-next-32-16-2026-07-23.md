# Mac mini Qwen3-Next 32/16 side-region tuning (2026-07-23)

## Decision

Use `EXPERT_SLOTS=32`, `POOL_SPEC_SLOTS=16` and `K=3` as the fixed profile for Leonard's 16 GB M4 Mac mini. In a controlled A/B/A/B comparison, 32/16 improved median speculative throughput by approximately 5.05% over the accepted 32/8 profile, preserved exact output and recovered safely after each process. This clears the repository's greater-than-5% tuning threshold.

The change is limited to speculative side-region capacity. K4/V3 KV quantisation, prefill chunk 2, adaptive threshold 0.3, maximum depth 3, the 32-slot real-region floor and all storage paths remain unchanged.

## Target and method

- Target: `leonardw@leonards-mac-mini`, Apple M4, 16 GB unified memory.
- Source checkout: clean `agent/qwen3-next-mlx-raid` at `0bd17f96627df63cdfc054ada05985a46471f6b7`.
- Four verifier-off `mlx_streaming.runtime.run_mtp_spec` processes in A/B/A/B order.
- Fixed settings: `EXPERT_SLOTS=32`, `K=3`, `MAXTOK=64`, `WARMUP_TOK=32`, `REPEAT=2` and the same default Chinese prompt.
- Fixed runtime path: hot experts on the internal SSD; canonical model and MTP assets on Leonard's RAID.
- Fixed fast path: stream loader, zero-copy dual source, native fused prefetch and persistent LFU enabled.
- Fixed memory/runtime settings: K4/V3 KV quantisation, prefill chunk 2, adaptive threshold 0.3 and maximum depth 3.
- `STG_VERIFY`, `UNION_PROF` and timing/recall/miss probes were unset.
- `DUMP_IDS=1` retained cross-run token evidence.

Each process produced two measured repeats after warm-up. Each repeat produced 64 greedy and 64 speculative tokens.

## Results

| Case | Spec repeats (tok/s) | Process median | Baseline median | Spec hit | Spec loads | GPU fast/fallback | MLX peak | Reported RSS |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 32/8 A | 4.91, 4.87 | 4.890 | 2.76 | 0.805 | 7,361 | 122 / 1,414 | 6.00 GB | 3.80 GB |
| 32/16 B | 5.11, 5.16 | 5.135 | 2.61 | 0.869 | 5,023 | 473 / 1,063 | 6.68 GB | 5.68 GB |
| 32/8 C | 4.80, 4.83 | 4.815 | 2.73 | 0.806 | 7,329 | 120 / 1,416 | 6.00 GB | 5.46 GB |
| 32/16 D | 5.08, 5.07 | 5.075 | 2.61 | 0.868 | 5,046 | 468 / 1,068 | 6.68 GB | 5.47 GB |

Across the four speculative repeats per capacity:

- 32/8 median: 4.85 tok/s;
- 32/16 median: 5.095 tok/s;
- net 32/16 gain: approximately 5.05%;
- process-median centre gain: approximately 5.2%;
- demand loads: approximately 31.5% lower at 32/16;
- layer fast-path calls: roughly 3.9 times higher at 32/16;
- MLX peak cost: 0.68 GB.

Both 32/16 processes exceeded their neighbouring 32/8 controls, so the result is not explained by a simple warm-page-cache order effect.

## Correctness and memory

All eight speculative repeats exactly matched greedy output. Every repeat reported `n_mismatch=0` and `fallback_replays=0`, and the dumped token sequences were identical across both capacities.

System observations were:

- before the first process: 87% system-wide free and 1,538.25 MiB swap used;
- sampled 32/8 load: 24% free and 2,997.94 MiB swap used;
- sampled 32/16 load: 22% free and 3,287.38 MiB swap used;
- after all four processes: 87% free and 2,045.12 MiB swap used.

The 32/16 sample used about 0.29 GB more swap than the 32/8 sample. No critical pressure, allocation failure, stale process or continuously growing post-run state was observed. Final swap was 507 MiB above the initial state after four consecutive model processes and below several intermediate post-run readings.

The initial 32/8 deployment qualification remains the historical byte-truth and 256-token soak evidence. This tuning benchmark was verifier-off and establishes the comparative performance, exact-output and observed-memory case for changing only the side-region capacity to 16.

## Historical 32/32 claim

A July 2026 Vates regression report recorded 9.6–10.77 speculative decode tok/s after a memory optimisation with 32 real rows, 32 side rows, `K=3` and `MAXTOK=48`. The report explicitly states that the 32 GB machine's swap was nearly full and that absolute throughput was untrustworthy. Model loading and prompt prefill were outside the timer, one output token was counted before the timer, and complete host, storage, command and raw-log provenance was not retained.

The historical range is therefore a regression signal, not a standardised serving-speed claim, not evidence for a particular M5 host and not directly comparable with Leonard's 16 GB Mac mini. The authoritative tracked source is `benchmarks/reports/peak-shrink-2026-07-03.md` at commit `da6ecb8bdefd6c6f7425e9afeb3417e61e012e49`.

## Preserved evidence

The raw A/B logs remain on Leonard's RAID and were not copied into Git:

- `qwen3-next-ab-32-8-a-20260723.log` — SHA-256 `664298110537fc8cbfef27ce935d59d52579a9cdcce90dca5cf5f9cc9a1e1cfa`;
- `qwen3-next-ab-32-16-b-20260723.log` — SHA-256 `4486897779163305961f584da75d6f77520c6564404d7af5d8a6781ab3c48507`;
- `qwen3-next-ab-32-8-c-20260723.log` — SHA-256 `e74f3d55c413928e6f62b243df119aa4ae09b6c14f61f62ddc4ccbd99f068c5f`;
- `qwen3-next-ab-32-16-d-20260723.log` — SHA-256 `5bd208b24e1584731df51f3824551b758d3c5618020c1fbc180f15ea738a2cf4`.
