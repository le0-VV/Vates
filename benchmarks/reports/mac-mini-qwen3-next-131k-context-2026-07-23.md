# Mac mini Qwen3-Next 131k context qualification (2026-07-23)

## Result

The fixed `EXPERT_SLOTS=32`, `POOL_SPEC_SLOTS=16`, `K=3` profile completed an
exact 131,072-token context on Leonard's 16 GB M4 Mac mini. One uninterrupted
process prefixed 131,071 deterministic prompt tokens into one cache, then
decoded one token and advanced every cache offset from 131,071 to exactly
131,072.

No allocation, capacity, byte-integrity or cache-offset failure occurred.

## Method

- Original 4-bit model and configuration: Leonard's RAID.
- Prepared MTP weights, split experts and result logs: internal SSD.
- KV cache: K4/V3 quantisation, group size 64, rotation enabled.
- Prefill chunk: 2 tokens.
- Model profile: 32 real expert slots, 16 speculative side-region slots and
  MTP `K=3`.
- Prompt: deterministic synthetic token IDs, split into progressive targets at
  32,768, 65,536, 98,304 and 131,071 tokens.
- Checkpoints: every 2,048 prompt tokens.
- The stage checks computed a candidate token without advancing the cache. Only
  the final decode advanced the cache from 131,071 to 131,072.

This is a context-capacity and runtime-throughput qualification. The synthetic
prompt is not a language-quality benchmark.

## Timing

| Boundary | Stage time | Stage throughput |
| --- | ---: | ---: |
| 0 → 32,768 | 4,882.21 s | 6.71 tok/s |
| 32,768 → 65,536 | 5,429.20 s | 6.04 tok/s |
| 65,536 → 98,304 | 6,113.84 s | 5.36 tok/s |
| 98,304 → 131,071 | 6,226.56 s | 5.26 tok/s |

- Prompt prefill: 22,651.82 s, or 6 h 17 min 31.82 s.
- Average prompt prefill: 5.786 tok/s.
- One-token boundary decode: 0.2598 s, equivalent to 3.849 tok/s for that
  single observation.
- Total measured run: 22,652.08 s, or 6 h 17 min 32.08 s.

The single boundary token is first-token latency evidence at full context, not
a sustained decode-throughput benchmark.

## Peak resources

- MLX peak allocation: 7.326 GB (6.823 GiB).
- Peak process RSS: 4.911 GB (4.574 GiB).
- Lowest sampled system-wide free memory: 32%.
- Highest sampled swap use: 3,326.19 MiB.
- Internal `disk0` read activity across 1,503 sampled intervals:
  approximately 4,035 reads/s and 1,406 MB/s on average, with observed peaks
  of approximately 6,057 reads/s and 1,703 MB/s.

Memory pressure remained non-critical. Swap fluctuated rather than growing
monotonically, and no resource-related failure was logged.

## Server recovery

After the stress process and monitor exited, the reviewed persistent OpenAI
server was restarted with the same 32/16/K=3 profile, RAID original model and
internal MTP/expert/log paths.

- `GET /health`: passed.
- `GET /v1/models`: advertised
  `qwen3-next-80b-a3b-instruct-4bit`.
- Short real completion: returned exactly `VATES_OK`.
- Short request wall time: 6.21 s.
- Listener: `0.0.0.0:8000`.

The temporary stress harness, monitor scripts and stale stress PID files were
removed. The result logs and live server PID were preserved.

## Preserved evidence

- `context-stress-131072-20260723.jsonl`:
  SHA-256 `f6853751751705436e935329ac967d00ca9d5e6e32a9631fada88e2fea0df190`.
- `context-stress-131072-resources-20260723.log`:
  SHA-256 `8593b874d7783fb2727bc046bf0675e444a8661521eb72070813daeae9f4ab97`.

Both logs remain under the internal runtime `logs` directory. Redundant
derived RAID preparation artefacts remain untouched pending explicit deletion
authorisation.
