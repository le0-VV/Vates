# Mac mini Qwen3-Next long-context optimisation surfaces (2026-07-23)

## Scope and constraints

This note records evidence-backed opportunities to accelerate the 4-bit
Qwen3-Next-80B-A3B runtime on Leonard's 16 GB M4 Mac mini. It is an
investigation ledger, not an approved persistent configuration.

The reviewed serving profile remains:

- 32 real expert slots, 16 speculative side-region slots and MTP `K=3`;
- two-token prefill chunks;
- K4/V3 quantised KV cache with group size 64 and rotation enabled;
- original model and configuration on Leonard's RAID;
- prepared MTP weights, split experts and logs on the internal SSD.

Any candidate must preserve correctness, bounded memory pressure and
operational stability. Shorter-context screens may reject or rank candidates,
but cannot establish a production improvement. A surviving candidate must
complete the exact 131,071-prompt-token plus one-token-decode boundary before
it can be recommended. The 262,144-token model limit remains a separate stretch
target.

## Established baseline

The reviewed profile completed the exact 131,072-token boundary:

- 131,071 prompt tokens in 22,651.82 seconds: 5.786 tok/s;
- one boundary decode in 0.2598 seconds;
- 7.326 GB peak MLX allocation and 4.911 GB peak RSS;
- minimum sampled free memory 32% and maximum swap use 3,326 MiB;
- internal SSD average approximately 1,406 MB/s and 4,035 reads/s.

A reproducible short diagnostic with 64 generated tokens measured:

- speculative decode: 5.02 tok/s;
- demand-core time, including synchronous expert-miss reads: 47.61 seconds;
- routed-ID/Metal wait: 11.29 seconds;
- Python-side prediction bookkeeping: 0.01 seconds;
- exact greedy/speculative token agreement.

The hot expert source is the internal SSD. The RAID supplies original model
weights during loading but is not the steady expert-demand bottleneck.

## Ranked performance surfaces

### 1. Reduce forward and routing boundaries

Two-token prefill creates 65,536 forwards at 131,071 prompt tokens. Across 48
MoE layers, this produces roughly 3.15 million routing synchronisation points.
Each native dual-pool acquisition evaluates routed IDs before demand loading,
and each new chunk drains preceding side-region work.

An isolated 8,192-token screen with three-token chunks measured:

| Profile | Prefill time | Throughput | Boundary decode |
| --- | ---: | ---: | ---: |
| Chunk 2 | 1,203.82 s | 6.805 tok/s | 0.3580 s |
| Chunk 3 | 969.25 s | 8.452 tok/s | 0.3293 s |

The measured throughput gain was 24.2%, with 6.1% fewer demand loads. Top-token
and boundary-decoded-token choices matched. However, final-logit cosine was
0.98727, below the repository's 0.99 quality gate, so chunk 3 is currently a
performance lead with a correctness concern, not an acceptable change.

Chunk 3 is the largest safe candidate with 32 real rows because
`3 × top_k(10) = 30`. Chunk 4 can route 40 positions and exceed the native
32-row real-pool authority even though the combined real/side physical storage
is larger.

Potential follow-up:

- preserve two-token numerical grouping while amortising host dispatch;
- batch multiple two-token groups inside one controlled command-buffer
  schedule;
- avoid a global side-region drain at every group when dependencies allow it;
- reduce per-layer routed-ID evaluation without delaying required demand data.

### 2. Increase expert residency

Expert misses dominate the measured wall time. More residency can reduce both
internal-SSD reads and the frequency of the whole-layer host fallback.

The completed one-variable screen measured:

| Profile | Throughput | Gain | Demand loads | Peak MLX |
| --- | ---: | ---: | ---: | ---: |
| 32 real / 16 speculative | 6.422 tok/s | baseline | 233,300 | 5.545 GB |
| 40 real / 16 speculative | 6.860 tok/s | +6.8% | 196,910 | 6.225 GB |
| 32 real / 24 speculative | 6.790 tok/s | +5.7% | 209,995 | 6.225 GB |

Both candidates produced final logits that were bit-identical to the fresh
baseline. Real-40 reduced demand loads more strongly and is the leading
residency candidate. The extra 0.68 GB MLX allocation left 46% minimum sampled
free memory for real-40 and 42% for speculative-24; neither produced a critical
pressure event in the short screen.

The 131k qualification left enough headroom for one cautious increase, but the
3.3 GB observed swap peak requires strict recovery and no-growth checks.
Increasing both regions simultaneously must not be attempted until each is
combined with the prediction-width lead independently and their joint
combination is justified.

### 3. Prevent speculative reads from delaying demand reads

Eight background workers serve demand and speculative expert reads. Demand
jobs have queue priority but cannot pre-empt an active speculative job.
A speculative side-region job can read up to 16 full expert records serially,
so all workers can be occupied when a demand miss arrives. The next prefill
group also waits in `sideregion_drain()`.

The worker-count screen measured:

| Profile | Throughput | Gain | Demand loads | Peak MLX |
| --- | ---: | ---: | ---: | ---: |
| 8 workers | 6.422 tok/s | baseline | 233,300 | 5.545 GB |
| 12 workers | 6.573 tok/s | +2.4% | 235,958 | 5.545 GB |

The 12-worker result was bit-identical but increased demand loads and delivered
only a marginal short-screen gain. It is not being combined yet. Higher-value
follow-up is a lower active-low-priority cap, bounded or pre-emptible
speculative fills, demand-ticket and queue-depth instrumentation, and
per-forward side-drain timing.

Worker count is a scheduling lever, not a guaranteed bandwidth gain. The
internal SSD already sustains roughly 1.4–1.6 GB/s and 4,000–5,000 reads/s;
more workers may simply increase contention.

### 4. Improve prediction usefulness rather than prediction volume

Historical evidence attributes misses to two classes:

- predicted experts that arrived too late or were evicted;
- experts that were not predicted.

Wider prediction can flood the low-priority queue and make useful reads arrive
later. The completed width screen measured:

| Profile | Throughput | Gain | Demand loads | Peak MLX |
| --- | ---: | ---: | ---: | ---: |
| Width 24 | 6.422 tok/s | baseline | 233,300 | 5.545 GB |
| Width 16 | 8.612 tok/s | +34.1% | 216,156 | 5.545 GB |

Width 16 produced bit-identical final logits and reduced measured internal-SSD
traffic from approximately 1,618 to 1,341 MB/s. Despite fewer GPU fast-path
acquisitions, it was substantially faster, which points to less speculative
I/O and side-drain contention rather than a pure Metal-kernel gain. This is the
strongest short-screen lead and now requires an order-controlled repeat.
Follow-up parameters should be changed one at a time:

- prediction width and per-layer budget;
- look-ahead distance;
- timing-miss versus eviction-miss split;
- realised recall, demand loads and side-read waste.

An improvement is useful only if reduced demand wait outweighs prediction
compute and speculative I/O.

### 5. Improve expert-read granularity and cache policy

The 131k resource trace implies many segmented reads, not full-record streaming.
Native demand reads cached segments, while side-region fills deliberately use
uncached full-record reads. `STREAM_BLOB_NOCACHE` does not control this mixed
native steady-state policy.

Potential work:

- coalesce adjacent demand segments or expert records where it reduces IOPS;
- retain hot expert pages without causing unbounded page-cache pressure;
- separate demand and speculative reader pools or enforce a low-priority
  concurrency cap;
- measure bytes read per useful resident hit;
- avoid repeated open/close work for side-region blobs where safe.

Page-cache results must include cold and warm processes, swap writes and
post-run memory recovery. A warm-cache-only speed-up is not a serving result.

### 6. Reduce long-context KV growth and attention work

All 12 full-attention layers scan the quantised KV cache. This makes boundary
decode inherently dependent on context length. The cache currently grows in
256-token steps, producing 512 reallocations/copies by 131,072 tokens.

An order-controlled 8,192-prompt-token A/B/A screen held prediction width 16
and 32/16 residency fixed:

| Growth step | Throughput | Growth events | Final capacity | Final KV bytes |
| --- | ---: | ---: | ---: | ---: |
| 256 A | 8.530 tok/s | 396 | 8,448 | 51,904,512 |
| 8,192 | 8.864 tok/s | 24 | 16,384 | 100,663,296 |
| 256 B | 8.751 tok/s | 396 | 8,448 | 51,904,512 |

Step 8,192 was 2.6% above the step-256 median, but only 1.3% above the faster
reference. All three final-logit arrays were bit-identical. However, the
8,193rd boundary token forced every full-attention cache to allocate a second
8,192-token region, almost doubling final KV bytes, and observed swap peaked at
2,459 MiB versus 1,593 MiB in the first reference. The result therefore fails
the equal-final-bytes gate and is too small to justify a boundary-aligned
follow-up. Step 256 remains the validation setting.

Longer-term work may still measure 32k, 64k, 96k and 131k decode-latency slope,
separate cache-copy time from attention scan time, or design a bounded
geometric growth policy that avoids both 512 small reallocations and a
whole-step over-allocation at boundaries.

This surface is especially relevant to 131k and 262k operation and may be
underrepresented by short decode benchmarks.

### 7. Improve Metal occupancy without mistaking waits for compute

The short diagnostic spent 11.29 of 68.49 seconds waiting for routed IDs.
Deleting that entire interval would provide a theoretical upper bound of
approximately 20%, and some of the interval is real GPU work. A prior
event-gated demand implementation changed host blocking into a Metal event wait
but measured 13.14 versus 13.20 tok/s: no speed-up, because expert-miss I/O
remained on the critical path.

Useful Metal work therefore includes:

- command-buffer and routing-boundary amortisation;
- overlapping independent compute with expert reads;
- reducing avoidable cache-growth copies;
- profiling per-layer gaps, not only aggregate GPU utilisation;
- improving MoE kernel occupancy after required expert bytes are resident.

Simply replacing a host wait with a GPU event is not sufficient.

### 8. Reduce MTP synchronisation and verification overhead

Decode contains additional host-visible synchronisation:

- per-draft argmax and adaptive-confidence conversion;
- batch-verification evaluation and ID extraction;
- drafter-cache synchronisation.

The 64-token diagnostic spent 12.38 seconds in verification versus 0.32 seconds
in drafting. Candidate work includes retaining decisions on Metal longer,
reducing verification barriers and improving accepted depth, while preserving
exact output and cache state. These changes require separate greedy/speculative
correctness tests and cannot be inferred from prefill gains.

### 9. Use memory more aggressively only when it removes critical-path work

The existing 1 GB MLX reusable-buffer limit is intentionally conservative.
Possible stability screens include an 8 GB wired-memory limit and a larger
bounded expert pool. Success criteria are lower tail latency or fewer demand
misses without:

- critical memory pressure;
- monotonically growing swap;
- failure to recover after process exit;
- reduced SSD page-cache effectiveness;
- allocation or byte-integrity errors.

Allocating memory merely to raise GPU utilisation is not a performance result.

## Completed one-variable matrix

A fresh-process 4,096-token matrix completed with fixed two-token grouping.
All final-logit SHA-256 values were identical to baseline:

| Case | Throughput | Gain | Demand loads | Minimum free memory | Maximum swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 6.422 tok/s | — | 233,300 | 52% | 2,273 MiB |
| Workers 12 | 6.573 tok/s | +2.4% | 235,958 | 51% | 1,921 MiB |
| Real 40 | 6.860 tok/s | +6.8% | 196,910 | 46% | 1,825 MiB |
| Speculative 24 | 6.790 tok/s | +5.7% | 209,995 | 42% | 1,809 MiB |
| Prediction width 16 | 8.612 tok/s | +34.1% | 216,156 | 49% | 1,801 MiB |

The reviewed 32/16/K=3 server was restored after the matrix; health, model
enumeration and a real `VATES_OK` completion passed. No persistent performance
setting was changed.

## Completed prediction-width and residency combinations

The order-controlled fresh-process screen reproduced both reference profiles
and then combined prediction width 16 with each residency candidate:

| Case | Throughput | Gain vs width-16 median | Demand loads | Peak MLX |
| --- | ---: | ---: | ---: | ---: |
| Baseline A | 6.352 tok/s | — | 233,905 | 5.545 GB |
| Width 16 A | 8.654 tok/s | -0.1% | 216,372 | 5.545 GB |
| Width 16 + real 40 | 9.123 tok/s | +5.3% | 185,528 | 6.225 GB |
| Width 16 + speculative 24 | 9.119 tok/s | +5.2% | 195,500 | 6.225 GB |
| Width 16 B | 8.679 tok/s | +0.1% | 216,819 | 5.545 GB |
| Baseline B | 6.323 tok/s | — | 233,362 | 5.545 GB |
| Width 16 + real 40 + speculative 24 | 9.542 tok/s | +10.1% | 168,519 | 6.904 GB |

The repeated width-16 median was 8.667 tok/s, 36.8% above the repeated
baseline median of 6.337 tok/s. The joint candidate was 50.6% above the
repeated baseline and reduced demand loads by 27.9%. Every case produced the
same final-logit SHA-256, next token and boundary-decoded token. The preserved
`.npy` files also share SHA-256
`e2f88986314f3b3051cd270d25ce4685ef7be90829e384cf4cc6adb99a818d48`.

The joint candidate retained 38% minimum sampled free memory. Swap fell from
1,601 to 1,593 MiB during the process, and the internal SSD averaged
approximately 1,256 MB/s between the first and last samples. Its boundary
decode took 0.2884 seconds. The screen's automatic gate required identical
logits, a gain from both individual combinations and at least 20% free memory
before permitting the joint case.

After the screen, the reviewed 32/16/K=3 server restored successfully and
independent health, model enumeration and real `VATES_OK` inference checks
passed. Temporary harness, bytecode and PID state were removed while all result
logs were preserved. The 40/24/width-16 profile remains a short-context lead,
not an approved persistent or long-context configuration.

## Exact 131,072-token candidate qualification

The joint prediction-width and residency lead completed the required exact
long-context qualification with:

- cross-layer prediction width 16;
- 40 real expert slots and 24 speculative slots;
- MTP `K=3`, two-token prefill chunks and quantised-KV growth step 256;
- no persistent profile or implementation change.

The fail-fast stage tokens at 32,768, 65,536, 98,304 and 131,071 prompt tokens
were 88, 13, 220 and 79, matching the preserved baseline. Feeding token 79
advanced every cache from 131,071 to exactly 131,072 tokens and produced the
required following token 220.

| Metric | Reviewed 32/16 baseline | Width16/real40/spec24 | Change |
| --- | ---: | ---: | ---: |
| Prompt prefill | 5.786 tok/s | 7.2446 tok/s | +25.2% |
| One-token boundary decode | 0.2598 s | 0.2085 s | -19.8% latency |
| Peak MLX allocation | 7.326 GB | 8.685 GB | +1.359 GB |
| Peak RSS | 4.911 GB | 6.208 GB | +1.297 GB |
| Minimum sampled free memory | 32% | 23% | -9 percentage points |
| Maximum sampled swap | 3,326 MiB | 2,668 MiB | -658 MiB |
| Final KV bytes | — | 805,306,368 | — |

The candidate recorded 4,997,147 demand loads, 57,774,037 resident hits,
1,454,003 GPU fast-path acquisitions and 1,691,773 fallbacks. System-wide
`disk0` counters increased by an average of approximately 937 MB/s and 3,231
transfers/s over the 18,100-second resource trace. The final-logit SHA-256 was
`7817f7d722692fa5cd50e11447bf217d2ee038828100e863697fa98b483709c4`.

Correctness, pressure and stability therefore passed at the production
validation length. The result qualifies the joint candidate for user
consideration, but does not authorise changing the reviewed persistent
32/16/K=3 profile.

## Exact 131,072-token w16/40r/24s/k3 repeat

A second fresh-process run repeated the exact candidate workload to measure
reproducibility and locate peak system pressure by context level. The profile
is abbreviated **w16/40r/24s/k3**: prediction width 16, 40 real expert slots,
24 speculative slots and MTP `K=3`.

All correctness evidence reproduced. Stage tokens were again 88, 13, 220 and
79; feeding token 79 advanced every cache from 131,071 to exactly 131,072
tokens and produced following token 220. The final-logit SHA-256 remained
`7817f7d722692fa5cd50e11447bf217d2ee038828100e863697fa98b483709c4`.

| Metric | Qualification | Repeat | Repeat change |
| --- | ---: | ---: | ---: |
| Prompt prefill | 7.2446 tok/s | 6.9507 tok/s | -4.1% |
| Prefill time | 18,092.26 s | 18,857.28 s | +765.02 s |
| Boundary decode | 0.2085 s | 0.2061 s | -1.1% latency |
| Peak MLX allocation | 8.685 GB | 8.685 GB | effectively identical |
| Peak RSS | 6.208 GB | 5.681 GB | -0.527 GB |
| Minimum sampled free memory | 23% | 21% | -2 percentage points |
| Maximum sampled swap | 2,668 MiB | 1,872.75 MiB | -795 MiB |
| Demand loads | 4,997,147 | 4,966,756 | -0.6% |
| Resident hits | 57,774,037 | 57,805,648 | +0.1% |

Repeat stage throughput was 9.052, 7.660, 6.449 and 5.573 tok/s across the
four 32k intervals. The lower whole-run throughput occurred despite slightly
fewer demand loads, so it reflects run-to-run system and scheduling variance
rather than a correctness or residency regression.

The lowest sampled free-memory reading was 21% at 15:40:31 UTC, first reached
between the 79,872- and 81,920-token checkpoints and closest to 81,920 tokens.
At that sample RSS was 5,365,776 KiB, swap use was 1,707.06 MiB and the
compressor occupied 225,816 pages. Free memory recovered afterwards. Swap
peaked later at 1,872.75 MiB near 90k–92k tokens, then fell to 1,824.75 MiB by
the final sample. There was no unbounded pressure trend, allocation failure or
cache-integrity failure.

The repeat therefore confirms bounded memory headroom but also shows that
w16/40r/24s/k3 is not excessively conservative on a 16 GB system: the
system-wide free-memory floor is already approximately one fifth under this
synthetic sustained load. A further memory experiment remains reasonable only
as a gated one-variable screen. Because real residency previously reduced
demand loads more strongly than speculative residency, the next candidate
should be **w16/48r/24s/k3**. The prior slot increments imply approximately
0.68 GB additional MLX allocation, but that is a projection requiring direct
measurement; it does not authorise a persistent change.

After the repeat, the supervisor restored the original RAID-backed model with
internal MTP and experts at fixed 32/16/K=3. Independent `/health`,
`/v1/models` and exact `VATES_OK` checks passed. Temporary harness, bytecode
and PID state were removed while all result logs were preserved.

## Exact 262,144-token stretch

The pinned model configuration declares `max_position_embeddings=262144`,
with no RoPE scaling and no sliding window. The same transient joint candidate
therefore attempted the separate 262,144-token stretch.

The first four stage tokens again matched 88, 13, 220 and 79. The ungated
196,608-token stage produced token 198. At 262,143 prompt tokens the model
produced token 20,221; feeding it advanced every cache to exactly 262,144 and
produced following token 9,370.

| Prompt interval | Stage throughput | Cumulative throughput |
| --- | ---: | ---: |
| 0–32,768 | 9.1799 tok/s | 9.1799 tok/s |
| 32,768–65,536 | 8.1014 tok/s | 8.6070 tok/s |
| 65,536–98,304 | 6.6328 tok/s | 7.8301 tok/s |
| 98,304–131,071 | 5.6370 tok/s | 7.1361 tok/s |
| 131,071–196,608 | 4.8801 tok/s | 6.1833 tok/s |
| 196,608–262,143 | 3.6878 tok/s | 5.2886 tok/s |

The full prefill took 49,567.44 seconds, or 13 hours 46 minutes 7 seconds.
The one-token boundary decode took 0.4609 seconds. Peak MLX allocation was
9.567 GB, peak RSS was 6.289 GB, minimum sampled free memory was 18%, and
maximum sampled swap use was 3,084 MiB. Final quantised KV capacity was 262,398
tokens and 1,612,173,312 bytes. System-wide `disk0` counters averaged
approximately 681 MB/s and 2,413 transfers/s across the 49,574-second resource
trace.

The run recorded 9,703,647 demand loads, 115,833,572 resident hits, 3,045,667
GPU fast-path acquisitions and 3,245,885 fallbacks. The final-logit SHA-256
was `6db97f1105eb2e9005bc9c9fc9e80ab2fc9172c0fcf09101a734820c1bcc14ea`.

The 131,071-to-196,608 interval did not emit the intended intermediate
2,048-token checkpoints. Because the stage began at an odd absolute position
and advanced in two-token chunks, intermediate positions remained odd and
could not satisfy `absolute % 2048 == 0`. Resource logging continued throughout
and the exact 196,608-token stage event passed. Future harnesses should compare
distance from the stage start rather than absolute modulo so observability is
not lost across an odd stage boundary.

This stretch proves exact 262,144-token model/runtime support, cache integrity,
bounded pressure and operational recovery. It does not establish a 262k
performance gain because no equivalent 262k reviewed-profile baseline exists.
Comparing its 5.2886 tok/s directly with the 5.786 tok/s 131k baseline would
confound profile and context length.

After the stretch, the supervisor restored the reviewed server with the
original RAID-backed model, internal MTP, internal experts and fixed
32/16/K=3. Independent checks passed `/health`, `/v1/models` and an exact
`VATES_OK` inference. The transient harness, PID and bytecode state was removed
while the JSONL, resource, supervisor and final-logit files were preserved.

## Evidence-backed choices

1. **Keep the reviewed 32/16/K=3 profile.** This retains the lower 7.326 GB
   peak MLX baseline and the existing operational margin. It forgoes the
   measured 25.2% 131k prefill gain and 19.8% boundary-decode latency reduction.
2. **Approve width16/real40/spec24 for the persistent profile.** This is the
   strongest evidence-backed speed choice at 131k: correct, stable and
   repeatable through the exact boundary, with demonstrated 262k support. It
   costs approximately 1.36 GB more peak MLX allocation and reduced minimum
   sampled free memory to 23% at 131k and 18% at 262k.
3. **Continue optimisation before changing the profile.** The next high-value
   work is either a gated w16/48r/24s/k3 residency screen or long-context
   attention/KV scaling, not increasing GPU utilisation for its own sake. The
   residency screen should proceed to an exact 131k run only if it reduces
   critical-path loads and improves throughput without crossing the pressure
   gate. A matched 262k 32/16 baseline would quantify the candidate's
   stretch-length gain, but costs roughly another half-day run. Partial width
   or residency profiles remain short-screen evidence only and should not be
   represented as long-context-qualified alternatives.

The persistent profile remains unchanged pending explicit user approval.

## Acceptance sequence

1. Reject unsafe, incorrect or unstable candidates in short screens.
2. Repeat promising cases in fresh A/B order to exclude cache/order effects.
3. Combine independently positive candidates once and remeasure because
   scheduling and residency effects are not additive.
4. Validate the final candidate at exactly 131,072 tokens, including a real
   boundary decode, logit/output checks, memory, swap and disk evidence.
5. Restore the reviewed persistent server and request approval before making
   any profile or implementation change permanent.
6. Attempt 262,144 tokens only after the 131k candidate is correct and stable.
