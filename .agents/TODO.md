# Stress-test the current 131k context configuration

- [x] Audit live storage and correct the launcher so only canonical original model sources remain RAID-backed in steady state.
- [x] Publish and verify the prepared MTP file on the internal SSD without changing 32/16/K=3.
- [x] Supersede stale Brick storage guidance, obtain independent review, push the branch and confirm PR CI.
- [x] Prepare an exact progressive 131,071-prompt-token plus one-token-decode harness and internal resource logging.
- [x] Complete checkpoints at 32,768, 65,536, 98,304 and 131,071 prompt tokens.
- [x] Verify one decoded token advances the cache to the 131,072-token boundary without allocation, capacity or byte-integrity failure.
- [x] Restart the OpenAI server with internal MTP, expert and log paths; verify health and a short real inference request.
- [x] Remove temporary harness/monitor state and report elapsed time, throughput, peak resources and any limiting failure.
- [ ] After explicit deletion authorisation, remove redundant derived RAID preparation artefacts and confirm only original source files remain.

# Investigate Qwen3-Next inference speed after the 131k baseline

- [x] Preserve the completed 32/16/K=3 stress result and establish reproducible short-context prefill and decode baselines for rapid iteration only.
- [x] Profile Metal utilisation, CPU synchronisation, expert-cache hit/miss behaviour, internal-SSD reads and swap writes to locate the dominant stalls.
- [ ] Compare memory-residency, expert-slot/read-scheduling and prefill/decode-path options one variable at a time, including peak memory, correctness and stability.
  - [x] Complete the fresh 4,096-token one-variable matrix for workers, real slots, speculative slots and prediction width.
  - [ ] Repeat the fresh baseline and width-16 lead, then screen width 16 with each positive residency candidate.
  - [ ] Measure KV-cache growth separately at a context long enough to expose allocation-copy cost.
- [ ] Validate surviving speed improvements at exactly 131,072 tokens; attempt 262,144 tokens as a separate stretch target after confirming model/runtime support.
- [ ] Rank candidates by reproducible long-context prefill/decode improvement with no artificial throughput floor; correctness, bounded pressure and stability remain hard gates.
- [ ] Present evidence-backed options and trade-offs for user approval before changing the persistent server profile or implementation.
- [ ] Restore and verify the reviewed persistent server after experiments, retaining only explicitly approved speed changes.
