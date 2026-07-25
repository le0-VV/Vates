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
- [x] Compare memory-residency, expert-slot/read-scheduling and prefill/decode-path options one variable at a time, including peak memory, correctness and stability.
  - [x] Complete the fresh 4,096-token one-variable matrix for workers, real slots, speculative slots and prediction width.
  - [x] Repeat the fresh baseline and width-16 lead, then screen width 16 with each positive residency candidate.
  - [x] Measure KV-cache growth separately at a context long enough to expose allocation-copy cost.
- [x] Validate surviving speed improvements at exactly 131,072 tokens; attempt 262,144 tokens as a separate stretch target after confirming model/runtime support.
- [x] Rank candidates by reproducible long-context prefill/decode improvement with no artificial throughput floor; correctness, bounded pressure and stability remain hard gates.
- [x] Present evidence-backed options and trade-offs for user approval before changing the persistent server profile or implementation.
- [x] Restore and verify the reviewed persistent server after experiments, retaining only explicitly approved speed changes.

# Repeat exact 131k stress at w16/40r/24s/k3

- [x] Reconstruct and review the exact 131,071-prompt-token plus one-token-decode harness from the preserved validation evidence.
- [x] Record the current reviewed server identity and health, then stop it only after the detached supervisor is ready.
- [x] Launch a transient w16/40r/24s/k3 run with fail-fast token gates, continuous resource sampling and automatic restoration of the reviewed 32/16/K=3 server.
- [x] Track every 2,048-token checkpoint, stage throughput, boundary decode, correctness, KV growth, peak MLX/RSS, macOS memory-pressure level, free memory and swap.
- [x] On completion or limitation, preserve result logs, remove only temporary harness/monitor state and independently verify health, model enumeration and exact `VATES_OK`.
- [x] Compare the repeat with the qualified w16/40r/24s/k3 run, document memory headroom and present evidence-backed options without changing the persistent profile.

# Promote w16/40r/24s/k3 to the persistent server profile

- [x] Locate every persistent profile default and the existing launcher regression coverage.
- [x] Add a regression test for w16/40r/24s/k3 defaults and verify that it fails for the old profile.
- [x] Change only the approved prediction-width and expert-residency defaults.
- [x] Run the focused regression test and the complete portable CI suite.
- [x] Restart the server and verify its live command, health, model enumeration and exact `VATES_OK` response.
- [x] Record the approved persistent profile in the report and Brick, validate, commit, push and confirm PR CI.

# Run Qwen3.5-35B-A3B through a general-purpose Vates MoE runtime

- [x] Research and agree the Qwen3.5 capability, context, storage, process and optimisation requirements.
- [x] Write, self-review, commit and obtain approval for the migration design.
- [x] Write and review a test-driven implementation plan with explicit component and qualification boundaries.
- [x] Stop every current Vates model process and reserve the Mac mini for Qwen3.5 experiments.
- [ ] Acquire and verify the pinned canonical Qwen3.5 MLX 4-bit source without deleting existing model files.
- [ ] Establish a small-context reference for text, thinking, non-thinking, image and tool-call behaviour.
- [x] Generalise Vates' model assembly and expert-streaming boundaries, then add the Qwen3.5 MoE adapter.
- [x] Add the non-speculative general generation engine with exact pending-token cache semantics.
  - [x] Preserve and run the focused RED engine tests.
  - [x] Implement chunked prefill, greedy decode, cancellation and per-request state isolation.
  - [x] Route the CLI and TUI backend through an engine-neutral boundary.
  - [x] Run focused engine/backend/portable verification and create a signed focused commit.
- [ ] Add standard OpenAI image attachments, reasoning fields and protocol-only tool calling.
  - [x] Add strict streaming reasoning separation and malformed-output gates.
  - [x] Validate OpenAI function schemas and parse Qwen tool XML without executing tools.
  - [x] Normalise bounded data-URL and public-HTTPS image attachments.
  - [ ] Integrate reasoning, tools and images into OpenAI request/response handling.
- [ ] Reach a correct, stable exact 131,072-token boundary with Qwen3.5 before any performance tuning.
- [ ] Run and publish the standardised text, reasoning, code, tool, vision and long-context intelligence suite.
- [ ] Optimise configuration and runtime performance without weakening correctness or pressure gates.
- [ ] Validate the complete repository, record Brick memory, commit signed changes, push the task branch and verify PR CI.
