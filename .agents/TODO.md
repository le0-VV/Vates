# Stress-test the current 131k context configuration

- [x] Audit live storage and correct the launcher so only canonical original model sources remain RAID-backed in steady state.
- [x] Publish and verify the prepared MTP file on the internal SSD without changing 32/16/K=3.
- [x] Supersede stale Brick storage guidance, obtain independent review, push the branch and confirm PR CI.
- [x] Prepare an exact progressive 131,071-prompt-token plus one-token-decode harness and internal resource logging.
- [ ] Complete checkpoints at 32,768, 65,536, 98,304 and 131,071 prompt tokens.
- [ ] Verify one decoded token advances the cache to the 131,072-token boundary without allocation, capacity or byte-integrity failure.
- [ ] Restart the OpenAI server with internal MTP, expert and log paths; verify health and a short real inference request.
- [ ] Remove temporary harness/monitor state and report elapsed time, throughput, peak resources and any limiting failure.
- [ ] After explicit deletion authorisation, remove redundant derived RAID preparation artefacts and confirm only original source files remain.
