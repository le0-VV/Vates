# Final OpenAI-compatible server review fixes

- [x] Read the approved design, implementation plan and complete current server/CLI/test/CI APIs.
- [x] Add strict server warm-up regression coverage, record RED, implement the minimal server-only fix and record GREEN.
- [x] Add request semantic-field validation coverage, record RED, implement minimal rejection without changing harmless metadata/no-auth behaviour and record GREEN.
- [x] Add bounded request-body and timeout coverage, record RED, implement the minimal HTTP framing protections and record GREEN.
- [x] Add portable CLI serve coverage and update hosted CI without importing MLX; record RED/GREEN evidence.
- [x] Add inference error-type, token-limit restoration and usage-accuracy coverage, record RED, implement minimal fixes and record GREEN.
- [x] Run focused portable/server/launcher tests, compileall and the full suite once.
- [x] Self-review the complete diff against every final-review finding and deployment invariant.
- [x] Write the evidence report and create one verified signed commit.
