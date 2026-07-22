# Brick Semantic Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install repository-local Brick tooling in Vates and configure its semantic retrieval to use the host embedding service at `http://127.0.0.1:8745/v1` while leaving canonical memory empty.

**Architecture:** Vendor Brick through its official installer, retain its generated repository wrapper and agent instructions, and keep device-specific embedding configuration and indexes gitignored. Prove semantic retrieval with one disposable Brick example memory, then remove it and rebuild the empty final index.

**Tech Stack:** POSIX shell, Git, Python 3.11+, uv, Brick, OpenAI-compatible embeddings

## Global Constraints

- Use Brick's official installer from `https://github.com/le0-VV/brick`.
- Keep the final canonical memory store empty.
- Configure embedding URL `http://127.0.0.1:8745/v1` and model `mlx-community/embeddinggemma-300m-4bit` only in `.agents/brick/config.local.json`.
- Execute embedding-dependent commands outside the Codex sandbox because the host-local endpoint is unavailable inside it.
- Do not track `.agents/TODO.md`, `.agents/brick/.venv/`, `.agents/brick/index/`, `.agents/brick/conflicts/`, `.agents/brick/config.local.json`, or `.agents/brick/update-state.json`.
- Before implementation, use `superpowers:using-git-worktrees` to create or switch to the `agent/setup-brick-semantic-search` task branch.
- Do not seed durable project memory; the packaged decision used for verification must be removed before completion.

---

### Task 1: Install the repository-local Brick package

**Files:**

- Create: `AGENTS.md`
- Create: `.gitattributes`
- Modify: `.gitignore`
- Create symlink: `brick` -> `.agents/brick/bin/brick`
- Create: `.agents/brick/AGENT_USAGE.md`
- Create: `.agents/brick/bin/brick`
- Create: `.agents/brick/config.example.json`
- Create: `.agents/brick/package-files.json`
- Create: `.agents/brick/pyproject.toml`
- Create: `.agents/brick/setup.py`
- Create: `.agents/brick/source.json`
- Create: `.agents/brick/templates/AGENTS.md`
- Create: `.agents/brick/src/brick/__init__.py`
- Create: `.agents/brick/src/brick/cli.py`
- Create: `.agents/brick/src/brick/conflicts.py`
- Create: `.agents/brick/src/brick/index.py`
- Create: `.agents/brick/src/brick/memory.py`
- Create: `.agents/brick/examples/llm-ingest/instructions.md`
- Create: `.agents/brick/examples/llm-ingest/memory-ingest.schema.json`
- Create: `.agents/brick/examples/memory-add/blocked-unsafe.json`
- Create: `.agents/brick/examples/memory-add/command.json`
- Create: `.agents/brick/examples/memory-add/decision.json`
- Create: `.agents/brick/examples/memory-add/routine.json`
- Create: `.agents/brick/examples/memory-add/skill.json`
- Create: `.agents/brick/examples/memory-files/command.md`
- Create: `.agents/brick/examples/memory-files/decision.md`
- Create: `.agents/brick/examples/memory-files/routine.md`
- Create: `.agents/brick/examples/memory-files/skill.md`
- Create, ignored: `.agents/TODO.md`
- Create, ignored: `.agents/brick/.venv/`
- Create, ignored: `.agents/brick/config.local.json`
- Create, ignored: `.agents/brick/update-state.json`
- Modify, local only: `.git/config`

**Interfaces:**

- Consumes: a clean Vates Git worktree, Git, Python 3.11+, network access to GitHub, and optional `uv`.
- Produces: executable `./brick`, the Brick Python package under `.agents/brick/`, root agent instructions, ignore rules, and the local `brick-memory` merge driver.

- [ ] **Step 1: Record the failing pre-installation check**

Run:

```bash
test -x ./brick
```

Expected: exit status `1` because Brick is not installed yet.

- [ ] **Step 2: Run the official installer outside the sandbox**

From the repository root, execute with sandbox escalation:

```bash
curl -fsSL https://github.com/le0-VV/brick/raw/refs/heads/main/install.sh \
  | env BRICK_SOURCE_BASE_URL=https://github.com/le0-VV/brick/raw/refs/heads/main \
      sh -s -- --json --pretty
```

Expected: JSON with `"status": "ok"`; actions include creating the repository-root symlink, extending `.gitignore` and `.gitattributes`, creating `AGENTS.md`, configuring the merge driver, and creating the Brick virtual environment.

- [ ] **Step 3: Read the installed instructions and create the ignored work checklist**

Read `AGENTS.md` and `.agents/brick/AGENT_USAGE.md` fully. Then create `.agents/TODO.md` with:

```markdown
# Brick setup

- [ ] Verify the official Brick installation.
- [ ] Configure the host-local embedding service.
- [ ] Add the sandbox-access instructions.
- [ ] Prove hybrid semantic retrieval with a disposable memory.
- [ ] Remove the disposable memory and verify the final empty state.
```

Expected: `git check-ignore .agents/TODO.md` prints `.agents/TODO.md`.

- [ ] **Step 4: Verify the installation is idempotent and valid**

Run:

```bash
./brick setup --json --pretty
./brick memory validate --pretty
test "$(readlink brick)" = ".agents/brick/bin/brick"
git config --local --get merge.brick-memory.driver
git check-ignore .agents/TODO.md .agents/brick/.venv .agents/brick/config.local.json .agents/brick/update-state.json
test ! -e AGENTS.md.brick-backup
```

Expected:

- setup returns `"status": "ok"`;
- validation returns `"status": "ok"` and `"checked": 0`;
- the symlink target is `.agents/brick/bin/brick`;
- the merge driver is `./brick merge-driver %O %A %B %L %P`;
- all four local paths are ignored;
- no agent-instruction backup exists because Vates had no pre-existing root `AGENTS.md`.

- [ ] **Step 5: Commit the coherent installer output**

Run:

```bash
git add AGENTS.md .gitattributes .gitignore brick .agents/brick
git diff --cached --check
git status --short
git commit -m "build(brick): install repository memory tooling"
```

Expected: the commit contains the tracked installer package, wrapper, root instructions, and Git metadata; no ignored local state or canonical memory is staged.

---

### Task 2: Configure and prove semantic retrieval

**Files:**

- Modify: `AGENTS.md`
- Modify: `.agents/brick/templates/AGENTS.md`
- Modify, ignored: `.agents/brick/config.local.json`
- Temporarily create, then delete: `.agents/memory/decision/01JX3Y1Y8H6TR4Y3Q38K1W9P2A-use-brick-memory-search-before-assuming-project-context.md`
- Modify, ignored: `.agents/TODO.md`
- Modify, ignored: `.agents/brick/index/brick.sqlite3`

**Interfaces:**

- Consumes: Task 1's `./brick` wrapper, packaged example decision, root and template agent instructions, and the host-local OpenAI-compatible service.
- Produces: device-local embedding configuration, durable sandbox-access instructions, a verified hybrid semantic round-trip, and a final empty canonical memory store.

- [ ] **Step 1: Record the failing pre-configuration search check**

Run:

```bash
./brick rebuild --json --pretty
./brick memory search "assuming project context" --pretty
```

Expected: rebuild succeeds with `"memory_count": 0`; search reports `"mode": "keyword"` and semantic reason `"embedding_url_not_configured"`.

- [ ] **Step 2: Replace the generated device-local embedding configuration**

Replace `.agents/brick/config.local.json` with exactly:

```json
{
  "embedding": {
    "url": "http://127.0.0.1:8745/v1",
    "model": "mlx-community/embeddinggemma-300m-4bit"
  }
}
```

Run:

```bash
git check-ignore .agents/brick/config.local.json
```

Expected: Git prints `.agents/brick/config.local.json`.

- [ ] **Step 3: Add durable host-access guidance to both agent-instruction copies**

Read `AGENTS.md` and `.agents/brick/templates/AGENTS.md` fully before editing. Append this section verbatim to both files so a same-version `./brick setup` remains idempotent:

```markdown

## Vates Brick embedding service

- Brick's device-local embedding base URL is `http://127.0.0.1:8745/v1`; the configured model is `mlx-community/embeddinggemma-300m-4bit`.
- The endpoint is host-local and cannot be reached from inside the Codex sandbox. Run embedding-dependent commands such as `./brick rebuild` and `./brick memory search "query" --pretty` with sandbox escalation.
- Treat an in-sandbox connection failure as an execution-context limitation. Confirm service availability outside the sandbox before diagnosing the embedding server as unavailable.
```

Run:

```bash
./brick setup --json --pretty
cmp AGENTS.md .agents/brick/templates/AGENTS.md
```

Expected: setup returns `"status": "ok"` without replacing the appended section, and `cmp` exits `0`.

- [ ] **Step 4: Confirm the endpoint and model outside the sandbox**

Execute with sandbox escalation:

```bash
curl -sS --max-time 10 http://127.0.0.1:8745/v1/models
```

Expected: HTTP success JSON whose `data` array contains model ID `mlx-community/embeddinggemma-300m-4bit`.

- [ ] **Step 5: Add and validate the disposable verification memory**

Run:

```bash
./brick memory add --pretty < .agents/brick/examples/memory-add/decision.json
./brick memory validate .agents/memory/decision/01JX3Y1Y8H6TR4Y3Q38K1W9P2A-use-brick-memory-search-before-assuming-project-context.md --pretty
```

Expected: memory add and validation both return `"status": "ok"`, and the added memory ID is `01JX3Y1Y8H6TR4Y3Q38K1W9P2A`.

- [ ] **Step 6: Prove hybrid semantic retrieval outside the sandbox**

Execute both commands with sandbox escalation:

```bash
./brick rebuild --json --pretty
./brick memory search "assuming project context" --pretty
```

Expected:

- rebuild reports `"memory_count": 1`, `"embedding_count": 1`, `"embedding_model": "mlx-community/embeddinggemma-300m-4bit"`, and a positive embedding dimension;
- search reports `"mode": "hybrid"`, semantic `"available": true`, the configured model, and a result with ID `01JX3Y1Y8H6TR4Y3Q38K1W9P2A`.

- [ ] **Step 7: Remove the disposable memory and rebuild the final empty index**

Delete only this setup-created file:

```diff
*** Begin Patch
*** Delete File: .agents/memory/decision/01JX3Y1Y8H6TR4Y3Q38K1W9P2A-use-brick-memory-search-before-assuming-project-context.md
*** End Patch
```

Then execute the rebuild with sandbox escalation and validate locally:

```bash
./brick rebuild --json --pretty
./brick memory validate --pretty
./brick memory search "assuming project context" --pretty
```

Expected:

- rebuild reports `"memory_count": 0` and `"embedding_count": 0`;
- validation reports `"status": "ok"` and `"checked": 0`;
- final search reports keyword mode with semantic reason `"index_has_no_embeddings"`, which is Brick's expected empty-index state.

- [ ] **Step 8: Verify tracking boundaries and commit durable guidance**

Tick every item in `.agents/TODO.md`, then run:

```bash
test -z "$(find .agents/memory -type f -name '*.md' -print -quit)"
git check-ignore .agents/TODO.md .agents/brick/.venv .agents/brick/index .agents/brick/config.local.json .agents/brick/update-state.json
git ls-files --error-unmatch .agents/brick/config.local.json
```

Expected: the empty-memory assertion succeeds; all local paths are ignored; `git ls-files --error-unmatch` exits `1`, proving local configuration is not tracked.

Run:

```bash
git add AGENTS.md .agents/brick/templates/AGENTS.md
git diff --cached --check
git diff --cached --name-only
git commit -m "docs(agents): document Brick embedding access"
git status --short
```

Expected: only the two instruction files are committed, and the final non-ignored worktree is clean.
