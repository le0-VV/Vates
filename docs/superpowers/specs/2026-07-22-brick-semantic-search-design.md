# Brick Semantic Search Design

## Goal

Set up Vates with the repository-local Brick memory tooling and enable semantic retrieval through the existing OpenAI-compatible embedding service on the host at `http://127.0.0.1:8745/v1`.

## Confirmed environment

- Vates is a Git repository with no existing root `AGENTS.md` and no existing Brick installation.
- The embedding service is reachable from outside the Codex sandbox.
- `GET http://127.0.0.1:8745/v1/models` returns the model identifier `mlx-community/embeddinggemma-300m-4bit`.
- The service is not reachable from inside the Codex sandbox, so commands that require embeddings must be executed with sandbox escalation.

## Approach

Use Brick's official installer from `https://github.com/le0-VV/brick`. This preserves Brick's supported repository layout, generated agent guidance, local environment, and update mechanism.

Do not seed canonical Brick memory during setup. The initial memory store remains empty.

## Repository and local state

The installer may add tracked Brick tooling, memory directories, Git attributes, ignore rules, and a root `AGENTS.md`. The coherent tracked installer output will be retained.

Semantic configuration will live only in the gitignored device-local file `.agents/brick/config.local.json`:

```json
{
  "embedding": {
    "url": "http://127.0.0.1:8745/v1",
    "model": "mlx-community/embeddinggemma-300m-4bit"
  }
}
```

The generated virtual environment, search index, conflict reports, update state, local configuration, and `.agents/TODO.md` must remain untracked.

## Agent instructions

Extend the installed root `AGENTS.md` with a project-specific instruction that:

- identifies `http://127.0.0.1:8745/v1` as the Brick embedding base URL;
- identifies `mlx-community/embeddinggemma-300m-4bit` as the configured model;
- states that the endpoint is host-local and unavailable inside the Codex sandbox;
- directs agents to request sandbox escalation for embedding-dependent commands such as `./brick rebuild` and `./brick memory search`, rather than diagnosing an in-sandbox connection failure as a server outage.

## Verification

Run Brick's supported checks from the repository root:

1. `./brick setup --json --pretty`
2. `./brick memory validate --pretty`
3. `./brick rebuild` outside the sandbox
4. `./brick memory search "project context" --pretty` outside the sandbox

Setup and validation must succeed. Search output must report hybrid retrieval with semantic search available and the configured model. Git status must show that `.agents/brick/config.local.json`, the Brick virtual environment, generated index, update state, conflicts, and `.agents/TODO.md` are not tracked.

Because canonical memory is intentionally empty, a semantic search may return no results; retrieval-mode metadata, rather than a non-empty result set, proves semantic retrieval is enabled.

## Error handling

- If installation or dependency setup fails because the network is unavailable in the sandbox, retry the official operation outside the sandbox.
- If the embedding endpoint cannot be reached outside the sandbox, stop without claiming semantic search is enabled.
- If the embedding response is malformed or reports a different model, keep the local configuration uncommitted and report the mismatch.
- If the installer creates `AGENTS.md.brick-backup`, review and merge applicable project-specific instructions before proceeding.

