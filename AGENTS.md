# AGENTS.md

MemDiff is a provenance- and belief-revision-aware agent memory engine evaluated across two matched-budget benchmark tracks, paired with a static React "Memory Time Machine" replay viewer.

## Architecture & Invariants

- **Matched-budget ablation ladder**: All 4 policies (`naive_rag`, `recency_rag`, `structured_extract`, `belief_revision`) implement `MemoryPolicy` (`src/memdiff/core/protocol.py`) under the same `top_k` retrieval budget. Keep `BeliefRevisionPolicy` and `StructuredExtractPolicy` decoupled from LLM backends via injected `embed_fn`, `extractor_fn`, and `reconciler_fn` callbacks (`src/memdiff/policies/llm_ops.py`) so `tests/` stay offline and deterministic.
- **Immutable belief ledger & time-travel**: Never hard-delete beliefs on revision. Mark old rows `superseded`, `refuted`, or `forgotten`, record directed `BeliefEdge` links, and append `TimelineEvent` entries so `BeliefStore.get_state_at_step()` (`src/memdiff/core/store.py`) and `computeStatusAtStep()` (`web/src/App.tsx`) reconstruct identical snapshots.
- **Vector index cache invalidation**: `BeliefStore` caches normalized `(N, D)` float32 matrices (`_chunk_index`, `_belief_index`) and fetches full rows in a second `WHERE id IN (...)` query. Any write to `chunks` or `beliefs` must clear the matching index cache before committing.
- **Schema parity**: Keep `ReplayBundle` and its nested types in `src/memdiff/core/models.py` 1:1 synchronized with `web/src/types/replay.ts`.

## Two-Track Benchmark Scope

- **Track 1 — LongMemEval-V2 (`src/memdiff/benchmarks/lmev2/`)**: Real ServiceNow (`enterprise`: 100 trajectories, 211 questions) and WebArena (`web`) logs. Here `dynamic-environment` tests interactive UI transitions (`pre-state + action -> post-state`), *not* cross-session temporal overrides. MemDiff targets failure→success `gotcha` revision, surface-scoped `closed_set_items` consolidation (for `*-abs` flawed-premise questions), and ephemeral task-noise filtering.
  - Keep `metrics.py` rubrics and `DOMAIN_SYSTEM_PROMPTS` in `runner.py` faithful to upstream `xiaowu0162/LongMemEval-V2`. On `*-abs` questions, bare `\boxed{UNKNOWN}` scores `0`; the response must identify why the premise is flawed.
  - Fetch data via `uv run python scripts/fetch_lmev2_small.py --domain enterprise --include-question-screenshots` (streams `trajectories.jsonl` down to ~94 MB + 2.6 MB question screenshots and skips the 5.6 GB trajectory screenshot archive).
- **Track 2 — DriftSim (`src/memdiff/benchmarks/drift_sim/`)**: Deterministic state-machine stress test owning 100% of ground truth for temporal fact contradictions (`Stale Belief Rate`) and early-session `Durable Rule Preservation`. `DriftSession.to_trajectory_dict()` adapts sessions to the exact trajectory dict shape used by Track 1.

## Local Ollama & `qwen3.5:9b` Runtime Conventions

- Run `uv run python scripts/check_ollama.py` before live evaluations. Configure endpoints and models in `.env` (see `.env.example`).
- Launch Ollama with `OLLAMA_CONTEXT_LENGTH=32768` on 12 GB VRAM hosts (`qwen3.5:9b` + `qwen3-embedding:0.6b` fit co-resident on GPU at 32K; default 4K truncates RAG/thinking prompts, while 128K spills to CPU).
- `qwen3.5:9b` thinks by default: keep `max_completion_tokens=16384` (and `llm_timeout_seconds=600`) across Reader, Extractor, Reconciler, and Judge so thinking never exhausts the token cap before emitting `content`.
- To disable thinking over Ollama's `/v1/chat/completions`, pass `think=False` to `CachedLLMClient.chat()` (`src/memdiff/llm/openai_client.py`), which sends `extra_body={"reasoning_effort": "none"}` (Ollama's `/v1` endpoint ignores `"think": false`).
- `CachedLLMClient` persists chat and embedding responses in `.cache/llm/llm_cache.db` and skips caching empty `""` chat outputs. When resuming interrupted `DriftSim` runs, pass `--resume` to `scripts/run_drift_sim.py` to reuse existing SQLite stores—re-running `belief_revision` from scratch generates new random `b_*` IDs that miss reconciler cache entries.

## Context Pointers

- `manual/` is a scratch board of human-facing explanations (code map, DB schema, run guide). It may be outdated, so don't treat it as a source of truth: verify against the code. Don't update it unless the user asks.
- Modifying belief lifecycle, SQLite storage, or time-travel replay → read `src/memdiff/core/models.py`, `src/memdiff/core/store.py`, and `tests/test_store.py`.
- Tuning extraction/reconciliation prompts or adding a policy → read `src/memdiff/policies/belief_revision.py` and `src/memdiff/policies/llm_ops.py`.
- Changing LongMemEval-V2 scoring, prompting, or dataset loading → read `src/memdiff/benchmarks/lmev2/metrics.py` and `src/memdiff/benchmarks/lmev2/runner.py`.
- Updating the Memory Time Machine viewer → read `src/memdiff/export/replay.py`, `web/src/types/replay.ts`, and `web/src/App.tsx`.

## Public Repo Hygiene

This repository is public. Keep `.env`, `/data/`, `/runs/`, `.cache/`, `*.db`, `*.sqlite`, and personal hostnames or IPs out of git. Use `<gpu-host>` and `127.0.0.1` placeholders in tracked docs, configs, and tests.

## Verification Gates

Run all relevant checks before finishing a task:

- **Python (`src/`, `tests/`, `scripts/`)**: `uv run pytest -v && uv run ruff check src tests scripts && uv run lint-imports` (all unit tests must stay fast, deterministic, and offline; keep live LLM execution inside `scripts/`).
- **Web UI (`web/`)**: `pnpm --dir web build` (`tsc -b && vite build` must succeed with zero errors).
