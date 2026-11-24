<img src="assets/icon.svg" width="64" height="64" alt="MemDiff">

# MemDiff

**Provenance- & Revision-Aware Agent Memory Engine + Memory Time Machine Debugger**

MemDiff treats agent memory as a versioned **Belief Ledger** rather than an append-only vector store. Every incoming trajectory triggers explicit belief lifecycle operations (`ADD`, `SUPERSEDE`, `CONSOLIDATE`, `REFUTE`, `IGNORE`, `FORGET`) backed by step-level provenance, enabling clean empirical ablations and an interactive timeline diff viewer.

## Evaluation Tracks

1. **Track 1 — LongMemEval-V2 (`LME-V2-Small`)**: Real-world ServiceNow (`enterprise`, 100 trajectories / 211 questions) and WebArena (`web`, 100 trajectories / 240 questions) agent trajectories across all 5 memory abilities (*Static State Recall*, *Dynamic State Tracking*, *Workflow Knowledge*, *Environment Gotchas*, *Premise Awareness*). Uses the official `LongMemEval-V2` deterministic + LLM-as-a-Judge scoring rubrics.
2. **Track 2 — `MemDiff-Drift` Simulator**: Deterministic procedural state-machine stress test with exact ground-truth tracking of **Stale Belief Rate** and **Durable Instruction Preservation** as contradiction rate, session horizon, and memory budgets vary.

## Policy Comparison (Matched Budgets)

| Policy | Description |
|---|---|
| `naive_rag` | Append-only chunk vector search |
| `recency_rag` | Time-decay weighted retrieval over raw chunks |
| `structured_extract` | Typed belief extraction without supersession or forgetting |
| `belief_revision` | **MemDiff**: provenance-backed beliefs with explicit `SUPERSEDE`, `CONSOLIDATE`, `REFUTE`, and `FORGET` |

## Repository Structure

```text
memdiff/
├── src/memdiff/
│   ├── core/            # Belief schemas, SQLite+vector BeliefStore, MemoryPolicy protocol
│   ├── policies/        # 4 matched-budget memory policies
│   ├── llm/             # OpenAI client with disk caching & token/cost accounting
│   ├── benchmarks/
│   │   ├── lmev2/       # LongMemEval-V2 loader, official metrics port, streaming fetcher
│   │   └── drift_sim/   # Deterministic temporal contradiction simulator
│   └── export/          # Static JSON replay bundle exporter for the web UI
├── scripts/             # CLI utilities (data fetch, eval runners)
├── tests/               # Deterministic pytest suite
└── web/                 # React + TypeScript + Vite "Memory Time Machine" UI
```

## Quickstart

### 1. Python Engine & Local Ollama (`uv`)
```bash
uv sync
cp .env.example .env
uv run python scripts/check_ollama.py  # verifies Ollama connectivity & pulled models
uv run pytest
```

**Local / Remote GPU Ollama setup (12GB VRAM-friendly):**
```bash
# 1) On your Ollama host, start `ollama serve` and pull the Qwen3.5 pair:
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b

# 2) If running Ollama on a remote GPU machine, forward port 11434 via SSH:
ssh -fN -L 11434:127.0.0.1:11434 <gpu-host>
```

### 2. Fetch `LongMemEval-V2-Small` (Enterprise MVP, skipping 5.6 GB trajectory screenshots)
```bash
uv run python scripts/fetch_lmev2_small.py --domain enterprise --include-question-screenshots
```

### 3. Memory Time Machine UI (`pnpm`)
```bash
cd web
pnpm install
pnpm dev
```
