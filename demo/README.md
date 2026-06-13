# NILS-JENS Demo — Next.js Frontend

Interactive UI for running inference with all NILS-JENS eval-grid systems (vector RAG, Graph RAG, RankRAG, SFT ablations).

## Architecture

```
Browser → Next.js (port 3000) → proxy → FastAPI (port 8080) → models + FAISS
```

The Next app rewrites `/api/*` and `/health` to the Python backend (`GROUNDED_API_URL`).

## Quick start

### One command (backend + frontend)

```bash
cd demo
pnpm install
pnpm dev
```

`pnpm dev` runs `scripts/run-dev.sh`, which:

1. Reuses a healthy API on `:8080`, or clears a stale listener and starts a new one
2. **Waits until `/health` returns JSON** before starting Next.js
3. Keeps the web UI down if the API fails to bind

- API: [http://localhost:8080](http://localhost:8080)
- UI: [http://localhost:3000](http://localhost:3000)

**Real inference by default** (`GROUNDED_DEMO_MOCK=0`): Llama-3.1-8B-Instruct (4-bit), RankRAG LoRA, SFT adapters, real BGE+FAISS+graph retrieval.

**Multi-GPU auto-layout** (excludes verifier GPUs 0–1):

| Free LLM GPUs (≥6 GB) | Placement |
|----------------------|-----------|
| **2+** | **Parallel** — RankRAG on one card, generator on another (both stay loaded) |
| **3+** (optional) | BGE embedder on a third card; otherwise embedder uses CPU |
| **1** | **Sequential** — swap RankRAG → generator on the same card |

`CUDA_VISIBLE_DEVICES` unset → demo discovers all non-verifier cards automatically. Pin manually if needed: `CUDA_VISIBLE_DEVICES=2,3 pnpm dev`.

The API binds to `:8080` within seconds; model preload continues in the background (~1–3 min). `/health` reports `status: loading` until `stack_ready: true`, then `gpu_mode` (`parallel` / `sequential`) and `gpu_layout`.

If you see **CUDA OOM**, check `nvidia-smi` and free a card outside GPUs 0–1.

**Mock mode** (template text, no 8B — dev/CI only):

```bash
pnpm dev:mock
# or: GROUNDED_DEMO_MOCK=1 pnpm dev
```

The UI shows amber **mock generation** badges when mock is active.

### Separate terminals (optional)

```bash
# API only
pnpm dev:api

# UI only
pnpm dev:web
```

## API endpoints (backend)

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Stack status, adapter paths |
| `/api/systems` | GET | All 11 eval systems + FActScore means |
| `/api/generate` | POST | `{ title, outline, system, top_k }` → abstract + passages |

## Supported systems

`zero_shot`, `zero_shot_with_sft`, `naive_rag`, `naive_rag_with_sft`, `graph_only`, `rankrag_only`, `full`, `full_minus_sft`, `full_minus_graph`, `full_minus_rerank`

Default in the UI: **`full_minus_sft`** (best FActScore on the Jun 2026 grid).

## Interactive features (toggle in UI)

Open **Features** in the header to enable optional demo modes:

| Feature | Description |
|---------|-------------|
| A/B compare | Run two systems side-by-side on the same prompt |
| Guided 3-act tour | Scripted zero-shot → champion → SFT reveal |
| Eval challenges | Held-out prompts from `public/eval-showcase.json` |
| Gold abstract reveal | Show human holdout abstract after generation |
| Quick system swap | Re-run same prompt under another system |
| Live FActScore | Score output via `/api/verify` (70B verifier when up) |
| Passage ↔ abstract link | Click passage to highlight overlapping abstract spans |
| Pre/post RankRAG | Toggle passage list before vs after reranking |
| Snapshot replay | Save/load recent runs from browser storage |
| Presentation layout | Larger abstract, compact chrome |
| Outline guardrails | Warn on vague outlines with poor retrieval overlap |
| QR code | Share URL for Q&A (`?compare=…&present=1`) |

**Pipeline stage progress** is always on during generation (real stages from API when available).

URL examples:

```
http://localhost:3000/?present=1&example=nils-jens-demo
http://localhost:3000/?compare=full_minus_sft,full&example=nils-jens-demo
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROUNDED_API_URL` | `http://127.0.0.1:8080` | Backend for Next rewrites |
| `GROUNDED_DEMO_MOCK` | `0` | Set `1` for template mock generation (no 8B) |
| `GROUNDED_DEMO_EMBED_DEVICE` | auto | `cpu` if &lt;3 GPUs; else cuda on embed card. Force `cpu` or `cuda` to override |
| `CUDA_VISIBLE_DEVICES` | auto | Comma-list to restrict/pin cards (e.g. `2,3`) |
