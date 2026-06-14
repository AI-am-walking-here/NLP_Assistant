# Grounded — Scientific Writing Assistant

**Team 1 · Peking University · Advanced NLP Final Project**

## Abstract

Grounded is a retrieval-augmented scientific writing assistant that generates NLP-style paper abstracts from a title and contribution outline. We built an end-to-end pipeline over ~8,800 arXiv cs.CL papers: LaTeX parsing and normalization, section-aware chunking, FAISS dense retrieval (BGE-large), optional Graph RAG and RankRAG reranking, domain SFT (QLoRA on Llama-3.1-8B), and evaluation with a 70B FActScore verifier. The system compares multiple retrieval and training regimes on an 80-prompt eval grid, reporting FActScore, reference overlap, and specificity to measure grounding quality.

---

## Repository layout

```
.
├── README.md              # this file
├── environment.yml        # conda env (recommended)
├── requirements.txt       # pip fallback
├── pyproject.toml
├── .env.example
├── src/grounded/          # library code (retrieval, training, eval, demo)
├── scripts/               # CLI entrypoints (one per pipeline stage)
├── configs/               # YAML configs (data, retrieval, eval, sft, graph)
├── data/                  # small metadata + eval prompts (large artifacts excluded)
│   ├── splits/            # train/eval paper ID lists
│   └── eval_set/          # 80 eval prompts + grid aggregate JSON
├── demo/                  # Next.js interactive UI (see demo/README.md)
├── results/               # reported main table (main_table.json / .md)
├── tests/                 # pytest suite
└── docs/                  # eval workflow, model transfer, data acquisition
```

**Not included in this repo (too large):** parsed papers (`data/parsed/`), FAISS index (`data/chunks/`, `data/indices/`), SFT training JSONL (`data/sft/`), model weights, and training checkpoints. See [Data artifacts](#data-artifacts) below.

---

## Setup

### 1. Environment

Recommended (CUDA + PyTorch):

```bash
conda install -n base mamba -c conda-forge -y   # one-time, if conda solver is slow
mamba env create -f environment.yml
conda activate NLPFinal
```

Alternative (pip only; you must install a CUDA-matched PyTorch yourself):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or use the project venv helper:

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

### 2. Configuration

```bash
cp .env.example .env
# Optional: S2_API_KEY, HF_TOKEN, WANDB_API_KEY
export GROUNDED_MODELS_ROOT=/path/to/models   # see docs/OFFLINE_MODEL_TRANSFER.md
```

### 3. Models (required for full eval / demo)

**The Python code does not auto-download models.** By default it loads weights from `GROUNDED_MODELS_ROOT` and raises an error if they are missing. Hub downloads only happen when you explicitly set `GROUNDED_ALLOW_MODEL_DOWNLOAD=1`.

Download weights once (on a machine with disk + Hugging Face access):

```bash
export GROUNDED_MODELS_ROOT=/path/to/models
bash scripts/download_models.sh          # BGE + 8B (~18 GB)
DOWNLOAD_VERIFIER=1 bash scripts/download_models.sh   # + 70B AWQ (~40 GB)
```

Or follow the manual steps in `docs/OFFLINE_MODEL_TRANSFER.md` (`huggingface-cli download` + rsync).

| Role | Model |
|------|-------|
| Embedder | `BAAI/bge-large-en-v1.5` |
| Generator | `meta-llama/Llama-3.1-8B-Instruct` |
| Verifier | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` |

```bash
export GROUNDED_MODELS_ROOT=/path/to/models
python scripts/check_models.py
```

### 4. Data artifacts

Rebuild or obtain locally before running the full pipeline:

| Artifact | Produced by | Approx. size |
|----------|-------------|--------------|
| `data/parsed/` | `scripts/normalize.py` / corpus export | ~GB |
| `data/chunks/chunks.parquet` | `scripts/build_index.py` | ~100 MB |
| `data/indices/faiss.index` | `scripts/build_index.py` | ~1 GB |
| `data/sft/train.jsonl` | `scripts/build_sft_data.py` | ~70 MB |
| SFT adapter | `scripts/sft_train.py` → `runs/seg5_sft_train_*/adapter` | ~200 MB |
| RankRAG adapter | `scripts/rankrag_train.py` → `runs/seg6_rankrag_*/adapter` | ~200 MB |

This repo **does** include: split ID lists (`data/splits/`), valid paper IDs (`data/parsed_valid.json`), eval prompts (`data/eval_set/prompts.jsonl`), and the reported aggregate scores (`results/main_table.json`). See `docs/DATA_ACQUISITION.md` for corpus download/export.

### 5. Smoke test (no GPU models)

```bash
export PYTHONPATH=src
pytest tests/ -q
python scripts/run_eval.py --system naive_rag --limit 2 --mock-gen --mock-verifier
```

---

## Reproducing the main table

The headline results are in `results/main_table.json` and `results/main_table.md`:

| System | FActScore | Ref overlap | Specificity |
|--------|----------:|------------:|------------:|
| `full` | 0.240 | 0.378 | 0.540 |
| `full_minus_rerank` | 0.222 | 0.374 | 0.540 |
| `full_minus_sft` | 0.496 | 0.275 | 0.637 |

> **Reading the table.** Rows are ablations of the `full` system. `full_minus_sft`
> (no domain SFT adapter) scores **higher** on FActScore than `full`: in this setup,
> domain SFT traded factual grounding for higher reference overlap. This is a
> deliberate, reported finding, not an error.
>
> **Reproducing these exact numbers requires the trained SFT and RankRAG adapters**
> (see the artifacts table above). They are not committed (too large); retrain them
> with `scripts/sft_train.py` and `scripts/rankrag_train.py`, or run the mock-mode
> sanity check below to exercise the harness without GPUs.

### Full reproduction (real stack)

**Prerequisites:** parsed corpus, FAISS index, SFT + RankRAG adapters, 2× GPUs for the 70B verifier.

```bash
export PYTHONPATH=src
export GROUNDED_MODELS_ROOT=/path/to/models

# Terminal 1 — FActScore verifier (70B, vLLM, 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python scripts/serve_verifier.py --host 127.0.0.1 --port 8765

# Terminal 2 — eval grid (80 prompts × 3 headline systems)
# Note: --systems takes one system per flag (repeat it), not a space-separated list.
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval_grid.py \
  --systems full --systems full_minus_rerank --systems full_minus_sft

# Aggregate → results/
cp data/eval_set/grid_runs.json results/main_table.json
PYTHONPATH=src python scripts/build_report.py --grid results/main_table.json --out results/main_table.md
```

See `docs/EVAL_WORKFLOW.md` for step-by-step details (verifier smoke, ablations, demo).

### Quick sanity check (mock, ~2 min)

```bash
export PYTHONPATH=src
python scripts/run_eval_grid.py \
  --systems full --systems full_minus_rerank --systems full_minus_sft \
  --limit 5 --mock-gen --mock-verifier --skip-verifier-check
```

This exercises the eval harness without GPU models; scores will **not** match the reported table.

---

## Estimated runtime

| Stage | Hardware | Time |
|-------|----------|------|
| Environment setup | CPU | 15–45 min |
| Index build (`build_index.py`) | 1× GPU | ~2–4 h |
| SFT training (`sft_train.py`, 2 epochs) | 1× GPU | ~3–5 h |
| RankRAG training (`rankrag_train.py`) | 1× GPU | ~2–4 h |
| Verifier server startup (70B AWQ) | 2× GPU | ~1–2 min |
| Single system eval (80 prompts, real stack) | 1× GPU + verifier | ~45–90 min |
| **Full headline grid (3 systems above)** | 2× GPU | **~3–5 h** |
| Full 9-system ablation grid | 2× GPU | ~8–15 h |

Times assume weights are local and the FAISS index is pre-built.

---

## Demo

Interactive Next.js UI + FastAPI backend (`demo/` + `src/grounded/demo/`).

```bash
# Terminal 1 — from repo root
bash scripts/setup_venv.sh
export GROUNDED_MODELS_ROOT=/path/to/models
export PYTHONPATH=src
python scripts/serve_demo.py --port 8080

# Terminal 2 — frontend (requires Node 20+ and pnpm)
cd demo
pnpm install
pnpm dev:web
```

Or one command (starts API + waits for health, then Next.js):

```bash
cd demo && pnpm install && pnpm dev
```

- UI: http://localhost:3000
- API: http://localhost:8080

Mock mode (no GPU models): `pnpm dev:mock`

Full details: `demo/README.md`.

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_index.py` | Chunk papers + build FAISS index |
| `scripts/build_sft_data.py` | Create SFT training JSONL |
| `scripts/sft_train.py` | QLoRA domain SFT |
| `scripts/rankrag_train.py` | RankRAG reranker training |
| `scripts/serve_verifier.py` | 70B FActScore verifier server |
| `scripts/run_eval.py` | Evaluate one system |
| `scripts/run_eval_grid.py` | Run multiple systems → `data/eval_set/grid_runs.json` |
| `scripts/compare_eval.py` | Paired bootstrap between two runs |
| `scripts/serve_demo.py` | FastAPI demo API |
| `scripts/download_models.sh` | Download HF weights to `GROUNDED_MODELS_ROOT` |
| `scripts/s3_pull.py` | arXiv corpus acquisition (Segment 1) |

Full index: `scripts/README.md`.

---

## Citation

Course final project, Team 1, Peking University, 2026.
