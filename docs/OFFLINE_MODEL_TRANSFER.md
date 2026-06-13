# Offline model transfer (home → university)

**Policy:** Do **not** run `huggingface-cli download` or large `from_pretrained` pulls on the university SSH host. Download at home, transfer files once, then load from disk on the server.

**Primary layout (recommended):** flat directories under a shared models root:

```text
/data/team1/models/
  BAAI/bge-large-en-v1.5/
  meta-llama/Llama-3.1-8B-Instruct/
  hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4/
```

`scripts/setup_venv.sh` sets `GROUNDED_MODELS_ROOT=/data/team1/models`. Paths are listed in `configs/models.yaml`. Check status:

```bash
python scripts/check_models.py
```

See `scripts/download_models.sh` for a one-shot Hugging Face download helper.

**Legacy Hub cache** (optional): `HF_HOME=/data/team1/llm-assistant-final/.hf-cache/hub/models--…/snapshots/…`

---

## What to download (from `configs/`)

| Model ID | Milestone | ~Disk | Gated? |
|----------|-----------|-------|--------|
| `BAAI/bge-large-en-v1.5` | M-3.2 real embeddings | ~1.3 GB | No |
| `meta-llama/Llama-3.1-8B-Instruct` | Generation, SFT, RankRAG | ~16 GB | **Yes** |
| AWQ 70B (see below) | M-4.2 verifier (when vLLM wired) | ~35–40 GB | **Yes** |

**Phase 1 (recommended first):** BGE + 8B only (~18 GB). Mock verifier stays fine until vLLM is wired.

**Phase 2 (optional):** One **AWQ INT4** Llama-3.1-70B checkpoint (not full FP16 ~140 GB). Search Hugging Face for `Llama-3.1-70B-Instruct AWQ` and pick a repo your account can access after accepting Meta’s license. Example IDs that often appear (verify before downloading):

- `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`
- `casperhansen/Meta-Llama-3.1-70B-Instruct-AWQ`

Update `configs/eval.yaml` / verifier wiring if your AWQ repo id differs from `meta-llama/Llama-3.1-70B-Instruct`.

**Not transferred:** SFT/RankRAG **adapters** — trained on the server into `runs/…/adapter/` after 8B is present.

---

## Before you download (both machines)

1. Create a Hugging Face account and accept the license for **Llama 3.1** on each model page.
2. Create a read token: https://huggingface.co/settings/tokens
3. On the **university** machine, put the token in `.env` (for gated repo metadata only — not for bulk download):

   ```bash
   HF_TOKEN=hf_...
   ```

---

## Download on Windows (WSL2 recommended)

### Setup (one time, in Ubuntu WSL)

```bash
sudo apt update && sudo apt install -y git git-lfs rsync openssh-client
git lfs install

python3 -m venv ~/hf-venv && source ~/hf-venv/bin/activate
pip install -U "huggingface_hub[cli]"

huggingface-cli login
# paste token when prompted
```

### Download into a Hub cache layout

```bash
export HF_HOME="$HOME/hf-export"   # e.g. D:\hf-export via /mnt/d/hf-export
mkdir -p "$HF_HOME"

# Open model (no gating)
huggingface-cli download BAAI/bge-large-en-v1.5

# Gated — requires login + license
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Optional phase 2 — replace with the AWQ repo you chose
# huggingface-cli download hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
```

This creates:

```text
~/hf-export/hub/models--BAAI--bge-large-en-v1.5/snapshots/<hash>/...
~/hf-export/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<hash>/...
```

**Native Windows (no WSL):** install Python 3.10+, `pip install huggingface_hub[cli]`, run the same commands in PowerShell with `set HF_HOME=D:\hf-export` (use `huggingface-cli` from that venv).

---

## Transfer to the university host

Replace `USER`, `HOST`, and paths.

### rsync (WSL — resumable)

```bash
rsync -avP --partial "$HOME/hf-export/hub/" \
  USER@HOST:/data/team1/llm-assistant-final/.hf-cache/hub/
```

If you also have `xet` / `assets` under `hf-export`, sync the whole tree:

```bash
rsync -avP --partial "$HOME/hf-export/" \
  USER@HOST:/data/team1/llm-assistant-final/.hf-cache/
```

### scp

```bash
scp -r "$HOME/hf-export/hub" \
  USER@HOST:/data/team1/llm-assistant-final/.hf-cache/
```

### WinSCP

Upload the local `hub` folder to:

`/data/team1/llm-assistant-final/.hf-cache/hub/`

---

## On the university machine (load only — no Hub pull)

```bash
cd /data/team1/llm-assistant-final
source .venv/bin/activate
export PYTHONPATH=src
export HF_HOME=/data/team1/llm-assistant-final/.hf-cache
export GROUNDED_ALLOW_MODEL_DOWNLOAD=1   # allows code to call load_pretrained (uses local cache)
# HF_TOKEN from .env if gated repos need it

# Sanity: BGE loads from disk
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-large-en-v1.5')
print('bge ok', m.get_sentence_embedding_dimension())
"

# Sanity: 8B tokenizer/weights visible (may take a minute; uses GPU RAM)
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('8b tokenizer ok')
"
```

### Turn on real pipeline pieces

1. **Index** — rebuild with real embeddings (not mock):

   ```bash
   python scripts/build_index.py --embed-only
   ```

2. **Generation** — in `configs/retrieval.yaml` set `mock_generation: false` when ready for real 8B inference.

3. **Eval** — `python scripts/run_eval.py --system naive_rag --no-mock-gen` (slow; uses GPU).

4. **SFT / RankRAG** — `python scripts/sft_train.py` / `rankrag_train.py` only after professor approves GPU time.

**Do not** run `huggingface-cli download` on the university host unless you intentionally replace a corrupted file.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Refusing to download models` | `export GROUNDED_ALLOW_MODEL_DOWNLOAD=1` |
| Gated repo 401 | `HF_TOKEN` in `.env`; license accepted on HF website |
| `File not found` after copy | Ensure `hub/models--…/snapshots/<hash>/` exists under `.hf-cache` |
| Still tries to download | Wrong `HF_HOME`; must be project `.hf-cache` |
| Out of disk on `/` | Project data is on `/data`; cache lives under project dir on `/data/team1` |

---

## Size checklist

| Bundle | Approx. total |
|--------|----------------|
| Phase 1: BGE + 8B | ~18 GB |
| Phase 2: + 70B AWQ | ~55 GB |
| Full 70B FP16 | ~140 GB+ — **do not use** |
