# FActScore verifier server (M-4.2)

## Backends

| `--backend` | Description |
|-------------|-------------|
| `mock` | Lexical overlap (fast, no GPU) |
| `awq` | 70B AWQ via AutoAWQ + transformers (`tensor_parallel` over 2 GPUs) |
| `vllm` | 70B AWQ via vLLM (`configs/eval.yaml` → `verifier_vllm`) |

Default CLI backend comes from **`configs/eval.yaml`** → `verifier_default_backend` (**`vllm`**). Use **`mock`** only for CI.

## Prerequisites (GPU)

1. Weights: `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` under `GROUNDED_MODELS_ROOT`
2. **2× GPUs** with ~24GB each (tensor parallel size 2 in `configs/eval.yaml`)
3. For **vLLM** only: Python dev headers (Triton JIT in workers needs `Python.h`):

```bash
bash scripts/setup_python_dev_headers.sh   # extracts libpython3.10-dev → .tmp/py310dev
# Or: sudo apt install python3.10-dev libpython3.10-dev
```

> The header version must match the venv's Python. The reference results were
> produced on Python 3.10. If your venv uses a different version (the conda env
> in `environment.yml` is 3.11), substitute `3.10` accordingly in the command
> above and in `setup_python_dev_headers.sh` / `verifier_server.py`.

`serve_verifier.py` runs the setup script automatically when headers are missing.

4. Install extras (project venv):

```bash
export TMPDIR=/data/team1/llm-assistant-final/.tmp
pip install vllm autoawq gptqmodel ninja
```

## Smoke tests

```bash
export PYTHONPATH=src GROUNDED_MODELS_ROOT=/data/team1/models
export CUDA_VISIBLE_DEVICES=0,1

# Fast
python scripts/serve_verifier.py --backend mock --smoke

# GPU (first load ~1–2 min)
python scripts/serve_verifier.py --backend awq --smoke --n-claims 3

# vLLM (after python3.10-dev / setup_python_dev_headers.sh)
python scripts/serve_verifier.py --backend vllm --smoke --n-claims 3 --max-seconds 900
```

## Run server

```bash
python scripts/serve_verifier.py --backend awq --host 127.0.0.1 --port 8765
```

```bash
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/verify \
  -H 'Content-Type: application/json' \
  -d '{"claim":"Transformers improve NLP benchmarks.","passages":["We evaluate on GLUE."]}'
```

Eval harness can use `HttpClaimVerifier("http://127.0.0.1:8765")` from `grounded.eval.factscore`.
