# Real evaluation workflow (M-4 / M-7)

Headline metrics require the **full stack**, not mock generators or lexical verifiers.

## 1. Models on disk

```bash
export GROUNDED_MODELS_ROOT=/data/team1/models
python scripts/check_models.py
```

All three roles must show `weights_ready: true` (embedder, generator_8b, verifier_70b_awq).

## 2. Start the 70B FActScore verifier (M-4.2)

On 2× GPUs (see `configs/eval.yaml` `verifier_vllm.tensor_parallel_size`):

```bash
export PYTHONPATH=src GROUNDED_MODELS_ROOT=/data/team1/models
export CUDA_VISIBLE_DEVICES=0,1
unset GROUNDED_ALLOW_MODEL_DOWNLOAD

# Smoke once
.venv/bin/python scripts/serve_verifier.py --backend vllm --smoke --n-claims 3

# Daemon (default backend from configs/eval.yaml → vllm)
.venv/bin/python scripts/serve_verifier.py --host 127.0.0.1 --port 8765
```

Verify: `curl -s http://127.0.0.1:8765/health`

## 3. Run eval (M-4.7)

Defaults: **real 8B**, **HTTP 70B verifier**, **real BGE index** (fails if `mock_embed: true`).

```bash
export PYTHONPATH=src GROUNDED_MODELS_ROOT=/data/team1/models CUDA_VISIBLE_DEVICES=0
.venv/bin/python scripts/run_eval.py --system naive_rag --limit 5
```

Dev / CI only:

```bash
.venv/bin/python scripts/run_eval.py --system naive_rag --limit 2 --mock-gen --mock-verifier
```

## 4. SFT smoke (M-5.4)

Requires verifier server + 8B + SFT adapter under `runs/seg5_sft_train_*/adapter`:

```bash
.venv/bin/python scripts/sft_smoke_eval.py --limit 10
```

## 5. Full grid (M-7.3)

```bash
.venv/bin/python scripts/run_eval_grid.py
```

Systems using RankRAG (`rankrag_only`, `full`, …) need a trained `seg6_rankrag_*` adapter unless `--mock-gen`.

## Graph pilot (M-6.2)

```bash
# Real 8B JSON extraction (cap chunks for smoke)
.venv/bin/python scripts/build_graph_pilot.py --extractor llm --max-chunks 5
```

## Demo (M-8.3)

```bash
export PYTHONPATH=src GROUNDED_MODELS_ROOT=/data/team1/models CUDA_VISIBLE_DEVICES=0
.venv/bin/python scripts/serve_demo.py
# Dev-only mock: GROUNDED_DEMO_MOCK=1
```
