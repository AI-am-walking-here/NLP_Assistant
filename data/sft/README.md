# SFT training data (v3.1)

| File | Produced by | Role |
|------|-------------|------|
| `train.jsonl` | `scripts/build_sft_data.py` | Chat-format `(title + outline) → abstract` from `data/splits/sft.txt` |
| `val.jsonl` | same (splits refresh) | Held-out validation examples |

Included in this repository for reproducibility. See [`../DATASET_CARD.md`](../DATASET_CARD.md) for collection, license, and biases.

```bash
source .venv/bin/activate
export PYTHONPATH=src GROUNDED_MODELS_ROOT=/data/team1/models
python scripts/build_sft_data.py
python scripts/check_models.py          # generator_8b.weights_ready must be true
python scripts/sft_train.py --dry-run
python scripts/sft_train.py             # full 1-epoch QLoRA (~GPU hours)
python scripts/sft_train.py --smoke-train   # 20-step sanity check
```

If weights were rsync'd into `.hf-cache/hub/`:

```bash
python scripts/import_generator_8b_from_hf_cache.py
```
