# External artifacts download (recommended)

Large pinned files are **not in Git** (avoids Git LFS quota issues and pointer-file clones).

## Size summary

| Bundle | Contents | Size |
|--------|----------|------|
| **Core** (corpus + adapters) | `papers.jsonl.gz`, SFT + RankRAG LoRA | **~376 MB** |
| **Index** | `faiss.index`, `embeddings.npy`, `chunks.parquet`, metadata | **~2.1 GB** |
| **Full archive** | Core + index | **~2.5 GB** |

| Component | Size |
|-----------|------|
| Corpus (`papers.jsonl.gz`) | 264 MB |
| SFT adapter | 43 MB |
| RankRAG adapter | 69 MB |
| FAISS index + embeddings + chunks | ~2.1 GB |

## Download

**Replace this URL before submission** with your course website, Google Drive, or Zenodo link:

```
TODO: https://YOUR_HOST/grounded-artifacts.tar.gz
```

Local copy on the university server (until decommission):

```
/data/team1/grounded-artifacts.tar.gz   # 2.3 GB compressed
/data/team1/grounded-artifacts/        # uncompressed tree (same contents)
```

## Install into repo root

```bash
bash scripts/install_artifacts.sh /path/to/grounded-artifacts.tar.gz
# or
export GROUNDED_ARTIFACTS_URL='https://YOUR_HOST/grounded-artifacts.tar.gz'
bash scripts/install_artifacts.sh
```

Extracts:

- `data/corpus/papers.jsonl.gz`
- `runs/seg5_sft_train_2026-06-05-0617/adapter/` (pins main table)
- `runs/seg6_rankrag_train_2026-06-05-0542/adapter/`
- `data/chunks/`, `data/indices/`

## Already in Git (no download needed)

- Training JSONL (`data/sft/`, `data/rankrag/`)
- Eval per-prompt outputs (`runs/seg4_eval_*_2026-06-11-*/`) — inspect abstracts + FActScore without GPU
- `results/main_table.json`

## Git LFS policy

We **do not use Git LFS**. Free GitHub LFS includes ~1 GB storage and 1 GB/month bandwidth; a 2.5 GB archive would exceed quota quickly, and clones without `git lfs pull` receive useless pointer files.

See `data/DATASET_CARD.md` for collection, license, and biases.
