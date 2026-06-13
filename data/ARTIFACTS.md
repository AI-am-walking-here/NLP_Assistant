# Large artifacts — not included in git

Place regenerated outputs here after running the pipeline locally.

| Subdirectory | Script | Notes |
|--------------|--------|-------|
| `parsed/` | `scripts/normalize.py`, `scripts/export_from_corpus.py` | One JSON per arXiv paper |
| `chunks/` | `scripts/build_index.py` | `chunks.parquet` |
| `indices/` | `scripts/build_index.py` | `faiss.index`, `embeddings.npy`, `index_meta.json` |
| `sft/` | `scripts/build_sft_data.py` | `train.jsonl`, `val.jsonl` |
| `rankrag/` | `scripts/build_rankrag_data.py` | RankRAG training pairs |

See root `README.md` → **Data artifacts** for sizes and setup.
