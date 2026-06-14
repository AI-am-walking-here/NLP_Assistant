# data/indices/

**This folder is empty in Git on purpose.** It is filled in when you extract the
external artifact archive `grounded-artifacts.tar.gz` (delivered separately, ~2.3 GB).

Expected contents after extraction:

- `faiss.index` — FAISS `IndexFlatIP` over L2-normalized BGE-large embeddings
- `embeddings.npy` — chunk embedding matrix
- `index_meta.json` — index metadata

Extract the tarball **from the repository root**, not into this folder — the archive
already contains the `data/indices/...` path:

```bash
tar -xzf /path/to/grounded-artifacts.tar.gz -C .   # run from the repo root
```

See [`../ARTIFACTS_DOWNLOAD.md`](../ARTIFACTS_DOWNLOAD.md) and the root `README.md`
("Data archive — extract this first") for full instructions.
