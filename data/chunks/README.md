# data/chunks/

**This folder is empty in Git on purpose.** It is filled in when you extract the
external artifact archive `grounded-artifacts.tar.gz` (delivered separately, ~2.3 GB).

Expected contents after extraction:

- `chunks.parquet` — section-aware 512-token chunks (64-token overlap) of the corpus

Extract the tarball **from the repository root**, not into this folder — the archive
already contains the `data/chunks/...` path:

```bash
tar -xzf /path/to/grounded-artifacts.tar.gz -C .   # run from the repo root
```

See [`../ARTIFACTS_DOWNLOAD.md`](../ARTIFACTS_DOWNLOAD.md) and the root `README.md`
("Data archive — extract this first") for full instructions.
