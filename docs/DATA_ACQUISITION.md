# Data acquisition

Two ways to obtain the ~8,900-paper cs.CL corpus.

## Option A — Re-export from frozen snapshot (recommended)

If you have `data/corpus/papers.jsonl.gz` (included in this repo via Git LFS):

```bash
export PYTHONPATH=src
git lfs pull   # fetch corpus if cloned without LFS files
python scripts/export_from_corpus.py --corpus data/corpus/papers.jsonl.gz
python scripts/seg2_bookkeeping.py --skip-s2
```

Default corpus path: `data/corpus/papers.jsonl.gz` (included via Git LFS; see `data/DATASET_CARD.md`).

## Option B — Download from scratch (Segment 1)

Enable sources in `configs/data.yaml` (`unarxive.enabled: true` and/or `arxiv_s3.enabled: true`), then:

```bash
python scripts/s3_pull.py filter-metadata
python scripts/s3_pull.py download-unarxive      # Zenodo unarXive
python scripts/s3_pull.py download-manifest      # arXiv S3 manifest
python scripts/s3_pull.py download-tarballs      # requires AWS credentials
python scripts/s3_pull.py extract-tarballs
python scripts/normalize.py
python scripts/seg2_bookkeeping.py
```

Sources are **disabled by default** (corpus frozen per v3.1). S3 downloads incur egress costs.

## Downstream artifacts

| Artifact | Script |
|----------|--------|
| `data/parsed/` | `export_from_corpus.py` or `normalize.py` |
| `data/splits/` | `seg2_bookkeeping.py` (included in repo) |
| `data/chunks/`, `data/indices/` | `build_index.py` |
| `data/sft/*.jsonl` | `build_sft_data.py` |
| `data/rankrag/*.jsonl` | `build_rankrag_data.py` |
