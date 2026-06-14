# Corpus snapshot

| File | Where |
|------|--------|
| `papers.jsonl.gz` | **External archive** (~264 MB) — not in Git |

Download and extract: [`ARTIFACTS_DOWNLOAD.md`](ARTIFACTS_DOWNLOAD.md)

```bash
bash scripts/install_artifacts.sh /path/to/grounded-artifacts.tar.gz
python scripts/export_from_corpus.py --corpus data/corpus/papers.jsonl.gz
```
