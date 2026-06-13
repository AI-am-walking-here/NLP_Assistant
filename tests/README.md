# Tests

Offline unit tests for `src/grounded/`. Run from the project root:

```bash
export PYTHONPATH=src
pytest tests/ -q
```

| Module | Area |
|--------|------|
| `test_bootstrap.py` | M-0 layout, schemas, `init_run` |
| `test_config.py` | YAML configs and path resolution |
| `test_schema.py` | Real parsed JSON round-trip |
| `test_filter.py`, `test_splits.py`, `test_citations.py` | M-2 data pipeline |
| `test_corpus_export.py`, `test_unarxive_manifest.py` | Corpus / manifest helpers |
| `test_chunker.py` | M-3 chunking |
| `test_factscore.py`, `test_eval_prompts.py`, `test_ragas_wrap.py`, `test_verifier_server.py` | M-4 eval |
| `test_prompts.py`, `test_pipeline.py`, `test_rerank.py` | M-5–7 generation |
| `test_sft_data.py`, `test_graph.py` | M-5–6 training |
| `test_stats.py` | Bootstrap CIs |
| `test_model_download_policy.py` | Hub download guard |

Shared fixtures live in `conftest.py` (`repo_root`, `sample_parsed_path`).
