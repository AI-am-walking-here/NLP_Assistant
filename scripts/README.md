# CLI entrypoints

One script per pipeline stage. Logic lives in `src/grounded/`; scripts parse args and call library code.

| Script | Stage |
|--------|-------|
| `bootstrap_test.py` | Config + run tracking smoke |
| `export_from_corpus.py` | Re-export parsed JSON from corpus |
| `fix_unarxive_manifest_paths.py` | Manifest path recovery |
| `normalize.py` | LaTeX parse + normalize → `data/parsed/` |
| `seg2_bookkeeping.py` | Filter, enrich, splits |
| `build_index.py` | Chunk papers + build FAISS index |
| `seg3_smoke.py` | Retrieval smoke test |
| `build_eval_prompts.py` | Build 80-prompt eval set |
| `serve_verifier.py` | 70B FActScore verifier server |
| `run_eval.py`, `run_eval_grid.py`, `compare_eval.py` | Evaluation + ablations |
| `run_ragas_sample.py` | RAGAS diagnostic |
| `build_sft_data.py`, `sft_train.py`, `sft_smoke_eval.py` | SFT pipeline |
| `build_sft_dpo_data.py`, `sft_dpo_train.py` | Optional DPO |
| `build_graph_pilot.py`, `build_graph_communities.py` | Graph RAG |
| `build_rankrag_data.py`, `rankrag_train.py` | RankRAG |
| `prepare_human_eval.py`, `prepare_factscore_audit.py` | Human eval prep |
| `serve_demo.py` | Web demo |
| `build_report.py` | Markdown report from eval grid |
| `preflight.py` | Host / artifact check |
| `check_models.py` | Verify local model weights |
| `download_models.sh` | Download HF weights to `GROUNDED_MODELS_ROOT` |
| `s3_pull.py` | Segment 1 corpus acquisition |
| `eval_verifier_smoke.py` | Verifier server smoke test |
| `setup_venv.sh`, `setup_python_dev_headers.sh` | Environment setup |
