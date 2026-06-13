# Data layout (canonical paths)

Downstream code reads only the paths below. Do not add new top-level JSON/JSONL blobs without updating `configs/data.yaml` and this file.

| Path | Role |
|------|------|
| `parsed/` | One `Paper` JSON per arxiv ID (source of truth after Segment 2) |
| `parsed_manifest.jsonl` | Per-paper parse summary index |
| `parsed_valid.json` | IDs passing M-2.3 quality filter |
| `papers_enriched.jsonl` | Slim metadata export after M-2.4 (not full papers) |
| `s2_cache.jsonl` | OpenAlex / S2 enrichment cache |
| `splits/` | `index.txt`, `eval_holdout.txt`, `eval_grid_80.txt`, `sft.txt` |
| `chunks/`, `indices/` | Segment 3 vector index |
| `eval_set/` | Eval prompts, grid results, human-eval / audit templates |
| `sft/`, `rankrag/`, `graph/` | Training / graph artifacts |
| `archive/` | Read-only Lud drop + `metadata/recovery/` legacy exports |

Recovery-only exports from `/data/team1/corpus/papers.jsonl.gz` live under `archive/metadata/recovery/` (do not use as canonical IDs — use `parsed_valid.json`).

**Eval scores:** only `eval_set/grid_runs.json` (see `eval_set/README.md`). Do not add stray comparison JSON files.

**Large artifacts** (`parsed/`, `chunks/`, `indices/`, `sft/`) are excluded from git. See `ARTIFACTS.md` and root `README.md` for how to regenerate locally.
