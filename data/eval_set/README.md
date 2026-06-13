# Eval set

| File | Produced by | Role |
|------|-------------|------|
| `prompts.jsonl` | `scripts/build_eval_prompts.py` | 80 eval prompts (included) |
| `grid_runs.json` | `scripts/run_eval_grid.py` | Aggregate scores (regenerated on eval; submitted copy in `results/main_table.json`) |
| `factscore_audit.jsonl` | `scripts/prepare_factscore_audit.py` | Manual audit sample |
| `human_eval_template.jsonl` | `scripts/prepare_human_eval.py` | Annotator template |

Reported results for submission: `results/main_table.json` and `results/main_table.md`.
