# Results

This folder holds the **reported main evaluation table** for the final project submission.

| File | Description |
|------|-------------|
| `main_table.json` | Canonical aggregate scores (FActScore, reference overlap, specificity) for headline systems |
| `main_table.md` | Human-readable markdown table generated from `main_table.json` |

Fresh evaluation runs write per-system outputs under `runs/seg4_eval_<system>_<timestamp>/` (`results.json`, `per_prompt.jsonl`, `meta.json`). After a full grid, `scripts/run_eval_grid.py` aggregates scores into `data/eval_set/grid_runs.json`.

To regenerate the markdown table:

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from grounded.eval.report_build import build_report_markdown
Path('results/main_table.md').write_text(
    build_report_markdown(grid_path=Path('results/main_table.json'))
)
"
```
