# Eval set

| File | Produced by | Role |
|------|-------------|------|
| `prompts.jsonl` | `scripts/build_eval_prompts.py` | 80 eval prompts (included) |
| `grid_runs.json` | `scripts/run_eval_grid.py` | Aggregate scores (regenerated on eval; submitted copy in `results/main_table.json`) |
| `factscore_audit.jsonl` | `scripts/prepare_factscore_audit.py` | Manual audit sample |
| `human_eval_template.jsonl` | `scripts/prepare_human_eval.py` | Annotator template |

Reported results for submission: `results/main_table.json` and `results/main_table.md`.

---

## Dataset card — `prompts.jsonl`

> The canonical, project-wide dataset card is [`../DATASET_CARD.md`](../DATASET_CARD.md).
> This section adds detail specific to the 80 evaluation prompts.

**Summary.** 80 abstract-generation tasks. Each line is one held-out paper: the
model receives the `title` and a section `outline` and must generate an abstract;
`gold_abstract` is the paper's own abstract, used as the reference.

**Fields.**

| Field | Description |
|-------|-------------|
| `arxiv_id` | Source arXiv identifier |
| `title` | Paper title (model input) |
| `outline` | Section outline (model input) |
| `gold_abstract` | Reference abstract (held out; never shown to the model) |
| `year` | Publication year |
| `source` | Provenance pipeline: `unarxive` (67/80) or `latex_s3` (13/80) |

**Collection.** Sampled from the project's frozen `cs.CL` corpus (~7,900 papers,
arXiv primary category `cs.CL`), built from two pipelines over arXiv LaTeX source:
the **unarXive** release of parsed arXiv (67 of the 80 prompts) and a **direct
arXiv S3 LaTeX export** (13 of 80). The 80 prompts are a subset of the 878-paper
evaluation holdout (`data/splits/eval_grid_80.txt`) and are **disjoint from the
SFT training split** (`data/splits/sft.txt`; verified 0 overlap). Regenerate with
`scripts/build_eval_prompts.py`.

**Known biases / limitations.** English-only; single domain (NLP / `cs.CL`);
publication years span 2016–2025 but concentrate in **2021–2022 (~66% of
prompts)**. Abstracts reflect arXiv author writing styles, and papers are limited
to those whose LaTeX source parsed successfully (a selection bias). Scores are
not representative of other domains, languages, or time periods.
