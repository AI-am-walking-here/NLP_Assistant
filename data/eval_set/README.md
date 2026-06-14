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
| `source` | Provenance tag (e.g. `unarxive`) |

**Collection.** Sampled from the project's frozen cs.CL corpus (~7,900 papers,
arXiv primary category `cs.CL`), built from the unarXive release of parsed arXiv
LaTeX. The 80 prompts are a subset of the 878-paper evaluation holdout
(`data/splits/eval_grid_80.txt`), disjoint from the SFT training split.
Regenerate with `scripts/build_eval_prompts.py`.

**Known biases / limitations.** English-only; single domain (NLP / cs.CL);
skewed toward recent papers; abstracts reflect arXiv author writing styles and
any selection bias in the unarXive parse. Scores are not representative of other
domains or languages.
