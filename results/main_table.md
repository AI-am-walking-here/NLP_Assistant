# Grounded — Results

Headline metric: **FActScore** (70B verifier via HTTP; see docs/EVAL_WORKFLOW.md).

## Eval grid

| System | FActScore (mean) | Ref overlap | Specificity | Mock gen | Mock rerank |
| --- | ---: | ---: | ---: | --- | --- |
| full | 0.2398015873015873 | 0.3775375000000001 | 0.53962 | False | False |
| full_minus_rerank | 0.22245535714285708 | 0.3735475 | 0.5398637499999998 | False | True |
| full_minus_sft | 0.4958333333333331 | 0.27484125 | 0.63708125 | False | False |

## Notes

- Grid source: `main_table.json` under `data/eval_set/`
- Index mock embed: `see data/indices/index_meta.json`
- Citation precision/recall metrics are not reported.
- Models are loaded from local disk; no large Hugging Face downloads at eval time (see README → Models).
