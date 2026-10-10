# swethaleena — IP-A: Intellectual Property Case Law

Model: Phi-4-mini-instruct. Corpus: 214 Australian Federal Court IP cases (2006–2009),
UCI Legal Case Reports dataset (AustLII). **Research use only, do not redistribute.**

| Required item | File |
|---|---|
| Original dataset | `assignment1_FINAL.zip` → `assignment1/corpus.jsonl` |
| Development dataset (50) | `dev_inputs.jsonl` |
| Evaluation dataset (50, no overlap with dev) | `eval_inputs.jsonl` |
| Chatlogs / probe dataset (50 prompts) | `partD_domain_specific_probe_set.jsonl` (original JSON is in the results zip) |
| Metrics, one per experiment | `results-20261002T222957Z-1-001.zip` |
| Code | `LLM_API_code.zip` (AS01 notebook, `extraction.py`, AS02 notebook) |
| AI disclosure | `AI_disclosure.md` |
| Reports | `AS02_Report_Swetaleena_Guha.pdf`, AS01 memo inside `assignment1_FINAL.zip` |

## Results zip: experiment → file

| Experiment | File |
|---|---|
| Baseline, default and tuned decoding (dev) | `baseline_eval1_dev50.jsonl`, `baseline_eval3_dev50.jsonl` |
| Structured output (eval) | `partC_eval1_structured_output_eval50.jsonl` |
| Tool calling (eval) | `partC_eval2_tool_calling_eval50.jsonl` |
| Input screening / output validation | `partC_eval3_input_screening_eval50.jsonl`, `partC_eval4_output_validation_eval50.jsonl` |
| Guardrail probes | `partD_guardrail_eval50.jsonl`, `partD_domain_guardrail_eval50.jsonl`, `partD_domain_baseline_eval50.jsonl`, `partD_guardrail_log.jsonl` |
| Baseline vs guarded model, human adjudication | `partD_baseline_vs_guardrail_model15.jsonl`, `partD_human_adjudication_model15.jsonl` |
| Indirect injection canary | `partD_indirect_injection_canary_baseline10.jsonl` |
| Summary metrics | `partD_final_summary.json`, `partE_technical_metrics.json`, `partE_cost_benefit_scenario.json` |

The dev/eval files were rebuilt without re-running any model: they contain the corpus
records whose `doc_id`s appear in the dev and eval result files, in the same order.
