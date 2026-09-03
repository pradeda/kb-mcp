# Corpus router evaluation data

This directory defines the dataset shape only. Phase A does not add synthetic calibration or holdout questions.

- `schema.json` is the normative record schema.
- `calibration.jsonl` will contain the 40 editable calibration questions.
- `holdout.jsonl` will contain the 20 locked holdout questions.
- Auto-generated paraphrases live in a separate future file and never count as independent holdout samples.

Every JSONL line must validate against `schema.json`. `expected_refs` uses corpus-qualified references. Once populated, dataset hashes are recorded with the `router_version` holdout report.
