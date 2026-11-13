# Experiment Log

This file records observations and decisions chronologically. Do not rewrite old entries when the protocol changes; append a new entry and identify which comparisons must be rerun.

## 2026-10-24 — Repository implementation

- Scope: implemented the CPU-verifiable repository and lazy A100 training/evaluation interfaces.
- Protocol state: exploratory pilot; final test data is not frozen.
- Model: `Qwen/Qwen3-4B-Thinking-2507` at revision `768f209d9ea81521153ed38c47d515654e938aea`.
- Candidate settings: BF16 rank-16 LoRA; four completions per prompt; 4,096-token probe; gamma candidates 0.1 and 0.2; `dr_grpo`; reward scaling disabled; KL beta 0.
- Evidence: local lint, tests, data audit, dry runs, and synthetic analysis are recorded after validation below.
- Not completed: model download, A100 resource probe, inference, GRPO updates, adapter save/reload/resume, completion-budget selection, gamma selection, final test freeze, main runs, and empirical report.

### Local CPU validation

- Environment: macOS arm64, Python 3.12.14 in the project-local `.venv`.
- `ruff check .`: passed.
- `pytest -m 'not gpu'`: 30 passed; the CUDA integration test was deselected.
- Pilot data build/audit: passed with 24 families and 36 records (16 train records, 18 development records across six paired families, two harder records, and no frozen test records).
- Config dry runs: passed for accuracy, efficiency at gamma 0.1, and initial-model evaluation without importing the GPU stack.
- Synthetic analysis: passed and produced summary JSON, CSV, all three figure groups, a blinded trace sample, and its codebook in a temporary test directory.
- Notebook: parsed and validated as nbformat 4.5.
- Interpretation: this establishes CPU pipeline behavior only; it is not model-performance evidence.

## 2026-11-13 — Development calibration workflow

- Added an executable budget probe using four samples per development item (72 pilot responses).
- Predeclared budget rule: smallest standard candidate at or above 110% of the correct,
  non-truncated p95, requiring at least 50 total samples, 20 correct samples, and no more than
  2% probe truncation.
- Added independent gamma 0.1 and 0.2 pilots with matched completion-budget overrides and
  matched evaluation provenance.
- Predeclared gamma rule: choose the smallest candidate with at least 15% token reduction on
  matched completions that both conditions answered correctly and no more than a
  five-percentage-point drop in overall or changed-premise accuracy.
- Added machine-readable `budget_decision.json`, `gamma_decision.json`, and
  `selected_protocol.yaml` artifacts. A null selection stops the workflow rather than tuning on
  final test data.
- Updated analysis to read training metrics from the actual training run directories, and expanded
  backup coverage to data, analysis, notebook, source, tests, runs, and configurations.
- No GPU experiment or scientific result has been produced by this update.
- CPU verification after this update: 33 non-GPU tests passed; the CUDA integration test remains
  intentionally deselected locally.
