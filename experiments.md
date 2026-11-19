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

## 2026-11-15 — Colab progress and interruption visibility

- Added immediate progress lines for dataset generation, model loading, every evaluation
  completion, training setup and optimization, calibration decisions, bootstrap comparisons, and
  figure rendering.
- Evaluation now reports completed/total predictions, remaining predictions, token count,
  correctness, finish reason, item latency, elapsed time, and estimated time remaining after every
  saved completion.
- Evaluation manifests now retain total and completed prediction counts. A keyboard interruption is
  recorded explicitly and the same command can resume from atomically saved predictions.
- TRL progress display is explicitly enabled for GRPO steps; these messages do not change seeds,
  sampling, rewards, datasets, or scientific comparisons.
- The first attempted GPU probe downloaded and loaded the pinned model but was interrupted before
  a completed evaluation was established. It is not scientific evidence and does not alter the
  predeclared calibration protocol.

## 2026-11-19 — Censored-response budget follow-up

- The completed 4,096-token development probe contained 72 predictions, including 68 correct,
  non-truncated predictions and four truncated predictions (5.56% truncation). Its correct,
  non-truncated p95 was 3,595 tokens and the 10% headroom requirement was 3,955 tokens.
- Because truncation exceeded the predeclared 2% threshold, 4,096 was rejected and the saved
  decision requested an 8,192-token follow-up.
- To avoid needlessly repeating 68 uncensored predictions, the follow-up protocol now reuses those
  immutable rows and regenerates only the four truncated keys with the same model, prompt, seed,
  and sampling settings at the larger ceiling.
- Every regenerated prediction must reproduce the exact SHA-256 hash of its original 4,096-token
  prefix. Any mismatch rejects the combined run. This is a computational amendment only; sample
  membership, budget thresholds, and the development-only selection rule are unchanged.
- No 8,192-token follow-up result or final budget had been produced when this amendment was added.

## 2026-11-19 — Censored-response continuation correction

- The first replay-based extension attempt demonstrated why exact stochastic replay is unsafe: the
  first previously truncated response instead sampled EOS at 3,309 tokens. No replacement
  prediction was saved.
- The replay protocol recorded immediately above is therefore superseded. Each saved 4,096-token
  prefix is retokenized with the pinned tokenizer and must reproduce its exact recorded token count
  and SHA-256 token-ID hash. The model then samples only the continuation using a deterministic,
  recorded seed offset.
- The 68 uncensored rows, sample membership, budget thresholds, and development-only selection rule
  remain unchanged.

## 2026-11-19 — Gamma pilot range amendment

- Budget calibration selected `B = 8192`: all 72 development completions were correct and
  non-truncated after the censored-response continuation, with a correct-response p95 of 4,128.3
  tokens and a 10% headroom requirement of 4,542 tokens.
- Because reward pressure per token is `gamma / B`, the originally planned gamma values 0.1 and
  0.2 are weaker under 8,192-token normalization than under the initial 4,096-token probe.
- Add gamma 0.4 as a third development-only efficiency pilot before any pilot training is run.
  This supplies a stronger condition while retaining the same completion cap, exact reward formula,
  selection thresholds, base revision, seed, data, LoRA setup, and five-step pilot duration.
- Gamma selection still chooses the smallest of 0.1, 0.2, and 0.4 meeting the predeclared accuracy,
  premise-accuracy, and token-reduction thresholds. Only the selected gamma may proceed to the main
  experiment; test data remains untouched.
