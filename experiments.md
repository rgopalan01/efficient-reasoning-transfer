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

## 2026-11-19 — Pilot interruption safety

- Before pilot training began, change the accuracy and efficiency training configurations from
  checkpoint-at-step-5 to checkpoint-after-every-step. This may add storage and checkpoint-writing
  overhead but limits a runtime interruption to the currently unfinished step. Retain the two newest
  checkpoints for a fallback without keeping all five.
- Training commands now use `--resume auto`, validate configuration and dataset identity, resume the
  numerically highest complete Trainer checkpoint, and verify and skip an already-completed adapter.
  An incomplete output directory cannot be silently restarted without a resume request.
- Place accuracy, gamma 0.1, gamma 0.2, and gamma 0.4 training in separate Colab cells. Evaluation
  remains response-level resumable through atomic predictions. No scientific thresholds, seeds,
  data, rewards, or model settings changed.

## 2026-11-19 — Colab TorchAO environment correction

- The first accuracy-pilot launch failed before adapter attachment, generation, optimization, or a
  checkpoint because Colab preinstalled `torchao==0.10.0`, which PEFT 0.21.2 rejects as an
  incompatible optional integration. No training result or scientific evidence was produced.
- This experiment uses unquantized BF16 LoRA and neither requests nor imports TorchAO functionality.
  The Colab setup now removes the unused optional TorchAO package after installing project
  dependencies. Rerunning the same `--resume auto` command safely starts accuracy training at step 0.

## 2026-11-20 — Transformers 5 warmup interface correction

- The next accuracy-pilot launch successfully loaded the base model and attached LoRA, then failed
  while constructing `GRPOConfig`: Transformers 5.20.2 replaced the `warmup_ratio` keyword with
  `warmup_steps`, whose fractional values retain ratio semantics. The failure occurred before the
  trainer, generation, optimization, or a checkpoint; it produced no scientific result.
- Keep the repository's stable `warmup_ratio` configuration field and translate it to
  `warmup_steps` at the TRL boundary. The configured value remains 0.0, so the optimization
  protocol is unchanged. GRPO configuration is now validated before the model is loaded, allowing
  future interface errors to fail quickly without incurring model setup time.

## 2026-11-23 — L4 microbatch preflight

- Colab repeatedly allocated an NVIDIA L4 with approximately 24 GB rather than an A100. The
  batch-4 GRPO configuration exhausted device memory before the first efficiency update. No
  efficiency adapter or scientific result was produced by that attempt.
- Before changing the pinned model, add a one-step efficiency memory preflight with
  `per_device_train_batch_size=1` and `gradient_accumulation_steps=4`. This retains an effective
  batch of four, `num_generations=4`, the 8,192-token ceiling, reward, decoding, and LoRA settings.
- If the preflight succeeds, rerun the accuracy control and every efficiency pilot under this same
  batching configuration using distinct `_l4` run directories. Retain the earlier batch-4 accuracy
  run only as provenance and exclude it from matched comparisons. Stage 1 budget calibration and
  baseline evaluation remain valid because neither depends on the training microbatch schedule.
- If the preflight fails, do not silently reduce the completion cap or mix configurations; record
  the failure before considering a smaller base model and repeating model-dependent calibration.

## 2026-11-25 — T4 protocol amendment

- Constraint discovered before any main training or held-out final-test evaluation: future Colab
  access can reliably provide only an NVIDIA T4. The 4B BF16 protocol is therefore not the final
  experimental lineage.
- Preserve all completed `Qwen/Qwen3-4B-Thinking-2507` L4 runs as feasibility-pilot provenance.
  The completed accuracy and gamma pilots may inform engineering, but their budget, adapters, and
  development measurements are excluded from the revised comparison.
- Start a fresh T4-compatible lineage using `Qwen/Qwen3-1.7B` at exact revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, unquantized FP16 rank-16 LoRA, four generations,
  microbatch 1, and four gradient-accumulation steps. FP16 remains fixed if a different CUDA GPU is
  allocated later.
- Repeat every model-dependent development decision: one-response inference preflight, 72-response
  budget probe, budget validation, one-step training preflight, matched accuracy and gamma pilots,
  development evaluation, and gamma selection. Reusing the deterministic puzzle manifest is
  permitted because puzzle generation is model-independent.
- Store revised artifacts only below `runs/qwen3_1_7b_t4/` and
  `analysis/qwen3_1_7b_t4/`. Configuration and analysis guards must reject cross-model or
  cross-precision comparisons.
- No QLoRA, SFT, reward change, decoding change, hidden bonus, or final-test access is introduced.
  Main sample sizes, steps, seeds, frozen test data, and scientific results remain deferred until
  this revised development calibration completes.
- Local amendment validation: Ruff passed; 49 non-GPU tests passed with one CUDA integration test
  deselected; all three T4 configurations dry-ran without importing the GPU stack; and the revised
  notebook parsed as valid JSON. No 1.7B weights were downloaded and no T4 result was fabricated
  locally.

## 2026-11-25 — Qwen3-1.7B strict-format prompt amendment

- The first 1.7B inference preflight and first four budget-probe completions all derived the exact
  correct assignment for the same development puzzle, but emitted bare JSON after `</think>` rather
  than the required literal `<answer>...</answer>` wrapper. The strict parser therefore recorded
  `missing_answer_block`; these are interface failures, not demonstrated reasoning failures.
- Stop the initial prompt-v1 budget probe after four saved responses. Exclude its preflight and
  partial probe from calibration, training, gamma selection, and scientific results.
- Keep the parser and output contract unchanged. Strengthen the shared prompt to state explicitly
  that both literal tags are mandatory, bare JSON is invalid, the answer block follows reasoning,
  and the response ends immediately after `</answer>`.
- Version every revised artifact below `runs/qwen3_1_7b_t4_prompt_v2/` and
  `analysis/qwen3_1_7b_t4_prompt_v2/`. The notebook now refuses to begin the 72-response probe unless
  the one-response preflight both completes and passes strict parsing.
- This correction was made during development before budget selection, adapter training, protocol
  selection, frozen test creation, or final-test access. It changes neither gold answers nor the
  parser, reward, model, precision, sampling settings, or selection thresholds.

## 2026-11-25 — Nonredundant budget-evidence gate

- The completed prompt-v2 probe produced 72 responses directly at 4,096 tokens, with 69 correct,
  non-truncated responses, zero truncations, a correct-response p95 of 1,992 tokens, and a
  headroom requirement of 2,192 tokens. The predeclared candidate rule selected `B = 4096`, equal
  to the actual probe cap.
- A second 36-response run at the identical cap, model, prompt, development records, and sampling
  policy would not test a newly selected cap and would consume GPU time without resolving an
  observed uncertainty. Reuse the completed probe as the budget evidence and baseline source.
- Generalize the notebook rather than special-casing this result: reuse a completed source only
  when selected budget equals source budget and source truncation is within the predeclared 2%
  threshold. If the selected cap differs, retain the fresh validation run and the same stopping
  rule.
- This changes no prediction, threshold, budget, reward, model, training setting, or held-out data.
  It only removes redundant inference before the one-step training preflight.
