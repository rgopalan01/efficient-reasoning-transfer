# Efficient Reasoning and Cognitive Flexibility

This repository tests whether training a reasoning model to use fewer generated tokens preserves its ability to adapt when a familiar logical structure changes.

The authoritative T4 protocol compares independently initialized FP16 LoRA adapters on
`Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`:

1. exact-assignment reward only (`gamma = 0`), and
2. exact assignment minus a bounded completion-length cost (`0 < gamma < 1`).

Both conditions use the same Knights and Knaves data, prompt schedule, update budget, decoding policy, and evaluation families. Evaluation distinguishes new originals, surface-only changes, answer-changing premise mutations, and harder puzzles.

Earlier `Qwen/Qwen3-4B-Thinking-2507` L4 runs are retained as feasibility-pilot provenance only.
They are not mixed with the 1.7B budget calibration, gamma selection, training, or evaluation.

## Cognitive-science framing

Generated completion tokens are treated as a limited proxy for the cost of externalized reasoning. The comparison is motivated by bounded/resource rationality, proceduralization and skill acquisition, logical representation, transfer, and connectionist learning. Shorter outputs do not establish lower internal computational cost, and model behavior does not directly demonstrate a human cognitive mechanism.

The controlled variants help distinguish ordinary held-out accuracy from robustness to renamed surfaces and adaptation to changed premises. Reusing an obsolete assignment is an observable error pattern, not evidence of a specific hidden mechanism. Generated chain-of-thought is supplementary behavior rather than privileged access to cognition.

The project uses synthetic puzzles and public model weights. It has no human participants or personal data, so the proposed design does not require CITI human-subjects training.

## Repository layout

```text
configs/              original 4B configs plus isolated T4/FP16 1.7B configs
src/effreason/data.py deterministic generation, oracle, variants, audits
src/effreason/scoring.py strict answer parser and reward contract
src/effreason/modeling.py lazy model/tokenizer/LoRA loading and generation
src/effreason/train.py TRL GRPO adapter training and checkpoint resumption
src/effreason/evaluate.py resumable initial/adaptor evaluation
src/effreason/calibrate.py development-only budget and gamma selection
src/effreason/analyze.py CPU metrics, clustered uncertainty, and figures
tests/                focused scientific-integrity tests
notebooks/colab.ipynb authoritative stage-by-stage T4 launcher with inference/training preflights
experiments.md        chronological decisions and run log
```

Generated datasets, runs, checkpoints, model weights, notebook output, and analysis directories are ignored by Git.

## CPU setup and validation

Python 3.11 or newer is required. The local validation target is Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

ruff check .
pytest -m 'not gpu'
python -m effreason.data build --config configs/t4_pilot.yaml
python -m effreason.data audit --manifest data/generated/pilot/manifest.json
python -m effreason.train --config configs/t4_accuracy.yaml --phase pilot --dry-run
python -m effreason.train --config configs/t4_efficiency.yaml --phase pilot --gamma 0.1 --dry-run
python -m effreason.evaluate --config configs/t4_pilot.yaml --condition initial --split dev --dry-run
```

CPU-only imports do not import Torch, Transformers, TRL, PEFT, Accelerate, or Datasets. Dataset records contain solutions and AST metadata, but model prompts receive only the rendered puzzle and the shared output instruction.

## Dataset contract

Each JSONL record includes a schema version, stable example/family IDs, variant and parent links, split, rendered question, canonical-role AST, visible character mapping, exact Boolean solution, structural hash, difficulty metadata, generator provenance, and mutation provenance.

The local oracle enumerates all assignments and implements the exact supported operators. Structural canonicalization checks all role permutations, sorts only genuinely commutative children, and preserves implication direction. The builder rejects cross-family structural collisions; the audit blocks split leakage, non-unique solutions, broken parent links, invalid surface variants, and premise mutations whose answer did not change.

The checked-in pilot configuration deliberately creates a small development dataset. It does not freeze a final test set.

## Parser and reward

The parser examines only the region after the last native `</think>` marker when present. It requires exactly one `<answer>...</answer>` JSON object, rejects duplicate keys, and requires every puzzle name exactly once with a `knight` or `knave` label. A generated opening `<think>` is not required because the pinned Qwen tokenizer adds it through the chat template.

The reward is:

```text
R = exact_accuracy - gamma * completion_tokens / completion_budget
```

Completion length comes from generated token IDs, includes generated delimiters and EOS when present, and excludes the prompt and padding. The completion budget equals `max_new_tokens` and is fixed within matched comparisons. With `gamma < 1`, any correct response within the cap has higher individual reward than any incorrect response, though optimization can still reduce accuracy.

## Colab calibration and pilot sequence

Use `notebooks/colab.ipynb` as the authoritative launcher. It uses `configs/t4_*.yaml`, the pinned
1.7B revision, and FP16 for both inference and training. FP16 does not change if Colab happens to
allocate an L4 or A100. The strict-format prompt-v2 artifacts are isolated below
`runs/qwen3_1_7b_t4_prompt_v2/` and `analysis/qwen3_1_7b_t4_prompt_v2/`; never copy the 4B or
pre-amendment 1.7B budget, gamma, adapter, or evaluation into this lineage.

Run the notebook in order:

1. Clone/update the repository, install dependencies, audit the deterministic puzzle data, and
   verify CUDA memory.
2. Run a single-response inference preflight with `--max-records 1`; the notebook requires a
   strictly valid `<answer>...</answer>` block before allowing calibration to begin.
3. Collect the 72-response development budget probe and select `B`. When the selected cap is the
   probe cap and truncation already passes, reuse the probe as budget evidence; otherwise run fresh
   validation. Extend only censored responses if the 2% rule requires a larger cap.
4. Run a one-step GRPO memory preflight using microbatch 1 and four accumulation steps.
5. Train fresh accuracy-only and gamma 0.1, 0.2, and 0.4 five-step pilots from the same base
   revision.
6. Evaluate all four adapters on the same development items and select gamma using the predeclared
   accuracy, changed-premise accuracy, and token-reduction thresholds.
7. Save the selected development protocol and back up the artifacts.

The budget rule requires at least 50 baseline completions and 20 correct, non-truncated
completions. It recommends the smallest standard cap at or above 110% of the correct-response p95,
provided probe truncation is at most 2%. The gamma selector chooses the smallest candidate that
reduces tokens by at least 15% on matched jointly-correct responses while keeping overall and
changed-premise accuracy within five percentage points of the accuracy-only condition. A null
decision stops the workflow rather than tuning on test data.

The budget evidence gate avoids redundant GPU inference: a completed probe is sufficient when it
was generated at the selected cap and already passes the truncation threshold. Fresh validation is
reserved for a cap not directly tested by the selected source run. Baseline analysis is computed
from whichever run the gate selects.

Evaluation saves every response atomically and resumes an identical command by prediction key.
Training saves after every optimization step and uses `--resume auto`. A disconnected runtime may
need to reload the model, but completed responses and checkpoints remain in Drive.

Do not pass `--phase main --allow-main` until the protocol, dataset hashes, completion budget, gamma, difficulty, training steps, and seeds have been recorded and the final test data is frozen. The guard is intentional.

After creating a non-empty final test split and completing the audit, freeze it explicitly:

```bash
python -m effreason.data freeze --manifest data/generated/main/manifest.json
```

Frozen manifests cannot be replaced by `data build --force`. Main training rejects an unfrozen manifest, a missing test split, and the five-step pilot budget.

## Evaluation and analysis

Every prediction retains the raw completion, parsed answer, exact token count, correctness, finish reason, truncation, latency, reward components, and variant linkage. Evaluation resumes only when its configuration, dataset, split, prompt variant, condition, and adapter identity match.

Long-running commands emit Colab-friendly progress lines. Evaluation reports every completion with
completed/total counts, remaining work, token count, latency, elapsed time, and ETA. Dataset builds
report every accepted family, TRL reports every configured training step, and analysis announces
each bootstrap comparison and figure. Interrupting evaluation preserves all atomically written
predictions; rerun the identical command to resume them.

Pilot training saves a complete Trainer checkpoint after every optimization step. Every notebook
training command uses `--resume auto`: it resumes the highest valid `checkpoint-N`, starts at step
0 only when no completed checkpoint exists, and skips a completed run after verifying the final
adapter hash. An incomplete run refuses to restart without an explicit resume request, preventing
accidental loss of training progress. At most the currently unfinished step is lost when a runtime
stops. The two newest checkpoints are retained to limit Drive usage while preserving a fallback.

```bash
python -m effreason.analyze \
  --run-dirs runs/accuracy_seed11_test runs/efficiency_seed11_test \
  --output analysis/main
```

Analysis reports counts and exact accuracy, overall and correct-only length summaries, paired surface/premise success, premise adaptation conditional on original success, obsolete-assignment rates, validity diagnostics, and measured token/time cost. Difference intervals resample whole puzzle families with 10,000 bootstrap draws. With only two training seeds, interpret across-seed uncertainty cautiously.

For two or more matched runs, analysis also emits a reproducibly selected, condition-blinded `trace_sample.jsonl` and a predeclared `trace_codebook.json`. The labels remain unset until a human codes the traces; the software does not invent qualitative observations.

## Reproducibility and limitations

Run manifests capture configuration and dataset hashes, package versions, model revision,
timestamps, Git state, hardware, and status. Complete training checkpoints retain
trainer/optimizer/scheduler/RNG state; an exported adapter alone is not a resumable training
checkpoint. Colab runtime-local model caches and Python variables do not survive a runtime reset,
so setup may download the base model again and rerun the lightweight decision cells.

The current training dependencies are bounded provisional ranges. After the first successful T4
FP16 save/reload/resume smoke test, export one verified Colab lock and record CUDA, driver, GPU,
and storage measurements. Do not describe a dependency file as verified before that test.

The experiment reduces exact overlap with this fine-tuning dataset; it cannot prove that the pretrained model never encountered related puzzle structures. Preserved transfer is compatible with reusable reasoning but does not prove proceduralization. Failure to detect a meaningful accuracy difference is not evidence of equivalence when intervals are wide.

## Upstream attribution

- [Reasoning Gym](https://github.com/open-thought/reasoning-gym), Apache-2.0
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), Apache-2.0 (authoritative T4 protocol)
- [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507), Apache-2.0 (earlier feasibility pilot)
- [Hugging Face TRL](https://huggingface.co/docs/trl/grpo_trainer)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/developer_guides/lora)
