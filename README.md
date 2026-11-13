# Efficient Reasoning and Cognitive Flexibility

This repository tests whether training a reasoning model to use fewer generated tokens preserves its ability to adapt when a familiar logical structure changes.

The experiment compares independently initialized LoRA adapters on `Qwen/Qwen3-4B-Thinking-2507`:

1. exact-assignment reward only (`gamma = 0`), and
2. exact assignment minus a bounded completion-length cost (`0 < gamma < 1`).

Both conditions use the same Knights and Knaves data, prompt schedule, update budget, decoding policy, and evaluation families. Evaluation distinguishes new originals, surface-only changes, answer-changing premise mutations, and harder puzzles.

## Cognitive-science framing

Generated completion tokens are treated as a limited proxy for the cost of externalized reasoning. The comparison is motivated by bounded/resource rationality, proceduralization and skill acquisition, logical representation, transfer, and connectionist learning. Shorter outputs do not establish lower internal computational cost, and model behavior does not directly demonstrate a human cognitive mechanism.

The controlled variants help distinguish ordinary held-out accuracy from robustness to renamed surfaces and adaptation to changed premises. Reusing an obsolete assignment is an observable error pattern, not evidence of a specific hidden mechanism. Generated chain-of-thought is supplementary behavior rather than privileged access to cognition.

The project uses synthetic puzzles and public model weights. It has no human participants or personal data, so the proposed design does not require CITI human-subjects training.

## Repository layout

```text
configs/              complete pilot configurations
src/effreason/data.py deterministic generation, oracle, variants, audits
src/effreason/scoring.py strict answer parser and reward contract
src/effreason/modeling.py lazy model/tokenizer/LoRA loading and generation
src/effreason/train.py TRL GRPO adapter training and checkpoint resumption
src/effreason/evaluate.py resumable initial/adaptor evaluation
src/effreason/calibrate.py development-only budget and gamma selection
src/effreason/analyze.py CPU metrics, clustered uncertainty, and figures
tests/                focused scientific-integrity tests
notebooks/colab.ipynb thin, stage-by-stage A100 launcher
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
python -m effreason.data build --config configs/pilot.yaml
python -m effreason.data audit --manifest data/generated/pilot/manifest.json
python -m effreason.train --config configs/accuracy.yaml --phase pilot --dry-run
python -m effreason.train --config configs/efficiency.yaml --phase pilot --gamma 0.1 --dry-run
python -m effreason.evaluate --config configs/pilot.yaml --condition initial --split dev --dry-run
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

## A100 calibration and pilot sequence

Use `notebooks/colab.ipynb` as the authoritative launcher. It stops after development
calibration and does not expose or launch final test runs. Run each stage independently and
inspect its decision file before proceeding.

```bash
python -m pip install -e '.[train,dev]'
python -m effreason.data build --config configs/pilot.yaml
python -m effreason.data audit --manifest data/generated/pilot/manifest.json

python -m effreason.evaluate \
  --config configs/pilot.yaml \
  --condition initial_budget_probe \
  --split dev \
  --samples-per-example 4 \
  --output runs/initial_budget_probe

python -m effreason.calibrate budget \
  --run-dir runs/initial_budget_probe \
  --output analysis/calibration/budget_decision.json \
  --candidates 512 1024 2048 4096 8192
```

The budget rule requires at least 50 baseline completions and 20 correct, non-truncated
completions. It recommends the smallest standard cap at or above 110% of the correct-response
p95, provided the probe truncation rate is at most 2%. Otherwise it explicitly requests more
samples or a larger probe. Load `recommended_budget` as `B`, then validate it with a fresh
development evaluation before training.

```bash
SELECTED_B=1024  # replace with budget_decision.json recommended_budget

python -m effreason.evaluate \
  --config configs/pilot.yaml \
  --condition initial_calibrated \
  --split dev \
  --samples-per-example 2 \
  --completion-budget $SELECTED_B \
  --output runs/initial_calibrated_dev

python -m effreason.train \
  --config configs/accuracy.yaml \
  --phase pilot \
  --completion-budget $SELECTED_B \
  --seed 11 \
  --output runs/accuracy_seed11_pilot

python -m effreason.train \
  --config configs/efficiency.yaml \
  --phase pilot \
  --gamma 0.1 \
  --completion-budget $SELECTED_B \
  --seed 11 \
  --output runs/efficiency_g01_seed11_pilot

python -m effreason.train \
  --config configs/efficiency.yaml \
  --phase pilot \
  --gamma 0.2 \
  --completion-budget $SELECTED_B \
  --seed 11 \
  --output runs/efficiency_g02_seed11_pilot
```

Evaluate all three adapters with the same `B`, passing the matching gamma explicitly. The gamma
selector chooses the smallest candidate that reduces mean tokens by at least 15% on matched
completions that both conditions answered correctly, while keeping overall and changed-premise
accuracy within five percentage points of the accuracy-only condition. These thresholds are
predeclared pilot defaults, not post-hoc choices.

```bash
python -m effreason.calibrate gamma \
  --reference-run runs/accuracy_seed11_pilot_eval \
  --candidate-runs runs/efficiency_g01_seed11_pilot_eval runs/efficiency_g02_seed11_pilot_eval \
  --output analysis/calibration/gamma_decision.json \
  --accuracy-margin-pp 5 \
  --premise-margin-pp 5 \
  --minimum-token-reduction-percent 15

python -m effreason.calibrate protocol \
  --budget-decision analysis/calibration/budget_decision.json \
  --gamma-decision analysis/calibration/gamma_decision.json \
  --output analysis/calibration/selected_protocol.yaml
```

If no candidate satisfies the rule, the selector records a null decision instead of silently
choosing the best-looking result. Record the development decision in `experiments.md` before
creating the main configuration.

Do not pass `--phase main --allow-main` until the protocol, dataset hashes, completion budget, gamma, difficulty, training steps, and seeds have been recorded and the final test data is frozen. The guard is intentional.

After creating a non-empty final test split and completing the audit, freeze it explicitly:

```bash
python -m effreason.data freeze --manifest data/generated/main/manifest.json
```

Frozen manifests cannot be replaced by `data build --force`. Main training rejects an unfrozen manifest, a missing test split, and the five-step pilot budget.

## Evaluation and analysis

Every prediction retains the raw completion, parsed answer, exact token count, correctness, finish reason, truncation, latency, reward components, and variant linkage. Evaluation resumes only when its configuration, dataset, split, prompt variant, condition, and adapter identity match.

```bash
python -m effreason.analyze \
  --run-dirs runs/accuracy_seed11_test runs/efficiency_seed11_test \
  --output analysis/main
```

Analysis reports counts and exact accuracy, overall and correct-only length summaries, paired surface/premise success, premise adaptation conditional on original success, obsolete-assignment rates, validity diagnostics, and measured token/time cost. Difference intervals resample whole puzzle families with 10,000 bootstrap draws. With only two training seeds, interpret across-seed uncertainty cautiously.

For two or more matched runs, analysis also emits a reproducibly selected, condition-blinded `trace_sample.jsonl` and a predeclared `trace_codebook.json`. The labels remain unset until a human codes the traces; the software does not invent qualitative observations.

## Reproducibility and limitations

Run manifests capture configuration and dataset hashes, package versions, model revision, timestamps, Git state, hardware, and status. Complete training checkpoints retain trainer/optimizer/scheduler/RNG state; an exported adapter alone is not a resumable training checkpoint.

The current training dependencies are bounded provisional ranges. After the first successful A100 save/reload/resume smoke test, export one verified Colab lock and record CUDA, driver, GPU, and storage measurements. Do not describe a dependency file as verified before that test.

The experiment reduces exact overlap with this fine-tuning dataset; it cannot prove that the pretrained model never encountered related puzzle structures. Preserved transfer is compatible with reusable reasoning but does not prove proceduralization. Failure to detect a meaningful accuracy difference is not evidence of equivalence when intervals are wide.

## Upstream attribution

- [Reasoning Gym](https://github.com/open-thought/reasoning-gym), Apache-2.0
- [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507), Apache-2.0
- [Hugging Face TRL](https://huggingface.co/docs/trl/grpo_trainer)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/developer_guides/lora)
