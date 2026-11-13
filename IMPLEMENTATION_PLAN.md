# Efficient Reasoning and Cognitive Flexibility

Updated: October 24, 2026. This revision supersedes the earlier, larger repository layout.

## Handoff to the coding agent

Build a small, reproducible post-training research repository for a Georgia Tech cognitive science project. This document is the implementation specification, not a report of completed experiments. No model training or performance measurements have been completed yet.

**Research question:** Does training a language model to solve logic problems with fewer generated tokens preserve its ability to adapt when familiar problem structures change?

**Chosen model:** `Qwen/Qwen3-4B-Thinking-2507`.

**Task:** Reasoning Gym's `knights_knaves` generator, wrapped with strict scoring and controlled evaluation variants.

**Method:** Reinforcement learning with verifiable rewards (RLVR), using Hugging Face TRL's GRPO trainer and PEFT LoRA adapters.

**Hardware:** One Google Colab A100. Detect actual VRAM and available storage; do not assume 80 GB. The first working configuration should target a 40 GB device. All resource and runtime claims require measurement.

**Deliverable:** A GitHub-ready Python package, thin Colab launcher, reproducible experiment configurations, validated evaluation data, trained adapters when experiments are run, and an evidence-based report. The repository should demonstrate post-training engineering and experimental judgment without becoming a general-purpose framework.

Implement in milestones. Complete the CPU pipeline and a small GPU pilot before scheduling full training. If hardware is unavailable, finish CPU-verifiable work and document the exact GPU command that remains untested. Never substitute fabricated results.

## 1. Why this project exists

The cognitive question concerns the relationship between learned efficiency and flexibility. A model might become shorter by removing redundant wording, learning a better strategy, skipping useful checks, or guessing early. Token reduction alone cannot distinguish these explanations.

We therefore evaluate behavior under controlled problem changes:

- New problems from the training difficulty range test ordinary generalization.
- Changes in names or wording test robustness when the underlying logic is preserved.
- Changes to a premise test whether the model adapts when the correct answer must change.
- Harder problems test transfer beyond the training difficulty range, separately from premise sensitivity.

This connects to bounded/resource rationality, skill acquisition and proceduralization, transfer, and Thagard's treatment of representations, rules, logic, and connectionism. The model already has reasoning capabilities. We study how an effort-sensitive training objective modifies those capabilities.

Interpretation boundaries:

- Generated tokens proxy the cost of externalized reasoning, not human working-memory capacity or all internal computation.
- Preserved transfer with shorter outputs is compatible with reusable approaches, but does not prove proceduralization.
- Reusing an obsolete answer on a changed puzzle is an observable error pattern, not proof of a particular internal mechanism.
- Chain-of-thought is supplementary behavioral output, not guaranteed access to internal reasoning.
- A cognitive cliff is exploratory. The main comparison cannot establish a threshold curve. Satisficing would require a more explicit stopping/acceptance design and is not a primary claim.

The project is successful if it produces an interpretable comparison, even if the length penalty harms performance or offers no advantage.

## 2. Scope and engineering principles

### Required for version 1

1. One model, one task family, two training objectives.
2. Deterministic dataset construction with an exact logical oracle.
3. One shared parser and scorer for training and evaluation.
4. Baseline evaluation, paired transfer evaluation, and a concise-prompt control.
5. Adapter training, resumable checkpoints, and explicit experiment provenance.
6. Local JSONL/CSV artifacts and reproducible analysis figures.
7. Targeted local CPU tests and a documented single-GPU smoke test. CI is optional later.

### Defer until evidence justifies them

- Additional models or task families; SFT warm-starts; multiple RL algorithms.
- Prime Intellect Verifiers, hosted training, distributed training, or a custom rollout service.
- Hidden-state probes, mechanistic claims, LLM judges, or an agent/tool environment.
- Exhaustive penalty sweeps, automatic curriculum learning, and elaborate dashboards.

Reasoning Gym supplies generation and task metadata; TRL performs optimization. Verifiers is optional and unnecessary for the initial single-turn experiment.

Prefer small functions and typed data records. Keep task logic independent of PyTorch and TRL. Do not implement a custom GRPO optimizer, plugin registry, deep class hierarchy, or generic benchmark framework. Add abstraction only when two real uses need it.

## 3. Minimal repository layout

Use the working repository name `efficient-reasoning-transfer` and Python package `effreason`.

```text
README.md
IMPLEMENTATION_PLAN.md
pyproject.toml
.gitignore
experiments.md
configs/
  pilot.yaml
  accuracy.yaml
  efficiency.yaml
src/effreason/
  __init__.py
  data.py
  scoring.py
  modeling.py
  train.py
  evaluate.py
  analyze.py
  utils.py
tests/
  test_data.py
  test_scoring.py
  test_analysis.py
notebooks/
  colab.ipynb
```

Start with this structure. Do not create empty modules or extra infrastructure for hypothetical future needs. Split a module only when actual complexity warrants it.

| File | Responsibility |
|---|---|
| `README.md` | Research question, setup, finalized experimental protocol, reproduction commands, results and limitations. |
| `IMPLEMENTATION_PLAN.md` | This implementation handoff; do not duplicate it into several design documents. |
| `pyproject.toml` | Package metadata, direct dependencies, optional training/development dependencies and lint/test settings. |
| `.gitignore` | Exclude generated datasets, large logs, checkpoints, caches and secrets. |
| `experiments.md` | Short chronological log of runs, findings, decisions and protocol revisions. |
| `data.py` | Generator adapter, exact solver, rendering, mutations, canonicalization, split construction and audits. Keep this CPU-only. |
| `scoring.py` | Parse final assignments, compute strict correctness and reward components, classify errors. CPU-only and shared by training/evaluation. |
| `modeling.py` | Load pinned model/tokenizer, apply native chat formatting, configure/load adapters and generation. |
| `train.py` | Convert data into the trainer format and configure TRL. Do not reimplement GRPO. |
| `evaluate.py` | Evaluate the original model or an adapter, invoke shared scoring and save outputs. |
| `analyze.py` | Read saved outputs on CPU and produce metrics, paired comparisons, uncertainty and figures. |
| `utils.py` | A few shared functions for YAML validation, seeds, hashes and JSON/file writing. No framework or unrelated utilities. |
| `__init__.py` | Mark the package; no eager GPU initialization. |

Use ordinary functions, dataclasses and argparse. Each runnable module gets a small `main()` function; invoke it with `python -m effreason.<module>`. No separate `cli.py`, `config.py`, `task.py` or `artifacts.py` is required initially.

The three YAML files hold complete, small configurations using one shared schema. Reuse their evaluation fields when evaluating the corresponding condition. Use explicit overrides for data split, checkpoint, prompt variant and output directory. Do not add a separate evaluation config unless it later removes real duplication.

**Optional later:** a short `AGENTS.md` if coding-agent conventions need it, a tested Colab lock file after the GPU setup works, and GitHub Actions after local tests are stable. Do not create a `docs/` hierarchy, a custom CLI entry point, an artifact manager, or an automatic sweep runner. Example fixtures may live in tests; generated datasets and figures are created only when used.

## 4. Dependencies and reproducibility

Core CPU dependencies: Reasoning Gym, PyYAML, NumPy, pandas, matplotlib, and pytest/ruff for development. Training dependencies: PyTorch, Transformers, TRL, PEFT, Accelerate, and datasets. Keep GPU imports lazy so CPU checks do not download a model or require CUDA.

Start with ordinary Transformers generation and BF16 LoRA. Add bitsandbytes for QLoRA only if needed. Add vLLM only after profiling demonstrates that generation throughput is a bottleneck worth the integration work.

Before implementation, inspect the installed versions and the relevant upstream code. Pin a tested compatible dependency set after the first successful GPU smoke test. Do not assume the newest releases interoperate. Record:

- Reasoning Gym version and source revision; retain its license and attribution if copying any code.
- Exact model and tokenizer revision IDs, including the chat template hash.
- Python, PyTorch, Transformers, TRL, PEFT, CUDA and driver versions.
- GPU model/VRAM, package freeze, repository commit, and uncommitted changes or patch hash.

Keep direct dependencies in `pyproject.toml`. Save the actual environment/package versions with runs from the outset. Once a GPU smoke test passes, export one tested `requirements-colab.lock.txt` (or an equivalent lock) and document how it is generated. Avoid manually maintaining multiple competing dependency lists. Document any PyTorch CUDA-index installation step separately. Do not label an untested dependency file as a verified lock.

## 5. Dataset and logical oracle

### 5.1 Data records

Use JSONL for generated datasets. Each `PuzzleRecord` should contain:

| Field | Meaning |
|---|---|
| `schema_version`, `example_id` | Versioned schema and stable identifier. |
| `family_id`, `variant_type`, `parent_id` | Link original/surface/premise examples; original has no parent. |
| `split` | `train`, `dev`, `test`, or `test_harder`. |
| `question` | The complete user-visible puzzle text. |
| `statements_ast` | Logical statements indexed by canonical character roles. |
| `characters` | Explicit mapping between canonical roles and visible names. |
| `solution` | Unique Boolean knight/knave assignment, keyed by canonical role. |
| `structural_hash` | Canonical structure hash used for overlap checks. |
| `difficulty` | Character count, depth, width, AST size, operator counts. |
| `generation` | Source seed/index, generator revision and configuration. |
| `mutation` | Edited statement/path/operator or rendering variant, if applicable. |

The model receives only the question and output instructions. Never include the solution, AST, parent answer, or correctness metadata in the prompt.

### 5.2 Strict oracle

For a truth assignment `t`, character `i` is consistent when `t[i] == evaluate(statement[i], t)`. Enumerate all `2**n` assignments for these small puzzles and retain only puzzles with exactly one solution.

Inspect Reasoning Gym's pinned `knights_knaves` implementation: native metadata and partial-credit scoring may differ across versions. Reuse its generator, but make experiment correctness binary. Never use a substring match or partial character credit as the primary reward.

Implement a small, auditable pure-Python evaluator for the actual supported AST operators. Cross-check it against the upstream solver on a generated sample and hand-checked fixtures. Do not use Python `eval` on statements. Put bounded retries on generation and mutation; log rejection reasons rather than hanging.

### 5.3 Structural separation

Different random seeds or names do not establish independent logical problems. For the small character counts in this study, canonicalize over character permutations: rename all variable references, reorder speakers consistently, serialize each candidate, and retain the lexicographically minimal serialization. Normalize child ordering for genuinely commutative operators. Preserve direction for implication.

This detects structural duplicates under the transformations implemented; it does not prove that every logically equivalent formula or familiar subpattern has been excluded. Document that limit.

The pretrained model may already have encountered Knights and Knaves or related logical patterns. Newly generated test instances reduce exact instance overlap with our fine-tuning data; they do not establish absence from pretraining or novelty of the reasoning skill.

Create original-plus-variant families before final split assignment. Assign all members of a family to one split. Connect families sharing any structural hash and keep each connected group in one split, or reject conflicting families deterministically. Audit all resulting train/dev/test overlaps, including premise variants, not only originals. Exact surface variants intentionally share structure within a family.

Save immutable dataset manifests and hashes. Freeze final test data before main training; use separate development data throughout pilots. Do not regenerate a favorable test set after seeing model results. Small puzzles can exhaust available structures; report this and reduce counts or adjust the development difficulty before freezing instead of accepting duplicates.

### 5.4 Controlled evaluation families

| Variant | Construction | Required check |
|---|---|---|
| Original | New valid puzzle at the selected difficulty. | Unique solution; no forbidden split overlap. |
| Surface | Rename characters and/or use a small set of manually verified templates. | Same AST and mapped solution. |
| Premise | Change one supported logical atom/operator in the AST; render anew. | Unique solution that differs from the original assignment. |
| Harder | Independently generated at greater character count or depth. | Unique solution; difficulty metadata and separate reporting. |

Start with one surface and one premise variant per original. Do not use an LLM to paraphrase or label puzzles in version 1. Keep ordinary `knight`/`knave` terminology in the initial wrapper so changes in synonym parsing do not confound the study.

Keep character count fixed within premise pairs and preserve AST size/operator profile where practical. Record remaining complexity changes and mutation rejection rates. Restrict pair-based claims to the subset of puzzles with valid mutations. Changed-premise puzzles are not automatically equal in difficulty: compare both trained conditions on the exact same examples, and use the initial model to contextualize any inherent difficulty difference.

## 6. Prompt, parser, and reward contract

### 6.1 Prompt and final-answer format

Use the checkpoint's native chat template. This exact model is thinking-only; its template supplies the opening `<think>` prefix. The generated continuation may therefore contain a closing `</think>` without a generated opening tag. Do not reject that valid behavior or add a second thinking prefix.

Use one concise output instruction shared by all primary conditions, for example:

> Solve the puzzle. End your response with one answer block assigning every named character: `<answer>{"Alice":"knight","Bob":"knave"}</answer>`. Use the character names in this puzzle.

The example is a syntax illustration; use it consistently and check for answer-pattern bias during development. Prefer a schema description without a fixed sample assignment if the example proves distracting.

Parsing rules:

- Extract one final answer block from the final-answer region, after the native thinking boundary when present.
- Do not search speculative assignments inside the reasoning for a correct answer.
- Require every expected name exactly once, valid role labels, and no unknown names. Reject duplicate JSON keys explicitly; standard `json.loads` alone silently overwrites them.
- Allow documented whitespace and capitalization normalization. Reject multiple conflicting answer blocks and malformed or incomplete output.
- Do not require a generated opening thinking tag. Missing thinking delimiters should be a separately logged event, with a documented fallback for an otherwise unambiguous final answer.
- Do not provide partial reward for valid JSON or thinking tags.

If format failures dominate the pilot, improve the shared prompt/parser before freezing. Do not silently add an SFT stage or change the parser for one condition.

### 6.2 Reward

Use the pitch's objective:

```math
R = A - gamma * L / B
```

- `A`: 1 for an exact complete assignment, otherwise 0.
- `L`: actual generated completion-token count, including reasoning, final answer and generated delimiters; exclude prompt and padding. Count EOS consistently when generated.
- `B`: maximum completion length selected after inspecting original-model behavior on development data, then fixed for the main comparison; validate `0 <= L <= B`.
- Accuracy condition: `gamma = 0`.
- Efficiency condition: one moderate `0 < gamma < 1`, selected on development data.

Use generated token IDs and true generation lengths/masks. Do not estimate tokens from characters or re-tokenized decoded text. When PAD and EOS share an ID, use sequence boundaries/masks rather than deleting every occurrence of that ID.

Keep `accuracy`, `length_fraction`, `length_penalty`, and `total_reward` separately in logs. At gamma below 1, every correct completion has higher individual reward than every incorrect completion within the cap. This does not guarantee optimization will preserve accuracy: short guessing can still be an attractive policy under uncertainty.

**Why normalize by B:** `L/B` is between zero and one, so gamma is the maximum penalty. Equivalently the per-token cost is `gamma/B`. Changing B therefore changes both the generation allowance and the penalty slope. Never update B dynamically from a batch's mean/maximum length or separately for different conditions.

**Penalty selection:** Use a small development-only comparison, initially gamma candidates 0.1 and 0.2 versus gamma 0, with matched short pilot budgets. This is not a large sweep. Before selection, record the acceptable development-accuracy loss and desired token reduction; account for uncertainty and premise-transfer results. Choose one moderate value for the main experiment, then keep it fixed throughout each run and across seeds. If neither candidate produces a useful tradeoff, document that and revise during development rather than selecting on the final test. No penalty schedule is planned.

Truncated output without a complete answer is incorrect. A complete answer that reaches the cap can be scored but must retain a truncation flag. Never silently discard truncated examples from evaluation.

### 6.3 GRPO-specific controls

Use TRL's supported reward callback and completion-token inputs for the pinned version. Keep the reward function pure; the callback should only adapt batch shape and return floats.

- Explicitly select `loss_type="dr_grpo"` if supported by the pinned version. Its constant normalization avoids the original objective's per-response-length normalization issue; it does not eliminate every possible training bias.
- Start with `scale_rewards=False` if supported, to keep reward magnitudes interpretable. Group standard-deviation scaling can amplify small length differences in all-correct groups, complicating interpretation of gamma. Record the choice and keep it fixed.
- Set KL coefficient, clipping, optimizer, learning rate, warmup, and truncation-mask behavior explicitly. Do not rely on changing library defaults. An initial `beta=0` pilot is acceptable and avoids a separate reference-model requirement; if instability motivates KL, make that a documented development decision applied to both conditions.
- Log proportions of all-correct, all-wrong and mixed-correctness rollout groups. Accuracy-only groups with identical rewards provide no group-relative learning signal; the length objective can still create differences in those groups.
- Confirm that example metadata aligns with repeated completions. Add a small integration fixture with deliberately different gold answers to detect misalignment.

Do not introduce hidden length shaping, format bonuses, or scheduled penalties in the primary experiment.

### 6.4 What the training code updates

The language model is the policy: it assigns probabilities to next tokens given the puzzle and generated prefix. For each puzzle, sample G full responses, compute their rewards, and let TRL calculate group-relative advantages and its policy loss. The verifier/reward is not differentiated. Its fixed numerical feedback weights gradients of the model's response log probabilities.

Freeze all original model parameters. Initially attach trainable rank-16 LoRA matrices to attention query, key, value and output projections, after checking their actual module names. Original attention weights, MLP weights, embeddings, normalization parameters and the output head remain frozen. Verify and log trainable names/counts. Gradients can pass through frozen computations to reach adapters. Use PEFT's supported initialization and saving/loading rather than custom adapter code.

Start with G=4 responses per puzzle, not four responses for the whole experiment. More responses can improve the chance of mixed correct/incorrect groups and provide a richer comparison, but increase generation cost and may reduce the number of distinct puzzles affordable. Inspect mixed-group frequency and throughput in development; consider G=8 only if justified. Freeze G across main conditions. Do not confuse changing concurrent generation batch size with changing the number of responses per puzzle.


## 7. Experimental conditions and configuration

| ID | Checkpoint | Training reward | Evaluation prompt |
|---|---|---|---|
| `initial` | Pinned original Qwen checkpoint | None | Standard |
| `accuracy` | LoRA trained from original | Exact accuracy | Standard |
| `efficiency` | Independently trained LoRA from original | Accuracy minus length cost | Standard |
| `accuracy_concise` | Same trained adapter as `accuracy` | No additional training | Standard plus a fixed concise-reasoning instruction |

The `initial` condition means the original publicly released checkpoint before **our** training. It is already a post-trained reasoning model, not a pretraining-only model.

**No SFT stage is planned.** Consider a shared SFT warm-start only if the baseline/prompt pilots reveal a concrete problem such as persistently unusable answer formatting or extremely sparse correct solutions. Record that as a design change. If adopted, evaluate the SFT checkpoint separately and initialize both GRPO objectives from that same SFT checkpoint, using training-only demonstrations. Never add SFT to only one branch.

The concise control changes wording, not the native thinking mode or output cap. It is an inexpensive control for whether prompting alone produces similar savings.

Plan two training seeds per trained condition: four main training runs. Pair seeds across objectives for initialization and prompt scheduling, while acknowledging sampled outputs will diverge. Reinitialize adapters from the same starting setup for each run. Never initialize efficiency training from an already accuracy-trained adapter in the main comparison.

Keep data, prompt sequence, adapter configuration, optimizer-step count, completions per prompt, decoding, maximum lengths, and evaluation sets matched. Equal update budgets do not imply equal token consumption or wall time; measure both. Use final scheduled checkpoints as the primary result. Any checkpoint selection based on development performance must use the same rule for both objectives and be declared in advance.

### Configuration shape

Use one validated schema. A run loads one YAML file plus explicit command-line overrides; no multi-level inheritance system is needed. Reject unknown keys. Save the full resolved configuration.

The following is an illustrative schema, not a tested TRL constructor:

```yaml
experiment:
  name: efficiency
  phase: pilot
  seed: 11
model:
  id: Qwen/Qwen3-4B-Thinking-2507
  revision: RESOLVE_AND_PIN
  dtype: bfloat16
  quantization: none
adapter:
  method: lora
  rank: 16
  alpha: 32
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj]
data:
  train_manifest: data/generated/pilot_train_manifest.json
  eval_manifest: data/generated/dev_manifest.json
reward:
  gamma: 0.2
generation:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  max_new_tokens: 4096
training:
  num_generations: 4
  loss_type: dr_grpo
  scale_rewards: false
  beta: 0.0
  learning_rate: 0.000005
  max_steps: 5
  gradient_checkpointing: true
artifacts:
  output_dir: runs/efficiency_seed11_pilot
```

The numeric values above are pilot candidates, not established optimal hyperparameters. Validate adapter target names on the loaded architecture. Record trainable parameter count. Define effective batch size in **unique prompts and generated completions**, and ensure the pinned trainer's batch/divisibility requirements are met. Do not blindly set microbatch size to one without checking generation grouping semantics.

For a main run, reject unresolved revision IDs, absent dataset hashes, or `phase=pilot` settings passed as a claimed frozen experiment. Record the finalized protocol in the README and its configuration/data hashes in run metadata so accidental changes are visible; a custom protocol-lock subsystem is unnecessary.

## 8. Baseline pilot and compute budget

The selected model can generate long reasoning. Its official card recommends much longer output allowances for broad reasoning benchmarks than a small Colab experiment can casually afford. A 4,096-token cap is only a resource probe, not proof that the model has had enough time to reason.

Every stage must be independently runnable and inspectable. No command should automatically launch baseline evaluation, both training branches and final evaluation in one sequence. Scripts should save their outputs and exit; the user can examine findings and choose when to run the next stage. A notebook is a thin launcher, not the only place experiment state exists.

Use this progression:

1. **CPU audit:** Validate hand-checked/generated fixtures, parser behavior, split separation and mutation correctness.
2. **Original-model inference smoke:** Run 5–10 development puzzles and inspect outputs, token boundaries, template behavior, memory and runtime.
3. **Original-model baseline pilot:** Evaluate roughly 60–100 original development puzzles across a few difficulties and a smaller paired subset. Report mean, median, p90/p95 length, correctness and truncation overall and by difficulty. Probe a 4,096-token cap; compare a larger allowance such as 8,192 on a small subset if truncation is material. Capped lengths underestimate natural response lengths, so do not set B equal to the observed mean. Choose a provisional B using completion/truncation behavior, accuracy sensitivity and compute availability.
4. **Accuracy-only pilot:** First run about 5 optimizer steps as a smoke test, including save/reload/resume. Then run a short development comparison to check learning signal, length drift and stability. Pilot training examples must be disjoint from the development examples used to judge the run.
5. **Accuracy-plus-length pilots:** Start each candidate independently from the original checkpoint. Evaluate the small gamma comparison described above. Use the same provisional B, data, update budget and other shared settings as the accuracy pilot. Inspect correctness, transfer, length, reward variation and runtime.
6. **Freeze the main protocol:** Record the selected B, gamma, difficulty, data sizes, rollout count, adapter/trainer settings, budgets, seeds and analysis rules. Finalize and audit held-out datasets without selecting examples based on model outcomes.
7. **Main runs, executed separately:** Reinitialize from the common original checkpoint for each objective and seed, using final settings. Pilots normally remain development evidence rather than final runs. Run final original-model evaluation and each trained/control evaluation independently with the same frozen evaluation settings.
8. **Analysis:** Recompute comparisons from saved predictions on CPU without rerunning inference.

**Changing settings between stages:** Development changes are expected. Record each in `experiments.md`. If B, data, prompt/parser, rollout count, learning rate or another shared setting changes, rerun any comparison needed under matched settings; do not compare an old accuracy run with a newly configured efficiency run as if only the reward changed. Keep superseded pilot outputs clearly labeled. Once final-test results have been examined, any redesign is exploratory or requires a new untouched confirmation set.

Changing the original-model pilot cap does not require retraining anything. Changing the main cap after a training pilot can require repeating that pilot because both generation and reward normalization changed. B stays fixed inside each finalized run and across the matched main conditions.

Choose difficulty that gives meaningful initial success and meaningful errors. A rough 30–80% band is a development heuristic, not a scientific requirement. Check success across multiple sampled completions: sparse mixed-correctness groups can impede GRPO even if aggregate accuracy looks acceptable.

Provisional data sizes: 500–2,000 unique training structures, 60–100 development originals, 100–200 held-out originals each with two variants, and 50–100 harder puzzles. Adjust before freezing based on structural availability and measured compute. Do not promise these counts before auditing feasibility.

Budget calculation must include initial evaluation, four training runs, all checkpoint/control evaluations, and reruns. Estimate GPU hours from measured seconds per training step and evaluation example, with headroom for startup and variability. A pilot range of 50–150 optimizer steps is a planning candidate only; choose a meaningful exposure budget based on the learning curves and actual prompt batch size.

If memory or runtime is excessive, in order:

1. Reduce microbatch/concurrent generation and use gradient accumulation while preserving grouping.
2. Use gradient checkpointing and an available supported attention implementation.
3. Reduce exploratory sweep breadth and evaluation counts while preserving paired design.
4. Consider QLoRA, documenting the change and applying it to both training conditions and their comparable starting baseline.
5. Consider an optimized rollout backend only with a measured benefit and a correctness check.

Do not silently lower the token cap until outputs are systematically cut off. If the chosen model cannot finish enough tractable puzzles within the practical budget, report the incompatibility and propose an explicit change.

## 9. Evaluation and analysis

### 9.1 Execution

- Every example is a new conversation. No original answer or neighboring puzzle is supplied.
- Use the same held-out examples and decoding policy for all conditions.
- Begin with one sampled completion per example and fixed evaluation seeds. More samples are optional; do not silently convert pass@1 into best-of-N performance.
- Separate sampling variability from training-seed variability. Matching RNG seeds does not guarantee identical random draws across models/backends.
- Shuffle or interleave variant order for execution and retain IDs for pairing.
- Store raw outputs before analysis so all scores can be recomputed without generation.
- Infrastructure exceptions are not wrong answers. Log them and resume the affected examples. Model format errors and truncation remain behavioral outcomes.

### 9.2 Required metrics

| Metric | Definition / interpretation |
|---|---|
| Exact accuracy | Correct complete assignments divided by attempted model responses; show counts. |
| Completion length | Mean, median, and upper quantile of generated tokens, overall and on correct answers. |
| Surface pair success | Fraction of families with both original and surface variant correct. |
| Premise pair success | Fraction of families with both original and changed-premise variant correct. |
| Adaptation conditional on original success | Correct changed-premise answer among families whose original was correct; report denominator and selection caveat. |
| Obsolete assignment rate | Changed-premise response equals the original gold assignment, although the new gold differs; report unconditional and valid-parse rates. |
| Validity diagnostics | Malformed answers, missing roles, conflicting blocks, repeated outputs, and truncation. |
| Cost | Actual generated tokens, wall/GPU time, and peak memory with the measurement method stated. |

Primary comparisons: efficiency versus accuracy training on held-out exact accuracy, token usage, and premise pair success. Report percentage-point accuracy differences and percentage token reduction separately, rather than a single opaque composite score.

Before final evaluation, state what accuracy/transfer loss would be practically meaningful, for example a proposed five-percentage-point margin justified against the pilot and course scope. Report interval estimates against that margin; failure to find a significant difference is not evidence of equivalence. If the sample is too small to support a preservation claim, say the estimate is inconclusive. Length measured only on correct responses compares potentially different problem subsets, so also report overall length and, optionally, length on the shared subset both conditions solve.

Compute a family-level contrast between changes in premise and surface performance across the two trained conditions. A disproportionately worse premise result is consistent with reduced flexibility, but does not establish a human mechanism or guarantee that the perturbations have equal difficulty.

Show initial-checkpoint performance to contextualize learning. The concise-prompt control helps distinguish gains obtainable through instruction from those associated with training.

### 9.3 Uncertainty and plots

Report each training seed separately. Bootstrap differences by resampling entire puzzle families with all variants together. Cluster connected families if they share structure. Do not treat variants as independent examples or many test puzzles as substitutes for independent training runs. With only two training seeds, avoid strong claims about across-training-seed statistical significance.

Required figures:

1. Accuracy versus mean completion length, with conditions labeled and training seeds visible.
2. Accuracy and paired success by evaluation type.
3. Training curves for correctness, completion length, total reward, and truncation.

Optional: compact error table and a few representative original/variant responses. A plot with two penalty strengths is not evidence of a capacity threshold.

### 9.4 Trace inspection

Select a small reproducible sample, for example 20–30 families stratified by success/error pattern. Use the same selection rule for each condition and blind condition names where practical. Code explicit assignment enumeration, consistency checking, self-correction, repetition, and unsupported final jumps. Define labels before inspecting the full sample. Report descriptive observations; no automated LLM judge is needed.

## 10. Run artifacts and resumption

Create only artifacts appropriate to the stage:

```text
runs/<run_id>/
  config.resolved.yaml     # exact settings, including overrides
  manifest.json            # provenance, hardware, status and dataset hashes
  metrics.jsonl            # training runs only
  predictions.jsonl        # evaluation runs only
  checkpoints/             # training state, managed by the trainer
  adapter/                 # final adapter export when training completes
```

A bounded sample of training rollouts is optional when needed for debugging. Summary JSON, tables and figures are derived outputs from analysis and need not be written by every command. Do not build a database or custom artifact lifecycle manager. The evaluation manifest must link to the exact base checkpoint or adapter it evaluated.

The manifest includes run ID, condition, seeds, git state, dependency/model revisions, dataset and prompt hashes, timestamps, hardware, and status (`started`, `interrupted`, `completed`, `failed`). Prediction records include example/family IDs, adapter checkpoint hash, raw completion, parsed assignment, token count, correctness, reward components, finish reason, and latency.

Save training metrics at a modest fixed interval and complete checkpoints at practical intervals and run end. A resumable checkpoint needs optimizer/scheduler/RNG/trainer state; an adapter export alone is insufficient. Test resume with a short interrupted smoke run. If exact resumed equivalence is unavailable, document the source of nondeterminism.

For evaluation, resume by stable `(run_id, example_id, sample_id)` keys with config/data hash checks. Do not append incompatible results to an old run. Write checkpoint manifests atomically and distinguish complete from partial copies.

Colab notebook responsibilities: environment setup, hardware report, repository checkout at a recorded commit, configuration selection, separate module invocations, and artifact backup to a user-selected persistent location. Train against local Colab disk; copy complete checkpoints to persistent storage periodically. Avoid keeping critical state only on an ephemeral VM. Keep credentials out of notebooks, logs, and Git.

## 11. Independent commands and developer workflow

Use `python -m` entry points with lightweight argparse handling inside each runnable module. The following are target interfaces to implement, not commands that currently exist:

```bash
# CPU setup and data validation
python -m pip install -e '.[dev]'
python -m effreason.data build --config configs/pilot.yaml
python -m effreason.data audit --manifest data/generated/dev_manifest.json

# Original checkpoint only: writes a baseline report, then exits
python -m effreason.evaluate --config configs/pilot.yaml --condition initial --split dev --output runs/initial_pilot

# Run these separately after inspecting the baseline and choosing provisional B
python -m effreason.train --config configs/accuracy.yaml --phase pilot --seed 11 --output runs/accuracy_pilot
python -m effreason.evaluate --config configs/accuracy.yaml --adapter runs/accuracy_pilot/adapter --split dev --output runs/accuracy_pilot_eval
python -m effreason.train --config configs/efficiency.yaml --phase pilot --gamma 0.1 --seed 11 --output runs/efficiency_g01_pilot
python -m effreason.evaluate --config configs/efficiency.yaml --adapter runs/efficiency_g01_pilot/adapter --split dev --output runs/efficiency_g01_pilot_eval

# After finalizing configurations: each command is one independent main run
python -m effreason.train --config configs/accuracy.yaml --phase main --seed 11 --output runs/accuracy_seed11
python -m effreason.train --config configs/efficiency.yaml --phase main --seed 11 --output runs/efficiency_seed11

# Example final evaluation and concise-prompt control
python -m effreason.evaluate --config configs/accuracy.yaml --adapter runs/accuracy_seed11/adapter --split test --output runs/accuracy_seed11_test
python -m effreason.evaluate --config configs/accuracy.yaml --adapter runs/accuracy_seed11/adapter --split test --prompt-variant concise --output runs/accuracy_seed11_concise_test

# CPU analysis of saved evaluation outputs; supply actual evaluation directories
python -m effreason.analyze --run-dirs runs/accuracy_seed11_test runs/efficiency_seed11_test
```

Repeat main training/evaluation for the second seed; evaluate initial and efficiency conditions on the same frozen test data as well. The examples illustrate separate stages, not a shell script to execute blindly. Final data paths and budgets must be configured before `--phase main`; that flag alone must not convert a five-step pilot into a completed scientific run. Analysis should reject comparisons that mix development/test splits or incompatible main settings.

Provide config validation without weight loading (for example `--dry-run`). Support explicit training resume using the trainer's checkpoint mechanism and evaluation resume through output IDs. Save independently runnable results even if the next stage is never started. Analysis must work from saved predictions on a CPU laptop.

During iteration, change one decision, run the relevant cheap checks, run the smallest informative pilot, and append a short entry to `experiments.md`: run/config IDs, observations, decision, and which comparisons need rerunning. Do not overwrite frozen or superseded runs.

## 12. Tests that protect the experiment

Keep tests focused on risks that could invalidate results, rather than mirroring every implementation detail.

- Oracle fixtures: known unique, inconsistent, and ambiguous puzzles; each supported logical operator.
- Property checks: character renaming preserves the mapped solution; accepted premise mutations change it and remain unique.
- Canonicalization: role permutations collide as intended; implication direction is preserved.
- Split audit: collisions introduced through a variant or family linkage are detected.
- Parser adversarial fixtures: multiple answers, duplicate JSON keys, incomplete names, unknown roles, a correct guess inside reasoning but wrong final answer, and native Qwen tag behavior.
- Reward fixtures: gamma zero equals accuracy; length costs match exact generated lengths; malformed/empty answers earn no correctness; short wrong does not outrank long correct under the specified bounded reward.
- Token-boundary fixture: prompt/padding excluded, EOS handled consistently, closing thinking tokens counted.
- Analysis fixtures: manually calculable paired metrics, obsolete assignments, and clustered bootstrap grouping.
- Tiny trainer integration: repeated-completion metadata aligns, adapters change, frozen base weights do not, save/reload works. GPU-tagged and excluded from ordinary local CPU tests.

Run targeted CPU tests and lint locally without downloading model weights or using secrets. Put oracle/mutation/split tests in `test_data.py`, scorer/reward tests in `test_scoring.py`, and metrics tests in `test_analysis.py`; add a separate GPU integration test only when there is working training code to exercise. CI can be added later as a thin wrapper around the same commands. A failing logical audit must block dataset use.

## 13. Implementation milestones and acceptance gates

| Milestone | Build | Evidence required before moving on |
|---|---|---|
| M1: Repository and task contract | Package, config, task wrapper, oracle, parser and reward. | CPU tests pass; a hand-checked puzzle can be rendered and scored. |
| M2: Controlled data | Families, canonicalization, manifests and split audits. | Unique solutions; validated mutations; no forbidden structural overlaps. |
| M3: Baseline pilot | Model loader, evaluator and thin Colab notebook. | Real outputs, accuracy/length/truncation measurements, and resource profile. |
| M4: Separate training pilots | Accuracy pilot, then independent length-penalty candidates; GRPO/LoRA, logs and resume. | Actual adapter updates, matched development comparisons, reload/resume verification; B and gamma finalized. |
| M5: Main experiment | Two objectives across two seeds and matched evaluation controls. | Complete manifests and interpretable results, including failures and deviations. |
| M6: Report and portfolio | Analysis, figures, reproducibility instructions and result summary. | Another reader can regenerate reported metrics from saved predictions and identify the exact trained setup. |

The midpoint deliverable is M1–M4: a working prototype and preliminary evidence, not a promised improvement. Approximately 100 hours covers the entire course project, including reading, writing, analysis and presentation. Keep implementation restrained so it does not consume that entire allowance. Unattended GPU runtime is a separate budget.

If a milestone fails, document the failure and the smallest corrective action. Reduce scope before adding infrastructure. Do not use final-test outcomes to choose gamma, task difficulty, or checkpoints.

## 14. GitHub presentation and resume evidence

README should lead with the cognitive research question, followed by the experimental comparison, a small real results table when available, and a clear reproduction path. Include:

- What RLVR, GRPO and LoRA do in this repository.
- Why controlled transfer evaluation is needed beyond average accuracy.
- Setup, CPU quickstart, separate Colab/GPU commands, measured resource requirements.
- The finalized protocol and its distinction from exploratory pilot changes recorded in `experiments.md`.
- Data construction, controls, limitations, source attribution and model license links.
- Which work is complete versus planned. Do not display placeholder improvements as results.

Commit source, tested configurations, the dependency lock, small synthetic fixtures, manifests, analysis scripts, and lightweight figures/results. Ignore model weights, full checkpoint directories, large rollout logs, caches, `.env`, and notebook output containing credentials. Export adapters separately when publishing is explicitly requested. Check upstream licensing before distributing copied code or generated assets; keep attribution near adapted code.

No Docker, web UI, Kubernetes, or experiment tracking service is required. Local files are the source of truth; optional W&B can mirror them later. Do not create or push a remote repository until the user requests that action.

Resume evidence to retain after execution:

| Potential claim | Required supporting artifact |
|---|---|
| Post-trained a 4B reasoning model with GRPO and LoRA | Trainer configs, real training logs, saved adapter, base/model revisions. |
| Designed verifiable rewards | Shared scorer/reward implementation and adversarial fixtures. |
| Built controlled transfer evaluation | Dataset generator, paired manifests, overlap audit, and oracle tests. |
| Reduced tokens by X% at Y accuracy change | Reproducible baseline comparison with denominators, task counts, and seeds. |
| Implemented reproducible GPU experiments | Environment lock, run manifests, resumption test and measured GPU costs. |

Use measured numbers only. An honest engineering contribution remains valuable even when efficiency training reduces flexibility or fails to outperform concise prompting.

## 15. Sources and implementation references

These links provide provenance and starting points. Inspect the exact pinned versions during implementation; do not copy version-sensitive tutorial settings without verification.

1. [Reasoning Gym repository](https://github.com/open-thought/reasoning-gym) and [paper](https://arxiv.org/abs/2505.24760): procedural generation and verifiable tasks.
2. [Knights and Knaves implementation](https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/logic/knights_knaves.py): inspect metadata, AST operators and scoring in the pinned revision.
3. [Chosen Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507): native template, thinking-only behavior, sampling and output-length guidance.
4. [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer): reward callbacks, completion token IDs, grouping, loss normalization and training configuration.
5. [PEFT LoRA documentation](https://huggingface.co/docs/peft/developer_guides/lora): adapter configuration and loading.
6. [Xie et al., On Memorization of Large Language Models in Logical Reasoning](https://arxiv.org/abs/2410.23123): direct precedent for Knights and Knaves perturbation/generalization analysis.
7. [Yuan et al., Shorten After You're Right](https://arxiv.org/abs/2505.12284): prior work on reasoning-length rewards; shortening itself is not a novelty claim.
8. [Lieder and Griffiths, Resource-rational analysis](https://cocosci.princeton.edu/papers/lieder_resource.pdf): cognitive rationale for balancing quality and computation.
9. [Anderson, Acquisition of cognitive skill](https://doi.org/10.1037/0033-295X.89.4.369): proceduralization and skill acquisition.
10. [Turpin et al., Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388): limits of interpreting generated explanations as mechanisms.
11. Paul Thagard, *Mind: Introduction to Cognitive Science*, 2nd edition, 2005; Herbert Simon, *The Sciences of the Artificial*, 3rd edition, 1996: course framing and bounded rationality.

## 16. First instruction to the coding agent

Read this plan and inspect the existing repository and any `AGENTS.md` before changing files. Implement M1 and M2 first with the minimal structure above and small test fixtures. Then build M3 and profile the chosen model on the available GPU. Make every stage independently runnable. Present original-model length/truncation findings before fixing B, then support separate accuracy and efficiency pilots before a large training run. Do not add deferred files merely because an earlier plan listed them. Keep a concise record of implementation decisions and actual validation. The goal is a small, inspectable experiment whose scientific conclusions can be traced back to data and code.
