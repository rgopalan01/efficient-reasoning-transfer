from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from .data import PuzzleRecord, load_manifest_records
from .modeling import attach_lora, gpu_hardware, load_policy, trainable_parameter_summary
from .scoring import build_prompt, compute_reward, parse_final_answer
from .utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_yaml,
    canonical_json,
    environment_manifest,
    format_duration,
    load_config,
    log_progress,
    resolve_path,
    sha256_path,
    sha256_text,
    utc_now,
)


def prepare_training_rows(records: list[PuzzleRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "prompt": [{"role": "user", "content": build_prompt(record.question)}],
                "example_id": record.example_id,
                "characters_json": canonical_json(record.characters),
                "solution_json": canonical_json(record.solution),
            }
        )
    return rows


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    raise TypeError(f"Unsupported completion format: {type(completion).__name__}")


def compute_batch_rewards(
    completions: list[Any],
    completion_ids: list[Any],
    characters_json: list[str],
    solution_json: list[str],
    *,
    budget: int,
    gamma: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    lengths = {len(completions), len(completion_ids), len(characters_json), len(solution_json)}
    if len(lengths) != 1:
        raise ValueError("Reward metadata is not aligned with completions")
    rewards: list[float] = []
    components: list[dict[str, Any]] = []
    for completion, ids, characters_raw, solution_raw in zip(
        completions, completion_ids, characters_json, solution_json, strict=True
    ):
        characters = json.loads(characters_raw)
        solution = json.loads(solution_raw)
        expected = {
            characters[role]: "knight" if solution[role] else "knave" for role in sorted(characters)
        }
        parsed = parse_final_answer(_completion_text(completion), list(expected))
        correct = bool(parsed.valid and parsed.assignment == expected)
        token_count = len(ids)
        reward = compute_reward(correct, token_count, budget, gamma)
        rewards.append(reward.total_reward)
        components.append(
            {
                **reward.to_dict(),
                "parse_error": parsed.error,
                "thinking_boundary_found": parsed.thinking_boundary_found,
            }
        )
    return rewards, components


class TrlRewardAdapter:
    def __init__(self, *, budget: int, gamma: float, num_generations: int) -> None:
        self.budget = budget
        self.gamma = gamma
        self.num_generations = num_generations
        self.history: list[dict[str, Any]] = []

    def __call__(
        self,
        completions: list[Any],
        completion_ids: list[Any],
        characters_json: list[str],
        solution_json: list[str],
        example_id: list[str],
        **_: Any,
    ) -> list[float]:
        if len(example_id) != len(completions):
            raise ValueError("Example metadata is not aligned with completions")
        example_groups = [
            example_id[index : index + self.num_generations]
            for index in range(0, len(example_id), self.num_generations)
        ]
        if any(
            len(group) != self.num_generations or len(set(group)) != 1
            for group in example_groups
        ):
            raise ValueError("Repeated-generation metadata is not grouped by example")
        rewards, components = compute_batch_rewards(
            completions,
            completion_ids,
            characters_json,
            solution_json,
            budget=self.budget,
            gamma=self.gamma,
        )
        correctness = [int(component["accuracy"]) for component in components]
        groups = [
            correctness[index : index + self.num_generations]
            for index in range(0, len(correctness), self.num_generations)
        ]
        group_counts = Counter(
            "all_correct" if all(group) else "all_wrong" if not any(group) else "mixed"
            for group in groups
            if len(group) == self.num_generations
        )
        self.history.append(
            {
                "schema_version": "1",
                "event": "reward_batch",
                "mean_accuracy": sum(correctness) / len(correctness) if correctness else 0.0,
                "mean_completion_tokens": sum(c["completion_tokens"] for c in components) / len(components),
                "mean_total_reward": sum(rewards) / len(rewards),
                "group_counts": dict(group_counts),
                "group_proportions": {
                    key: value / sum(group_counts.values())
                    for key, value in group_counts.items()
                }
                if group_counts
                else {},
                "truncation_rate": sum(
                    int(component["completion_tokens"] == self.budget) for component in components
                )
                / len(components),
            }
        )
        return rewards


def _training_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError("Install the training dependencies with 'pip install -e .[train]'") from error
    return Dataset, GRPOConfig, GRPOTrainer, TrainerCallback


def _make_progress_callback(
    callback_base: type,
    *,
    total_steps: int,
    clock: Any = time.perf_counter,
) -> Any:
    """Create a lazy Transformers callback with explicit Colab-friendly step progress."""

    class ColabProgressCallback(callback_base):
        def __init__(self) -> None:
            self._training_started: float | None = None
            self._step_started: float | None = None
            self._initial_step = 0

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            self._training_started = clock()
            self._initial_step = int(state.global_step)
            remaining = max(total_steps - self._initial_step, 0)
            log_progress(
                "train",
                (
                    f"progress initialized: completed={self._initial_step}/{total_steps}; "
                    f"remaining={remaining}"
                ),
            )

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, state, control, kwargs
            self._step_started = clock()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            now = clock()
            completed = int(state.global_step)
            remaining = max(total_steps - completed, 0)
            session_steps = max(completed - self._initial_step, 1)
            session_elapsed = (
                now - self._training_started if self._training_started is not None else 0.0
            )
            average_step_time = session_elapsed / session_steps
            step_time = now - self._step_started if self._step_started is not None else 0.0
            log_progress(
                "train",
                (
                    f"completed step {completed}/{total_steps}; remaining={remaining}; "
                    f"step_time={format_duration(step_time)}; "
                    f"session_elapsed={format_duration(session_elapsed)}; "
                    f"eta={format_duration(average_step_time * remaining)}"
                ),
            )

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            log_progress(
                "train",
                f"checkpoint saved after step {int(state.global_step)}/{total_steps}",
            )

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            elapsed = (
                clock() - self._training_started if self._training_started is not None else 0.0
            )
            log_progress(
                "train",
                (
                    f"training loop finished at step {int(state.global_step)}/{total_steps}; "
                    f"session_elapsed={format_duration(elapsed)}"
                ),
            )

    return ColabProgressCallback()


def _build_grpo_config(
    config_class: Any,
    config: dict[str, Any],
    checkpoints_dir: Path,
) -> Any:
    """Translate the repository's stable training schema to the installed TRL API."""
    training = config["training"]
    generation = config["generation"]
    warmup_ratio = float(training.get("warmup_ratio", 0.0))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    dtype = str(config["model"]["dtype"])
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError("GRPO training requires model.dtype to be bfloat16 or float16")
    return config_class(
        output_dir=str(checkpoints_dir),
        num_generations=int(training["num_generations"]),
        max_completion_length=int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        loss_type=training["loss_type"],
        scale_rewards=training["scale_rewards"],
        beta=float(training["beta"]),
        learning_rate=float(training["learning_rate"]),
        max_steps=int(training["max_steps"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        save_steps=int(training.get("save_steps", 5)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        logging_steps=int(training.get("logging_steps", 1)),
        # Transformers 5 replaces warmup_ratio with warmup_steps; a float in [0, 1)
        # retains ratio semantics.
        warmup_steps=warmup_ratio,
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        optim=training["optim"],
        lr_scheduler_type=training["lr_scheduler_type"],
        epsilon=float(training["epsilon"]),
        epsilon_high=float(training["epsilon_high"]),
        mask_truncated_completions=bool(training["mask_truncated_completions"]),
        seed=int(training.get("seed", config["experiment"]["seed"])),
        remove_unused_columns=False,
        report_to="none",
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        disable_tqdm=False,
    )


def _write_status(path: Path, manifest: dict[str, Any], status: str, error: str | None = None) -> None:
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    if error:
        manifest["error"] = error
    else:
        manifest.pop("error", None)
    atomic_write_json(path, manifest)


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    state_path = path / "trainer_state.json"
    if match is None or not path.is_dir() or not state_path.is_file():
        return None
    step = int(match.group(1))
    try:
        with state_path.open(encoding="utf-8") as stream:
            state = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return step if int(state.get("global_step", -1)) == step else None


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    candidates = [
        (step, path)
        for path in checkpoints_dir.glob("checkpoint-*")
        if (step := _checkpoint_step(path)) is not None
    ]
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _resolve_resume_checkpoint(requested: str | None, checkpoints_dir: Path) -> Path | None:
    if requested is None:
        return None
    if requested.casefold() == "auto":
        return _latest_checkpoint(checkpoints_dir)
    checkpoint = Path(requested).resolve()
    if checkpoint.parent != checkpoints_dir.resolve():
        raise ValueError(f"Resume checkpoint must be inside {checkpoints_dir.resolve()}")
    if _checkpoint_step(checkpoint) is None:
        raise ValueError(f"Not a valid trainer checkpoint: {checkpoint}")
    return checkpoint


def _load_existing_training_manifest(
    manifest_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as stream:
        existing = json.load(stream)
    comparable = (
        "run_id",
        "run_type",
        "condition",
        "phase",
        "seed",
        "config_hash",
        "dataset_hash",
        "base_model",
        "prompt_hash",
    )
    for field in comparable:
        if existing.get(field) != expected.get(field):
            raise ValueError(f"Cannot reuse training output: manifest field {field!r} changed")
    return existing


def _validate_main_inputs(resolved: Any) -> None:
    config = resolved.values
    if config["experiment"]["phase"] != "main":
        return
    if int(config["training"]["max_steps"]) <= 5:
        raise ValueError("Main training refuses the five-step pilot budget")
    data_path = resolve_path(
        resolved.path, config["data"].get("train_manifest") or config["data"]["manifest"]
    )
    with data_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("frozen", False) or not manifest.get("dataset_hash"):
        raise ValueError("Main training requires a frozen, hashed dataset manifest")
    if int(manifest["splits"].get("test", {}).get("families", 0)) <= 0:
        raise ValueError("Main training requires a non-empty held-out test split")


def run_training(args: argparse.Namespace) -> None:
    resolved = load_config(
        args.config,
        phase=args.phase,
        gamma=args.gamma,
        completion_budget=args.completion_budget,
        output_dir=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        allow_main=args.allow_main,
    )
    config = resolved.values
    _validate_main_inputs(resolved)
    if args.dry_run:
        print(canonical_json({"config_hash": resolved.digest, "config": config}))
        return
    output_dir = resolve_path(resolved.path, config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    project_dir = resolved.path.parent.parent
    data_manifest_path = resolve_path(
        resolved.path, config["data"].get("train_manifest") or config["data"]["manifest"]
    )
    data_manifest, records = load_manifest_records(data_manifest_path, "train")
    log_progress(
        "train",
        (
            f"dataset ready: records={len(records)}, condition={config['experiment']['condition']}, "
            f"steps={config['training']['max_steps']}, generations_per_prompt="
            f"{config['training']['num_generations']}"
        ),
    )
    expected_manifest: dict[str, Any] = {
        "schema_version": "1",
        "run_id": output_dir.name,
        "run_type": "training",
        "condition": config["experiment"]["condition"],
        "phase": config["experiment"]["phase"],
        "seed": config["experiment"]["seed"],
        "config_hash": resolved.digest,
        "dataset_hash": data_manifest["dataset_hash"],
        "base_model": config["model"],
        "prompt_hash": sha256_text(build_prompt("<QUESTION>")),
        "started_at": utc_now(),
        "status": "started",
        "environment": environment_manifest(project_dir),
    }
    existing_manifest = _load_existing_training_manifest(manifest_path, expected_manifest)
    if existing_manifest is not None and existing_manifest.get("status") == "completed":
        adapter_dir = output_dir / "adapter"
        if not adapter_dir.is_dir():
            raise ValueError("Completed training manifest exists but the final adapter is missing")
        recorded_adapter_hash = existing_manifest.get("adapter_hash")
        if not recorded_adapter_hash or sha256_path(adapter_dir) != recorded_adapter_hash:
            raise ValueError("Completed training adapter does not match its manifest hash")
        log_progress(
            "train",
            f"run is already complete; verified adapter under {adapter_dir}; no model load required",
        )
        return
    if existing_manifest is not None and args.resume is None:
        raise ValueError(
            "An incomplete training run already exists. Pass '--resume auto' to preserve its "
            "latest checkpoint, or choose a new output directory."
        )
    atomic_write_yaml(output_dir / "config.resolved.yaml", config)
    manifest = existing_manifest or expected_manifest
    if existing_manifest is not None:
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["resumed_at"] = utc_now()
        manifest["environment"] = expected_manifest["environment"]
    _write_status(manifest_path, manifest, "started")
    checkpoints_dir = output_dir / "checkpoints"
    resume_checkpoint = _resolve_resume_checkpoint(args.resume, checkpoints_dir)
    if resume_checkpoint is not None and existing_manifest is None:
        raise ValueError(
            "A checkpoint exists without its training manifest; refusing an unverified resume"
        )
    if args.resume is not None:
        if resume_checkpoint is None:
            log_progress("train", "no completed checkpoint found; starting from step 0")
        else:
            manifest["resume_checkpoint"] = str(resume_checkpoint)
            manifest["resume_checkpoint_step"] = _checkpoint_step(resume_checkpoint)
            atomic_write_json(manifest_path, manifest)
            log_progress("train", f"resuming from {resume_checkpoint}")
    try:
        setup_started = time.perf_counter()
        log_progress("train", "loading training libraries")
        dataset_class, grpo_config_class, trainer_class, callback_base = _training_imports()
        log_progress("train", "validating the installed TRL training interface")
        trainer_args = _build_grpo_config(grpo_config_class, config, checkpoints_dir)
        log_progress(
            "train",
            f"loading base model {config['model']['id']} (the first run downloads the pinned weights)",
        )
        model, tokenizer = load_policy(config["model"])
        log_progress(
            "train",
            f"base model ready after {format_duration(time.perf_counter() - setup_started)}",
        )
        log_progress("train", "freezing base weights and attaching LoRA adapters")
        model = attach_lora(model, config["adapter"])
        manifest["chat_template_hash"] = sha256_text(tokenizer.chat_template)
        manifest["trainable_parameters"] = trainable_parameter_summary(model)
        manifest["hardware"] = gpu_hardware()
        atomic_write_json(manifest_path, manifest)
        dataset = dataset_class.from_list(prepare_training_rows(records))
        training = config["training"]
        reward_adapter = TrlRewardAdapter(
            budget=int(config["reward"]["completion_budget"]),
            gamma=float(config["reward"]["gamma"]),
            num_generations=int(training["num_generations"]),
        )
        trainer = trainer_class(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_adapter,
            args=trainer_args,
            train_dataset=dataset,
        )
        trainer.add_callback(
            _make_progress_callback(
                callback_base,
                total_steps=int(training["max_steps"]),
            )
        )
        log_progress(
            "train",
            (
                f"starting GRPO: {training['max_steps']} steps; explicit step timing, ETA, "
                "checkpoint progress, and TRL metrics will appear below"
            ),
        )
        training_started = time.perf_counter()
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
        )
        log_progress(
            "train",
            f"optimization complete after {format_duration(time.perf_counter() - training_started)}",
        )
        log_progress("train", f"saving adapter to {output_dir / 'adapter'}")
        trainer.save_model(str(output_dir / "adapter"))
        manifest["adapter_hash"] = sha256_path(output_dir / "adapter")
        metrics = [
            {"schema_version": "1", **metric} for metric in trainer.state.log_history
        ] + reward_adapter.history
        atomic_write_jsonl(output_dir / "metrics.jsonl", metrics)
        manifest["hardware"] = gpu_hardware()
        _write_status(manifest_path, manifest, "completed")
        log_progress("train", f"complete: adapter and metrics saved under {output_dir}")
    except KeyboardInterrupt:
        _write_status(manifest_path, manifest, "interrupted")
        log_progress(
            "train",
            "interrupted; rerun the same command with '--resume auto'",
        )
        raise
    except Exception as error:
        _write_status(manifest_path, manifest, "failed", "".join(traceback.format_exception_only(error)))
        log_progress("train", f"failed: {error}")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA policy with TRL GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["pilot", "main"])
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--completion-budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_training(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
