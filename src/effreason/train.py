from __future__ import annotations

import argparse
import json
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
    load_config,
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


def _training_imports() -> tuple[Any, Any, Any]:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError("Install the training dependencies with 'pip install -e .[train]'") from error
    return Dataset, GRPOConfig, GRPOTrainer


def _write_status(path: Path, manifest: dict[str, Any], status: str, error: str | None = None) -> None:
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    if error:
        manifest["error"] = error
    atomic_write_json(path, manifest)


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
        allow_main=args.allow_main,
    )
    config = resolved.values
    _validate_main_inputs(resolved)
    if args.dry_run:
        print(canonical_json({"config_hash": resolved.digest, "config": config}))
        return
    output_dir = resolve_path(resolved.path, config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(output_dir / "config.resolved.yaml", config)
    manifest_path = output_dir / "manifest.json"
    project_dir = resolved.path.parent.parent
    data_manifest_path = resolve_path(
        resolved.path, config["data"].get("train_manifest") or config["data"]["manifest"]
    )
    data_manifest, records = load_manifest_records(data_manifest_path, "train")
    manifest: dict[str, Any] = {
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
    atomic_write_json(manifest_path, manifest)
    try:
        dataset_class, grpo_config_class, trainer_class = _training_imports()
        model, tokenizer = load_policy(config["model"])
        model = attach_lora(model, config["adapter"])
        manifest["chat_template_hash"] = sha256_text(tokenizer.chat_template)
        manifest["trainable_parameters"] = trainable_parameter_summary(model)
        manifest["hardware"] = gpu_hardware()
        atomic_write_json(manifest_path, manifest)
        dataset = dataset_class.from_list(prepare_training_rows(records))
        training = config["training"]
        generation = config["generation"]
        trainer_args = grpo_config_class(
            output_dir=str(output_dir / "checkpoints"),
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
            logging_steps=int(training.get("logging_steps", 1)),
            warmup_ratio=float(training.get("warmup_ratio", 0.0)),
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
            bf16=True,
        )
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
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(str(output_dir / "adapter"))
        manifest["adapter_hash"] = sha256_path(output_dir / "adapter")
        metrics = [
            {"schema_version": "1", **metric} for metric in trainer.state.log_history
        ] + reward_adapter.history
        atomic_write_jsonl(output_dir / "metrics.jsonl", metrics)
        manifest["hardware"] = gpu_hardware()
        _write_status(manifest_path, manifest, "completed")
    except Exception as error:
        _write_status(manifest_path, manifest, "failed", "".join(traceback.format_exception_only(error)))
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA policy with TRL GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["pilot", "main"])
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--completion-budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_training(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
