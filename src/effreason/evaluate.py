from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Any

from .data import PuzzleRecord, load_manifest_records
from .modeling import generate_completion, gpu_hardware, load_policy
from .scoring import (
    build_prompt,
    classify_repetition,
    compute_reward,
    expected_visible_assignment,
    score_assignment,
)
from .utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_yaml,
    canonical_json,
    environment_manifest,
    load_config,
    read_jsonl,
    resolve_path,
    sha256_file,
    sha256_object,
    sha256_path,
    sha256_text,
    utc_now,
)


def _resume_state(
    output_dir: Path,
    expected_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], set[tuple[str, str, int]]]:
    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.jsonl"
    if not manifest_path.exists():
        return expected_manifest, [], set()
    with manifest_path.open(encoding="utf-8") as stream:
        existing = json.load(stream)
    comparable = [
        "run_id",
        "config_hash",
        "dataset_hash",
        "split",
        "condition",
        "prompt_variant",
        "adapter",
        "adapter_hash",
    ]
    for key in comparable:
        if existing.get(key) != expected_manifest.get(key):
            raise ValueError(f"Cannot resume: manifest field {key!r} changed")
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {
        (row["run_id"], row["example_id"], int(row["sample_id"])) for row in predictions
    }
    existing["status"] = "started"
    existing["updated_at"] = utc_now()
    return existing, predictions, completed


def _parent_assignment(record: PuzzleRecord, by_id: dict[str, PuzzleRecord]) -> dict[str, str] | None:
    if record.variant_type != "premise" or not record.parent_id:
        return None
    parent = by_id.get(record.parent_id)
    if parent is None:
        return None
    return expected_visible_assignment(parent.characters, parent.solution)


def run_evaluation(args: argparse.Namespace) -> None:
    resolved = load_config(
        args.config,
        gamma=args.gamma,
        completion_budget=args.completion_budget,
        samples_per_example=args.samples_per_example,
        allow_main=args.allow_main,
    )
    config = resolved.values
    split = args.split
    prompt_variant = args.prompt_variant or config["evaluation"]["prompt_variant"]
    adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    condition = args.condition or ("adapter" if adapter else "initial")
    if args.dry_run:
        print(
            canonical_json(
                {
                    "config_hash": resolved.digest,
                    "condition": condition,
                    "split": split,
                    "prompt_variant": prompt_variant,
                    "adapter": adapter,
                    "samples_per_example": config["evaluation"]["samples_per_example"],
                }
            )
        )
        return
    if not args.output:
        raise ValueError("--output is required unless --dry-run is used")
    data_path = resolve_path(
        resolved.path, config["data"].get("eval_manifest") or config["data"]["manifest"]
    )
    data_manifest, records = load_manifest_records(data_path, split)
    if config["experiment"]["phase"] == "main" and split in {"test", "test_harder"}:
        if not data_manifest.get("frozen", False):
            raise ValueError("Main test evaluation requires a frozen dataset manifest")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(output_dir / "config.resolved.yaml", config)
    project_dir = resolved.path.parent.parent
    expected_manifest: dict[str, Any] = {
        "schema_version": "1",
        "run_id": output_dir.name,
        "run_type": "evaluation",
        "condition": condition,
        "split": split,
        "prompt_variant": prompt_variant,
        "adapter": adapter,
        "adapter_hash": sha256_path(adapter) if adapter else None,
        "config_hash": resolved.digest,
        "dataset_hash": data_manifest["dataset_hash"],
        "data_manifest": str(data_path),
        "base_model": config["model"],
        "prompt_hash": sha256_text(build_prompt("<QUESTION>", variant=prompt_variant)),
        "started_at": utc_now(),
        "status": "started",
        "environment": environment_manifest(project_dir),
    }
    manifest, predictions, completed = _resume_state(output_dir, expected_manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    by_id = {record.example_id: record for record in records}
    execution_records = list(records)
    random.Random(int(config["evaluation"]["seed"])).shuffle(execution_records)
    try:
        model, tokenizer = load_policy(config["model"], adapter_path=adapter)
        manifest["chat_template_hash"] = sha256_text(tokenizer.chat_template)
        manifest["hardware"] = gpu_hardware()
        atomic_write_json(output_dir / "manifest.json", manifest)
        samples = int(config["evaluation"]["samples_per_example"])
        base_seed = int(config["evaluation"]["seed"])
        for record_index, record in enumerate(execution_records):
            for sample_id in range(samples):
                key = (output_dir.name, record.example_id, sample_id)
                if key in completed:
                    continue
                result = generate_completion(
                    model,
                    tokenizer,
                    record.question,
                    config["generation"],
                    prompt_variant=prompt_variant,
                    seed=base_seed + record_index * samples + sample_id,
                )
                score = score_assignment(result.completion, record.characters, record.solution)
                reward = compute_reward(
                    score.correct,
                    result.completion_tokens,
                    int(config["reward"]["completion_budget"]),
                    float(config["reward"]["gamma"]),
                )
                parent_assignment = _parent_assignment(record, by_id)
                obsolete = bool(
                    parent_assignment is not None
                    and score.parse.valid
                    and score.parse.assignment == parent_assignment
                )
                predictions.append(
                    {
                        "schema_version": "1",
                        "run_id": output_dir.name,
                        "example_id": record.example_id,
                        "family_id": record.family_id,
                        "parent_id": record.parent_id,
                        "variant_type": record.variant_type,
                        "split": record.split,
                        "sample_id": sample_id,
                        "condition": condition,
                        "prompt_variant": prompt_variant,
                        "adapter_hash": manifest["adapter_hash"],
                        "raw_completion": result.completion,
                        "parsed_assignment": score.parse.assignment,
                        "parse_valid": score.parse.valid,
                        "parse_error": score.parse.error,
                        "thinking_boundary_found": score.parse.thinking_boundary_found,
                        "correct": score.correct,
                        "error_type": score.error_type,
                        "obsolete_assignment": obsolete,
                        "repeated_output": classify_repetition(result.completion),
                        "completion_tokens": result.completion_tokens,
                        "completion_ids_sha256": sha256_object(result.completion_ids),
                        "truncated": result.truncated,
                        "finish_reason": result.finish_reason,
                        "latency_seconds": result.latency_seconds,
                        "reward": reward.to_dict(),
                    }
                )
                atomic_write_jsonl(output_dir / "predictions.jsonl", predictions)
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["predictions_sha256"] = sha256_file(output_dir / "predictions.jsonl")
        manifest["hardware"] = gpu_hardware()
        atomic_write_json(output_dir / "manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = "".join(traceback.format_exception_only(error))
        manifest["updated_at"] = utc_now()
        atomic_write_json(output_dir / "manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the initial model or a LoRA adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition")
    parser.add_argument("--adapter")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--completion-budget", type=int)
    parser.add_argument("--samples-per-example", type=int)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "test_harder"])
    parser.add_argument("--prompt-variant", choices=["standard", "concise"])
    parser.add_argument("--output")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_evaluation(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
