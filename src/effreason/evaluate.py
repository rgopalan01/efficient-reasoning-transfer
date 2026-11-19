from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

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
    format_duration,
    load_config,
    log_progress,
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
        "extension_source_run_id",
        "extension_source_predictions_sha256",
        "extension_source_budget",
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


def _prediction_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["example_id"]), int(row["sample_id"])


def _verify_extension_prefix(
    completion_ids: list[int], source_row: dict[str, Any]
) -> int:
    if not bool(source_row.get("truncated", False)):
        raise ValueError("Extension source row is not truncated")
    prefix_tokens = int(source_row["completion_tokens"])
    if len(completion_ids) < prefix_tokens:
        raise ValueError(
            f"Extended completion ended at {len(completion_ids)} tokens before the "
            f"{prefix_tokens}-token source prefix"
        )
    actual = sha256_object(completion_ids[:prefix_tokens])
    expected = str(source_row["completion_ids_sha256"])
    if actual != expected:
        raise ValueError(
            "Extended completion did not reproduce the saved truncated token prefix; "
            "refusing to mix nondeterministic predictions"
        )
    return prefix_tokens


def _prepare_extension_source(
    source_dir: Path,
    *,
    expected_manifest: dict[str, Any],
    target_config: dict[str, Any],
    records: list[PuzzleRecord],
    samples: int,
    output_run_id: str,
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    manifest_path = source_dir / "manifest.json"
    config_path = source_dir / "config.resolved.yaml"
    predictions_path = source_dir / "predictions.jsonl"
    for path in (manifest_path, config_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Extension source is missing {path}")
    with manifest_path.open(encoding="utf-8") as stream:
        source_manifest = json.load(stream)
    with config_path.open(encoding="utf-8") as stream:
        source_config = yaml.safe_load(stream)
    if source_manifest.get("status") != "completed":
        raise ValueError("Extension source evaluation must have status='completed'")
    if source_manifest.get("run_type") != "evaluation":
        raise ValueError("Extension source must be an evaluation run")
    for field in ("dataset_hash", "split", "prompt_variant", "adapter_hash"):
        if source_manifest.get(field) != expected_manifest.get(field):
            raise ValueError(f"Extension source differs on manifest field {field!r}")
    if source_manifest.get("base_model") != expected_manifest.get("base_model"):
        raise ValueError("Extension source uses a different base model")
    actual_predictions_hash = sha256_file(predictions_path)
    recorded_predictions_hash = source_manifest.get("predictions_sha256")
    if recorded_predictions_hash and recorded_predictions_hash != actual_predictions_hash:
        raise ValueError("Extension source predictions hash does not match its manifest")
    source_generation = dict(source_config["generation"])
    target_generation = dict(target_config["generation"])
    source_budget = int(source_generation.pop("max_new_tokens"))
    target_budget = int(target_generation.pop("max_new_tokens"))
    if source_generation != target_generation:
        raise ValueError("Extension source uses different generation settings")
    if target_budget <= source_budget:
        raise ValueError("Extension completion budget must be larger than the source budget")
    source_evaluation = source_config["evaluation"]
    target_evaluation = target_config["evaluation"]
    if int(source_evaluation["samples_per_example"]) != samples:
        raise ValueError("Extension source uses a different samples-per-example value")
    if int(source_evaluation["seed"]) != int(target_evaluation["seed"]):
        raise ValueError("Extension source uses a different evaluation seed")
    rows = read_jsonl(predictions_path)
    expected_keys = {
        (record.example_id, sample_id) for record in records for sample_id in range(samples)
    }
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _prediction_key(row)
        if key in rows_by_key:
            raise ValueError(f"Extension source contains duplicate prediction key {key}")
        rows_by_key[key] = row
    if set(rows_by_key) != expected_keys:
        missing = len(expected_keys - set(rows_by_key))
        unexpected = len(set(rows_by_key) - expected_keys)
        raise ValueError(
            f"Extension source prediction keys differ: missing={missing}, unexpected={unexpected}"
        )
    truncated = {key: row for key, row in rows_by_key.items() if bool(row.get("truncated"))}
    if not truncated:
        raise ValueError("Extension source contains no truncated predictions")
    source_run_id = str(source_manifest["run_id"])
    reused: list[dict[str, Any]] = []
    target_budget_for_reward = int(target_config["reward"]["completion_budget"])
    target_gamma = float(target_config["reward"]["gamma"])
    for key, row in rows_by_key.items():
        if key in truncated:
            continue
        copied = dict(row)
        copied["run_id"] = output_run_id
        copied["condition"] = condition
        copied["reward"] = compute_reward(
            bool(copied["correct"]),
            int(copied["completion_tokens"]),
            target_budget_for_reward,
            target_gamma,
        ).to_dict()
        copied["extension_source_run_id"] = source_run_id
        copied["extension_reused_without_regeneration"] = True
        reused.append(copied)
    info = {
        "extension_source_run_id": source_run_id,
        "extension_source_path": str(source_dir),
        "extension_source_predictions_sha256": actual_predictions_hash,
        "extension_source_budget": source_budget,
        "extension_target_budget": target_budget,
        "extension_reused_predictions": len(reused),
        "extension_regenerated_predictions": len(truncated),
        "extension_prefix_verification": "sha256 of exact source-length token ID prefix",
    }
    return info, reused, truncated


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
                    "extend_truncated_from": args.extend_truncated_from,
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
    samples = int(config["evaluation"]["samples_per_example"])
    if config["experiment"]["phase"] == "main" and split in {"test", "test_harder"}:
        if not data_manifest.get("frozen", False):
            raise ValueError("Main test evaluation requires a frozen dataset manifest")
    output_dir = Path(args.output).resolve()
    extension_source_dir = (
        Path(args.extend_truncated_from).resolve() if args.extend_truncated_from else None
    )
    if extension_source_dir == output_dir:
        raise ValueError("--extend-truncated-from must differ from --output")
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
    extension_truncated: dict[tuple[str, int], dict[str, Any]] = {}
    extension_reused: list[dict[str, Any]] = []
    if extension_source_dir is not None:
        extension_info, extension_reused, extension_truncated = _prepare_extension_source(
            extension_source_dir,
            expected_manifest=expected_manifest,
            target_config=config,
            records=records,
            samples=samples,
            output_run_id=output_dir.name,
            condition=condition,
        )
        expected_manifest.update(extension_info)
    manifest, predictions, completed = _resume_state(output_dir, expected_manifest)
    if extension_reused:
        existing_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in predictions:
            key = _prediction_key(row)
            if key in existing_by_key:
                raise ValueError(f"Target run contains duplicate prediction key {key}")
            existing_by_key[key] = row
        for reused_row in extension_reused:
            key = _prediction_key(reused_row)
            existing = existing_by_key.get(key)
            if existing is None:
                predictions.append(reused_row)
                existing_by_key[key] = reused_row
            elif existing.get("completion_ids_sha256") != reused_row.get(
                "completion_ids_sha256"
            ):
                raise ValueError(f"Target run conflicts with reused prediction key {key}")
        atomic_write_jsonl(output_dir / "predictions.jsonl", predictions)
        completed = {
            (output_dir.name, row["example_id"], int(row["sample_id"]))
            for row in predictions
        }
    atomic_write_json(output_dir / "manifest.json", manifest)
    by_id = {record.example_id: record for record in records}
    execution_records = list(records)
    random.Random(int(config["evaluation"]["seed"])).shuffle(execution_records)
    total_predictions = len(execution_records) * samples
    completed_count = len(completed)
    if completed_count > total_predictions:
        raise ValueError(
            f"Resume state has {completed_count} predictions, expected at most {total_predictions}"
        )
    log_progress(
        "evaluate",
        (
            f"dataset ready: split={split}, records={len(execution_records)}, "
            f"samples_per_record={samples}, total={total_predictions}, "
            f"already_completed={completed_count}"
        ),
    )
    if extension_source_dir is not None:
        log_progress(
            "evaluate",
            (
                f"targeted extension: reused={len(extension_reused)}, "
                f"to_regenerate={len(extension_truncated)}, source={extension_source_dir}"
            ),
        )
    if completed_count == total_predictions:
        manifest["status"] = "completed"
        manifest["completed_at"] = manifest.get("completed_at", utc_now())
        manifest["total_predictions"] = total_predictions
        manifest["completed_predictions"] = completed_count
        manifest["predictions_sha256"] = sha256_file(output_dir / "predictions.jsonl")
        atomic_write_json(output_dir / "manifest.json", manifest)
        log_progress("evaluate", "run is already complete; no model load is required")
        return
    try:
        load_started = time.perf_counter()
        log_progress(
            "evaluate",
            f"loading {config['model']['id']} (the first run may download about 8 GB)",
        )
        model, tokenizer = load_policy(config["model"], adapter_path=adapter)
        log_progress(
            "evaluate",
            f"model ready after {format_duration(time.perf_counter() - load_started)}",
        )
        manifest["chat_template_hash"] = sha256_text(tokenizer.chat_template)
        manifest["hardware"] = gpu_hardware()
        manifest["total_predictions"] = total_predictions
        manifest["completed_predictions"] = completed_count
        atomic_write_json(output_dir / "manifest.json", manifest)
        base_seed = int(config["evaluation"]["seed"])
        generation_started = time.perf_counter()
        starting_completed = completed_count
        for record_index, record in enumerate(execution_records):
            for sample_id in range(samples):
                key = (output_dir.name, record.example_id, sample_id)
                if key in completed:
                    continue
                log_progress(
                    "evaluate",
                    (
                        f"starting {completed_count + 1}/{total_predictions}: "
                        f"variant={record.variant_type}, sample={sample_id + 1}/{samples}, "
                        f"example={record.example_id[:12]}"
                    ),
                )
                result = generate_completion(
                    model,
                    tokenizer,
                    record.question,
                    config["generation"],
                    prompt_variant=prompt_variant,
                    seed=base_seed + record_index * samples + sample_id,
                )
                source_row = extension_truncated.get((record.example_id, sample_id))
                extension_fields: dict[str, Any] = {}
                if extension_source_dir is not None:
                    if source_row is None:
                        raise ValueError(
                            "Targeted extension attempted to regenerate a non-truncated source key"
                        )
                    prefix_tokens = _verify_extension_prefix(result.completion_ids, source_row)
                    extension_fields = {
                        "extension_source_run_id": manifest["extension_source_run_id"],
                        "extension_reused_without_regeneration": False,
                        "extension_prefix_tokens": prefix_tokens,
                        "extension_prefix_verified": True,
                    }
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
                        **extension_fields,
                    }
                )
                atomic_write_jsonl(output_dir / "predictions.jsonl", predictions)
                completed.add(key)
                completed_count += 1
                completed_this_process = completed_count - starting_completed
                elapsed = time.perf_counter() - generation_started
                average = elapsed / completed_this_process
                remaining = total_predictions - completed_count
                manifest["completed_predictions"] = completed_count
                manifest["updated_at"] = utc_now()
                atomic_write_json(output_dir / "manifest.json", manifest)
                log_progress(
                    "evaluate",
                    (
                        f"completed {completed_count}/{total_predictions}; remaining={remaining}; "
                        f"tokens={result.completion_tokens}; correct={score.correct}; "
                        f"finish={result.finish_reason}; "
                        f"item_time={format_duration(result.latency_seconds)}; "
                        f"elapsed={format_duration(elapsed)}; eta={format_duration(remaining * average)}"
                    ),
                )
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["predictions_sha256"] = sha256_file(output_dir / "predictions.jsonl")
        manifest["hardware"] = gpu_hardware()
        atomic_write_json(output_dir / "manifest.json", manifest)
        log_progress(
            "evaluate",
            (
                f"complete: {completed_count}/{total_predictions} predictions saved to "
                f"{output_dir / 'predictions.jsonl'}"
            ),
        )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_predictions"] = completed_count
        manifest["updated_at"] = utc_now()
        atomic_write_json(output_dir / "manifest.json", manifest)
        log_progress(
            "evaluate",
            f"interrupted after {completed_count}/{total_predictions}; rerun the same command to resume",
        )
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = "".join(traceback.format_exception_only(error))
        manifest["updated_at"] = utc_now()
        atomic_write_json(output_dir / "manifest.json", manifest)
        log_progress("evaluate", f"failed after {completed_count}/{total_predictions}: {error}")
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
    parser.add_argument(
        "--extend-truncated-from",
        help="Reuse non-truncated predictions from a completed smaller-budget evaluation",
    )
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_evaluation(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
