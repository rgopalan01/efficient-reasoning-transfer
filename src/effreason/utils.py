from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the shared schema."""


CONFIG_VERSION = "1"


CONFIG_SCHEMA: dict[str, set[str]] = {
    "experiment": {"name", "phase", "seed", "condition"},
    "model": {"id", "revision", "dtype", "quantization", "attention_implementation"},
    "adapter": {"method", "rank", "alpha", "dropout", "target_modules"},
    "data": {
        "output_dir",
        "manifest",
        "train_manifest",
        "eval_manifest",
        "max_generation_attempts",
        "splits",
        "variants_for_splits",
    },
    "reward": {"gamma", "completion_budget"},
    "generation": {"temperature", "top_p", "top_k", "max_new_tokens", "do_sample"},
    "training": {
        "num_generations",
        "loss_type",
        "scale_rewards",
        "beta",
        "learning_rate",
        "max_steps",
        "gradient_checkpointing",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "save_steps",
        "logging_steps",
        "warmup_ratio",
        "max_grad_norm",
        "weight_decay",
        "seed",
        "optim",
        "lr_scheduler_type",
        "epsilon",
        "epsilon_high",
        "mask_truncated_completions",
    },
    "evaluation": {"samples_per_example", "prompt_variant", "seed"},
    "analysis": {"bootstrap_resamples", "confidence", "accuracy_margin_pp", "seed"},
    "artifacts": {"output_dir"},
}

REQUIRED_SECTIONS = set(CONFIG_SCHEMA)
REQUIRED_KEYS: dict[str, set[str]] = {
    "experiment": {"name", "phase", "seed", "condition"},
    "model": {"id", "revision", "dtype", "quantization"},
    "adapter": {"method", "rank", "alpha", "dropout", "target_modules"},
    "data": {"output_dir", "manifest", "max_generation_attempts", "splits", "variants_for_splits"},
    "reward": {"gamma", "completion_budget"},
    "generation": {"temperature", "top_p", "top_k", "max_new_tokens", "do_sample"},
    "training": {
        "num_generations",
        "loss_type",
        "scale_rewards",
        "beta",
        "learning_rate",
        "max_steps",
        "gradient_checkpointing",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "optim",
        "lr_scheduler_type",
        "epsilon",
        "epsilon_high",
        "mask_truncated_completions",
    },
    "evaluation": {"samples_per_example", "prompt_variant", "seed"},
    "analysis": {"bootstrap_resamples", "confidence", "accuracy_margin_pp", "seed"},
    "artifacts": {"output_dir"},
}


@dataclass(frozen=True)
class ResolvedConfig:
    path: Path
    values: dict[str, Any]
    digest: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    entries = [
        (str(item.relative_to(target)), sha256_file(item))
        for item in sorted(target.rglob("*"))
        if item.is_file()
    ]
    return sha256_object(entries)


def _atomic_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_replace(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    _atomic_replace(Path(path), text)


def atomic_write_yaml(path: str | Path, value: Any) -> None:
    _atomic_replace(Path(path), yaml.safe_dump(value, sort_keys=False, allow_unicode=False))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _validate_split_config(splits: Any) -> None:
    if not isinstance(splits, dict) or not splits:
        raise ConfigError("data.splits must be a non-empty mapping")
    allowed = {"count", "n_people", "depth", "width"}
    for split, values in splits.items():
        if split not in {"train", "dev", "test", "test_harder"}:
            raise ConfigError(f"Unknown split: {split}")
        if not isinstance(values, dict):
            raise ConfigError(f"data.splits.{split} must be a mapping")
        unknown = set(values) - allowed
        if unknown:
            raise ConfigError(f"Unknown keys in data.splits.{split}: {sorted(unknown)}")
        missing = allowed - set(values)
        if missing:
            raise ConfigError(f"Missing keys in data.splits.{split}: {sorted(missing)}")
        if int(values["count"]) < 0 or int(values["n_people"]) < 1:
            raise ConfigError(f"Invalid count or n_people in data.splits.{split}")


def validate_config(config: Mapping[str, Any], *, allow_main: bool = False) -> None:
    unknown_sections = set(config) - REQUIRED_SECTIONS - {"schema_version"}
    missing_sections = REQUIRED_SECTIONS - set(config)
    if unknown_sections:
        raise ConfigError(f"Unknown configuration sections: {sorted(unknown_sections)}")
    if missing_sections:
        raise ConfigError(f"Missing configuration sections: {sorted(missing_sections)}")
    if config.get("schema_version") != CONFIG_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_VERSION!r}")
    for section, allowed in CONFIG_SCHEMA.items():
        value = config[section]
        if not isinstance(value, dict):
            raise ConfigError(f"{section} must be a mapping")
        unknown = set(value) - allowed
        missing = REQUIRED_KEYS[section] - set(value)
        if unknown:
            raise ConfigError(f"Unknown keys in {section}: {sorted(unknown)}")
        if missing:
            raise ConfigError(f"Missing keys in {section}: {sorted(missing)}")

    _validate_split_config(config["data"]["splits"])
    phase = config["experiment"]["phase"]
    if phase not in {"pilot", "main"}:
        raise ConfigError("experiment.phase must be 'pilot' or 'main'")
    if phase == "main" and not allow_main:
        raise ConfigError("Main runs require an explicit --allow-main acknowledgement")
    if phase == "main" and "RESOLVE" in str(config["model"]["revision"]).upper():
        raise ConfigError("Main runs require a pinned model revision")
    if config["model"]["quantization"] != "none":
        raise ConfigError("Version 1 supports only unquantized BF16 LoRA")
    if config["adapter"]["method"] != "lora":
        raise ConfigError("Version 1 supports only LoRA adapters")
    if config["training"]["loss_type"] != "dr_grpo":
        raise ConfigError("training.loss_type must be dr_grpo")
    if config["training"]["scale_rewards"] is not False:
        raise ConfigError("training.scale_rewards must be false")
    gamma = float(config["reward"]["gamma"])
    if not 0 <= gamma < 1:
        raise ConfigError("reward.gamma must satisfy 0 <= gamma < 1")
    budget = int(config["reward"]["completion_budget"])
    if budget <= 0 or budget != int(config["generation"]["max_new_tokens"]):
        raise ConfigError("reward.completion_budget must equal generation.max_new_tokens")
    generations = int(config["training"]["num_generations"])
    effective_batch = (
        int(config["training"]["per_device_train_batch_size"])
        * int(config["training"]["gradient_accumulation_steps"])
    )
    if effective_batch % generations:
        raise ConfigError("Effective prompt batch must be divisible by training.num_generations")
    variants = config["data"]["variants_for_splits"]
    if not isinstance(variants, list) or any(v not in {"dev", "test"} for v in variants):
        raise ConfigError("data.variants_for_splits may contain only dev and test")


def load_config(
    path: str | Path,
    *,
    phase: str | None = None,
    gamma: float | None = None,
    completion_budget: int | None = None,
    samples_per_example: int | None = None,
    output_dir: str | None = None,
    seed: int | None = None,
    allow_main: bool = False,
) -> ResolvedConfig:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = deepcopy(raw)
    if phase is not None:
        config["experiment"]["phase"] = phase
    if gamma is not None:
        config["reward"]["gamma"] = float(gamma)
    if completion_budget is not None:
        config["reward"]["completion_budget"] = int(completion_budget)
        config["generation"]["max_new_tokens"] = int(completion_budget)
    if samples_per_example is not None:
        if samples_per_example <= 0:
            raise ConfigError("samples_per_example must be positive")
        config["evaluation"]["samples_per_example"] = int(samples_per_example)
    if output_dir is not None:
        config["artifacts"]["output_dir"] = output_dir
    if seed is not None:
        config["experiment"]["seed"] = int(seed)
        config["training"]["seed"] = int(seed)
        config["evaluation"]["seed"] = int(seed)
    validate_config(config, allow_main=allow_main)
    return ResolvedConfig(config_path, config, sha256_object(config))


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def package_freeze() -> list[str]:
    installed = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(installed, key=str.casefold)


def git_state(directory: str | Path) -> dict[str, Any]:
    cwd = str(Path(directory).resolve())

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
        "diff_hash": sha256_text(run("diff", "--binary") or ""),
    }


def environment_manifest(project_dir: str | Path) -> dict[str, Any]:
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(
            [
                "reasoning-gym",
                "numpy",
                "pandas",
                "matplotlib",
                "torch",
                "transformers",
                "trl",
                "peft",
                "accelerate",
                "datasets",
            ]
        ),
        "package_freeze": package_freeze(),
        "git": git_state(project_dir),
    }


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()
