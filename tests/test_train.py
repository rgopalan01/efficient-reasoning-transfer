from __future__ import annotations

import json
from pathlib import Path

import pytest

from effreason.train import (
    _build_grpo_config,
    _latest_checkpoint,
    _load_existing_training_manifest,
    _resolve_resume_checkpoint,
    _write_status,
)
from effreason.utils import atomic_write_json, load_config


def _checkpoint(root, step: int, *, complete: bool = True):
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    if complete:
        atomic_write_json(path / "trainer_state.json", {"global_step": step})
    return path


def test_auto_resume_uses_highest_complete_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    first = _checkpoint(checkpoints, 1)
    latest = _checkpoint(checkpoints, 3)
    _checkpoint(checkpoints, 4, complete=False)
    assert _latest_checkpoint(checkpoints) == latest
    assert _resolve_resume_checkpoint("auto", checkpoints) == latest
    assert _resolve_resume_checkpoint(str(first), checkpoints) == first


def test_auto_resume_starts_fresh_without_a_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    assert _resolve_resume_checkpoint("auto", checkpoints) is None
    invalid = _checkpoint(checkpoints, 1, complete=False)
    with pytest.raises(ValueError, match="valid trainer checkpoint"):
        _resolve_resume_checkpoint(str(invalid), checkpoints)
    outside = _checkpoint(tmp_path / "other", 2)
    with pytest.raises(ValueError, match="must be inside"):
        _resolve_resume_checkpoint(str(outside), checkpoints)


def test_existing_training_manifest_requires_identical_run_identity(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    expected = {
        "run_id": "run",
        "run_type": "training",
        "condition": "accuracy",
        "phase": "pilot",
        "seed": 11,
        "config_hash": "config",
        "dataset_hash": "dataset",
        "base_model": {"id": "model", "revision": "revision"},
        "prompt_hash": "prompt",
    }
    atomic_write_json(manifest_path, {**expected, "status": "interrupted"})
    loaded = _load_existing_training_manifest(manifest_path, expected)
    assert loaded is not None and loaded["status"] == "interrupted"
    with pytest.raises(ValueError, match="config_hash"):
        _load_existing_training_manifest(
            manifest_path,
            {**expected, "config_hash": "changed"},
        )


def test_started_status_clears_stale_error(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {"status": "failed", "error": "old failure"}
    _write_status(manifest_path, manifest, "started")
    with manifest_path.open(encoding="utf-8") as stream:
        saved = json.load(stream)
    assert saved["status"] == "started"
    assert "error" not in saved


def test_grpo_config_translates_fractional_warmup_to_transformers_5(tmp_path):
    captured = {}

    class FakeGRPOConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = {
        "experiment": {"seed": 11},
        "generation": {
            "max_new_tokens": 8192,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        "training": {
            "num_generations": 4,
            "loss_type": "dr_grpo",
            "scale_rewards": False,
            "beta": 0.0,
            "learning_rate": 5e-6,
            "max_steps": 5,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "warmup_ratio": 0.1,
            "optim": "adamw_torch_fused",
            "lr_scheduler_type": "linear",
            "epsilon": 0.2,
            "epsilon_high": 0.2,
            "mask_truncated_completions": False,
        },
    }
    _build_grpo_config(FakeGRPOConfig, config, tmp_path / "checkpoints")
    assert captured["warmup_steps"] == 0.1
    assert "warmup_ratio" not in captured


def test_grpo_config_rejects_invalid_warmup_ratio(tmp_path):
    config = {
        "experiment": {"seed": 11},
        "generation": {},
        "training": {"warmup_ratio": 1.0},
    }
    with pytest.raises(ValueError, match="warmup_ratio"):
        _build_grpo_config(object, config, tmp_path / "checkpoints")


def test_training_memory_overrides_preserve_effective_batch():
    config_path = Path(__file__).parents[1] / "configs" / "efficiency.yaml"
    resolved = load_config(
        config_path,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    )
    training = resolved.values["training"]
    assert training["max_steps"] == 1
    assert training["per_device_train_batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 4
    assert (
        training["per_device_train_batch_size"]
        * training["gradient_accumulation_steps"]
        == training["num_generations"]
    )
