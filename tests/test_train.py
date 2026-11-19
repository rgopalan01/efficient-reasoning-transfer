from __future__ import annotations

import json

import pytest

from effreason.train import (
    _latest_checkpoint,
    _load_existing_training_manifest,
    _resolve_resume_checkpoint,
    _write_status,
)
from effreason.utils import atomic_write_json


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
