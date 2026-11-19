from __future__ import annotations

from types import SimpleNamespace

import pytest

from effreason.evaluate import _prepare_extension_source, _verify_extension_prefix
from effreason.utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_yaml,
    sha256_file,
    sha256_object,
)


def _extension_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source_config = {
        "model": {"id": "model", "revision": "revision"},
        "generation": {
            "max_new_tokens": 4,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "do_sample": True,
        },
        "evaluation": {
            "samples_per_example": 1,
            "prompt_variant": "standard",
            "seed": 101,
        },
        "reward": {"completion_budget": 4, "gamma": 0.0},
    }
    rows = [
        {
            "run_id": "source",
            "example_id": "a",
            "sample_id": 0,
            "condition": "initial",
            "correct": True,
            "completion_tokens": 3,
            "completion_ids_sha256": sha256_object([1, 2, 3]),
            "truncated": False,
            "reward": {"total": 1.0},
        },
        {
            "run_id": "source",
            "example_id": "b",
            "sample_id": 0,
            "condition": "initial",
            "correct": False,
            "completion_tokens": 4,
            "completion_ids_sha256": sha256_object([4, 5, 6, 7]),
            "truncated": True,
            "reward": {"total": 0.0},
        },
    ]
    atomic_write_yaml(source / "config.resolved.yaml", source_config)
    atomic_write_jsonl(source / "predictions.jsonl", rows)
    atomic_write_json(
        source / "manifest.json",
        {
            "status": "completed",
            "run_type": "evaluation",
            "run_id": "source",
            "dataset_hash": "dataset",
            "split": "dev",
            "prompt_variant": "standard",
            "adapter_hash": None,
            "base_model": source_config["model"],
            "predictions_sha256": sha256_file(source / "predictions.jsonl"),
        },
    )
    target_config = {
        **source_config,
        "generation": {**source_config["generation"], "max_new_tokens": 8},
        "reward": {"completion_budget": 8, "gamma": 0.1},
    }
    expected_manifest = {
        "dataset_hash": "dataset",
        "split": "dev",
        "prompt_variant": "standard",
        "adapter_hash": None,
        "base_model": source_config["model"],
    }
    records = [SimpleNamespace(example_id="a"), SimpleNamespace(example_id="b")]
    return source, target_config, expected_manifest, records, rows


def test_extension_reuses_only_nontruncated_predictions(tmp_path):
    source, target_config, expected_manifest, records, _ = _extension_fixture(tmp_path)
    info, reused, truncated = _prepare_extension_source(
        source,
        expected_manifest=expected_manifest,
        target_config=target_config,
        records=records,
        samples=1,
        output_run_id="extended",
        condition="initial_budget_probe_8",
    )
    assert info["extension_reused_predictions"] == 1
    assert info["extension_regenerated_predictions"] == 1
    assert len(reused) == 1
    assert reused[0]["example_id"] == "a"
    assert reused[0]["run_id"] == "extended"
    assert reused[0]["extension_reused_without_regeneration"] is True
    assert set(truncated) == {("b", 0)}


def test_extension_prefix_requires_exact_saved_token_prefix(tmp_path):
    _, _, _, _, rows = _extension_fixture(tmp_path)
    source_row = rows[1]
    assert _verify_extension_prefix([4, 5, 6, 7, 8, 9], source_row) == 4
    with pytest.raises(ValueError, match="did not reproduce"):
        _verify_extension_prefix([4, 5, 0, 7, 8], source_row)
    with pytest.raises(ValueError, match="ended"):
        _verify_extension_prefix([4, 5, 6], source_row)


def test_extension_rejects_changed_seed(tmp_path):
    source, target_config, expected_manifest, records, _ = _extension_fixture(tmp_path)
    target_config["evaluation"]["seed"] = 999
    with pytest.raises(ValueError, match="evaluation seed"):
        _prepare_extension_source(
            source,
            expected_manifest=expected_manifest,
            target_config=target_config,
            records=records,
            samples=1,
            output_run_id="extended",
            condition="initial_budget_probe_8",
        )
