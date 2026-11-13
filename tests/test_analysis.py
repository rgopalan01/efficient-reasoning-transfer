from __future__ import annotations

import pytest

from effreason.analyze import analyze_runs, clustered_bootstrap_difference, summarize_predictions
from effreason.utils import atomic_write_json, atomic_write_jsonl, atomic_write_yaml


def _row(family, variant, correct, tokens, *, valid=True, obsolete=False):
    return {
        "family_id": family,
        "variant_type": variant,
        "sample_id": 0,
        "correct": correct,
        "completion_tokens": tokens,
        "parse_valid": valid,
        "parse_error": None if valid else "malformed_json",
        "error_type": None if correct else "wrong_assignment",
        "obsolete_assignment": obsolete,
        "truncated": False,
        "repeated_output": False,
        "latency_seconds": 1.0,
    }


def test_manually_calculable_metrics_and_obsolete_rate():
    rows = [
        _row("a", "original", True, 10),
        _row("a", "surface", True, 8),
        _row("a", "premise", False, 7, obsolete=True),
        _row("b", "original", True, 12),
        _row("b", "surface", False, 9),
        _row("b", "premise", True, 11),
    ]
    summary = summarize_predictions(rows)
    assert summary["accuracy"] == pytest.approx(4 / 6)
    assert summary["surface_pair_success"]["rate"] == pytest.approx(0.5)
    assert summary["premise_pair_success"]["rate"] == pytest.approx(0.5)
    assert summary["premise_adaptation_given_original_success"]["rate"] == pytest.approx(0.5)
    assert summary["obsolete_assignment"]["rate"] == pytest.approx(0.5)
    assert summary["completion_tokens"]["total"] == 57


def test_clustered_bootstrap_resamples_families_and_is_reproducible():
    rows_a = [_row("a", "original", False, 10), _row("b", "original", True, 20)]
    rows_b = [_row("a", "original", True, 8), _row("b", "original", True, 18)]
    first = clustered_bootstrap_difference(
        rows_a,
        rows_b,
        value=lambda group: sum(row["correct"] for row in group) / len(group),
        resamples=200,
        seed=3,
    )
    second = clustered_bootstrap_difference(
        rows_a,
        rows_b,
        value=lambda group: sum(row["correct"] for row in group) / len(group),
        resamples=200,
        seed=3,
    )
    assert first == second
    assert first["families"] == 2
    assert first["difference_b_minus_a"] == pytest.approx(0.5)


def test_synthetic_runs_produce_tables_and_all_required_figures(tmp_path):
    config = {
        "experiment": {"name": "fixture", "condition": "fixture", "seed": 1},
        "reward": {"gamma": 0.0},
        "artifacts": {"output_dir": "unused"},
        "shared": {"decoding": "matched"},
    }
    rows_a = [
        _row("a", "original", False, 10),
        _row("a", "surface", True, 9),
        _row("a", "premise", False, 8),
        _row("b", "original", True, 12),
        _row("b", "surface", True, 11),
        _row("b", "premise", True, 10),
    ]
    rows_b = [{**row, "correct": True, "completion_tokens": row["completion_tokens"] - 2} for row in rows_a]
    run_dirs = []
    for condition, rows in [("accuracy", rows_a), ("efficiency", rows_b)]:
        run_dir = tmp_path / condition
        run_dir.mkdir()
        run_config = {**config, "experiment": {**config["experiment"], "condition": condition}}
        atomic_write_yaml(run_dir / "config.resolved.yaml", run_config)
        atomic_write_json(
            run_dir / "manifest.json",
            {
                "status": "completed",
                "split": "dev",
                "dataset_hash": "same-data",
                "condition": condition,
                "prompt_variant": "standard",
            },
        )
        atomic_write_jsonl(run_dir / "predictions.jsonl", rows)
        run_dirs.append(run_dir)
    output = tmp_path / "analysis"
    summary = analyze_runs(run_dirs, output, resamples=100, confidence=0.95, seed=4)
    assert len(summary["runs"]) == 2
    assert (output / "summary.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "accuracy_vs_length.png").stat().st_size > 0
    assert (output / "accuracy_and_pairs.png").stat().st_size > 0
    assert (output / "training_curves.png").stat().st_size > 0
    assert (output / "trace_sample.jsonl").is_file()
    assert (output / "trace_codebook.json").is_file()
