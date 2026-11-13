from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .utils import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_object


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _pair_metrics(rows: list[dict[str, Any]], variant: str) -> tuple[int, int, float | None]:
    families: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        families[(row["family_id"], int(row.get("sample_id", 0)))][row["variant_type"]] = bool(
            row["correct"]
        )
    eligible = [values for values in families.values() if "original" in values and variant in values]
    successes = sum(values["original"] and values[variant] for values in eligible)
    return successes, len(eligible), successes / len(eligible) if eligible else None


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No predictions to analyze")
    correctness = [bool(row["correct"]) for row in rows]
    lengths = [int(row["completion_tokens"]) for row in rows]
    correct_lengths = [
        int(row["completion_tokens"]) for row in rows if bool(row["correct"])
    ]
    valid = [row for row in rows if bool(row.get("parse_valid", False))]
    premise = [row for row in rows if row["variant_type"] == "premise"]
    premise_valid = [row for row in premise if bool(row.get("parse_valid", False))]
    surface_success, surface_total, surface_rate = _pair_metrics(rows, "surface")
    premise_success, premise_total, premise_rate = _pair_metrics(rows, "premise")
    families: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        families[(row["family_id"], int(row.get("sample_id", 0)))][row["variant_type"]] = bool(
            row["correct"]
        )
    original_success = [
        values for values in families.values() if values.get("original") is True and "premise" in values
    ]
    conditional_success = sum(values["premise"] for values in original_success)
    errors = Counter(
        str(row.get("parse_error") or row.get("error_type") or "none") for row in rows
    )
    total_latency = sum(float(row.get("latency_seconds", 0.0)) for row in rows)
    accuracy_by_variant = {}
    evaluation_types = {
        "harder" if row.get("split") == "test_harder" else row["variant_type"] for row in rows
    }
    for variant in sorted(evaluation_types):
        variant_rows = [
            row
            for row in rows
            if ("harder" if row.get("split") == "test_harder" else row["variant_type"]) == variant
        ]
        accuracy_by_variant[variant] = {
            "correct": sum(bool(row["correct"]) for row in variant_rows),
            "attempts": len(variant_rows),
            "accuracy": mean(bool(row["correct"]) for row in variant_rows),
        }
    return {
        "attempts": len(rows),
        "correct": sum(correctness),
        "accuracy": mean(correctness),
        "accuracy_by_variant": accuracy_by_variant,
        "completion_tokens": {
            "total": sum(lengths),
            "mean": mean(lengths),
            "median": median(lengths),
            "p90": _quantile(lengths, 0.90),
            "p95": _quantile(lengths, 0.95),
        },
        "correct_completion_tokens": {
            "count": len(correct_lengths),
            "mean": mean(correct_lengths) if correct_lengths else None,
            "median": median(correct_lengths) if correct_lengths else None,
            "p90": _quantile(correct_lengths, 0.90),
        },
        "surface_pair_success": {
            "successes": surface_success,
            "families": surface_total,
            "rate": surface_rate,
        },
        "premise_pair_success": {
            "successes": premise_success,
            "families": premise_total,
            "rate": premise_rate,
        },
        "premise_adaptation_given_original_success": {
            "successes": conditional_success,
            "families": len(original_success),
            "rate": conditional_success / len(original_success) if original_success else None,
        },
        "obsolete_assignment": {
            "count": sum(bool(row.get("obsolete_assignment", False)) for row in premise),
            "premise_attempts": len(premise),
            "rate": (
                sum(bool(row.get("obsolete_assignment", False)) for row in premise) / len(premise)
                if premise
                else None
            ),
            "valid_count": sum(bool(row.get("obsolete_assignment", False)) for row in premise_valid),
            "valid_premise_attempts": len(premise_valid),
            "valid_rate": (
                sum(bool(row.get("obsolete_assignment", False)) for row in premise_valid)
                / len(premise_valid)
                if premise_valid
                else None
            ),
        },
        "validity": {
            "valid": len(valid),
            "valid_rate": len(valid) / len(rows),
            "missing_thinking_boundary": sum(
                not bool(row.get("thinking_boundary_found", False)) for row in rows
            ),
            "truncated": sum(bool(row.get("truncated", False)) for row in rows),
            "repeated": sum(bool(row.get("repeated_output", False)) for row in rows),
            "errors": dict(sorted(errors.items())),
        },
        "cost": {"total_generated_tokens": sum(lengths), "total_latency_seconds": total_latency},
    }


def _family_values(
    rows: list[dict[str, Any]], value: Callable[[list[dict[str, Any]]], float]
) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family_id"]].append(row)
    return {family: value(group) for family, group in groups.items()}


def clustered_bootstrap_difference(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    value: Callable[[list[dict[str, Any]]], float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> dict[str, float | int]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    values_a = _family_values(rows_a, value)
    values_b = _family_values(rows_b, value)
    families = sorted(set(values_a).intersection(values_b))
    if not families:
        raise ValueError("Runs do not share any family IDs")
    observed = mean(values_b[family] - values_a[family] for family in families)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sample = [rng.choice(families) for _ in families]
        draws.append(mean(values_b[family] - values_a[family] for family in sample))
    alpha = (1 - confidence) / 2
    return {
        "families": len(families),
        "difference_b_minus_a": observed,
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1 - alpha)),
        "confidence": confidence,
        "resamples": resamples,
    }


def _load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "completed":
        raise ValueError(f"Run is not complete: {path}")
    with (path / "config.resolved.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rows = read_jsonl(path / "predictions.jsonl")
    return manifest, config, rows


def _comparison_signature(config: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(config))
    normalized["experiment"].pop("name", None)
    normalized["experiment"].pop("condition", None)
    normalized["reward"].pop("gamma", None)
    normalized["artifacts"].pop("output_dir", None)
    return sha256_object(normalized)


def _validate_comparable(runs: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]) -> None:
    splits = {manifest["split"] for manifest, _, _ in runs}
    datasets = {manifest["dataset_hash"] for manifest, _, _ in runs}
    if len(splits) != 1:
        raise ValueError(f"Cannot compare mixed splits: {sorted(splits)}")
    if len(datasets) != 1:
        raise ValueError("Cannot compare different dataset hashes")
    # Prompt wording is intentionally allowed to differ for the concise control.
    signatures = {_comparison_signature(config) for _, config, _ in runs}
    if len(signatures) != 1:
        raise ValueError("Runs differ in settings beyond condition, gamma, output path, or prompt control")


def _figure_accuracy_length(results: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for result in results:
        axis.scatter(
            result["metrics"]["completion_tokens"]["mean"],
            result["metrics"]["accuracy"],
            s=75,
            label=result["label"],
        )
    axis.set_xlabel("Mean completion tokens")
    axis.set_ylabel("Exact accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _figure_pair_metrics(results: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [result["label"] for result in results]
    categories = ["original", "surface", "premise", "harder", "surface_pair", "premise_pair"]
    x = np.arange(len(categories))
    width = 0.8 / len(results)
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, result in enumerate(results):
        metrics = result["metrics"]
        by_variant = metrics["accuracy_by_variant"]
        values = [
            by_variant.get("original", {}).get("accuracy") or 0,
            by_variant.get("surface", {}).get("accuracy") or 0,
            by_variant.get("premise", {}).get("accuracy") or 0,
            by_variant.get("harder", {}).get("accuracy") or 0,
            metrics["surface_pair_success"]["rate"] or 0,
            metrics["premise_pair_success"]["rate"] or 0,
        ]
        axis.bar(x + (index - (len(results) - 1) / 2) * width, values, width, label=labels[index])
    axis.set_xticks(
        x,
        ["Original", "Surface", "Premise", "Harder", "Original + surface", "Original + premise"],
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proportion correct")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _transfer_contrast(group: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in group:
        grouped[row["variant_type"]].append(float(bool(row["correct"])))
    variants = {variant: mean(values) for variant, values in grouped.items()}
    if not {"original", "surface", "premise"}.issubset(variants):
        raise ValueError("Transfer contrast requires original, surface, and premise variants")
    return (variants["premise"] - variants["original"]) - (
        variants["surface"] - variants["original"]
    )


def _complete_transfer_rows(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {"original", "surface", "premise"}

    def complete(rows: list[dict[str, Any]]) -> set[str]:
        variants: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            variants[row["family_id"]].add(row["variant_type"])
        return {family for family, present in variants.items() if required.issubset(present)}

    shared = complete(rows_a).intersection(complete(rows_b))
    return (
        [row for row in rows_a if row["family_id"] in shared],
        [row for row in rows_b if row["family_id"] in shared],
    )


def _trace_sample(
    runs: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    *,
    maximum_families: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_rows: dict[str, list[list[dict[str, Any]]]] = {}
    shared = set.intersection(*({row["family_id"] for row in rows} for _, _, rows in runs))
    for family in shared:
        family_rows[family] = [
            [row for row in rows if row["family_id"] == family] for _, _, rows in runs
        ]
    strata: dict[str, list[str]] = defaultdict(list)
    for family, per_run in family_rows.items():
        run_success = [all(bool(row["correct"]) for row in rows) for rows in per_run]
        pattern = "".join("1" if success else "0" for success in run_success)
        strata[pattern].append(family)
    rng = random.Random(seed)
    for families in strata.values():
        rng.shuffle(families)
    selected: list[str] = []
    while len(selected) < min(maximum_families, len(shared)):
        changed = False
        for pattern in sorted(strata):
            if strata[pattern] and len(selected) < maximum_families:
                selected.append(strata[pattern].pop())
                changed = True
        if not changed:
            break
    blinded = {index: f"condition_{chr(65 + index)}" for index in range(len(runs))}
    rows_out: list[dict[str, Any]] = []
    for family in selected:
        for run_index, rows in enumerate(family_rows[family]):
            for row in rows:
                rows_out.append(
                    {
                        "schema_version": "1",
                        "family_id": family,
                        "condition": blinded[run_index],
                        "variant_type": row["variant_type"],
                        "raw_completion": row.get("raw_completion"),
                        "correct": row["correct"],
                        "labels": {
                            "explicit_assignment_enumeration": None,
                            "consistency_checking": None,
                            "self_correction": None,
                            "repetition": None,
                            "unsupported_final_jump": None,
                        },
                    }
                )
    codebook = {
        "schema_version": "1",
        "selection": "Up to 30 shared families, round-robin stratified by whole-family success pattern.",
        "blinding": blinded,
        "labels": {
            "explicit_assignment_enumeration": "Explicitly tests two or more complete role assignments.",
            "consistency_checking": "Checks a candidate assignment against every speaker's truth status.",
            "self_correction": "Rejects or revises an earlier candidate after detecting a conflict.",
            "repetition": "Substantively repeats a prior reasoning step without adding a constraint.",
            "unsupported_final_jump": (
                "Final assignment appears without an explicit link to prior constraints."
            ),
        },
        "values": "Set each label to true, false, or uncertain during manual review.",
    }
    return rows_out, codebook


def _figure_training_curves(run_dirs: list[Path], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    keys = [
        ("Total Reward", ["mean_total_reward", "reward"]),
        ("Completion Length", ["mean_completion_tokens", "completions/mean_length"]),
        ("Exact Accuracy", ["mean_accuracy", "accuracy"]),
        ("Truncation Rate", ["truncation_rate", "completions/clipped_ratio"]),
    ]
    plotted = False
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        rows = read_jsonl(metrics_path)
        steps = [row.get("step", index) for index, row in enumerate(rows)]
        for axis, (title, candidates) in zip(axes.flat, keys, strict=True):
            values = [
                next((row.get(key) for key in candidates if row.get(key) is not None), None)
                for row in rows
            ]
            points = [(step, value) for step, value in zip(steps, values, strict=True) if value is not None]
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], label=run_dir.name)
                plotted = True
            axis.set_title(title)
            axis.set_xlabel("Step")
    if not plotted:
        axes[0, 0].text(0.5, 0.5, "No training metrics supplied", ha="center", va="center")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze_runs(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    resamples: int,
    confidence: float,
    seed: int,
    training_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    runs = [_load_run(path) for path in run_dirs]
    _validate_comparable(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for run_dir, (manifest, _, rows) in zip(run_dirs, runs, strict=True):
        label = f"{manifest['condition']}:{manifest['prompt_variant']}:{run_dir.name}"
        results.append({"label": label, "manifest": manifest, "metrics": summarize_predictions(rows)})
    comparisons: list[dict[str, Any]] = []
    if len(runs) >= 2:
        reference_rows = runs[0][2]
        for index in range(1, len(runs)):
            rows = runs[index][2]
            transfer_reference, transfer_rows = _complete_transfer_rows(reference_rows, rows)
            accuracy_comparison = clustered_bootstrap_difference(
                reference_rows,
                rows,
                value=lambda group: mean(bool(row["correct"]) for row in group),
                resamples=resamples,
                confidence=confidence,
                seed=seed,
            )
            token_comparison = clustered_bootstrap_difference(
                reference_rows,
                rows,
                value=lambda group: mean(int(row["completion_tokens"]) for row in group),
                resamples=resamples,
                confidence=confidence,
                seed=seed + 1,
            )
            mean_reference_tokens = results[0]["metrics"]["completion_tokens"]["mean"]
            comparisons.append(
                {
                    "a": results[0]["label"],
                    "b": results[index]["label"],
                    "accuracy": accuracy_comparison,
                    "accuracy_difference_pp": accuracy_comparison["difference_b_minus_a"] * 100,
                    "completion_tokens": token_comparison,
                    "token_reduction_percent": (
                        -100 * token_comparison["difference_b_minus_a"] / mean_reference_tokens
                    ),
                    "premise_minus_surface_contrast": (
                        clustered_bootstrap_difference(
                            transfer_reference,
                            transfer_rows,
                            value=_transfer_contrast,
                            resamples=resamples,
                            confidence=confidence,
                            seed=seed + 2,
                        )
                        if transfer_reference
                        else None
                    ),
                }
            )
    summary = {"schema_version": "1", "runs": results, "comparisons": comparisons}
    atomic_write_json(output_dir / "summary.json", summary)
    table_rows = [
        {
            "schema_version": "1",
            "label": result["label"],
            "attempts": result["metrics"]["attempts"],
            "accuracy": result["metrics"]["accuracy"],
            "mean_completion_tokens": result["metrics"]["completion_tokens"]["mean"],
            "surface_pair_success": result["metrics"]["surface_pair_success"]["rate"],
            "premise_pair_success": result["metrics"]["premise_pair_success"]["rate"],
        }
        for result in results
    ]
    pd.DataFrame(table_rows).to_csv(output_dir / "metrics.csv", index=False)
    _figure_accuracy_length(results, output_dir / "accuracy_vs_length.png")
    _figure_pair_metrics(results, output_dir / "accuracy_and_pairs.png")
    _figure_training_curves(training_run_dirs or run_dirs, output_dir / "training_curves.png")
    if len(runs) >= 2:
        trace_rows, codebook = _trace_sample(runs, maximum_families=30, seed=seed)
        atomic_write_jsonl(output_dir / "trace_sample.jsonl", trace_rows)
        atomic_write_json(output_dir / "trace_codebook.json", codebook)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze saved evaluation predictions on CPU")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--training-run-dirs", nargs="+")
    parser.add_argument("--output", default="analysis")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not 0 < args.confidence < 1 or math.isnan(args.confidence):
        raise ValueError("--confidence must be between zero and one")
    summary = analyze_runs(
        [Path(path).resolve() for path in args.run_dirs],
        Path(args.output).resolve(),
        resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
        training_run_dirs=(
            [Path(path).resolve() for path in args.training_run_dirs]
            if args.training_run_dirs
            else None
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
