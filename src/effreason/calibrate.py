from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml

from .utils import atomic_write_json, atomic_write_yaml, read_jsonl, sha256_file, utc_now


def recommend_budget(
    rows: list[dict[str, Any]],
    *,
    current_budget: int,
    candidates: list[int],
    quantile: float = 0.95,
    headroom: float = 1.10,
    max_truncation_rate: float = 0.02,
    min_samples: int = 50,
    min_correct_samples: int = 20,
) -> dict[str, Any]:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    if headroom < 1:
        raise ValueError("headroom must be at least one")
    ordered = sorted({int(candidate) for candidate in candidates if int(candidate) > 0})
    if not ordered:
        raise ValueError("at least one positive budget candidate is required")
    truncation_rate = sum(bool(row.get("truncated", False)) for row in rows) / len(rows) if rows else 0
    correct_lengths = [
        int(row["completion_tokens"])
        for row in rows
        if bool(row.get("correct")) and not bool(row.get("truncated", False))
    ]
    result: dict[str, Any] = {
        "schema_version": "1",
        "method": "smallest candidate >= headroom * correct non-truncated length quantile",
        "current_probe_budget": current_budget,
        "sample_count": len(rows),
        "correct_nontruncated_count": len(correct_lengths),
        "truncation_rate": truncation_rate,
        "quantile": quantile,
        "headroom": headroom,
        "max_truncation_rate": max_truncation_rate,
        "candidates": ordered,
        "recommended_budget": None,
    }
    if len(rows) < min_samples:
        result["status"] = "insufficient_samples"
        result["required_samples"] = min_samples
        return result
    if len(correct_lengths) < min_correct_samples:
        result["status"] = "insufficient_correct_samples"
        result["required_correct_samples"] = min_correct_samples
        return result
    measured_quantile = float(np.quantile(np.asarray(correct_lengths), quantile))
    required = math.ceil(measured_quantile * headroom)
    result["measured_correct_length_quantile"] = measured_quantile
    result["required_with_headroom"] = required
    if truncation_rate > max_truncation_rate:
        result["status"] = "rerun_with_larger_probe"
        result["next_probe_budget"] = next(
            (candidate for candidate in ordered if candidate > current_budget), None
        )
        return result
    recommendation = next((candidate for candidate in ordered if candidate >= required), None)
    if recommendation is None:
        result["status"] = "no_candidate_large_enough"
        return result
    result["status"] = "selected"
    result["recommended_budget"] = recommendation
    return result


def _accuracy(rows: list[dict[str, Any]], variant: str | None = None) -> float | None:
    selected = rows if variant is None else [row for row in rows if row["variant_type"] == variant]
    return mean(bool(row["correct"]) for row in selected) if selected else None


def _correct_mean_tokens(rows: list[dict[str, Any]]) -> float | None:
    values = [int(row["completion_tokens"]) for row in rows if bool(row["correct"])]
    return mean(values) if values else None


def _paired_correct_tokens(
    reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> tuple[int, float | None, float | None]:
    def key(row: dict[str, Any]) -> tuple[str, int]:
        return str(row["example_id"]), int(row.get("sample_id", 0))

    reference = {key(row): row for row in reference_rows}
    candidate = {key(row): row for row in candidate_rows}
    shared = [
        item
        for item in sorted(set(reference).intersection(candidate))
        if bool(reference[item]["correct"]) and bool(candidate[item]["correct"])
    ]
    if not shared:
        return 0, None, None
    return (
        len(shared),
        mean(int(reference[item]["completion_tokens"]) for item in shared),
        mean(int(candidate[item]["completion_tokens"]) for item in shared),
    )


def select_gamma(
    reference_rows: list[dict[str, Any]],
    candidates: list[tuple[float, list[dict[str, Any]], str]],
    *,
    accuracy_margin_pp: float = 5.0,
    premise_margin_pp: float = 5.0,
    minimum_token_reduction_percent: float = 15.0,
) -> dict[str, Any]:
    reference_accuracy = _accuracy(reference_rows)
    reference_premise = _accuracy(reference_rows, "premise")
    reference_tokens = _correct_mean_tokens(reference_rows)
    if reference_accuracy is None or reference_tokens is None:
        raise ValueError("Reference run has no usable accuracy or correct-response lengths")
    reports: list[dict[str, Any]] = []
    for gamma, rows, label in sorted(candidates, key=lambda item: item[0]):
        accuracy = _accuracy(rows)
        premise = _accuracy(rows, "premise")
        tokens = _correct_mean_tokens(rows)
        paired_count, paired_reference_tokens, paired_candidate_tokens = _paired_correct_tokens(
            reference_rows, rows
        )
        accuracy_drop = 100 * (reference_accuracy - accuracy) if accuracy is not None else None
        premise_drop = (
            100 * (reference_premise - premise)
            if reference_premise is not None and premise is not None
            else None
        )
        token_reduction = (
            100 * (paired_reference_tokens - paired_candidate_tokens) / paired_reference_tokens
            if paired_reference_tokens is not None and paired_candidate_tokens is not None
            else None
        )
        eligible = bool(
            accuracy_drop is not None
            and accuracy_drop <= accuracy_margin_pp
            and (premise_drop is None or premise_drop <= premise_margin_pp)
            and token_reduction is not None
            and token_reduction >= minimum_token_reduction_percent
        )
        reports.append(
            {
                "label": label,
                "gamma": gamma,
                "accuracy": accuracy,
                "accuracy_drop_pp": accuracy_drop,
                "premise_accuracy": premise,
                "premise_accuracy_drop_pp": premise_drop,
                "correct_mean_tokens": tokens,
                "paired_correct_count": paired_count,
                "paired_reference_mean_tokens": paired_reference_tokens,
                "paired_candidate_mean_tokens": paired_candidate_tokens,
                "paired_correct_token_reduction_percent": token_reduction,
                "eligible": eligible,
            }
        )
    selected = next((report for report in reports if report["eligible"]), None)
    return {
        "schema_version": "1",
        "status": "selected" if selected else "no_candidate_met_rule",
        "selection_rule": "smallest gamma meeting accuracy, premise, and token-reduction thresholds",
        "thresholds": {
            "accuracy_margin_pp": accuracy_margin_pp,
            "premise_margin_pp": premise_margin_pp,
            "minimum_token_reduction_percent": minimum_token_reduction_percent,
        },
        "reference": {
            "accuracy": reference_accuracy,
            "premise_accuracy": reference_premise,
            "correct_mean_tokens": reference_tokens,
        },
        "candidates": reports,
        "selected_gamma": selected["gamma"] if selected else None,
    }


def _load_evaluation_run(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "completed" or manifest.get("run_type") != "evaluation":
        raise ValueError(f"Expected a completed evaluation run: {path}")
    with (path / "config.resolved.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rows = read_jsonl(path / "predictions.jsonl")
    if not rows:
        raise ValueError(f"Evaluation has no predictions: {path}")
    return manifest, config, rows


def budget_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    manifest, config, rows = _load_evaluation_run(run_dir)
    report = recommend_budget(
        rows,
        current_budget=int(config["generation"]["max_new_tokens"]),
        candidates=args.candidates,
        quantile=args.quantile,
        headroom=args.headroom,
        max_truncation_rate=args.max_truncation_rate,
        min_samples=args.min_samples,
        min_correct_samples=args.min_correct_samples,
    )
    report.update(
        {
            "created_at": utc_now(),
            "source_run": str(run_dir),
            "source_dataset_hash": manifest["dataset_hash"],
            "source_predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        }
    )
    atomic_write_json(args.output, report)
    return report


def gamma_report(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = Path(args.reference_run).resolve()
    reference_manifest, _, reference_rows = _load_evaluation_run(reference_path)
    candidate_data: list[tuple[float, list[dict[str, Any]], str]] = []
    source_runs: list[dict[str, Any]] = []
    for raw_path in args.candidate_runs:
        path = Path(raw_path).resolve()
        manifest, config, rows = _load_evaluation_run(path)
        for field in ("dataset_hash", "split", "prompt_variant"):
            if manifest.get(field) != reference_manifest.get(field):
                raise ValueError(f"Candidate {path} differs on {field}")
        gamma = float(config["reward"]["gamma"])
        candidate_data.append((gamma, rows, path.name))
        source_runs.append(
            {
                "path": str(path),
                "gamma": gamma,
                "predictions_sha256": sha256_file(path / "predictions.jsonl"),
            }
        )
    report = select_gamma(
        reference_rows,
        candidate_data,
        accuracy_margin_pp=args.accuracy_margin_pp,
        premise_margin_pp=args.premise_margin_pp,
        minimum_token_reduction_percent=args.minimum_token_reduction_percent,
    )
    report.update(
        {
            "created_at": utc_now(),
            "reference_run": str(reference_path),
            "reference_predictions_sha256": sha256_file(reference_path / "predictions.jsonl"),
            "candidate_runs": source_runs,
            "dataset_hash": reference_manifest["dataset_hash"],
            "split": reference_manifest["split"],
        }
    )
    atomic_write_json(args.output, report)
    return report


def protocol_report(args: argparse.Namespace) -> dict[str, Any]:
    budget_path = Path(args.budget_decision).resolve()
    gamma_path = Path(args.gamma_decision).resolve()
    with budget_path.open(encoding="utf-8") as stream:
        budget = json.load(stream)
    with gamma_path.open(encoding="utf-8") as stream:
        gamma = json.load(stream)
    if budget.get("status") != "selected" or gamma.get("status") != "selected":
        raise ValueError("Both budget and gamma decisions must have status='selected'")
    report = {
        "schema_version": "1",
        "created_at": utc_now(),
        "status": "development_selected_test_not_frozen",
        "completion_budget": int(budget["recommended_budget"]),
        "gamma": float(gamma["selected_gamma"]),
        "budget_decision": str(budget_path),
        "budget_decision_sha256": sha256_file(budget_path),
        "gamma_decision": str(gamma_path),
        "gamma_decision_sha256": sha256_file(gamma_path),
        "note": "Development choice only; record main steps/seeds and freeze test data before main runs.",
    }
    atomic_write_yaml(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select development-only budget and gamma")
    subparsers = parser.add_subparsers(dest="command", required=True)
    budget = subparsers.add_parser("budget")
    budget.add_argument("--run-dir", required=True)
    budget.add_argument("--output", required=True)
    budget.add_argument("--candidates", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    budget.add_argument("--quantile", type=float, default=0.95)
    budget.add_argument("--headroom", type=float, default=1.10)
    budget.add_argument("--max-truncation-rate", type=float, default=0.02)
    budget.add_argument("--min-samples", type=int, default=50)
    budget.add_argument("--min-correct-samples", type=int, default=20)
    gamma = subparsers.add_parser("gamma")
    gamma.add_argument("--reference-run", required=True)
    gamma.add_argument("--candidate-runs", nargs="+", required=True)
    gamma.add_argument("--output", required=True)
    gamma.add_argument("--accuracy-margin-pp", type=float, default=5.0)
    gamma.add_argument("--premise-margin-pp", type=float, default=5.0)
    gamma.add_argument("--minimum-token-reduction-percent", type=float, default=15.0)
    protocol = subparsers.add_parser("protocol")
    protocol.add_argument("--budget-decision", required=True)
    protocol.add_argument("--gamma-decision", required=True)
    protocol.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "budget":
        report = budget_report(args)
    elif args.command == "gamma":
        report = gamma_report(args)
    else:
        report = protocol_report(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
