from __future__ import annotations

import pytest

from effreason.calibrate import recommend_budget, select_gamma


def _row(index: int, *, correct: bool, tokens: int, variant: str = "original", truncated=False):
    return {
        "example_id": f"example-{index}",
        "family_id": f"family-{index // 3}",
        "variant_type": variant,
        "correct": correct,
        "completion_tokens": tokens,
        "truncated": truncated,
    }


def test_budget_uses_correct_length_quantile_headroom_and_standard_candidate():
    rows = [_row(index, correct=True, tokens=600) for index in range(60)]
    report = recommend_budget(
        rows,
        current_budget=4096,
        candidates=[512, 1024, 2048, 4096],
    )
    assert report["status"] == "selected"
    assert report["required_with_headroom"] == 660
    assert report["recommended_budget"] == 1024


def test_budget_requires_larger_probe_when_truncation_is_too_high():
    rows = [
        _row(index, correct=True, tokens=4096, truncated=index < 3) for index in range(60)
    ]
    report = recommend_budget(
        rows,
        current_budget=4096,
        candidates=[1024, 2048, 4096, 8192],
    )
    assert report["status"] == "rerun_with_larger_probe"
    assert report["next_probe_budget"] == 8192


def test_gamma_selects_smallest_candidate_that_meets_predeclared_rule():
    reference = [
        _row(index, correct=True, tokens=1000, variant="premise" if index % 3 == 0 else "original")
        for index in range(30)
    ]
    gamma_01 = [
        {**row, "completion_tokens": 800, "correct": row["correct"]}
        for row in reference
    ]
    gamma_02 = [
        {**row, "completion_tokens": 600, "correct": row["correct"]}
        for row in reference
    ]
    report = select_gamma(
        reference,
        [(0.2, gamma_02, "gamma-02"), (0.1, gamma_01, "gamma-01")],
        accuracy_margin_pp=5,
        premise_margin_pp=5,
        minimum_token_reduction_percent=15,
    )
    assert report["status"] == "selected"
    assert report["selected_gamma"] == pytest.approx(0.1)
