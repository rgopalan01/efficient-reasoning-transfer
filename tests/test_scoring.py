from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from effreason.scoring import compute_reward, parse_final_answer, score_assignment
from effreason.train import TrlRewardAdapter, compute_batch_rewards, prepare_training_rows

NAMES = ["Alice", "Bob"]
CHARACTERS = {"role_0": "Alice", "role_1": "Bob"}
SOLUTION = {"role_0": True, "role_1": False}


def test_native_qwen_closing_boundary_is_valid_without_opening_tag():
    result = parse_final_answer(
        'reasoning only</think>\n<answer>{"Alice":"Knight", "Bob":"KNAVE"}</answer>', NAMES
    )
    assert result.valid
    assert result.thinking_boundary_found
    assert result.assignment == {"Alice": "knight", "Bob": "knave"}


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('<answer>{"Alice":"knight"}</answer>', "incomplete_names"),
        ('<answer>{"Alice":"knight","Eve":"knave"}</answer>', "unknown_name"),
        ('<answer>{"Alice":"wizard","Bob":"knave"}</answer>', "unknown_role"),
        ('<answer>{"Alice":"knight",}</answer>', "malformed_json"),
        (
            '<answer>{"Alice":"knight","Alice":"knave","Bob":"knave"}</answer>',
            "duplicate_json_key",
        ),
        (
            '<answer>{"Alice":"knight","Bob":"knave"}</answer>'
            '<answer>{"Alice":"knave","Bob":"knight"}</answer>',
            "multiple_answer_blocks",
        ),
    ],
)
def test_adversarial_parser_cases(text, error):
    result = parse_final_answer(text, NAMES)
    assert not result.valid
    assert result.error == error


def test_correct_guess_in_reasoning_does_not_override_wrong_final_answer():
    completion = (
        '{"Alice":"knight","Bob":"knave"}</think>'
        '<answer>{"Alice":"knave","Bob":"knight"}</answer>'
    )
    result = score_assignment(completion, CHARACTERS, SOLUTION)
    assert not result.correct
    assert result.error_type == "wrong_assignment"


def test_reward_contract_and_bounded_ordering():
    assert compute_reward(True, 80, 100, 0).total_reward == 1
    penalized = compute_reward(True, 50, 100, 0.2)
    assert penalized.length_penalty == pytest.approx(0.1)
    assert penalized.total_reward == pytest.approx(0.9)
    assert compute_reward(True, 100, 100, 0.2).total_reward > compute_reward(
        False, 0, 100, 0.2
    ).total_reward
    with pytest.raises(ValueError):
        compute_reward(True, 101, 100, 0.2)


def test_batch_rewards_use_exact_completion_id_lengths_and_align_gold():
    completions = [
        '<answer>{"Alice":"knight","Bob":"knave"}</answer>',
        '<answer>{"Alice":"knave","Bob":"knight"}</answer>',
    ]
    characters = [json.dumps(CHARACTERS)] * 2
    solutions = [json.dumps(SOLUTION), json.dumps({"role_0": False, "role_1": True})]
    rewards, components = compute_batch_rewards(
        completions,
        [[10, 11, 12], [20, 21]],
        characters,
        solutions,
        budget=10,
        gamma=0.2,
    )
    assert [component["completion_tokens"] for component in components] == [3, 2]
    assert [component["accuracy"] for component in components] == [1, 1]
    assert rewards == pytest.approx([0.94, 0.96])


def test_batch_rewards_reject_metadata_misalignment():
    with pytest.raises(ValueError, match="not aligned"):
        compute_batch_rewards(["x"], [[1]], [], [], budget=10, gamma=0.1)


def test_reward_adapter_verifies_repeated_generation_grouping():
    adapter = TrlRewardAdapter(budget=10, gamma=0.1, num_generations=2)
    completion = '<answer>{"Alice":"knight","Bob":"knave"}</answer>'
    shared = {
        "completions": [completion, completion],
        "completion_ids": [[1], [1, 2]],
        "characters_json": [json.dumps(CHARACTERS)] * 2,
        "solution_json": [json.dumps(SOLUTION)] * 2,
    }
    rewards = adapter(**shared, example_id=["same", "same"])
    assert len(rewards) == 2
    with pytest.raises(ValueError, match="not grouped"):
        adapter(**shared, example_id=["first", "second"])


def test_training_rows_use_conversational_format_for_native_chat_template():
    record = SimpleNamespace(
        question="Alice says that Bob is a knave.",
        example_id="example-1",
        characters=CHARACTERS,
        solution=SOLUTION,
    )
    row = prepare_training_rows([record])[0]
    assert row["prompt"][0]["role"] == "user"
    assert row["prompt"][0]["content"].endswith("include every name exactly once and no other keys.")
