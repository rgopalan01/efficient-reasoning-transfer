from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
OUTPUT_INSTRUCTION = (
    "Solve the puzzle. End your response with exactly one <answer>...</answer> block. Inside "
    'it, write one valid JSON object mapping every puzzle name to "knight" or "knave"; '
    "include every name exactly once and no other keys."
)
CONCISE_INSTRUCTION = "Reason as concisely as possible while preserving correctness."


class DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class ParseResult:
    valid: bool
    assignment: dict[str, str] | None
    error: str | None
    thinking_boundary_found: bool
    answer_block_count: int
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    parse: ParseResult
    error_type: str | None
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "correct": self.correct,
            "error_type": self.error_type,
            "parse": self.parse.to_dict(),
        }


@dataclass(frozen=True)
class RewardResult:
    accuracy: float
    completion_tokens: int
    length_fraction: float
    length_penalty: float
    total_reward: float
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prompt(question: str, *, variant: str = "standard") -> str:
    if variant not in {"standard", "concise"}:
        raise ValueError(f"Unknown prompt variant: {variant}")
    parts = [question.strip()]
    if variant == "concise":
        parts.append(CONCISE_INSTRUCTION)
    parts.append(OUTPUT_INSTRUCTION)
    return "\n\n".join(parts)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate key: {key}")
        result[key] = value
    return result


def parse_final_answer(completion: str, expected_names: list[str]) -> ParseResult:
    if not isinstance(completion, str) or not completion.strip():
        return ParseResult(False, None, "empty_output", False, 0)
    boundary = completion.rfind("</think>")
    boundary_found = boundary >= 0
    final_region = completion[boundary + len("</think>") :] if boundary_found else completion
    blocks = ANSWER_PATTERN.findall(final_region)
    if len(blocks) != 1:
        error = "missing_answer_block" if not blocks else "multiple_answer_blocks"
        return ParseResult(False, None, error, boundary_found, len(blocks))
    try:
        value = json.loads(blocks[0], object_pairs_hook=_unique_object)
    except DuplicateKeyError:
        return ParseResult(False, None, "duplicate_json_key", boundary_found, 1)
    except json.JSONDecodeError:
        return ParseResult(False, None, "malformed_json", boundary_found, 1)
    if not isinstance(value, dict):
        return ParseResult(False, None, "answer_not_object", boundary_found, 1)

    canonical_names = {name.casefold(): name for name in expected_names}
    if len(canonical_names) != len(expected_names):
        raise ValueError("Expected names must be unique under case normalization")
    normalized: dict[str, str] = {}
    for raw_name, raw_role in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_role, str):
            return ParseResult(False, None, "non_string_assignment", boundary_found, 1)
        key = raw_name.strip().casefold()
        if key not in canonical_names:
            return ParseResult(False, None, "unknown_name", boundary_found, 1)
        visible_name = canonical_names[key]
        if visible_name in normalized:
            return ParseResult(False, None, "duplicate_normalized_name", boundary_found, 1)
        role = raw_role.strip().casefold()
        if role not in {"knight", "knave"}:
            return ParseResult(False, None, "unknown_role", boundary_found, 1)
        normalized[visible_name] = role
    if set(normalized) != set(expected_names):
        return ParseResult(False, None, "incomplete_names", boundary_found, 1)
    return ParseResult(True, normalized, None, boundary_found, 1)


def expected_visible_assignment(
    characters: dict[str, str], solution: dict[str, bool]
) -> dict[str, str]:
    if set(characters) != set(solution):
        raise ValueError("Character and solution roles do not align")
    return {
        characters[role]: "knight" if bool(solution[role]) else "knave"
        for role in sorted(characters)
    }


def score_assignment(
    completion: str,
    characters: dict[str, str],
    solution: dict[str, bool],
) -> ScoreResult:
    expected = expected_visible_assignment(characters, solution)
    parsed = parse_final_answer(completion, list(expected))
    if not parsed.valid:
        return ScoreResult(False, parsed, parsed.error)
    correct = parsed.assignment == expected
    return ScoreResult(correct, parsed, None if correct else "wrong_assignment")


def compute_reward(correct: bool, completion_tokens: int, budget: int, gamma: float) -> RewardResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 <= completion_tokens <= budget:
        raise ValueError(f"completion token count {completion_tokens} outside [0, {budget}]")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")
    accuracy = float(bool(correct))
    fraction = completion_tokens / budget
    penalty = gamma * fraction
    return RewardResult(accuracy, completion_tokens, fraction, penalty, accuracy - penalty)


def classify_repetition(completion: str) -> bool:
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    return len(lines) >= 4 and len(set(lines[-4:])) == 1
