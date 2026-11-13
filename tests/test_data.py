from __future__ import annotations

import itertools

import pytest

from effreason.data import (
    _make_record,
    audit_manifest,
    canonical_structure,
    evaluate_statement,
    find_solutions,
    mutate_premise,
    structural_hash,
)
from effreason.utils import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_object

TT0 = ("telling-truth", 0)
TT1 = ("telling-truth", 1)


@pytest.mark.parametrize(
    ("statement", "assignment", "expected"),
    [
        (("telling-truth", 0), (True, False), True),
        (("lying", 0), (True, False), False),
        (("not", ("telling-truth", 0)), (True, False), False),
        (("and", TT0, TT1), (True, False), False),
        (("or", TT0, TT1), (True, False), True),
        (("->", TT0, TT1), (True, False), False),
        (("<=>", TT0, TT1), (True, False), False),
    ],
)
def test_all_supported_operators(statement, assignment, expected):
    assert evaluate_statement(statement, assignment) is expected


def test_oracle_unique_ambiguous_and_inconsistent():
    unique = [("and", TT0, TT1), ("lying", 0)]
    assert find_solutions(unique) == [(False, True)]
    assert set(find_solutions([TT0])) == {(False,), (True,)}
    assert find_solutions([("lying", 0)]) == []


def _permute_roles(statements, permutation):
    old_to_new = {old: new for new, old in enumerate(permutation)}

    def remap(statement):
        if statement[0] in {"telling-truth", "lying"}:
            return (statement[0], old_to_new[statement[1]])
        return (statement[0], *(remap(child) for child in statement[1:]))

    return [remap(statements[old]) for old in permutation]


def test_canonicalization_collides_under_role_permutation():
    statements = [("and", TT0, TT1), ("lying", 0)]
    for permutation in itertools.permutations(range(2)):
        assert canonical_structure(_permute_roles(statements, permutation)) == canonical_structure(
            statements
        )


def test_implication_direction_is_preserved():
    forward = [("->", TT0, TT1), ("lying", 0)]
    reverse = [("->", TT1, TT0), ("lying", 0)]
    assert structural_hash(forward) != structural_hash(reverse)


def test_biconditional_children_are_canonicalized_as_commutative():
    forward = [("<=>", TT0, TT1), ("lying", 0)]
    reverse = [("<=>", TT1, TT0), ("lying", 0)]
    assert structural_hash(forward) == structural_hash(reverse)


def test_accepted_premise_mutation_changes_unique_solution():
    statements = [("and", TT0, TT1), ("lying", 0)]
    original = find_solutions(statements)[0]
    result = mutate_premise(statements, original, seed=7)
    assert result is not None
    mutated, solution, metadata = result
    assert find_solutions(mutated) == [solution]
    assert solution != original
    assert metadata["before"] != metadata["after"]


def test_surface_renaming_changes_only_rendered_names_and_template():
    statements = [("and", TT0, TT1), ("lying", 0)]
    solution = find_solutions(statements)[0]
    shared = {
        "split": "dev",
        "family_id": "family",
        "statements": statements,
        "solution": solution,
        "generation": {"source": "fixture"},
    }
    original = _make_record(
        **shared,
        variant_type="original",
        parent_id=None,
        names=["Alice", "Bob"],
        mutation=None,
    )
    surface = _make_record(
        **shared,
        variant_type="surface",
        parent_id=original.example_id,
        names=["Nora", "Omar"],
        mutation={"kind": "rename_and_template", "template": 1},
        template=1,
    )
    assert surface.structural_hash == original.structural_hash
    assert surface.solution == original.solution
    assert surface.statements_ast == original.statements_ast
    assert surface.characters != original.characters
    assert surface.question != original.question


def test_split_audit_detects_structural_collision(tmp_path):
    statements = [("and", TT0, TT1), ("lying", 0)]
    solution = find_solutions(statements)[0]
    generation = {"source": "fixture"}
    first = _make_record(
        split="train",
        variant_type="original",
        family_id="family-a",
        parent_id=None,
        statements=statements,
        names=["Alice", "Bob"],
        solution=solution,
        generation=generation,
        mutation=None,
    )
    second = _make_record(
        split="dev",
        variant_type="original",
        family_id="family-b",
        parent_id=None,
        statements=statements,
        names=["Nora", "Omar"],
        solution=solution,
        generation=generation,
        mutation=None,
    )
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    atomic_write_jsonl(train_path, [first.to_dict()])
    atomic_write_jsonl(dev_path, [second.to_dict()])
    splits = {
        "train": {"path": str(train_path), "sha256": sha256_file(train_path), "records": 1},
        "dev": {"path": str(dev_path), "sha256": sha256_file(dev_path), "records": 1},
    }
    manifest = {"schema_version": "1", "splits": splits, "dataset_hash": sha256_object(splits)}
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="structure crosses splits"):
        audit_manifest(manifest_path)
