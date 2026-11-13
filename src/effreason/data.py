from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    read_jsonl,
    resolve_path,
    sha256_file,
    sha256_object,
    utc_now,
)

SCHEMA_VERSION = "1"
DEFAULT_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Erin",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "James",
]
SURFACE_NAMES = [
    "Nora",
    "Omar",
    "Priya",
    "Quinn",
    "Rosa",
    "Sam",
    "Talia",
    "Uma",
    "Victor",
    "Wren",
]
PRIMITIVES = {"telling-truth", "lying"}
OPERATORS = PRIMITIVES | {"not", "and", "or", "->", "<=>"}


@dataclass(frozen=True)
class PuzzleRecord:
    schema_version: str
    example_id: str
    family_id: str
    variant_type: str
    parent_id: str | None
    split: str
    question: str
    statements_ast: list[Any]
    characters: dict[str, str]
    solution: dict[str, bool]
    structural_hash: str
    difficulty: dict[str, int]
    generation: dict[str, Any]
    mutation: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PuzzleRecord:
        return cls(**value)


def _tuple_ast(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_tuple_ast(item) for item in value)
    return value


def evaluate_statement(statement: Any, assignment: tuple[bool, ...] | list[bool]) -> bool:
    statement = _tuple_ast(statement)
    operator = statement[0]
    if operator == "telling-truth":
        return bool(assignment[statement[1]])
    if operator == "lying":
        return not bool(assignment[statement[1]])
    if operator == "not":
        return not evaluate_statement(statement[1], assignment)
    if operator == "and":
        return all(evaluate_statement(child, assignment) for child in statement[1:])
    if operator == "or":
        return any(evaluate_statement(child, assignment) for child in statement[1:])
    if operator == "->":
        return (not evaluate_statement(statement[1], assignment)) or evaluate_statement(
            statement[2], assignment
        )
    if operator == "<=>":
        return evaluate_statement(statement[1], assignment) == evaluate_statement(
            statement[2], assignment
        )
    raise KeyError(f"Unknown statement operator: {operator}")


def find_solutions(statements: list[Any] | tuple[Any, ...]) -> list[tuple[bool, ...]]:
    statements = [_tuple_ast(statement) for statement in statements]
    solutions: list[tuple[bool, ...]] = []
    for assignment in itertools.product([False, True], repeat=len(statements)):
        consistent = all(
            bool(assignment[index]) == evaluate_statement(statement, assignment)
            for index, statement in enumerate(statements)
        )
        if consistent:
            solutions.append(assignment)
    return solutions


def _remap_ast(statement: Any, old_to_new: dict[int, int]) -> Any:
    statement = _tuple_ast(statement)
    operator = statement[0]
    if operator in PRIMITIVES:
        return (operator, old_to_new[int(statement[1])])
    children = [_remap_ast(child, old_to_new) for child in statement[1:]]
    if operator in {"and", "or", "<=>"}:
        children.sort(key=_serialize_ast)
    return (operator, *children)


def _serialize_ast(statement: Any) -> str:
    statement = _tuple_ast(statement)
    operator = statement[0]
    if operator in PRIMITIVES:
        return f"{operator}:{statement[1]}"
    return f"{operator}({','.join(_serialize_ast(child) for child in statement[1:])})"


def canonical_structure(statements: list[Any] | tuple[Any, ...]) -> str:
    statements = [_tuple_ast(statement) for statement in statements]
    candidates: list[str] = []
    for new_to_old in itertools.permutations(range(len(statements))):
        old_to_new = {old: new for new, old in enumerate(new_to_old)}
        reordered = [_remap_ast(statements[old], old_to_new) for old in new_to_old]
        candidates.append("|".join(_serialize_ast(statement) for statement in reordered))
    return min(candidates)


def structural_hash(statements: list[Any] | tuple[Any, ...]) -> str:
    return sha256_object(canonical_structure(statements))


def _ast_stats(statement: Any, depth: int = 1) -> tuple[int, int, Counter[str]]:
    statement = _tuple_ast(statement)
    operator = statement[0]
    counts = Counter([operator])
    if operator in PRIMITIVES:
        return 1, depth, counts
    size = 1
    maximum_depth = depth
    for child in statement[1:]:
        child_size, child_depth, child_counts = _ast_stats(child, depth + 1)
        size += child_size
        maximum_depth = max(maximum_depth, child_depth)
        counts.update(child_counts)
    return size, maximum_depth, counts


def difficulty_metadata(statements: list[Any]) -> dict[str, int]:
    sizes: list[int] = []
    depths: list[int] = []
    counts: Counter[str] = Counter()
    for statement in statements:
        size, depth, operators = _ast_stats(statement)
        sizes.append(size)
        depths.append(depth)
        counts.update(operators)
    result = {
        "n_people": len(statements),
        "ast_size": sum(sizes),
        "max_depth": max(depths, default=0),
    }
    result.update({f"operator_{key}": value for key, value in sorted(counts.items())})
    return result


def _format_statement(statement: Any, names: list[str], *, nested: bool = False) -> str:
    statement = _tuple_ast(statement)
    operator = statement[0]
    if operator == "telling-truth":
        return f"{names[statement[1]]} is a knight"
    if operator == "lying":
        return f"{names[statement[1]]} is a knave"
    if operator == "not":
        child = _format_statement(statement[1], names, nested=True)
        return f"it is not the case that ({child})"
    if operator in {"and", "or"}:
        text = f" {operator} ".join(
            f"({_format_statement(child, names, nested=True)})" for child in statement[1:]
        )
        return text
    if operator == "->":
        left = _format_statement(statement[1], names, nested=True)
        right = _format_statement(statement[2], names, nested=True)
        return f"if ({left}) then ({right})"
    if operator == "<=>":
        left = _format_statement(statement[1], names, nested=True)
        right = _format_statement(statement[2], names, nested=True)
        return f"({left}) if and only if ({right})"
    raise KeyError(f"Unknown statement operator: {operator}")


def render_question(statements: list[Any], names: list[str], template: int = 0) -> str:
    introductions = [
        "On an island, knights always tell the truth and knaves always lie.",
        "Each person below is either a knight, who always tells the truth, or a knave, who always lies.",
    ]
    lines = [introductions[template % len(introductions)]]
    for index, statement in enumerate(statements):
        lines.append(f'{names[index]} says: "{_format_statement(statement, names)}."')
    lines.append("Determine whether each person is a knight or a knave.")
    return "\n".join(lines)


def _replace_at_path(statement: Any, path: tuple[int, ...], replacement: Any) -> Any:
    statement = _tuple_ast(statement)
    if not path:
        return _tuple_ast(replacement)
    child_index = path[0]
    items = list(statement)
    items[child_index] = _replace_at_path(items[child_index], path[1:], replacement)
    return tuple(items)


def _mutation_candidates(
    statement: Any, path: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], Any, str]]:
    statement = _tuple_ast(statement)
    operator = statement[0]
    candidates: list[tuple[tuple[int, ...], Any, str]] = []
    if operator in PRIMITIVES:
        toggled = "lying" if operator == "telling-truth" else "telling-truth"
        candidates.append((path, (toggled, statement[1]), f"{operator}->{toggled}"))
        for target in range(max(int(statement[1]) + 1, 2)):
            if target != statement[1]:
                candidates.append((path, (operator, target), f"role_{statement[1]}->role_{target}"))
        return candidates
    if operator == "and":
        candidates.append((path, ("or", *statement[1:]), "and->or"))
    elif operator == "or":
        candidates.append((path, ("and", *statement[1:]), "or->and"))
    elif operator == "->":
        candidates.append((path, ("<=>", statement[1], statement[2]), "implication->biconditional"))
    elif operator == "<=>":
        candidates.append((path, ("->", statement[1], statement[2]), "biconditional->implication"))
    for child_index in range(1, len(statement)):
        candidates.extend(_mutation_candidates(statement[child_index], (*path, child_index)))
    return candidates


def mutate_premise(
    statements: list[Any], original_solution: tuple[bool, ...], seed: int
) -> tuple[list[Any], tuple[bool, ...], dict[str, Any]] | None:
    rng = random.Random(seed)
    candidates: list[tuple[int, tuple[int, ...], Any, str]] = []
    for speaker, statement in enumerate(statements):
        for path, replacement, description in _mutation_candidates(statement):
            candidates.append((speaker, path, replacement, description))
    rng.shuffle(candidates)
    n_people = len(statements)
    for speaker, path, replacement, description in candidates:
        # Candidate reference edits are generated conservatively; reject out-of-range references.
        if replacement[0] in PRIMITIVES and int(replacement[1]) >= n_people:
            continue
        updated = list(map(_tuple_ast, statements))
        before = updated[speaker]
        updated[speaker] = _replace_at_path(before, path, replacement)
        solutions = find_solutions(updated)
        if len(solutions) == 1 and solutions[0] != original_solution:
            return updated, solutions[0], {
                "speaker_role": f"role_{speaker}",
                "path": list(path),
                "change": description,
                "before": _serialize_ast(before),
                "after": _serialize_ast(updated[speaker]),
            }
    return None


def _role_mapping(names: list[str]) -> dict[str, str]:
    return {f"role_{index}": name for index, name in enumerate(names)}


def _solution_mapping(solution: tuple[bool, ...]) -> dict[str, bool]:
    return {f"role_{index}": value for index, value in enumerate(solution)}


def _make_record(
    *,
    split: str,
    variant_type: str,
    family_id: str,
    parent_id: str | None,
    statements: list[Any],
    names: list[str],
    solution: tuple[bool, ...],
    generation: dict[str, Any],
    mutation: dict[str, Any] | None,
    template: int = 0,
) -> PuzzleRecord:
    statements_json = [listify_ast(statement) for statement in statements]
    payload = {
        "family_id": family_id,
        "variant_type": variant_type,
        "statements": statements_json,
        "characters": _role_mapping(names),
        "solution": _solution_mapping(solution),
        "mutation": mutation,
    }
    return PuzzleRecord(
        schema_version=SCHEMA_VERSION,
        example_id=sha256_object(payload),
        family_id=family_id,
        variant_type=variant_type,
        parent_id=parent_id,
        split=split,
        question=render_question(statements, names, template),
        statements_ast=statements_json,
        characters=_role_mapping(names),
        solution=_solution_mapping(solution),
        structural_hash=structural_hash(statements),
        difficulty=difficulty_metadata(statements),
        generation=generation,
        mutation=mutation,
    )


def listify_ast(statement: Any) -> list[Any]:
    statement = _tuple_ast(statement)
    return [listify_ast(child) if isinstance(child, tuple) else child for child in statement]


def _reasoning_gym_example(
    seed: int, index: int, n_people: int, depth: int, width: int
) -> tuple[list[Any], tuple[bool, ...]]:
    try:
        from reasoning_gym.logic.knights_knaves import KnightsKnavesConfig, KnightsKnavesDataset
    except ImportError as error:
        raise RuntimeError("reasoning-gym is required for dataset construction") from error
    config = KnightsKnavesConfig(
        n_people=n_people,
        depth_constraint=depth,
        width_constraint=width,
        size=1,
        seed=seed + index,
    )
    entry = KnightsKnavesDataset(config)[0]
    statements = [_tuple_ast(statement) for statement in entry["metadata"]["statements"]]
    solutions = find_solutions(statements)
    if len(solutions) != 1:
        raise ValueError(f"Upstream example did not have one solution: {len(solutions)}")
    upstream_solution = tuple(bool(value) for value in entry["metadata"]["solution"])
    if solutions[0] != upstream_solution:
        raise ValueError("Local oracle disagrees with Reasoning Gym")
    return statements, solutions[0]


def build_records(config: dict[str, Any]) -> tuple[list[PuzzleRecord], dict[str, int]]:
    seed = int(config["experiment"]["seed"])
    max_attempts = int(config["data"]["max_generation_attempts"])
    variants_for = set(config["data"]["variants_for_splits"])
    accepted_hashes: set[str] = set()
    records: list[PuzzleRecord] = []
    rejection_counts: Counter[str] = Counter()
    source_index = 0

    for split, split_config in config["data"]["splits"].items():
        target = int(split_config["count"])
        accepted = 0
        attempts = 0
        while accepted < target and attempts < max_attempts:
            attempts += 1
            source_index += 1
            try:
                statements, solution = _reasoning_gym_example(
                    seed,
                    source_index,
                    int(split_config["n_people"]),
                    int(split_config["depth"]),
                    int(split_config["width"]),
                )
            except (ValueError, IndexError) as error:
                rejection_counts[type(error).__name__] += 1
                continue
            base_hash = structural_hash(statements)
            if base_hash in accepted_hashes:
                rejection_counts["duplicate_structure"] += 1
                continue
            names = DEFAULT_NAMES[: len(statements)]
            family_id = sha256_object(
                {"seed": seed, "source_index": source_index, "structure": base_hash}
            )
            generation = {
                "source": "reasoning_gym.knights_knaves",
                "reasoning_gym_version": "0.1.25",
                "seed": seed,
                "source_index": source_index,
                "n_people": int(split_config["n_people"]),
                "depth": int(split_config["depth"]),
                "width": int(split_config["width"]),
            }
            original = _make_record(
                split=split,
                variant_type="original",
                family_id=family_id,
                parent_id=None,
                statements=statements,
                names=names,
                solution=solution,
                generation=generation,
                mutation=None,
            )
            family = [original]
            family_hashes = {base_hash}

            if split in variants_for:
                surface_names = SURFACE_NAMES[: len(statements)]
                surface = _make_record(
                    split=split,
                    variant_type="surface",
                    family_id=family_id,
                    parent_id=original.example_id,
                    statements=statements,
                    names=surface_names,
                    solution=solution,
                    generation=generation,
                    mutation={"kind": "rename_and_template", "template": 1},
                    template=1,
                )
                premise_result = mutate_premise(statements, solution, seed + source_index * 7919)
                if premise_result is None:
                    rejection_counts["no_valid_premise_mutation"] += 1
                    continue
                premise_statements, premise_solution, mutation = premise_result
                premise = _make_record(
                    split=split,
                    variant_type="premise",
                    family_id=family_id,
                    parent_id=original.example_id,
                    statements=premise_statements,
                    names=names,
                    solution=premise_solution,
                    generation=generation,
                    mutation=mutation,
                )
                family.extend([surface, premise])
                family_hashes.add(premise.structural_hash)

            if accepted_hashes.intersection(family_hashes):
                rejection_counts["cross_family_structure_collision"] += 1
                continue
            accepted_hashes.update(family_hashes)
            records.extend(family)
            accepted += 1
        if accepted < target:
            raise RuntimeError(
                f"Could only construct {accepted}/{target} families for {split} after {attempts} attempts"
            )
    return records, dict(sorted(rejection_counts.items()))


def _records_for_split(records: list[PuzzleRecord], split: str) -> list[PuzzleRecord]:
    return [record for record in records if record.split == split]


def write_dataset(
    config_path: Path,
    config: dict[str, Any],
    records: list[PuzzleRecord],
    rejections: dict[str, int],
    force: bool,
) -> Path:
    output_dir = resolve_path(config_path, config["data"]["output_dir"])
    manifest_path = resolve_path(config_path, config["data"]["manifest"])
    if manifest_path.exists():
        import json

        with manifest_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("frozen", False):
            raise PermissionError(f"Refusing to replace frozen dataset manifest: {manifest_path}")
        if not force:
            raise FileExistsError(f"Manifest already exists: {manifest_path}; pass --force to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_entries: dict[str, Any] = {}
    for split in config["data"]["splits"]:
        path = output_dir / f"{split}.jsonl"
        split_records = _records_for_split(records, split)
        atomic_write_jsonl(path, (record.to_dict() for record in split_records))
        split_entries[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(split_records),
            "families": len({record.family_id for record in split_records}),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "frozen": False,
        "config_hash": sha256_object(config),
        "splits": split_entries,
        "rejections": rejections,
    }
    manifest["dataset_hash"] = sha256_object(manifest["splits"])
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def load_manifest_records(manifest_path: str | Path, split: str) -> tuple[dict[str, Any], list[PuzzleRecord]]:
    import json

    path = Path(manifest_path).resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if split not in manifest.get("splits", {}):
        raise ValueError(f"Split {split!r} not present in manifest")
    split_path = Path(manifest["splits"][split]["path"])
    if not split_path.is_absolute():
        split_path = (path.parent / split_path).resolve()
    actual_hash = sha256_file(split_path)
    if actual_hash != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Split hash mismatch for {split}")
    records = [PuzzleRecord.from_dict(row) for row in read_jsonl(split_path)]
    return manifest, records


def audit_manifest(manifest_path: str | Path) -> dict[str, Any]:
    import json

    path = Path(manifest_path).resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_dataset_hash = sha256_object(manifest.get("splits", {}))
    if manifest.get("dataset_hash") != expected_dataset_hash:
        raise ValueError("Dataset audit failed:\ndataset hash mismatch")
    records: list[PuzzleRecord] = []
    hash_splits: dict[str, set[str]] = {}
    family_splits: dict[str, set[str]] = {}
    ids: set[str] = set()
    parents: dict[str, PuzzleRecord] = {}
    errors: list[str] = []
    for split in manifest.get("splits", {}):
        _, split_records = load_manifest_records(path, split)
        entry = manifest["splits"][split]
        if int(entry.get("records", -1)) != len(split_records):
            errors.append(f"record count mismatch {split}")
        actual_families = len({record.family_id for record in split_records})
        if int(entry.get("families", actual_families)) != actual_families:
            errors.append(f"family count mismatch {split}")
        for record in split_records:
            records.append(record)
            if record.example_id in ids:
                errors.append(f"duplicate example_id {record.example_id}")
            ids.add(record.example_id)
            parents[record.example_id] = record
            family_splits.setdefault(record.family_id, set()).add(record.split)
            hash_splits.setdefault(record.structural_hash, set()).add(record.split)
            statements = [_tuple_ast(statement) for statement in record.statements_ast]
            solutions = find_solutions(statements)
            expected = tuple(record.solution[f"role_{index}"] for index in range(len(statements)))
            if solutions != [expected]:
                errors.append(f"oracle mismatch {record.example_id}")
            if structural_hash(statements) != record.structural_hash:
                errors.append(f"structural hash mismatch {record.example_id}")
            identity_payload = {
                "family_id": record.family_id,
                "variant_type": record.variant_type,
                "statements": record.statements_ast,
                "characters": record.characters,
                "solution": record.solution,
                "mutation": record.mutation,
            }
            if sha256_object(identity_payload) != record.example_id:
                errors.append(f"example ID hash mismatch {record.example_id}")
    by_family: dict[str, list[PuzzleRecord]] = {}
    for record in records:
        by_family.setdefault(record.family_id, []).append(record)
    for family, family_records in by_family.items():
        originals = [record for record in family_records if record.variant_type == "original"]
        if len(originals) != 1:
            errors.append(f"family has {len(originals)} originals {family}")
            continue
        original = originals[0]
        generation = original.generation
        if {"seed", "source_index"}.issubset(generation):
            expected_family = sha256_object(
                {
                    "seed": generation["seed"],
                    "source_index": generation["source_index"],
                    "structure": original.structural_hash,
                }
            )
            if expected_family != family:
                errors.append(f"family ID hash mismatch {family}")
    for family, splits in family_splits.items():
        if len(splits) != 1:
            errors.append(f"family crosses splits {family}: {sorted(splits)}")
    for value, splits in hash_splits.items():
        if len(splits) != 1:
            errors.append(f"structure crosses splits {value}: {sorted(splits)}")
    for record in records:
        if record.variant_type == "original":
            if record.parent_id is not None:
                errors.append(f"original has parent {record.example_id}")
            continue
        parent = parents.get(record.parent_id or "")
        if parent is None:
            errors.append(f"missing parent {record.example_id}")
            continue
        if parent.family_id != record.family_id or parent.split != record.split:
            errors.append(f"parent linkage mismatch {record.example_id}")
        if record.variant_type == "surface":
            if record.structural_hash != parent.structural_hash or record.solution != parent.solution:
                errors.append(f"invalid surface variant {record.example_id}")
        if record.variant_type == "premise" and record.solution == parent.solution:
            errors.append(f"premise solution unchanged {record.example_id}")
    summary = {
        "valid": not errors,
        "records": len(records),
        "families": len(family_splits),
        "splits": {split: sum(record.split == split for record in records) for split in manifest["splits"]},
        "errors": errors,
    }
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors))
    return summary


def freeze_manifest(manifest_path: str | Path) -> dict[str, Any]:
    import json

    path = Path(manifest_path).resolve()
    summary = audit_manifest(path)
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest["splits"].get("test", {}).get("families", 0)) <= 0:
        raise ValueError("Refusing to freeze a manifest without held-out test families")
    manifest["frozen"] = True
    manifest["frozen_at"] = utc_now()
    manifest["audit"] = summary
    atomic_write_json(path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and audit controlled Knights and Knaves data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--force", action="store_true")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        resolved = load_config(args.config)
        records, rejections = build_records(resolved.values)
        manifest_path = write_dataset(
            resolved.path, resolved.values, records, rejections, force=args.force
        )
        summary = audit_manifest(manifest_path)
        print(f"Wrote and audited {manifest_path}: {summary}")
    elif args.command == "audit":
        summary = audit_manifest(args.manifest)
        print(summary)
    else:
        manifest = freeze_manifest(args.manifest)
        print(f"Frozen {args.manifest} with dataset hash {manifest['dataset_hash']}")


if __name__ == "__main__":
    main()
