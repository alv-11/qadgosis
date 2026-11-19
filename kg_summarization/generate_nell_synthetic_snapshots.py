#!/usr/bin/env python3
"""Create a reproducible synthetic snapshot sequence from a NELL fact file.

The snapshots are intended for testing dynamic-graph behavior. Their changes
are sampled from the input facts and do not represent NELL's historical state.
The snapshot files retain each original NELL entity and predicate name. The
runtime cache creates reversible integer indices only for the C++ walk engine,
which currently accepts integer node IDs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


DEFAULT_INPUT = Path.home() / "Downloads" / "nell" / "nell.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "nell_named_synthetic_3snap"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_facts(path: Path) -> list[tuple[str, str, str]]:
    facts = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"Expected exactly three columns in {path}:{line_number}; "
                    f"found {len(fields)}."
                )
            facts.append(tuple(fields))
    if not facts:
        raise ValueError(f"No triples found in {path}.")
    if len(facts) != len(set(facts)):
        raise ValueError(f"Duplicate triples found in {path}; refusing ambiguous input.")
    return facts


def write_snapshot(path: Path, facts: set[tuple[str, str, str]]) -> None:
    with path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", fileobj=raw_output, mode="wb", mtime=0) as output:
            for subject, predicate, obj in sorted(facts):
                line = f"<{subject}> <{predicate}> <{obj}> .\n"
                output.write(line.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-fraction", type=float, default=0.80)
    parser.add_argument("--addition-fraction", type=float, default=0.10)
    parser.add_argument("--removal-fraction", type=float, default=0.05)
    args = parser.parse_args()

    for name in ("initial_fraction", "addition_fraction", "removal_fraction"):
        value = getattr(args, name)
        if not 0 < value < 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output directory is not empty: {output_dir}. Choose a new directory "
            "so existing snapshots, caches, and query bundles are preserved."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    facts = load_facts(input_path)
    entities = {entity for subject, _, obj in facts for entity in (subject, obj)}

    rng = random.Random(args.seed)
    shuffled = list(facts)
    rng.shuffle(shuffled)

    initial_count = int(len(shuffled) * args.initial_fraction)
    per_step_add_count = int(len(shuffled) * args.addition_fraction)
    active = set(shuffled[:initial_count])
    held_back = list(shuffled[initial_count:])

    snapshots = []
    transition_counts = []
    snapshot_path = output_dir / "snapshot_0.nt.gz"
    write_snapshot(snapshot_path, active)
    snapshots.append({"file": snapshot_path.name, "triple_count": len(active)})

    for timestep in (1, 2):
        remove_count = min(int(len(active) * args.removal_fraction), len(active))
        removed = set(rng.sample(sorted(active), remove_count)) if remove_count else set()
        active.difference_update(removed)

        add_count = min(per_step_add_count, len(held_back))
        added = set(held_back[:add_count])
        del held_back[:add_count]
        active.update(added)

        snapshot_path = output_dir / f"snapshot_{timestep}.nt.gz"
        write_snapshot(snapshot_path, active)
        snapshots.append({"file": snapshot_path.name, "triple_count": len(active)})
        transition_counts.append({
            "from_timestep": timestep - 1,
            "to_timestep": timestep,
            "triples_removed": len(removed),
            "triples_added": len(added),
        })

    manifest = {
        "dataset_kind": "synthetic_temporal_sequence_from_static_nell_dump",
        "warning": (
            "These additions and removals are random experimental perturbations; "
            "they are not historical NELL changes."
        ),
        "source_file": str(input_path),
        "source_sha256": sha256_file(input_path),
        "random_seed": args.seed,
        "source_triple_count": len(facts),
        "entity_count": len(entities),
        "predicate_count": len({predicate for _, predicate, _ in facts}),
        "sampling": {
            "initial_fraction": args.initial_fraction,
            "addition_fraction_of_original_per_transition": args.addition_fraction,
            "removal_fraction_of_active_per_transition": args.removal_fraction,
        },
        "snapshots": snapshots,
        "transitions": transition_counts,
        "source_triples_not_yet_added_after_final_snapshot": len(held_back),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")

    print(f"Created {len(snapshots)} snapshots in {output_dir}")
    for item in snapshots:
        print(f"  {item['file']}: {item['triple_count']:,} triples")
    for item in transition_counts:
        print(
            f"  {item['from_timestep']} -> {item['to_timestep']}: "
            f"+{item['triples_added']:,} / -{item['triples_removed']:,} triples"
        )
    print(f"Entity labels retained in snapshot files: {len(entities):,}")
    print("NOTE: the temporal changes are synthetic, not NELL history.")


if __name__ == "__main__":
    main()
