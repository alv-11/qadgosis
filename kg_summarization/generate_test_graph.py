from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "test_small"
DEFAULT_TOTAL_ENTITIES = 500
DEFAULT_NUM_SNAPSHOTS = 4

Triple = tuple[str, str, str]

ENTITY_GROUP_SPECS = (
    ("people", 260),
    ("papers", 80),
    ("labs", 20),
    ("projects", 30),
    ("cities", 20),
    ("topics", 40),
    ("venues", 15),
    ("orgs", 20),
    ("tools", 15),
)

ENTITY_GROUPS: dict[str, list[str]] = {}
ENTITY_GROUP_COUNTS: dict[str, int] = {}
PEOPLE: list[str] = []
PAPERS: list[str] = []
LABS: list[str] = []
PROJECTS: list[str] = []
CITIES: list[str] = []
TOPICS: list[str] = []
VENUES: list[str] = []
ORGS: list[str] = []
TOOLS: list[str] = []
TOTAL_ENTITY_COUNT = 0


def _allocate_group_counts(total_entities: int) -> dict[str, int]:
    group_names = [name for name, _ in ENTITY_GROUP_SPECS]
    min_total_entities = len(group_names)
    if total_entities < min_total_entities:
        raise ValueError(
            f"total_entities must be at least {min_total_entities} so every entity group is represented."
        )

    minimum_per_group = 1
    remaining_entities = total_entities - min_total_entities
    weighted_sizes = {name: base_count - minimum_per_group for name, base_count in ENTITY_GROUP_SPECS}
    total_weight = sum(weighted_sizes.values())

    counts = {name: minimum_per_group for name in group_names}
    fractional_remainders: list[tuple[float, str]] = []

    for name, weight in weighted_sizes.items():
        scaled_extra = remaining_entities * weight / total_weight
        extra_count = int(scaled_extra)
        counts[name] += extra_count
        fractional_remainders.append((scaled_extra - extra_count, name))

    assigned_entities = sum(counts.values())
    for _, name in sorted(fractional_remainders, reverse=True)[: total_entities - assigned_entities]:
        counts[name] += 1

    return counts


def configure_entity_groups(total_entities: int) -> dict[str, int]:
    global ENTITY_GROUPS, ENTITY_GROUP_COUNTS
    global PEOPLE, PAPERS, LABS, PROJECTS, CITIES, TOPICS, VENUES, ORGS, TOOLS
    global TOTAL_ENTITY_COUNT

    counts = _allocate_group_counts(total_entities)
    max_group_size = max(counts.values())
    block_size = max(1000, max_group_size + 100)

    groups: dict[str, list[str]] = {}
    for index, (name, _) in enumerate(ENTITY_GROUP_SPECS):
        start = 1001 + index * block_size
        groups[name] = [str(entity_id) for entity_id in range(start, start + counts[name])]

    ENTITY_GROUPS = groups
    ENTITY_GROUP_COUNTS = counts
    PEOPLE = groups["people"]
    PAPERS = groups["papers"]
    LABS = groups["labs"]
    PROJECTS = groups["projects"]
    CITIES = groups["cities"]
    TOPICS = groups["topics"]
    VENUES = groups["venues"]
    ORGS = groups["orgs"]
    TOOLS = groups["tools"]
    TOTAL_ENTITY_COUNT = sum(counts.values())
    return counts


def _format_group_counts(group_counts: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in group_counts.items())


configure_entity_groups(DEFAULT_TOTAL_ENTITIES)


def _pick(group: list[str], index: int, *, multiplier: int = 1, offset: int = 0) -> str:
    return group[(index * multiplier + offset) % len(group)]


def _base_triples() -> set[Triple]:
    triples: set[Triple] = set()

    for index, person in enumerate(PEOPLE):
        triples.update(
            {
                (person, "rel_knows", _pick(PEOPLE, index, offset=1)),
                (person, "rel_collaborates_with", _pick(PEOPLE, index, offset=17)),
                (person, "rel_advises", _pick(PEOPLE, index, offset=41)),
                (person, "rel_authored", _pick(PAPERS, index)),
                (person, "rel_reads", _pick(PAPERS, index, multiplier=3, offset=7)),
                (person, "rel_works_at", _pick(LABS, index)),
                (person, "rel_member_of", _pick(ORGS, index, multiplier=5, offset=3)),
                (person, "rel_lives_in", _pick(CITIES, index, multiplier=7, offset=2)),
                (person, "rel_works_on", _pick(PROJECTS, index, multiplier=11, offset=5)),
                (person, "rel_interested_in", _pick(TOPICS, index, multiplier=13, offset=1)),
                (person, "rel_uses", _pick(TOOLS, index, multiplier=2, offset=1)),
            }
        )

    for index, paper in enumerate(PAPERS):
        triples.update(
            {
                (paper, "rel_cites", _pick(PAPERS, index, offset=1)),
                (paper, "rel_cites", _pick(PAPERS, index, offset=9)),
                (paper, "rel_about", _pick(TOPICS, index)),
                (paper, "rel_about", _pick(TOPICS, index, offset=17)),
                (paper, "rel_presented_at", _pick(VENUES, index)),
                (paper, "rel_references_project", _pick(PROJECTS, index, multiplier=3, offset=4)),
                (paper, "rel_uses", _pick(TOOLS, index, multiplier=5, offset=2)),
            }
        )

    for index, lab in enumerate(LABS):
        triples.update(
            {
                (lab, "rel_located_in", _pick(CITIES, index)),
                (lab, "rel_funded_by", _pick(ORGS, index, offset=4)),
                (lab, "rel_hosts", _pick(PROJECTS, index, multiplier=2, offset=1)),
                (lab, "rel_hosts", _pick(TOPICS, index, multiplier=3, offset=2)),
                (lab, "rel_uses", _pick(TOOLS, index)),
                (lab, "rel_collaborates_with", _pick(LABS, index, offset=1)),
            }
        )

    for index, project in enumerate(PROJECTS):
        triples.update(
            {
                (project, "rel_targets", _pick(TOPICS, index)),
                (project, "rel_uses", _pick(TOOLS, index, offset=3)),
                (project, "rel_funded_by", _pick(ORGS, index, multiplier=2, offset=1)),
                (project, "rel_coordinated_by", _pick(LABS, index, multiplier=3, offset=2)),
                (project, "rel_deployed_in", _pick(CITIES, index, multiplier=2, offset=1)),
            }
        )

    for index, org in enumerate(ORGS):
        triples.update(
            {
                (org, "rel_based_in", _pick(CITIES, index)),
                (org, "rel_sponsors", _pick(PROJECTS, index, multiplier=2, offset=5)),
                (org, "rel_sponsors", _pick(TOPICS, index, multiplier=3, offset=7)),
                (org, "rel_partners_with", _pick(ORGS, index, offset=1)),
            }
        )

    for index, venue in enumerate(VENUES):
        triples.update(
            {
                (venue, "rel_held_in", _pick(CITIES, index)),
                (venue, "rel_features", _pick(TOPICS, index, offset=2)),
                (venue, "rel_features", _pick(TOPICS, index, offset=11)),
            }
        )

    for index, tool in enumerate(TOOLS):
        triples.update(
            {
                (tool, "rel_supports", _pick(TOPICS, index)),
                (tool, "rel_supports", _pick(PROJECTS, index, multiplier=2, offset=1)),
                (tool, "rel_maintained_by", _pick(ORGS, index, offset=6)),
            }
        )

    for index, city in enumerate(CITIES):
        triples.update(
            {
                (city, "rel_twinned_with", _pick(CITIES, index, offset=1)),
                (city, "rel_hosts", _pick(VENUES, index, offset=3)),
            }
        )

    return triples


def _snapshot_one_updates() -> tuple[set[Triple], set[Triple]]:
    additions: set[Triple] = set()
    removals: set[Triple] = set()

    for index in range(0, len(PEOPLE), 2):
        person = PEOPLE[index]
        additions.add((person, "rel_mentions", _pick(TOPICS, index, multiplier=5, offset=9)))

    for index in range(0, len(PEOPLE), 3):
        person = PEOPLE[index]
        additions.add((person, "rel_reviews", _pick(PAPERS, index, multiplier=7, offset=3)))

    for index, lab in enumerate(LABS):
        additions.add((lab, "rel_tracks", _pick(TOPICS, index, multiplier=2, offset=9)))

    for index, venue in enumerate(VENUES):
        additions.add((venue, "rel_showcases", _pick(PROJECTS, index, multiplier=2, offset=1)))

    for index, org in enumerate(ORGS):
        additions.add((org, "rel_invests_in", _pick(TOOLS, index, offset=2)))

    for index in range(0, len(PEOPLE), 6):
        person = PEOPLE[index]
        removals.add((person, "rel_reads", _pick(PAPERS, index, multiplier=3, offset=7)))

    for index in range(0, len(PAPERS), 8):
        paper = PAPERS[index]
        removals.add((paper, "rel_cites", _pick(PAPERS, index, offset=9)))

    return additions, removals


def _snapshot_two_updates() -> tuple[set[Triple], set[Triple]]:
    additions: set[Triple] = set()
    removals: set[Triple] = set()

    for index in range(0, len(PEOPLE), 4):
        person = PEOPLE[index]
        additions.add((person, "rel_attends", _pick(VENUES, index, multiplier=3, offset=2)))

    for index in range(0, len(PAPERS), 2):
        paper = PAPERS[index]
        additions.add((paper, "rel_extends", _pick(PROJECTS, index, multiplier=5, offset=6)))

    for index, project in enumerate(PROJECTS):
        additions.add((project, "rel_reports_to", _pick(ORGS, index, multiplier=3, offset=4)))

    for index, tool in enumerate(TOOLS):
        additions.add((tool, "rel_benchmarks", _pick(PAPERS, index, multiplier=4, offset=8)))

    for index, city in enumerate(CITIES):
        additions.add((city, "rel_connects_to", _pick(PROJECTS, index, multiplier=2, offset=7)))

    for index in range(0, len(PEOPLE), 7):
        person = PEOPLE[index]
        removals.add((person, "rel_collaborates_with", _pick(PEOPLE, index, offset=17)))

    for index in range(0, len(LABS), 5):
        lab = LABS[index]
        removals.add((lab, "rel_hosts", _pick(TOPICS, index, multiplier=3, offset=2)))

    return additions, removals


def _snapshot_three_updates() -> tuple[set[Triple], set[Triple]]:
    additions: set[Triple] = set()
    removals: set[Triple] = set()

    for index in range(1, len(PEOPLE), 3):
        person = PEOPLE[index]
        additions.add((person, "rel_contributes_to", _pick(PROJECTS, index, multiplier=4, offset=11)))

    for index, lab in enumerate(LABS):
        additions.add((lab, "rel_incubates", _pick(PROJECTS, index, multiplier=3, offset=5)))

    for index in range(0, len(PAPERS), 3):
        paper = PAPERS[index]
        additions.add((paper, "rel_discusses", _pick(TOPICS, index, multiplier=2, offset=13)))

    for index, venue in enumerate(VENUES):
        additions.add((venue, "rel_streams", _pick(TOPICS, index, multiplier=4, offset=6)))

    for index, org in enumerate(ORGS):
        additions.add((org, "rel_supports", _pick(TOOLS, index, multiplier=2, offset=3)))

    for index in range(0, len(PEOPLE), 9):
        person = PEOPLE[index]
        removals.add((person, "rel_advises", _pick(PEOPLE, index, offset=41)))

    for index in range(0, len(PROJECTS), 6):
        project = PROJECTS[index]
        removals.add((project, "rel_uses", _pick(TOOLS, index, offset=3)))

    return additions, removals


def _extended_snapshot_updates(phase: int) -> tuple[set[Triple], set[Triple]]:
    additions: set[Triple] = set()
    removals: set[Triple] = set()
    phase_offset = phase - 4

    people_step = 2 + (phase_offset % 3)
    review_step = 3 + (phase_offset % 4)
    attend_step = 4 + (phase_offset % 3)
    contribution_step = 3 + (phase_offset % 5)

    for index in range(phase_offset % people_step, len(PEOPLE), people_step):
        person = PEOPLE[index]
        additions.add(
            (
                person,
                "rel_mentions",
                _pick(TOPICS, index, multiplier=5 + (phase_offset % 4), offset=9 + phase_offset * 3),
            )
        )

    for index in range((phase_offset + 1) % review_step, len(PEOPLE), review_step):
        person = PEOPLE[index]
        additions.add(
            (
                person,
                "rel_reviews",
                _pick(PAPERS, index, multiplier=7 + (phase_offset % 5), offset=3 + phase_offset * 2),
            )
        )

    for index in range((phase_offset + 2) % attend_step, len(PEOPLE), attend_step):
        person = PEOPLE[index]
        additions.add(
            (
                person,
                "rel_attends",
                _pick(VENUES, index, multiplier=3 + (phase_offset % 4), offset=2 + phase_offset),
            )
        )

    for index in range((phase_offset + 1) % contribution_step, len(PEOPLE), contribution_step):
        person = PEOPLE[index]
        additions.add(
            (
                person,
                "rel_contributes_to",
                _pick(PROJECTS, index, multiplier=4 + (phase_offset % 6), offset=11 + phase_offset * 2),
            )
        )

    for index, lab in enumerate(LABS):
        additions.add(
            (
                lab,
                "rel_tracks",
                _pick(TOPICS, index, multiplier=2 + (phase_offset % 3), offset=9 + phase_offset),
            )
        )
        additions.add(
            (
                lab,
                "rel_incubates",
                _pick(PROJECTS, index, multiplier=3 + (phase_offset % 4), offset=5 + phase_offset),
            )
        )

    for index, paper in enumerate(PAPERS):
        if (index + phase_offset) % 2 == 0:
            additions.add(
                (
                    paper,
                    "rel_extends",
                    _pick(PROJECTS, index, multiplier=5 + (phase_offset % 4), offset=6 + phase_offset),
                )
            )
        if (index + phase_offset) % 3 == 0:
            additions.add(
                (
                    paper,
                    "rel_discusses",
                    _pick(TOPICS, index, multiplier=2 + (phase_offset % 5), offset=13 + phase_offset * 2),
                )
            )

    for index, venue in enumerate(VENUES):
        additions.add(
            (
                venue,
                "rel_showcases",
                _pick(PROJECTS, index, multiplier=2 + (phase_offset % 3), offset=1 + phase_offset),
            )
        )
        additions.add(
            (
                venue,
                "rel_streams",
                _pick(TOPICS, index, multiplier=4 + (phase_offset % 3), offset=6 + phase_offset),
            )
        )

    for index, org in enumerate(ORGS):
        additions.add(
            (
                org,
                "rel_supports",
                _pick(TOOLS, index, multiplier=2 + (phase_offset % 4), offset=3 + phase_offset),
            )
        )
        additions.add(
            (
                org,
                "rel_invests_in",
                _pick(PROJECTS, index, multiplier=3 + (phase_offset % 4), offset=4 + phase_offset),
            )
        )

    for index in range(phase_offset % 6, len(PEOPLE), 6):
        person = PEOPLE[index]
        removals.add((person, "rel_reads", _pick(PAPERS, index, multiplier=3, offset=7)))

    for index in range((phase_offset + 1) % 7, len(PEOPLE), 7):
        person = PEOPLE[index]
        removals.add((person, "rel_collaborates_with", _pick(PEOPLE, index, offset=17)))

    for index in range((phase_offset + 2) % 9, len(PEOPLE), 9):
        person = PEOPLE[index]
        removals.add((person, "rel_advises", _pick(PEOPLE, index, offset=41)))

    for index in range(phase_offset % 8, len(PAPERS), 8):
        paper = PAPERS[index]
        removals.add((paper, "rel_cites", _pick(PAPERS, index, offset=9)))

    for index in range((phase_offset + 3) % 6, len(PROJECTS), 6):
        project = PROJECTS[index]
        removals.add((project, "rel_uses", _pick(TOOLS, index, offset=3)))

    return additions, removals


def _snapshot_updates(phase: int) -> tuple[set[Triple], set[Triple]]:
    if phase == 1:
        return _snapshot_one_updates()
    if phase == 2:
        return _snapshot_two_updates()
    if phase == 3:
        return _snapshot_three_updates()
    return _extended_snapshot_updates(phase)


def _snapshot_filename(index: int) -> str:
    return f"snapshot_{index}.nt.gz"


def build_snapshots(num_snapshots: int = DEFAULT_NUM_SNAPSHOTS) -> list[list[Triple]]:
    if num_snapshots < 1:
        raise ValueError("num_snapshots must be at least 1.")

    snapshots: list[list[Triple]] = []
    current = _base_triples()
    snapshots.append(sorted(current))

    for phase in range(1, num_snapshots):
        additions, removals = _snapshot_updates(phase)
        current |= additions
        current -= removals
        snapshots.append(sorted(current))

    return snapshots


def write_snapshot(snapshot_path: Path, triples: list[Triple]) -> None:
    with gzip.open(snapshot_path, "wt", encoding="utf-8") as handle:
        for subject, predicate, obj in triples:
            handle.write(f"<{subject}> <{predicate}> <{obj}> .\n")


def generate_dataset(
    output_dir: Path,
    *,
    total_entities: int = DEFAULT_TOTAL_ENTITIES,
    num_snapshots: int = DEFAULT_NUM_SNAPSHOTS,
    overwrite: bool = False,
) -> tuple[list[Path], dict[str, int]]:
    group_counts = configure_entity_groups(total_entities)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths: list[Path] = []
    for index, triples in enumerate(build_snapshots(num_snapshots=num_snapshots)):
        snapshot_path = output_dir / _snapshot_filename(index)
        write_snapshot(snapshot_path, triples)
        snapshot_paths.append(snapshot_path)

    return snapshot_paths, group_counts


def validate_dataset(snapshot_paths: list[Path], expected_entity_count: int) -> None:
    from src.base import DBPedia

    for snapshot_path in snapshot_paths:
        graph = DBPedia(rdf_gz=str(snapshot_path))
        graph.load()
        head_entities = len(graph.triples_)
        non_numeric_entities = [entity for entity in graph.entities() if not str(entity).isdigit()]

        if graph.number_of_entities() != expected_entity_count:
            raise ValueError(
                f"{snapshot_path.name} loaded {graph.number_of_entities()} entities; "
                f"expected {expected_entity_count}."
            )

        if head_entities < 5:
            raise ValueError(
                f"{snapshot_path.name} only has {head_entities} head entities; "
                "the query generator expects several valid topic nodes."
            )

        if non_numeric_entities:
            preview = ", ".join(sorted(non_numeric_entities)[:5])
            raise ValueError(
                f"{snapshot_path.name} contains non-numeric entity IDs ({preview}). "
                "main.py currently converts query seeds and graph nodes with int(...)."
            )

        print(
            f"Validated {snapshot_path.name}: "
            f"{graph.number_of_triples()} triples, "
            f"{graph.number_of_entities()} entities, "
            f"{graph.number_of_relationships()} relation types, "
            f"{head_entities} head entities"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a larger synthetic snapshot dataset compatible with kg_summarization."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write snapshot_*.nt.gz files into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip loading the generated snapshots back through src.base.DBPedia.",
    )
    parser.add_argument(
        "--total-entities",
        type=int,
        default=DEFAULT_TOTAL_ENTITIES,
        help=f"Total number of entities to generate across all groups (default: {DEFAULT_TOTAL_ENTITIES}).",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=DEFAULT_NUM_SNAPSHOTS,
        help=f"Number of snapshot_*.nt.gz files to generate (default: {DEFAULT_NUM_SNAPSHOTS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_paths, group_counts = generate_dataset(
        args.output_dir,
        total_entities=args.total_entities,
        num_snapshots=args.num_snapshots,
        overwrite=args.overwrite,
    )

    for snapshot_path in snapshot_paths:
        print(f"Wrote {snapshot_path}")

    print(f"Configured entity groups: {_format_group_counts(group_counts)}")
    print(f"Generated {len(snapshot_paths)} snapshots")

    if not args.skip_validate:
        validate_dataset(snapshot_paths, expected_entity_count=sum(group_counts.values()))

    print("\nDataset ready.")
    print(f"Point subgraph_dir at {args.output_dir} to use these snapshots.")


if __name__ == "__main__":
    main()