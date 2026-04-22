from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.path import ENCODING


EDGE_WEIGHT = 0.01
CACHE_VERSION = 1
CACHE_DIR_NAME = ".sankofa_cache"
MANIFEST_FILE = "manifest.json"
INITIAL_GRAPH_FILE = "initial_graph.pkl"
DELTAS_DIR_NAME = "deltas"
PROJECT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Graph = dict[int, dict[int, float]]
GraphDelta = dict[str, object]
GENERATED_CACHE_STATUS = "generated"
REUSED_CACHE_STATUS = "reused"


@dataclass(frozen=True)
class SankofaDeltaCache:
    subgraph_dir: Path
    cache_dir: Path
    snapshot_files: tuple[str, ...]
    manifest: dict[str, object]
    cache_status: str

    @property
    def initial_graph_path(self) -> Path:
        return self.cache_dir / INITIAL_GRAPH_FILE

    @property
    def deltas_dir(self) -> Path:
        return self.cache_dir / DELTAS_DIR_NAME

    def delta_path(self, timestep: int) -> Path:
        return self.deltas_dir / _delta_filename(timestep)


def extract_snapshot_number(filename: str) -> int:
    base = Path(filename).name
    num_part = base.replace("snapshot_", "").replace(".nt.gz", "")
    return int(num_part)


def list_snapshot_files(subgraph_dir: str | Path) -> list[str]:
    directory = Path(subgraph_dir)
    snapshot_files = [
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith("snapshot_") and path.name.endswith(".nt.gz")
    ]
    return sorted(snapshot_files, key=extract_snapshot_number)


def graph_node_count(graph: Graph) -> int:
    return len(graph)


def graph_edge_count(graph: Graph) -> int:
    return sum(len(neighbors) for neighbors in graph.values())


def parse_snapshot_graph(snapshot_path: str | Path) -> Graph:
    graph: Graph = {}

    with gzip.open(snapshot_path, "rt", encoding=ENCODING) as handle:
        for line_number, line in enumerate(handle, start=1):
            fact = _parse_snapshot_line(line)
            if fact is None:
                continue

            source_str, target_str = fact

            try:
                source = int(source_str)
                target = int(target_str)
            except ValueError as exc:
                raise ValueError(
                    f"Snapshot {snapshot_path} contains a non-integer entity at line {line_number}: {line.strip()}"
                ) from exc

            graph.setdefault(source, {})
            graph.setdefault(target, {})
            graph[source][target] = EDGE_WEIGHT

    return graph


def ensure_sankofa_delta_cache(subgraph_dir: str | Path, *, rebuild: bool = False, verbose: bool = True) -> SankofaDeltaCache:
    subgraph_path = Path(subgraph_dir)
    snapshot_files = list_snapshot_files(subgraph_path)
    if not snapshot_files:
        raise ValueError(f"No snapshot_*.nt.gz files found in {subgraph_path}")

    cache_dir = resolve_cache_dir(subgraph_path)
    manifest = _load_manifest(cache_dir)
    needs_rebuild = rebuild or not _cache_is_valid(cache_dir, manifest, subgraph_path, snapshot_files)

    if needs_rebuild:
        if verbose:
            reason = "rebuild requested" if rebuild else "cache missing or out of date"
            print(f"Sankofa cache: generating new cache at {cache_dir} ({reason}).")
        return _build_sankofa_delta_cache(subgraph_path, snapshot_files)

    if verbose:
        print(f"Sankofa cache: using preexisting cache at {cache_dir}.")

    return SankofaDeltaCache(
        subgraph_dir=subgraph_path,
        cache_dir=cache_dir,
        snapshot_files=tuple(snapshot_files),
        manifest=manifest,
        cache_status=REUSED_CACHE_STATUS,
    )


def load_initial_graph(cache: SankofaDeltaCache) -> Graph:
    with cache.initial_graph_path.open("rb") as handle:
        return pickle.load(handle)


def load_delta(cache: SankofaDeltaCache, timestep: int) -> GraphDelta:
    with cache.delta_path(timestep).open("rb") as handle:
        return pickle.load(handle)


def apply_delta_to_graph(current_graph: Graph, delta: GraphDelta) -> Graph:
    next_graph = current_graph.copy()

    nodes_added = set(delta["nodes_added"])
    nodes_removed = set(delta["nodes_removed"])
    edges_added = [tuple(edge) for edge in delta["edges_added"]]
    edges_removed = [tuple(edge) for edge in delta["edges_removed"]]

    changed_keys = {source for source, _ in edges_added}
    changed_keys.update(source for source, _ in edges_removed)
    changed_keys.update(nodes_added)
    changed_keys.update(nodes_removed)

    for node in changed_keys:
        next_graph[node] = current_graph.get(node, {}).copy()

    for source, target in edges_removed:
        neighbors = next_graph.get(source)
        if neighbors is None:
            continue
        neighbors.pop(target, None)

    for source, target in edges_added:
        next_graph.setdefault(source, {})
        next_graph[source][target] = EDGE_WEIGHT

    for node in nodes_added:
        next_graph.setdefault(node, {})

    for node in nodes_removed:
        next_graph.pop(node, None)

    return next_graph


def validate_sankofa_delta_cache(subgraph_dir: str | Path, *, rebuild: bool = False) -> dict[str, object]:
    cache = ensure_sankofa_delta_cache(subgraph_dir, rebuild=rebuild, verbose=False)
    snapshot_paths = [cache.subgraph_dir / file_name for file_name in cache.snapshot_files]

    reconstructed = load_initial_graph(cache)
    expected = parse_snapshot_graph(snapshot_paths[0])
    if reconstructed != expected:
        raise ValueError("Cached initial Sankofa graph does not match snapshot_0.")

    edge_counts = [graph_edge_count(reconstructed)]

    for timestep in range(len(cache.snapshot_files) - 1):
        reconstructed = apply_delta_to_graph(reconstructed, load_delta(cache, timestep))
        expected = parse_snapshot_graph(snapshot_paths[timestep + 1])

        if reconstructed != expected:
            raise ValueError(
                f"Cached Sankofa delta reconstruction diverged at timestep {timestep + 1}."
            )

        edge_counts.append(graph_edge_count(reconstructed))

    return {
        "snapshot_count": len(cache.snapshot_files),
        "edge_counts": edge_counts,
        "cache_dir": str(cache.cache_dir),
        "cache_status": cache.cache_status,
    }


def _build_sankofa_delta_cache(subgraph_dir: Path, snapshot_files: list[str]) -> SankofaDeltaCache:
    cache_dir = resolve_cache_dir(subgraph_dir)
    deltas_dir = cache_dir / DELTAS_DIR_NAME

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    deltas_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths = [subgraph_dir / file_name for file_name in snapshot_files]
    current_graph = parse_snapshot_graph(snapshot_paths[0])

    with (cache_dir / INITIAL_GRAPH_FILE).open("wb") as handle:
        pickle.dump(current_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest: dict[str, object] = {
        "version": CACHE_VERSION,
        "edge_weight": EDGE_WEIGHT,
        "snapshot_files": [_snapshot_metadata(path) for path in snapshot_paths],
        "initial_graph": INITIAL_GRAPH_FILE,
        "deltas_dir": DELTAS_DIR_NAME,
        "delta_count": max(0, len(snapshot_paths) - 1),
        "initial_node_count": graph_node_count(current_graph),
        "initial_edge_count": graph_edge_count(current_graph),
    }

    for timestep in range(len(snapshot_paths) - 1):
        next_graph = parse_snapshot_graph(snapshot_paths[timestep + 1])
        delta = _compute_graph_delta(current_graph, next_graph)

        with (deltas_dir / _delta_filename(timestep)).open("wb") as handle:
            pickle.dump(delta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        current_graph = next_graph

    with (cache_dir / MANIFEST_FILE).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return SankofaDeltaCache(
        subgraph_dir=subgraph_dir,
        cache_dir=cache_dir,
        snapshot_files=tuple(snapshot_files),
        manifest=manifest,
        cache_status=GENERATED_CACHE_STATUS,
    )


def _cache_is_valid(
    cache_dir: Path,
    manifest: dict[str, object] | None,
    subgraph_dir: Path,
    snapshot_files: list[str],
) -> bool:
    if manifest is None:
        return False

    if manifest.get("version") != CACHE_VERSION:
        return False

    if not (cache_dir / INITIAL_GRAPH_FILE).exists():
        return False

    if manifest.get("delta_count") != max(0, len(snapshot_files) - 1):
        return False

    manifest_snapshots = manifest.get("snapshot_files")
    if not isinstance(manifest_snapshots, list) or len(manifest_snapshots) != len(snapshot_files):
        return False

    for timestep in range(max(0, len(snapshot_files) - 1)):
        if not (cache_dir / DELTAS_DIR_NAME / _delta_filename(timestep)).exists():
            return False

    current_metadata = [_snapshot_metadata(subgraph_dir / file_name) for file_name in snapshot_files]
    return manifest_snapshots == current_metadata


def _load_manifest(cache_dir: Path) -> dict[str, object] | None:
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _delta_filename(timestep: int) -> str:
    return f"delta_{timestep:04d}.pkl"


def _compute_graph_delta(current_graph: Graph, next_graph: Graph) -> GraphDelta:
    current_nodes = set(current_graph)
    next_nodes = set(next_graph)

    current_edges = {
        (source, target)
        for source, neighbors in current_graph.items()
        for target in neighbors
    }
    next_edges = {
        (source, target)
        for source, neighbors in next_graph.items()
        for target in neighbors
    }

    edges_added = sorted(next_edges - current_edges)
    edges_removed = sorted(current_edges - next_edges)

    return {
        "nodes_added": sorted(next_nodes - current_nodes),
        "nodes_removed": sorted(current_nodes - next_nodes),
        "edges_added": edges_added,
        "edges_removed": edges_removed,
    }


def _parse_snapshot_line(line: str) -> tuple[str, str] | None:
    stripped = line.rstrip("\n")
    if len(stripped) < 3:
        return None

    fact = (
        stripped[:-2]
        .replace("/", "-")
        .replace("<", "")
        .replace(">", "")
        .replace(":", "")
        .replace("*", "")
        .replace("%", "")
        .split(" ")
    )

    if len(fact) < 3:
        return None

    source, _relation = fact[:2]
    target = " ".join(fact[2:])

    if not source or not target:
        return None

    return source, target


def _snapshot_metadata(snapshot_path: Path) -> dict[str, int | str]:
    stat = snapshot_path.stat()
    return {
        "name": snapshot_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def resolve_cache_dir(subgraph_dir: str | Path) -> Path:
    subgraph_path = Path(subgraph_dir).resolve()
    data_dir = PROJECT_DATA_DIR.resolve()

    if subgraph_path.is_relative_to(data_dir):
        return subgraph_path / CACHE_DIR_NAME

    dataset_name = subgraph_path.name or "dataset"
    dataset_key = hashlib.sha1(str(subgraph_path).encode("utf-8")).hexdigest()[:12]
    return data_dir / CACHE_DIR_NAME / f"{dataset_name}-{dataset_key}"