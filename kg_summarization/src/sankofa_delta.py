from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.base import KnowledgeGraph
from src.path import ENCODING
from src.progress import format_bytes, format_seconds, progress, progress_write


EDGE_WEIGHT = 0.01
CACHE_VERSION = 4
CACHE_DIR_NAME = ".sankofa_cache"
CACHE_LOCK_FILE_NAME = ".sankofa_cache.lock"
MANIFEST_FILE = "manifest.json"
INITIAL_GRAPH_FILE = "initial_graph.pkl"
DELTAS_DIR_NAME = "deltas"
TRIPLE_DELTAS_DIR_NAME = "triple_deltas"
SNAPSHOT_TRIPLES_DIR_NAME = "snapshot_triples"
PROJECT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Graph = dict[int, dict[int, float]]
GraphDelta = dict[str, object]
SnapshotTriple = tuple[str, str, str]
SnapshotTriples = list[SnapshotTriple]
TripleDelta = dict[str, SnapshotTriples]
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

    @property
    def triple_deltas_dir(self) -> Path:
        return self.cache_dir / TRIPLE_DELTAS_DIR_NAME

    def triple_delta_path(self, timestep: int) -> Path:
        return self.triple_deltas_dir / _delta_filename(timestep)

    @property
    def snapshot_triples_dir(self) -> Path:
        return self.cache_dir / SNAPSHOT_TRIPLES_DIR_NAME

    def snapshot_triples_path(self, timestep: int) -> Path:
        return self.snapshot_triples_dir / _snapshot_triples_filename(timestep)


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
    return _graph_from_snapshot_triples(parse_snapshot_triples(snapshot_path), snapshot_path)


def parse_snapshot_triples(snapshot_path: str | Path) -> SnapshotTriples:
    triples: SnapshotTriples = []

    with gzip.open(snapshot_path, "rt", encoding=ENCODING) as handle:
        for line in handle:
            triple = _parse_snapshot_triple(line)
            if triple is None:
                continue
            triples.append(triple)

    return triples


def ensure_sankofa_delta_cache(subgraph_dir: str | Path, *, rebuild: bool = False, verbose: bool = True) -> SankofaDeltaCache:
    subgraph_path = Path(subgraph_dir)
    snapshot_files = list_snapshot_files(subgraph_path)
    if not snapshot_files:
        raise ValueError(f"No snapshot_*.nt.gz files found in {subgraph_path}")

    cache_dir = resolve_cache_dir(subgraph_path)
    with _cache_build_lock(subgraph_path, verbose=verbose):
        manifest = _load_manifest(cache_dir)
        needs_rebuild = rebuild or not _cache_is_valid(cache_dir, manifest, subgraph_path, snapshot_files)

        if needs_rebuild:
            if verbose:
                reason = "rebuild requested" if rebuild else "cache missing or out of date"
                progress_write(
                    f"Sankofa cache: generating new cache at {cache_dir} "
                    f"for {len(snapshot_files)} snapshots ({reason}; walk deltas + evaluation snapshots)."
                )
            return _build_sankofa_delta_cache(subgraph_path, snapshot_files)

        if verbose:
            progress_write(
                f"Sankofa cache: using preexisting cache at {cache_dir} "
                f"for {len(snapshot_files)} snapshots (walk deltas + evaluation snapshots)."
            )

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


def load_snapshot_triples(cache: SankofaDeltaCache, timestep: int) -> SnapshotTriples:
    with cache.snapshot_triples_path(timestep).open("rb") as handle:
        return pickle.load(handle)


def load_triple_delta(cache: SankofaDeltaCache, timestep: int) -> TripleDelta:
    with cache.triple_delta_path(timestep).open("rb") as handle:
        return pickle.load(handle)


def build_knowledge_graph(snapshot_triples: SnapshotTriples, *, progress_label: str | None = None) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    total_triples = len(snapshot_triples)
    started = time.perf_counter()
    report_every = max(10_000, total_triples // 20) if total_triples > 0 else 1

    if progress_label is not None:
        progress_write(f"{progress_label}: loading {total_triples:,} triples into evaluation graph...")

    for index, triple in enumerate(snapshot_triples, start=1):
        graph.add_triple(triple)

        if (
            progress_label is not None
            and index < total_triples
            and index % report_every == 0
        ):
            progress_write(
                f"{progress_label}: loaded {index:,}/{total_triples:,} triples "
                f"({index / total_triples:.0%}) in {format_seconds(time.perf_counter() - started)}"
            )

    if progress_label is not None:
        progress_write(
            f"{progress_label}: ready with {graph.number_of_entities():,} entities, "
            f"{graph.number_of_relationships():,} relations, {graph.number_of_triples():,} triples "
            f"in {format_seconds(time.perf_counter() - started)}"
        )

    return graph


def load_cached_snapshot_kg(cache: SankofaDeltaCache, timestep: int, *, progress_label: str | None = None) -> KnowledgeGraph:
    return build_knowledge_graph(load_snapshot_triples(cache, timestep), progress_label=progress_label)


def initialize_streamed_evaluation_graph(cache: SankofaDeltaCache, *, progress_label: str | None = None) -> KnowledgeGraph:
    return load_cached_snapshot_kg(cache, 0, progress_label=progress_label)


def initialize_streamed_apex_graph(cache: SankofaDeltaCache, *, progress_label: str | None = None) -> KnowledgeGraph:
    initial_snapshot = [tuple(triple) for triple in load_snapshot_triples(cache, 0)]
    reserved_triples = []
    seen_triples = set()

    def remember(triple):
        if triple in seen_triples:
            return
        seen_triples.add(triple)
        reserved_triples.append(triple)

    for triple in initial_snapshot:
        remember(triple)

    for timestep in range(len(cache.snapshot_files) - 1):
        delta = load_triple_delta(cache, timestep)
        for triple in delta["triples_added"]:
            remember(tuple(triple))

    graph = KnowledgeGraph()
    graph.reserve_triples(reserved_triples)
    graph.apply_triple_delta(triples_added=initial_snapshot, triples_removed=[])

    if progress_label is not None:
        progress_write(
            f"{progress_label}: ready with {graph.number_of_entities():,} reserved entities, "
            f"{graph.number_of_relationships():,} reserved relations, {graph.number_of_triples():,} active triples"
        )

    return graph


def apply_triple_delta_to_knowledge_graph(graph: KnowledgeGraph, delta: TripleDelta) -> bool:
    triples_removed = [tuple(triple) for triple in delta["triples_removed"]]
    triples_added = [tuple(triple) for triple in delta["triples_added"]]
    return graph.apply_triple_delta(
        triples_added=triples_added,
        triples_removed=triples_removed,
    )


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

    validation_steps = progress(
        range(len(cache.snapshot_files) - 1),
        desc="Validating Sankofa cache",
        unit="delta",
        leave=False,
    )
    for timestep in validation_steps:
        started = time.perf_counter()
        reconstructed = apply_delta_to_graph(reconstructed, load_delta(cache, timestep))
        expected = parse_snapshot_graph(snapshot_paths[timestep + 1])

        if reconstructed != expected:
            raise ValueError(
                f"Cached Sankofa delta reconstruction diverged at timestep {timestep + 1}."
            )

        current_edge_count = graph_edge_count(reconstructed)
        edge_counts.append(current_edge_count)
        validation_steps.set_postfix(
            snapshot=snapshot_paths[timestep + 1].name,
            dt=format_seconds(time.perf_counter() - started),
            edges=current_edge_count,
        )

    return {
        "snapshot_count": len(cache.snapshot_files),
        "edge_counts": edge_counts,
        "cache_dir": str(cache.cache_dir),
        "cache_status": cache.cache_status,
    }


@contextlib.contextmanager
def _cache_build_lock(subgraph_dir: Path, *, verbose: bool):
    lock_path = subgraph_dir / CACHE_LOCK_FILE_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        waiting_logged = False
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if verbose and not waiting_logged:
                    progress_write(f"Sankofa cache: waiting for cache build lock at {lock_path}...")
                    waiting_logged = True
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                break

        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _build_sankofa_delta_cache(subgraph_dir: Path, snapshot_files: list[str]) -> SankofaDeltaCache:
    build_started = time.perf_counter()
    cache_dir = resolve_cache_dir(subgraph_dir)
    deltas_dir = cache_dir / DELTAS_DIR_NAME
    triple_deltas_dir = cache_dir / TRIPLE_DELTAS_DIR_NAME
    snapshot_triples_dir = cache_dir / SNAPSHOT_TRIPLES_DIR_NAME
    snapshot_paths = [subgraph_dir / file_name for file_name in snapshot_files]

    progress_write(
        f"Sankofa cache: building walk deltas and evaluation snapshots from {len(snapshot_paths)} snapshots "
        f"under {subgraph_dir}."
    )

    if cache_dir.exists():
        progress_write(f"Sankofa cache: clearing stale cache contents in {cache_dir}.")
        shutil.rmtree(cache_dir)

    deltas_dir.mkdir(parents=True, exist_ok=True)
    triple_deltas_dir.mkdir(parents=True, exist_ok=True)
    snapshot_triples_dir.mkdir(parents=True, exist_ok=True)

    initial_started = time.perf_counter()
    progress_write(f"Sankofa cache: parsing initial snapshot {snapshot_paths[0].name}...")
    current_triples = parse_snapshot_triples(snapshot_paths[0])
    current_graph = _graph_from_snapshot_triples(current_triples, snapshot_paths[0])

    progress_write(
        f"Sankofa cache: initial snapshot {snapshot_paths[0].name} ready "
        f"nodes={graph_node_count(current_graph):,} edges={graph_edge_count(current_graph):,} "
        f"triples={len(current_triples):,} dt={format_seconds(time.perf_counter() - initial_started)}"
    )

    with (cache_dir / INITIAL_GRAPH_FILE).open("wb") as handle:
        pickle.dump(current_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    initial_snapshot_cache_path = snapshot_triples_dir / _snapshot_triples_filename(0)
    with initial_snapshot_cache_path.open("wb") as handle:
        pickle.dump(current_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cache_bytes = (
        (cache_dir / INITIAL_GRAPH_FILE).stat().st_size +
        initial_snapshot_cache_path.stat().st_size
    )

    manifest: dict[str, object] = {
        "version": CACHE_VERSION,
        "edge_weight": EDGE_WEIGHT,
        "snapshot_files": [_snapshot_metadata(path) for path in snapshot_paths],
        "initial_graph": INITIAL_GRAPH_FILE,
        "deltas_dir": DELTAS_DIR_NAME,
        "triple_deltas_dir": TRIPLE_DELTAS_DIR_NAME,
        "snapshot_triples_dir": SNAPSHOT_TRIPLES_DIR_NAME,
        "delta_count": max(0, len(snapshot_paths) - 1),
        "snapshot_triples_count": len(snapshot_paths),
        "initial_node_count": graph_node_count(current_graph),
        "initial_edge_count": graph_edge_count(current_graph),
    }

    if len(snapshot_paths) > 1:
        progress_write(
            f"Sankofa cache: computing {len(snapshot_paths) - 1} graph deltas, "
            f"triple deltas, and snapshot triple caches..."
        )

    cache_steps = progress(
        range(len(snapshot_paths) - 1),
        desc="Building Sankofa cache",
        unit="delta",
        leave=False,
    )
    for timestep in cache_steps:
        started = time.perf_counter()
        next_triples = parse_snapshot_triples(snapshot_paths[timestep + 1])
        next_graph = _graph_from_snapshot_triples(next_triples, snapshot_paths[timestep + 1])
        delta = _compute_graph_delta(current_graph, next_graph)
        triple_delta = _compute_triple_delta(current_triples, next_triples)

        delta_path = deltas_dir / _delta_filename(timestep)
        with delta_path.open("wb") as handle:
            pickle.dump(delta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        triple_delta_path = triple_deltas_dir / _delta_filename(timestep)
        with triple_delta_path.open("wb") as handle:
            pickle.dump(triple_delta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        snapshot_cache_path = snapshot_triples_dir / _snapshot_triples_filename(timestep + 1)
        with snapshot_cache_path.open("wb") as handle:
            pickle.dump(next_triples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        growth_bytes = (
            delta_path.stat().st_size +
            triple_delta_path.stat().st_size +
            snapshot_cache_path.stat().st_size
        )
        cache_bytes += growth_bytes
        edge_delta_added = len(delta["edges_added"])
        edge_delta_removed = len(delta["edges_removed"])
        triple_delta_added = len(triple_delta["triples_added"])
        triple_delta_removed = len(triple_delta["triples_removed"])
        cache_steps.set_postfix(
            snapshot=snapshot_paths[timestep + 1].name,
            dt=format_seconds(time.perf_counter() - started),
            edges=f"+{edge_delta_added}/-{edge_delta_removed}",
            triples=f"+{triple_delta_added}/-{triple_delta_removed}",
            cache=format_bytes(cache_bytes),
            growth=f"+{format_bytes(growth_bytes)}",
        )

        current_graph = next_graph
        current_triples = next_triples

    manifest_path = cache_dir / MANIFEST_FILE
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    progress_write(
        f"Sankofa cache: ready at {cache_dir} snapshots={len(snapshot_paths)} "
        f"deltas={max(0, len(snapshot_paths) - 1)} size={format_bytes(cache_bytes)} "
        f"dt={format_seconds(time.perf_counter() - build_started)}"
    )
    progress_write(
        f"Sankofa cache: final graph nodes={graph_node_count(current_graph):,} "
        f"edges={graph_edge_count(current_graph):,}; manifest={manifest_path}"
    )

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

    if manifest.get("snapshot_triples_count") != len(snapshot_files):
        return False

    if manifest.get("snapshot_triples_dir") != SNAPSHOT_TRIPLES_DIR_NAME:
        return False

    if manifest.get("triple_deltas_dir") != TRIPLE_DELTAS_DIR_NAME:
        return False

    manifest_snapshots = manifest.get("snapshot_files")
    if not isinstance(manifest_snapshots, list) or len(manifest_snapshots) != len(snapshot_files):
        return False

    for timestep in range(max(0, len(snapshot_files) - 1)):
        if not (cache_dir / DELTAS_DIR_NAME / _delta_filename(timestep)).exists():
            return False
        if not (cache_dir / TRIPLE_DELTAS_DIR_NAME / _delta_filename(timestep)).exists():
            return False

    for timestep in range(len(snapshot_files)):
        if not (cache_dir / SNAPSHOT_TRIPLES_DIR_NAME / _snapshot_triples_filename(timestep)).exists():
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


def _snapshot_triples_filename(timestep: int) -> str:
    return f"snapshot_{timestep:04d}.pkl"


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


def _compute_triple_delta(current_triples: SnapshotTriples, next_triples: SnapshotTriples) -> TripleDelta:
    current_set = set(current_triples)
    next_set = set(next_triples)

    return {
        "triples_added": [triple for triple in next_triples if triple not in current_set],
        "triples_removed": [triple for triple in current_triples if triple not in next_set],
    }


def _parse_snapshot_line(line: str) -> tuple[str, str] | None:
    triple = _parse_snapshot_triple(line)
    if triple is None:
        return None

    source, _relation, target = triple
    return source, target


def _parse_snapshot_triple(line: str) -> SnapshotTriple | None:
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

    source, relation = fact[:2]
    target = " ".join(fact[2:])

    if not source or not target:
        return None

    return source, relation, target


def _graph_from_snapshot_triples(snapshot_triples: SnapshotTriples, snapshot_path: str | Path) -> Graph:
    graph: Graph = {}

    for line_number, (source_str, _relation, target_str) in enumerate(snapshot_triples, start=1):
        try:
            source = int(source_str)
            target = int(target_str)
        except ValueError as exc:
            raise ValueError(
                f"Snapshot {snapshot_path} contains a non-integer entity near parsed triple {line_number}: {(source_str, target_str)}"
            ) from exc

        graph.setdefault(source, {})
        graph.setdefault(target, {})
        graph[source][target] = EDGE_WEIGHT

    return graph


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