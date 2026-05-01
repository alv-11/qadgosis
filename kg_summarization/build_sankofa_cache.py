import argparse
from pathlib import Path

from src.sankofa_delta import ensure_sankofa_delta_cache, resolve_cache_dir, validate_sankofa_delta_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prebuild the Sankofa delta cache for a snapshot dataset."
    )
    parser.add_argument(
        "graph_path",
        help="Path to a directory containing snapshot_*.nt.gz files.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration even if a valid cache already exists.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Reconstruct snapshots from the cache after building or loading it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph_path)

    print(f"Dataset: {graph_path.resolve()}")
    print(f"Target cache directory: {resolve_cache_dir(graph_path)}")

    cache = ensure_sankofa_delta_cache(graph_path, rebuild=args.rebuild, verbose=True)

    print(f"Sankofa cache: ready for later executions ({cache.cache_status}).")
    print(f"Snapshots discovered: {len(cache.snapshot_files)}")

    if args.validate:
        validation = validate_sankofa_delta_cache(graph_path, rebuild=False)
        print(
            "Sankofa cache: validation passed "
            f"({validation['snapshot_count']} snapshots, cache at {validation['cache_dir']})."
        )


if __name__ == "__main__":
    main()