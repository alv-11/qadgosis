
"""Multi-run driver for main.py.

- Creates a unique timestamped batch directory under <subgraph_dir>/simulations/ for each invocation.
- Prewarms the Sankofa cache and query bundles before launching child runs.
- Executes run bundles in conservative parallel batches based on graph size and host memory.
- Saves all run outputs directly into per-run directories without shared-file moves.
- Writes a batch overview manifest with execution time, graph, queries, settings, and run status.
- Computes averaged numeric CSV results across successful runs at the end.
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.progress import format_seconds, progress, progress_write
from src.sankofa_delta import ensure_sankofa_delta_cache
from src.summary import prepare_queries_and_splits


DEFAULT_SUBGRAPH_DIR = "/mnt/c/Users/kaspe/OneDrive/Skrivebord/Reps/qadgosis/kg_summarization/data/test_750"
DEFAULT_NUM_RUNS = 5
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARENT_DIR_NAME = "simulations"
ORIGINAL_SCRIPT = SCRIPT_DIR / "main.py"
PYTHON_BIN = sys.executable
EXPECTED_CSV_NAMES = []
CSV_GLOB = "*.csv"
GIB = 1024 ** 3
DEFAULT_LIVE_OUTPUT_MODE = "first"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def format_bytes_human(num_bytes):
    if num_bytes is None:
        return None

    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0


def resolve_num_runs():
    return max(1, int(os.environ.get("QADGOSIS_NUM_RUNS", DEFAULT_NUM_RUNS)))


def resolve_subgraph_dir():
    return Path(os.environ.get("QADGOSIS_SUBGRAPH_DIR", DEFAULT_SUBGRAPH_DIR)).resolve()


def resolve_parent_dir(subgraph_dir):
    override = os.environ.get("QADGOSIS_MULTI_RUNS_DIR")
    if override:
        return Path(override).resolve()
    return (subgraph_dir / DEFAULT_PARENT_DIR_NAME).resolve()


def resolve_parallel_cap():
    raw_value = os.environ.get("QADGOSIS_MAX_PARALLEL")
    if not raw_value:
        return None
    return max(1, int(raw_value))


def resolve_live_output_mode():
    mode = os.environ.get("QADGOSIS_LIVE_RUN_OUTPUT", DEFAULT_LIVE_OUTPUT_MODE).strip().lower()
    if mode not in {"none", "first", "all"}:
        return DEFAULT_LIVE_OUTPUT_MODE
    return mode


def dataset_label(subgraph_dir):
    name = subgraph_dir.name.strip()
    return name or "dataset"


def create_batch_dir(parent_dir, subgraph_dir, num_runs):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{timestamp}_{dataset_label(subgraph_dir)}_runs{num_runs}"
    candidate = parent_dir / base_name
    suffix = 1

    while candidate.exists():
        candidate = parent_dir / f"{base_name}_{suffix:02d}"
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_overview(batch_dir, overview):
    overview_path = batch_dir / "overview.json"
    with overview_path.open("w", encoding="utf-8") as handle:
        json.dump(overview, handle, indent=2)
    return overview_path


def snapshot_paths(subgraph_dir):
    return sorted(subgraph_dir.glob("snapshot_*.nt.gz"))


def estimate_graph_size_bytes(subgraph_dir):
    total_bytes = 0
    file_count = 0
    for path in snapshot_paths(subgraph_dir):
        try:
            total_bytes += path.stat().st_size
            file_count += 1
        except OSError:
            continue

    if file_count == 0:
        return None, 0

    return total_bytes, file_count


def detect_total_memory_bytes():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and total_pages > 0:
            return int(page_size * total_pages)
    except (AttributeError, ValueError, OSError):
        pass

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
        except OSError:
            return None

    return None


def choose_parallel_workers(num_runs, subgraph_dir):
    graph_bytes, snapshot_count = estimate_graph_size_bytes(subgraph_dir)
    total_memory_bytes = detect_total_memory_bytes()
    safe_memory_budget = int(total_memory_bytes * 0.35) if total_memory_bytes is not None else None
    estimated_per_run_bytes = max(2 * GIB, (graph_bytes or 0) * 10)
    cpu_cap = max(1, min(4, os.cpu_count() or 1))

    if safe_memory_budget is None:
        memory_cap = 1
    else:
        memory_cap = max(1, safe_memory_budget // estimated_per_run_bytes)

    conservative_limit = max(1, min(num_runs, cpu_cap, memory_cap))
    user_cap = resolve_parallel_cap()
    if user_cap is not None:
        conservative_limit = min(conservative_limit, user_cap)

    return conservative_limit, {
        "graph_size_bytes": graph_bytes,
        "snapshot_count": snapshot_count,
        "total_memory_bytes": total_memory_bytes,
        "safe_memory_budget_bytes": safe_memory_budget,
        "estimated_per_run_bytes": estimated_per_run_bytes,
        "cpu_cap": cpu_cap,
        "conservative_parallel_limit": conservative_limit,
        "user_parallel_cap": user_cap,
    }


def prewarm_runtime_artifacts(subgraph_dir, num_runs):
    progress_write(f"Prewarming Sankofa cache for {subgraph_dir}...")
    cache = ensure_sankofa_delta_cache(subgraph_dir, verbose=True)

    bundle_steps = progress(
        range(1, num_runs + 1),
        desc="Preparing query bundles",
        unit="bundle",
    )
    for bundle_index in bundle_steps:
        _training_data, _test_data, _queries, snapshot_files = prepare_queries_and_splits(
            subgraph_dir,
            query_bundle_index=bundle_index,
        )
        bundle_steps.set_postfix(bundle=bundle_index, timesteps=len(snapshot_files))

    return cache


def build_run_spec(batch_dir, run_index):
    run_dir = batch_dir / f"run_{run_index:03d}_bundle_{run_index:03d}"
    log_path = run_dir / "execution.log"
    cmd = [PYTHON_BIN, str(ORIGINAL_SCRIPT), str(run_dir), str(run_index)]
    return {
        "run_index": run_index,
        "bundle_index": run_index,
        "run_dir": run_dir,
        "log_path": log_path,
        "command": cmd,
    }


def should_live_stream_run(run_index, live_output_mode):
    if live_output_mode == "all":
        return True
    if live_output_mode == "first":
        return run_index == 1
    return False


def _stream_child_output(spec):
    stdout = spec.get("stdout_pipe")
    log_handle = spec["log_handle"]
    if stdout is None:
        return

    prefix = f"[run {spec['run_index']}]"
    try:
        for line in stdout:
            log_handle.write(line)
            log_handle.flush()
            text = line.rstrip()
            if text:
                progress_write(f"{prefix} {text}")
    finally:
        stdout.close()


def load_json_if_exists(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_all_csv_names(run_dirs):
    names = set()
    for run_dir in run_dirs:
        for path in run_dir.glob(CSV_GLOB):
            names.add(path.name)
    return sorted(names)


def finalize_run_record(spec, exit_code):
    run_dir = spec["run_dir"]
    return {
        "run_index": spec["run_index"],
        "bundle_index": spec["bundle_index"],
        "run_dir": str(run_dir),
        "log_path": str(spec["log_path"]),
        "command": spec["command"],
        "started_at": spec["started_at"],
        "finished_at": utc_now_iso(),
        "duration_seconds": time.perf_counter() - spec["started_perf"],
        "exit_code": exit_code,
        "status": "success" if exit_code == 0 else "failed",
        "output_files": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
        "main_run_metadata": load_json_if_exists(run_dir / "run_metadata.json"),
    }


def average_successful_runs(batch_dir, successful_records):
    if not successful_records:
        progress_write("No successful runs available for averaging.")
        return []

    run_dirs = [Path(record["run_dir"]) for record in successful_records]
    csv_names = EXPECTED_CSV_NAMES or collect_all_csv_names(run_dirs)
    if not csv_names:
        progress_write("No CSV files found in successful run folders. Skipping averages.")
        return []

    progress_write(f"Averaging CSV files: {csv_names}")
    average_steps = progress(csv_names, desc="Averaging result files", unit="file")
    saved_outputs = []

    for csv_name in average_steps:
        frames = []
        for record in successful_records:
            path = Path(record["run_dir"]) / csv_name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                df["run"] = record["run_index"]
                frames.append(df)
            except Exception as exc:
                progress_write(f"Warning: failed to read {path}: {exc}")

        if not frames:
            progress_write(f"No data for {csv_name}; skipping.")
            continue

        combined = pd.concat(frames, ignore_index=True)
        if "Timestep" in combined.columns:
            averaged = combined.groupby("Timestep").mean(numeric_only=True).reset_index()
        else:
            averaged = combined.mean(numeric_only=True).to_frame().T

        out_path = batch_dir / f"average_{csv_name}"
        averaged.to_csv(out_path, index=False)
        average_steps.set_postfix(file=csv_name, runs=len(frames), rows=len(combined))
        progress_write(f"Saved averaged results to {out_path}")
        saved_outputs.append(str(out_path))

    return saved_outputs


def main():
    num_runs = resolve_num_runs()
    subgraph_dir = resolve_subgraph_dir()
    parent_dir = resolve_parent_dir(subgraph_dir)

    parent_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = create_batch_dir(parent_dir, subgraph_dir, num_runs)

    cache = prewarm_runtime_artifacts(subgraph_dir, num_runs)
    parallel_workers, parallel_info = choose_parallel_workers(num_runs, subgraph_dir)

    progress_write(
        "Launching multi-run batch "
        f"{batch_dir.name} with up to {parallel_workers} parallel run(s) for graph {subgraph_dir}."
    )

    overview = {
        "batch_dir": str(batch_dir),
        "created_at": utc_now_iso(),
        "completed_at": None,
        "requested_runs": num_runs,
        "parallel_workers": parallel_workers,
        "parallel_heuristic": {
            "graph_size_bytes": parallel_info["graph_size_bytes"],
            "graph_size_human": format_bytes_human(parallel_info["graph_size_bytes"]),
            "snapshot_count": parallel_info["snapshot_count"],
            "total_memory_bytes": parallel_info["total_memory_bytes"],
            "total_memory_human": format_bytes_human(parallel_info["total_memory_bytes"]),
            "safe_memory_budget_bytes": parallel_info["safe_memory_budget_bytes"],
            "safe_memory_budget_human": format_bytes_human(parallel_info["safe_memory_budget_bytes"]),
            "estimated_per_run_bytes": parallel_info["estimated_per_run_bytes"],
            "estimated_per_run_human": format_bytes_human(parallel_info["estimated_per_run_bytes"]),
            "cpu_cap": parallel_info["cpu_cap"],
            "user_parallel_cap": parallel_info["user_parallel_cap"],
        },
        "graph": {
            "subgraph_dir": str(subgraph_dir),
            "cache_dir": str(cache.cache_dir),
            "snapshot_count": len(cache.snapshot_files),
        },
        "query_bundles": list(range(1, num_runs + 1)),
        "runs": [],
        "averaged_outputs": [],
    }
    write_overview(batch_dir, overview)

    pending = [build_run_spec(batch_dir, run_index) for run_index in range(1, num_runs + 1)]
    active = []
    completed = []
    live_output_mode = resolve_live_output_mode()

    child_env = os.environ.copy()
    child_env["QADGOSIS_SUBGRAPH_DIR"] = str(subgraph_dir)

    run_steps = progress(range(num_runs), desc="Simulation runs", unit="run")
    success_count = 0
    failure_count = 0

    while pending or active:
        while pending and len(active) < parallel_workers:
            spec = pending.pop(0)
            spec["run_dir"].mkdir(parents=True, exist_ok=True)
            spec["started_perf"] = time.perf_counter()
            spec["started_at"] = utc_now_iso()
            spec["live_output"] = should_live_stream_run(spec["run_index"], live_output_mode)
            log_handle = spec["log_path"].open("w", encoding="utf-8")
            spec["log_handle"] = log_handle
            child_env_for_run = dict(child_env)

            if spec["live_output"]:
                child_env_for_run["QADGOSIS_LIVE_RUN_PROGRESS"] = "1"
                child_env_for_run["PYTHONUNBUFFERED"] = "1"
                spec["process"] = subprocess.Popen(
                    spec["command"],
                    cwd=str(SCRIPT_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=child_env_for_run,
                    text=True,
                    bufsize=1,
                )
                spec["stdout_pipe"] = spec["process"].stdout
                spec["stream_thread"] = threading.Thread(
                    target=_stream_child_output,
                    args=(spec,),
                    daemon=True,
                )
                spec["stream_thread"].start()
            else:
                spec["process"] = subprocess.Popen(
                    spec["command"],
                    cwd=str(SCRIPT_DIR),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=child_env_for_run,
                )
            active.append(spec)
            progress_write(
                f"Started run {spec['run_index']} with bundle {spec['bundle_index']} -> {spec['run_dir']}"
            )

        remaining = []
        completed_this_tick = []
        for spec in active:
            exit_code = spec["process"].poll()
            if exit_code is None:
                remaining.append(spec)
                continue

            if spec.get("stream_thread") is not None:
                spec["stream_thread"].join(timeout=2.0)
            spec["log_handle"].close()
            record = finalize_run_record(spec, exit_code)
            completed.append(record)
            completed_this_tick.append(record)

        active = remaining

        for record in completed_this_tick:
            run_steps.update(1)
            if record["status"] == "success":
                success_count += 1
            else:
                failure_count += 1

            run_steps.set_postfix(ok=success_count, fail=failure_count, active=len(active))
            progress_write(
                f"Finished run {record['run_index']} ({record['status']}) in {format_seconds(record['duration_seconds'])}."
            )

        if pending or active:
            time.sleep(0.2)

    successful_records = [record for record in completed if record["status"] == "success"]
    averaged_outputs = average_successful_runs(batch_dir, successful_records)

    overview["completed_at"] = utc_now_iso()
    overview["success_count"] = len(successful_records)
    overview["failure_count"] = len(completed) - len(successful_records)
    overview["runs"] = completed
    overview["averaged_outputs"] = averaged_outputs
    if successful_records and successful_records[0]["main_run_metadata"] is not None:
        overview["main_settings"] = successful_records[0]["main_run_metadata"]

    overview_path = write_overview(batch_dir, overview)
    progress_write(f"Saved batch overview to {overview_path}")
    progress_write("Done.")


if __name__ == "__main__":
    main()
