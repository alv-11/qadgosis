
"""Multi-run script to execute main.py multiple times,
- Runs the main script N_RUNS times.
- Reuses stored query bundles when available and generates missing ones on demand.
- Saves each run's outputs (CSV + PKL) into multi_runs/run_{i}/
- Computes averaged numeric results across all runs at the end.

"""
import subprocess
import shutil
import time
from pathlib import Path
from glob import glob
import pandas as pd
import sys

from src.progress import format_seconds, progress, progress_write

# ---------- CONFIG ----------
NUM_RUNS = 5
PARENT_DIR = Path("multi_runs")
ORIGINAL_SCRIPT = "main.py"   
PYTHON_BIN = sys.executable        


CSV_PATTERNS_TO_COLLECT = [
    "*.csv"   
]

PKL_PATTERNS_TO_COLLECT = [
    "*.pkl"
]

EXPECTED_CSV_NAMES = []  

def move_matching_files(patterns, dest_dir):
    """Move files matching patterns in current directory to dest_dir."""
    moved = []
    for pat in patterns:
        for p in glob(pat):
            src = Path(p)
            if src.exists():
                try:
                    shutil.move(str(src), str(dest_dir / src.name))
                    moved.append(str(dest_dir / src.name))
                except Exception as e:
                    progress_write(f"Warning: failed to move {src}: {e}")
    return moved

def collect_all_csv_names(parent_dir):
    """Return sorted unique CSV filenames that appear across run folders."""
    names = set()
    for run_dir in sorted(parent_dir.glob("run_*")):
        for f in run_dir.glob("*.csv"):
            names.add(f.name)
    return sorted(names)

# ---------- Main ----------
def main():
    PARENT_DIR.mkdir(exist_ok=True)

    success_count = 0
    failure_count = 0
    run_steps = progress(range(1, NUM_RUNS + 1), desc="Simulation runs", unit="run")
    for r in run_steps:
        run_dir = PARENT_DIR / f"run_{r}"
        started = time.perf_counter()
        progress_write(f"=== RUN {r}/{NUM_RUNS} => {run_dir} ===")

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        progress_write(f"Selecting query bundle slot {r} for this run.")

        cmd = [PYTHON_BIN, ORIGINAL_SCRIPT, str(run_dir), str(r)]
        progress_write("Running:", " ".join(cmd))
        run_steps.clear()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            run_steps.refresh()
            failure_count += 1
            run_steps.set_postfix(
                run=run_dir.name,
                ok=success_count,
                fail=failure_count,
                dt=format_seconds(time.perf_counter() - started),
                code=e.returncode,
            )
            progress_write(f"Error: original script failed with exit code {e.returncode}.")
            progress_write("Continuing to next run.")
            continue

        run_steps.refresh()

        moved_csvs = move_matching_files(CSV_PATTERNS_TO_COLLECT, run_dir)
        moved_pkls = move_matching_files(PKL_PATTERNS_TO_COLLECT, run_dir)
        success_count += 1
        run_steps.set_postfix(
            run=run_dir.name,
            ok=success_count,
            fail=failure_count,
            dt=format_seconds(time.perf_counter() - started),
            files=len(moved_csvs) + len(moved_pkls),
        )

        progress_write(f"Moved {len(moved_csvs)} CSV(s) and {len(moved_pkls)} PKL(s) into {run_dir}.")

    # ---------- compute averages ----------
    progress_write("=== All runs complete — computing averages ===")

    if EXPECTED_CSV_NAMES:
        csv_names = EXPECTED_CSV_NAMES
    else:
        csv_names = collect_all_csv_names(PARENT_DIR)
        if not csv_names:
            progress_write("No CSV files found in run folders. Exiting.")
            return

    progress_write("Averaging CSV files:", csv_names)

    average_steps = progress(csv_names, desc="Averaging result files", unit="file")
    for csv_name in average_steps:
        frames = []
        for r in range(1, NUM_RUNS + 1):
            p = PARENT_DIR / f"run_{r}" / csv_name
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    df["run"] = r
                    frames.append(df)
                except Exception as e:
                    progress_write(f"Warning: failed to read {p}: {e}")

        if not frames:
            progress_write(f"No data for {csv_name}; skipping.")
            continue

        combined = pd.concat(frames, ignore_index=True)

        if "Timestep" in combined.columns:
            averaged = combined.groupby("Timestep").mean(numeric_only=True).reset_index()
        else:
            averaged = combined.mean(numeric_only=True).to_frame().T

        average_steps.set_postfix(
            file=csv_name,
            runs=len(frames),
            rows=len(combined),
        )

        out_path = PARENT_DIR / f"average_{csv_name}"
        averaged.to_csv(out_path, index=False)
        progress_write(f"Saved averaged results to {out_path}")

    progress_write("Done.")

if __name__ == "__main__":
    main()
