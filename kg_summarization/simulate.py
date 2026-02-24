
"""Multi-run script to execute main.py multiple times,
- Runs the main script N_RUNS times.
- Deletes any existing query files BEFORE each run to force regeneration.
- Saves each run's outputs (CSV + PKL) into multi_runs/run_{i}/
- Computes averaged numeric results across all runs at the end.

"""
import subprocess
import shutil
from pathlib import Path
from glob import glob
import pandas as pd
import sys

# ---------- CONFIG ----------
NUM_RUNS = 1
PARENT_DIR = Path("multi_runs")
ORIGINAL_SCRIPT = "main.py"   
PYTHON_BIN = sys.executable        

QUERY_PATTERNS_TO_DELETE = [
    "query_log_snapshot*.pkl",
    "generated_queries*.pkl",
    "train_queries*.pkl",
    "test_queries*.pkl",
    "query_log_*.pkl",
]


CSV_PATTERNS_TO_COLLECT = [
    "*.csv"   
]

PKL_PATTERNS_TO_COLLECT = [
    "*.pkl"
]

EXPECTED_CSV_NAMES = []  


def remove_matching_files(patterns, directory="."):
    """Delete files matching patterns in directory (non-recursive)."""
    dirp = Path(directory)
    removed = []
    for pat in patterns:
        for p in dirp.glob(pat):
            try:
                p.unlink()
                removed.append(str(p))
            except Exception as e:
                print(f"Warning: failed to delete {p}: {e}")
    return removed

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
                    print(f"Warning: failed to move {src}: {e}")
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

    for r in range(1, NUM_RUNS + 1):
        run_dir = PARENT_DIR / f"run_{r}"
        print(f"\n=== RUN {r}/{NUM_RUNS} => {run_dir} ===")

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)


        removed = remove_matching_files(QUERY_PATTERNS_TO_DELETE, directory=".")
        if removed:
            print(f"Deleted {len(removed)} existing query files before run {r}.")
        else:
            print("No matching query files found to delete (OK).")


        cmd = [PYTHON_BIN, ORIGINAL_SCRIPT, str(run_dir)]
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: original script failed with exit code {e.returncode}.")
            print("Continuing to next run.")
            continue

        moved_csvs = move_matching_files(CSV_PATTERNS_TO_COLLECT, run_dir)
        moved_pkls = move_matching_files(PKL_PATTERNS_TO_COLLECT, run_dir)

        print(f"Moved {len(moved_csvs)} CSV(s) and {len(moved_pkls)} PKL(s) into {run_dir}.")

    # ---------- compute averages ----------
    print("\n=== All runs complete — computing averages ===")

    if EXPECTED_CSV_NAMES:
        csv_names = EXPECTED_CSV_NAMES
    else:
        csv_names = collect_all_csv_names(PARENT_DIR)
        if not csv_names:
            print("No CSV files found in run folders. Exiting.")
            return

    print("Averaging CSV files:", csv_names)

    for csv_name in csv_names:
        frames = []
        for r in range(1, NUM_RUNS + 1):
            p = PARENT_DIR / f"run_{r}" / csv_name
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    df["run"] = r
                    frames.append(df)
                except Exception as e:
                    print(f"Warning: failed to read {p}: {e}")

        if not frames:
            print(f"No data for {csv_name}; skipping.")
            continue

        combined = pd.concat(frames, ignore_index=True)

        if "Timestep" in combined.columns:
            averaged = combined.groupby("Timestep").mean(numeric_only=True).reset_index()
        else:
            averaged = combined.mean(numeric_only=True).to_frame().T

        out_path = PARENT_DIR / f"average_{csv_name}"
        averaged.to_csv(out_path, index=False)
        print(f"Saved averaged results to {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
