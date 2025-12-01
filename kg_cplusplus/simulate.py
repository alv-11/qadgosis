# import os
# import shutil
# import subprocess
# import pandas as pd
# from glob import glob

# N_RUNS = 5
# RUNS_DIR = "runs"

# QUERY_LOG_PATTERNS = [
#     "*generated_queries*.pkl",
#     "*train_queries*.pkl",
#     "*test_queries*.pkl"
# ]

# # Make runs folder
# os.makedirs(RUNS_DIR, exist_ok=True)

# # -----------------------------
# #  RUN EXPERIMENT MULTIPLE TIMES
# # -----------------------------
# for r in range(N_RUNS):
#     print(f"\n============================")
#     print(f"        RUN {r+1}/{N_RUNS}")
#     print(f"============================\n")

#     run_dir = os.path.join(RUNS_DIR, f"run_{r}")
#     os.makedirs(run_dir, exist_ok=True)

#     # -----------------------
#     # 1. Run your main script
#     # -----------------------
#     subprocess.run(["python3", "montecarlo.py"], check=True)

#     # ------------------------------------------------------
#     # 2. Move RESULT CSVs AND QUERY LOGS to run_X folder
#     # ------------------------------------------------------
#     # Move CSV files
#     for csv_file in glob("*.csv"):
#         shutil.move(csv_file, os.path.join(run_dir, csv_file))

#     # Move PKL query files
#     for pattern in QUERY_LOG_PATTERNS:
#         for pkl in glob(pattern):
#             shutil.move(pkl, os.path.join(run_dir, pkl))

#     print(f"Saved run {r} outputs to {run_dir}")

# print("\nAll runs completed.\n")

# # -----------------------------
# #  COMPUTE AVERAGES
# # -----------------------------
# print("Computing average results...\n")

# all_frames = []

# for r in range(N_RUNS):
#     run_dir = os.path.join(RUNS_DIR, f"run_{r}")
#     csv_files = glob(os.path.join(run_dir, "*.csv"))

#     for fpath in csv_files:
#         df = pd.read_csv(fpath)
#         df["run"] = r
#         all_frames.append(df)

# if not all_frames:
#     raise RuntimeError("No CSV results found in any run folders.")

# combined = pd.concat(all_frames, ignore_index=True)

# # Average across runs by timestep
# averaged = (
#     combined
#     .groupby("Timestep")
#     .mean(numeric_only=True)
#     .reset_index()
# )

# averaged.to_csv("averaged_results.csv", index=False)
# print("Averaged results written to averaged_results.csv")


#!/usr/bin/env python3
"""
master_runner.py

- Runs your original script N_RUNS times.
- Deletes any existing query files BEFORE each run to force regeneration.
- Saves each run's outputs (CSV + PKL) into multi_runs/run_{i}/
- Computes averaged numeric results across all runs at the end.

Adjust ORIGINAL_SCRIPT and the filename lists below to match your project.
"""
import subprocess
import shutil
from pathlib import Path
from glob import glob
import pandas as pd
import os
import sys

# ---------- CONFIG ----------
NUM_RUNS = 1
PARENT_DIR = Path("multi_runs")
ORIGINAL_SCRIPT = "montecarlo.py"   # change if your script has a different name
PYTHON_BIN = sys.executable         # uses same python interpreter that's running this script

# Patterns of query files the original script writes (we will delete these before each run)
QUERY_PATTERNS_TO_DELETE = [
    "query_log_snapshot*.pkl",
    "generated_queries*.pkl",
    "train_queries*.pkl",
    "test_queries*.pkl",
    "query_log_*.pkl",
]

# CSV patterns to collect after each run (adjust if your script writes other CSVs)
CSV_PATTERNS_TO_COLLECT = [
    "*.csv"   # collect all csvs produced by original script; you may refine this
]

# PKL patterns to collect after each run (so each run keeps its queries)
PKL_PATTERNS_TO_COLLECT = [
    "*.pkl"
]

# Names of CSV files you expect and want averaged (if left empty, script will discover CSVs)
EXPECTED_CSV_NAMES = []  # e.g. ["sankofa_results.csv", "all_results.csv"]

# ---------- Helpers ----------
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
    # prepare parent dir
    PARENT_DIR.mkdir(exist_ok=True)

    for r in range(1, NUM_RUNS + 1):
        run_dir = PARENT_DIR / f"run_{r}"
        print(f"\n=== RUN {r}/{NUM_RUNS} => {run_dir} ===")

        # ensure clean run dir
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1) Delete query files from working dir to force regeneration
        removed = remove_matching_files(QUERY_PATTERNS_TO_DELETE, directory=".")
        if removed:
            print(f"Deleted {len(removed)} existing query files before run {r}.")
        else:
            print("No matching query files found to delete (OK).")

        # 2) Run original script, passing run_dir so the script can save there
        #    (Your original script should accept a folder path argument and save into it.)
        cmd = [PYTHON_BIN, ORIGINAL_SCRIPT, str(run_dir)]
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: original script failed with exit code {e.returncode}.")
            print("Continuing to next run.")
            continue

        # 3) Move CSVs and PKLs that were created by the original script into run_dir
        moved_csvs = move_matching_files(CSV_PATTERNS_TO_COLLECT, run_dir)
        moved_pkls = move_matching_files(PKL_PATTERNS_TO_COLLECT, run_dir)

        print(f"Moved {len(moved_csvs)} CSV(s) and {len(moved_pkls)} PKL(s) into {run_dir}.")

    # ---------- After all runs: compute averages ----------
    print("\n=== All runs complete — computing averages ===")

    # discover CSVs if not explicitly provided
    if EXPECTED_CSV_NAMES:
        csv_names = EXPECTED_CSV_NAMES
    else:
        csv_names = collect_all_csv_names(PARENT_DIR)
        if not csv_names:
            print("No CSV files found in run folders. Exiting.")
            return

    print("Averaging CSV files:", csv_names)

    # For each CSV name, gather from runs and compute average of numeric columns grouped by Timestep (if present)
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

        # If 'Timestep' is present, groupby it and take numeric mean; otherwise take numeric mean across whole df
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
