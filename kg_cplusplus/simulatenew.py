#!/usr/bin/env python3

import os
import shutil
import subprocess
import pandas as pd
from glob import glob
from pathlib import Path

N_RUNS = 10
PARENT_DIR = Path("multi_runs")

QUERY_LOG_PATTERNS = [
    "*generated_queries*.pkl",
    "*train_queries*.pkl",
    "*test_queries*.pkl",
    "*query_log*.pkl"
]

# Ensure parent folder exists
PARENT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------
#             MULTI-RUN EXPERIMENT
# ---------------------------------------------
for r in range(1, N_RUNS + 1):

    print(f"\n============================")
    print(f"        RUN {r}/{N_RUNS}")
    print(f"============================\n")

    run_dir = PARENT_DIR / f"run_{r}"
    run_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------
    # COPY THIS RUN'S QUERY FILES BACK TO WORKING DIRECTORY
    # -------------------------------------------------------
    for pattern in QUERY_LOG_PATTERNS:
        for pkl in run_dir.glob(pattern):
            shutil.copy(pkl, ".")
    print(f"✔ Reusing query files from {run_dir}\n")

    # -----------------------
    # 1. Run your main script
    # -----------------------
    subprocess.run(["python3", "montecarlo.py"], check=True)

    # ------------------------------------------------------
    # 2. Move RESULT CSVs AND QUERY LOGS to run_r/ folder
    # ------------------------------------------------------
    # Move CSV results
    for csv_file in Path(".").glob("*.csv"):
        shutil.move(str(csv_file), str(run_dir / csv_file.name))

    # Move PKL query files produced (or reused) this run
    for pattern in QUERY_LOG_PATTERNS:
        for pkl in Path(".").glob(pattern):
            shutil.move(str(pkl), str(run_dir / pkl.name))

    print(f"✔ Saved run {r} outputs to {run_dir}")


print("\nAll runs completed.\n")

