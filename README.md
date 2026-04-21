# SANKOFA
Online Query-aware Dynamic Graph Summaries

## Requirements
- Python >= 3.11

## Data
- The datasets are available for download
[here](https://drive.google.com/drive/folders/1PVlh5KGdQ4qkyEzSnDTCYGmXRuMGwgCT?usp=sharing). 

- To use, unzip and move the dataset to `kg_summarization/data/` and update the path in `main.py`


## C++
The walks component of Algorithm 2 of SANKOFA is implemented in C++ using pybind11. To compile the code, navigate to `kg_summarization/src/core` and run:

`python setup.py clean`

`python setup.py build_ext --inplace`

`python setup.py install`


## Running the code

- For a single run of SANKOFA and the baselines, return to the main directory `kg_summarization` and run:

    `python3 main.py` 

    This reuses the existing query logs.

- To generate a new set of queries or compute an average over multiple runs, execute:

    `python3 simulate.py`

    The output is stored in `multi_runs` with a corresponding folder for each run.

## Synthetic Test Graph Workflow

The repository includes a synthetic graph generator that writes deterministic
`snapshot_*.nt.gz` files in the same format as the normal loader expects.

### 1. Generate a test graph

- Default example: generate the standard test dataset in `kg_summarization/data/test_small/`:

```bash
python3 kg_summarization/generate_test_graph.py --overwrite
```

- Larger example: generate a custom dataset with more entities and snapshots:

```bash
python3 kg_summarization/generate_test_graph.py \
    --total-entities 750 \
    --num-snapshots 6 \
    --output-dir kg_summarization/data/test_750_6snap \
    --overwrite
```

- Useful options:

    - `--total-entities`: total number of entities across all node groups.
    - `--num-snapshots`: number of `snapshot_*.nt.gz` files to emit.
    - `--output-dir`: destination directory for the generated dataset.
    - `--overwrite`: replace an existing output directory.
    - `--skip-validate`: skip the round-trip `DBPedia` validation step.

The generator validates the output by loading each snapshot back through `src.base.DBPedia`.

### 2. Point SANKOFA at the generated dataset

`main.py` currently uses a hardcoded dataset path. Update `subgraph_dir` in
`kg_summarization/main.py` to the directory you generated.

Example:

```python
subgraph_dir = "data/test_750_6snap"
```

Use a path relative to the `kg_summarization/` directory if you run `main.py`
from there.

### 3. Clear old query logs when switching datasets

If you change `subgraph_dir` and run `main.py` directly, clear the cached query
logs first so you do not reuse queries generated for a different graph:

```bash
cd kg_summarization
rm -f query_log_snapshot*.pkl generated_queries*.pkl train_queries*.pkl test_queries*.pkl query_log_*.pkl
```

`simulate.py` already deletes these files before each run.

### 4. Run the generated test graph

- Single run:

```bash
cd kg_summarization
python3 main.py
```

- Multi-run average with fresh query generation on each run:

```bash
cd kg_summarization
python3 simulate.py
```

`simulate.py` stores per-run outputs in `kg_summarization/multi_runs/run_<n>/`
and writes averaged CSVs into `kg_summarization/multi_runs/`.

### 5. Minimal command flow example

```bash
python3 kg_summarization/generate_test_graph.py \
    --total-entities 500 \
    --num-snapshots 6 \
    --output-dir kg_summarization/data/test_6snap \
    --overwrite

cd kg_summarization
# edit main.py and set: subgraph_dir = "data/test_6snap"
python3 simulate.py
```