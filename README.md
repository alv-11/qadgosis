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

    This reuses a stored query bundle for the dataset when one matches the
    current generation settings. If none exists yet, a new bundle is generated
    and stored with the dataset.

- To generate a new set of queries or compute an average over multiple runs, execute:

    `python3 simulate.py`

    The output is stored in `multi_runs` with a corresponding folder for each run.
    Query bundles are no longer deleted between runs. Instead, run `1` uses query
    bundle `1`, run `2` uses query bundle `2`, and so on, generating any missing
    stored bundles on demand.

## Stored Query Logs

Generated query logs are stored with the dataset snapshots under:

`<subgraph_dir>/query_logs/`

Each reusable query bundle is kept in its own numbered folder whose name encodes
the generation settings and the bundle id. For example:

`tc5_cr0p4_q200_s1_p0_001`

The folder contains:

- `query_log_snapshot_<timestep>.pkl` files for each snapshot.
- `bundle_metadata.json` with generation settings and snapshot file metadata.

Reuse behavior:

- A normal `main.py` run reuses the first stored bundle that matches the current query generation settings and snapshot files.
- If no matching bundle exists, `main.py` generates a new one and stores it.
- `simulate.py` uses bundle `1`, `2`, `3`, ... across runs so multi-run experiments stay reusable.
- Old query logs should not be deleted anymore; they are meant to be reused.

## Sankofa Delta Cache

`main.py` now avoids reparsing every full snapshot during Sankofa Phase 1.
The first run against a dataset builds a cache under the repository data area
and later runs reuse it.

You can also prebuild the cache before running the framework:

```bash
cd kg_summarization
python3 build_sankofa_cache.py data/test_small
python3 build_sankofa_cache.py /path/to/snapshots --rebuild --validate
```

Both the prebuild script and normal framework execution print whether Sankofa is
generating a new cache or using a preexisting one.

Cache location:

- If the dataset already lives under `kg_summarization/data/...`, the cache is stored next to it in `<subgraph_dir>/.sankofa_cache/`.
- If the dataset lives somewhere else, the cache is stored in `kg_summarization/data/.sankofa_cache/<dataset-name>-<hash>/`.

The cache contains:

- `initial_graph.pkl`: the graph reconstructed from `snapshot_0.nt.gz`.
- `deltas/delta_<timestep>.pkl`: raw node/edge additions and removals between consecutive snapshots.
- `manifest.json`: cache metadata used to validate that the snapshots on disk still match the cached data.

Cache behavior:

- Snapshots on disk are still full graph states. The cache is a preprocessing layer on top of them.
- The cache rebuilds automatically if the snapshot list, file sizes, or mtimes change.
- Query-dependent dummy-node overlays are not cached. They are still applied at runtime inside Sankofa.
- This optimization currently targets Sankofa Phase 1 only. The later evaluation and baseline steps still load full snapshots through the existing Python path.

If you want to force a rebuild, delete the corresponding cache directory or call
`ensure_sankofa_delta_cache(..., rebuild=True)` from Python.

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

### 3. Reuse or grow stored query bundles

If you change `subgraph_dir`, SANKOFA automatically looks inside that dataset's
own `query_logs/` directory. Matching bundles are reused, and missing ones are
generated automatically. No manual cleanup is needed.

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