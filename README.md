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

### Per-snapshot random-walk behavior

The streaming run builds the initial walk-segment bank from every entity in
snapshot 0, with `QADGOSIS_NUM_WALKS` segments per entity (20 by default). For
each later snapshot it applies the incoming graph delta first, then removes
segments for deleted entities, repairs segments that cross deleted edges, and
creates segments for newly added entities. Repairs use the updated snapshot,
so newly added edges can be taken while a damaged segment is regenerated.

For each timestep, the current query seed entities are temporarily connected to
a dummy node. SANKOFA runs its PPR BigWalk from that node, removes the dummy
from entity ranking, and uses decayed visit counts to select the top-k entities.
The walk bank contains graph entities only; the query dummy is never retained
between timesteps.

In this code, `epsilon` is the restart/teleport probability, so it is the
complement of the paper's damping parameter: `epsilon = 1 - beta`. The default
`epsilon=0.90` therefore means `beta=0.10`, the practical setting recommended
in the paper's damping-factor analysis. Walk-segment lengths are sampled using
the geometric stopping rule described by the algorithm. The paper also reports
a separate experimental setup with `beta=0.9` and walk length 5, which is not
the same parameterization as that practical setting.

The current implementation recalculates the global visit distribution from
the walk bank at each timestep instead of maintaining its counters
incrementally as the paper pseudocode describes. This produces the same global
distribution for the current bank, with extra computation. Enabling RDF2Vec
does not change these SANKOFA walks or PPR rankings; it adds the separate
embedding-based scoring used during boundary and fallback triple selection.


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

## RDF2Vec-guided Summary Selection

RDF2Vec-style node embeddings can be enabled to guide the non-internal summary
triple selection. The embedding model is trained separately from each current
snapshot, using all current query seed nodes plus up to 10,000 sampled graph
nodes as walk starts. Set `QADGOSIS_USE_RDF2VEC=1` to enable it; set
`QADGOSIS_RDF2VEC_WEIGHT` between 0 and 1 to control the embedding contribution
when ranking both boundary triples and the remaining outside/random triples
(default `0.5`). The complementary weight is applied to normalized PPR scores:
0 means PPR-only ranking, 1 means embedding-only ranking, and 0.5 gives both
equal weight. If no candidate embeddings are available, the existing PPR-based
weighted-sampling fallback is used.

The supplied embedding routine learns vectors for entity nodes from walks on
the entity adjacency graph. In this initial integration, the query vector is
the mean vector of its training seed nodes, and a candidate triple vector is
the mean of its subject and object vectors. Predicate labels are not embedded
yet, so triples between the same entities but with different predicates get
the same embedding similarity. Embedding training is run once per timestep and
its runtime is included in `sankofa_results.csv`.

Install the added dependency with `python -m pip install -r requirements.txt`.
The VS Code launch configuration enables RDF2Vec for the small test graph.
For a terminal run, set `QADGOSIS_USE_RDF2VEC=1` before `python main.py`.

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

## NELL Snapshot Experiment

`kg_summarization/generate_nell_synthetic_snapshots.py` turns a three-column
NELL fact file into three snapshots while retaining the original entity names
and predicates in the snapshot files. By default it reads
`~/Downloads/nell/nell.txt` and writes to
`kg_summarization/data/nell_named_synthetic_3snap/`. The additions and removals
are sampled randomly from the static input; they are synthetic test changes,
not NELL's historical evolution. A manifest records the input hash and change
counts. No numeric IDs are written into the NELL snapshot files. At runtime,
the cache creates a reversible integer index only at the boundary to the
existing C++ walk engine, whose graph interface accepts integers; graph
queries, embeddings, and evaluation retain the original NELL names. The script
refuses to overwrite a non-empty output folder.

From `kg_summarization/`, generate snapshots with:

```bash
../.venv/bin/python generate_nell_synthetic_snapshots.py
```

Run SANKOFA without RDF2Vec, then with embedding-only ranking, using the same
saved query bundle and graph. This uses 3,000 queries per snapshot across 50
topics, leaving 600 held-out queries per snapshot:

```bash
QADGOSIS_SUBGRAPH_DIR="$PWD/data/nell_named_synthetic_3snap" \
PYTHONHASHSEED=42 \
QADGOSIS_RANDOM_SEED=42 \
QADGOSIS_NUM_WALKS=20 \
QADGOSIS_BUDGET_RATIO=0.05 \
QADGOSIS_QUERY_NUM_QUERIES=3000 \
QADGOSIS_QUERY_TOPIC_COUNT=50 \
QADGOSIS_QUERY_RANDOM_QUERY_PROB=0 \
QADGOSIS_QUERY_SHUFFLE=1 \
QADGOSIS_QUERY_HARD=1 \
QADGOSIS_QUERY_MIN_SHIFT_HOPS=1 \
QADGOSIS_QUERY_MIN_CHAIN_LEN=1 \
QADGOSIS_QUERY_MAX_CHAIN_LEN=1 \
QADGOSIS_USE_RDF2VEC=0 \
../.venv/bin/python main.py experiments/nell_named_synthetic_3snap/baseline_5pct_3000q

QADGOSIS_SUBGRAPH_DIR="$PWD/data/nell_named_synthetic_3snap" \
PYTHONHASHSEED=42 \
QADGOSIS_RANDOM_SEED=42 \
QADGOSIS_NUM_WALKS=20 \
QADGOSIS_BUDGET_RATIO=0.05 \
QADGOSIS_QUERY_NUM_QUERIES=3000 \
QADGOSIS_QUERY_TOPIC_COUNT=50 \
QADGOSIS_QUERY_RANDOM_QUERY_PROB=0 \
QADGOSIS_QUERY_SHUFFLE=1 \
QADGOSIS_QUERY_HARD=1 \
QADGOSIS_QUERY_MIN_SHIFT_HOPS=1 \
QADGOSIS_QUERY_MIN_CHAIN_LEN=1 \
QADGOSIS_QUERY_MAX_CHAIN_LEN=1 \
QADGOSIS_USE_RDF2VEC=1 \
QADGOSIS_RDF2VEC_WEIGHT=1 \
../.venv/bin/python main.py experiments/nell_named_synthetic_3snap/embeddings_5pct_3000q
```

The experiment outputs and measured results from the current run are described
in [`kg_summarization/experiments/nell_named_synthetic_3snap/README.md`](kg_summarization/experiments/nell_named_synthetic_3snap/README.md).
