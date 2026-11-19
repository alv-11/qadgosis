# NELL one-hop summary-budget sweep (1,000 queries)

Runs compare SANKOFA without RDF2Vec (`baseline`) against SANKOFA with RDF2Vec-guided selection (`embeddings`) at four requested summary budgets. The input is the same three-snapshot NELL synthetic dataset used in the earlier experiments.

## Shared workload and settings

- 1,000 one-hop query records requested per snapshot; 20 topics; hard queries; chain length exactly 1.
- Each snapshot query log is split 80/20 by the current evaluator: 800 training and 200 held-out test queries.
- All eight runs reused the same saved query bundle. The `queries.json` files have identical SHA-256 hashes, including across budgets.
- Random seed 42; SANKOFA settings: 20 walks, `c=2`, `epsilon=0.9`, decay `0.8`, window size 100.
- Baseline runs set RDF2Vec off. Embedding runs set RDF2Vec on with embedding weight 1.0 and one worker.
- The dataset has 145,452 initial triples. The actual summary-size values below come from each run’s metadata.

## Results

“Future F1” and “Future Macro F1” are simple arithmetic means of the three timestep values in each CSV. “Past unseen F1” is the mean over timesteps 1 and 2 because timestep 0 has no prior snapshot. These means are a compact comparison; the CSVs contain the exact timestep results.

| Requested budget | Actual summary triples | Future F1 baseline / embeddings | Future Macro F1 baseline / embeddings | Past unseen F1 baseline / embeddings |
|---:|---:|---:|---:|---:|
| 0.5% | 727 | 0.9073 / 0.9073 | 0.9387 / 0.9387 | 0.8889 / 0.8889 |
| 0.05% | 72 | 0.5749 / 0.5571 | 0.6121 / 0.5987 | 0.2645 / 0.2470 |
| 0.005% | 7 | 0.0773 / 0.1092 | 0.0490 / 0.0814 | 0.0000 / 0.0000 |
| 0.0005% | 4 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

## Budget floor note

The 0.0005% budget would allow fewer than one triple from a 145,452-triple graph. The implementation enforces a minimum summary size of four triples, so both runs at this setting contain 4 triples (about 0.00275% of the initial graph). This is why the run name records the requested budget while `run_metadata.json` records the actual size.

## Run folders

- **0.5%:** [`baseline_0p5pct`](./baseline_0p5pct/) and [`embeddings_0p5pct`](./embeddings_0p5pct/)
- **0.05%:** [`baseline_0p05pct`](./baseline_0p05pct/) and [`embeddings_0p05pct`](./embeddings_0p05pct/)
- **0.005%:** [`baseline_0p005pct`](./baseline_0p005pct/) and [`embeddings_0p005pct`](./embeddings_0p005pct/)
- **0.0005%:** [`baseline_0p0005pct`](./baseline_0p0005pct/) and [`embeddings_0p0005pct`](./embeddings_0p0005pct/)

Each run folder contains `sankofa_results.csv`, `sankofa_summary_composition.csv`, `queries.json`, `run_metadata.json`, and `run.log`.
