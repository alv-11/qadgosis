# NELL experiments with corrected SANKOFA walks

These eight runs use the corrected per-snapshot walk-bank implementation. The
previous experiment folders were left intact. Each run folder contains its
`sankofa_results.csv`, `sankofa_summary_composition.csv`, `queries.json`,
`run_metadata.json`, and captured `run.log`.

## Shared settings

- Same three synthetic NELL snapshots as the earlier experiment; the graph
  changes are synthetic perturbations, not historical NELL versions.
- Random seeds: `PYTHONHASHSEED=42`, `QADGOSIS_RANDOM_SEED=42`.
- SANKOFA: 20 walks per entity, `c=2`, `epsilon=0.90` (`beta=0.10`), decay
  rate `0.8`, window size `100`.
- Query logs use the same saved bundle in each baseline/embedding pair. The
  `queries.json` SHA-256 hashes match within all four pairs.
- Embedding runs use RDF2Vec weight `1.0` and one Word2Vec worker. Baseline
  means SANKOFA without RDF2Vec; no separate APEX or other method was enabled.
- The two-hop workload requested 1,000 queries and reused the saved generated
  bundle, which contains 952, 900, and 915 records across snapshots 0, 1, and 2.

## Mean results

`Future Test F1` and `Future Test Macro F1` are averaged over the three
snapshots. `Past Unseen Query F1` is averaged over snapshots 1 and 2 because
snapshot 0 has no past queries.

| Workload | Future micro F1, baseline / embeddings | Future macro F1, baseline / embeddings | Past F1, baseline / embeddings |
| --- | ---: | ---: | ---: |
| One-hop, 1%, 1,000 queries | 0.9103 / 0.9054 | 0.9428 / 0.9416 | 0.9023 / 0.8942 |
| One-hop, 5%, 1,000 queries | 1.0000 / 0.9428 | 1.0000 / 0.9720 | 0.9345 / 0.9176 |
| Two-hop, 1%, 1,000 requested | 0.0668 / 0.0429 | 0.1008 / 0.1610 | 0.0325 / 0.0249 |
| One-hop, 5%, 3,000 queries | 1.0000 / 0.8186 | 1.0000 / 0.7511 | 0.9970 / 0.7985 |

The embedding-only ranking did not improve the mean micro F1 in these runs.
For the two-hop workload, it raised mean macro F1 while lowering mean micro F1;
the CSVs contain the per-snapshot scores for closer inspection.

## Run folders

- `baseline_1pct/` and `embeddings_1pct/`
- `baseline_5pct/` and `embeddings_5pct/`
- `baseline_2hop_1pct/` and `embeddings_2hop_1pct/`
- `baseline_5pct_3000q/` and `embeddings_5pct_3000q/`

These are one fixed-seed run on synthetic snapshots and generated queries.
They are useful for a controlled baseline/embedding comparison, but do not
represent natural historical graph evolution or real user query logs.
