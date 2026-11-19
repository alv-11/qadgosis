# NELL named-entity snapshot experiment

## Data

The source was `~/Downloads/nell/nell.txt` (231,634 facts, 46,682 entities,
821 predicates). The generated `.nt.gz` files retain NELL terms as written;
for example:

```text
<concept:academicfield:academic_affairs> <concept:academicprogramatuniversity> <concept:university:college> .
```

The graph updates are synthetic random perturbations of one static dump, not
NELL's historical changes:

| Snapshot | Triples | Change from prior snapshot |
| --- | ---: | --- |
| 0 | 185,307 | Initial 80% sample |
| 1 | 199,205 | +23,163 / −9,265 |
| 2 | 212,408 | +23,163 / −9,960 |

The input hash, random seed, and exact counts are in
`../../data/nell_named_synthetic_3snap/manifest.json`. The cache builds a
reversible string-to-integer index only for the existing C++ SANKOFA walker,
whose graph type accepts integers. The saved facts, Python evaluation graph,
queries, and RDF2Vec tokens retain NELL's names.

## Query workloads

Each workload enabled hard-query mode, used no random replacement queries,
shuffled its query log, and held out the final 20% for testing. The same saved
bundle was reused for the no-embedding and embedding runs.

| Workload | Requested per snapshot | Generated per snapshot | Held out per snapshot | Path length |
| --- | ---: | ---: | ---: | ---: |
| One-hop | 1,000 | 1,000 | 200 | 1 predicate |
| Larger one-hop | 3,000 | 3,000 | 600 | 1 predicate |
| Two-hop | 1,000 | 952 / 900 / 915 | 191 / 180 / 183 | 2 predicates |

The larger one-hop run used 50 topics per snapshot; the 1,000-query one-hop
and two-hop runs used 20. The two-hop generator omits attempts for which it
cannot construct a valid query, so its saved log has fewer than 1,000 records.
Some two-hop queries have hundreds of correct answers, which is why both
pooled and per-query F1 are reported below.

## Run controls

All runs used the same graph, query bundle, `PYTHONHASHSEED=42`, and
`QADGOSIS_RANDOM_SEED=42`. SANKOFA used 20 walks. Embedding runs used
`QADGOSIS_RDF2VEC_WEIGHT=1` and one Word2Vec worker for repeatability. Summary
budgets were 1% (1,454 triples) and 5% (7,272 triples), calculated from the
initial snapshot's directed entity-edge count and held constant over time.

`Future Test F1` is pooled across all answers in that timestep (micro F1).
`Future Test Macro F1` computes F1 per query, then gives every query equal
weight. The “mean” below is the simple mean of the three timestep scores.

## Results

| Workload / budget | Timestep | Test queries | Micro F1: no embeddings | Micro F1: embeddings | Macro F1: no embeddings | Macro F1: embeddings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One-hop / 1% | 0 | 200 | 1.0000 | 0.9926 | 1.0000 | 0.9983 |
| One-hop / 1% | 1 | 200 | 0.8523 | 0.8523 | 0.9390 | 0.9390 |
| One-hop / 1% | 2 | 200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| One-hop / 5% | 0 | 200 | 1.0000 | 0.9926 | 1.0000 | 0.9977 |
| One-hop / 5% | 1 | 200 | 0.9941 | 0.8911 | 0.9938 | 0.9494 |
| One-hop / 5% | 2 | 200 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Larger one-hop / 5% | 0 | 600 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Larger one-hop / 5% | 1 | 600 | 0.9931 | 0.9885 | 0.9886 | 0.9797 |
| Larger one-hop / 5% | 2 | 600 | 0.9341 | 0.9076 | 0.9848 | 0.9764 |
| Two-hop / 1% | 0 | 191 | 0.0969 | 0.0426 | 0.1864 | 0.1748 |
| Two-hop / 1% | 1 | 180 | 0.0400 | 0.0400 | 0.1728 | 0.1728 |
| Two-hop / 1% | 2 | 183 | 0.0561 | 0.0561 | 0.1593 | 0.1593 |

| Workload / budget | Mean micro F1: no embeddings | Mean micro F1: embeddings | Mean macro F1: no embeddings | Mean macro F1: embeddings |
| --- | ---: | ---: | ---: | ---: |
| One-hop / 1% | 0.9508 | 0.9483 | 0.9797 | 0.9791 |
| One-hop / 5% | 0.9980 | 0.9612 | 0.9979 | 0.9824 |
| Larger one-hop / 5% | 0.9757 | 0.9654 | 0.9911 | 0.9854 |
| Two-hop / 1% | 0.0643 | 0.0462 | 0.1728 | 0.1690 |

The larger one-hop comparison has 600 held-out queries per snapshot across 50
topics. Its mean micro F1 was 0.9757 without embeddings and 0.9654 with them;
mean macro F1 was 0.9911 and 0.9854, respectively. Embeddings therefore did not
improve F1 in this workload. The smaller 1% one-hop workload was nearly tied,
and the 5% one-hop and 1% two-hop runs also scored lower with embeddings. At
1%, timesteps 1 and 2 spent the entire summary budget on internal triples,
leaving no boundary or outside candidates for embeddings to rerank; the scores
are identical there for that reason.

Embedding training added about 5–6 seconds per timestep on this machine; the
no-embedding run took about 0.03–0.13 seconds per timestep. The summaries did
select different triples, but that did not recover more held-out answers here.

## Limits

- These are synthetic graph changes from one static NELL file, not a test on
  historical NELL snapshots.
- There is one generated query bundle and one fixed seed. Repeated bundles or
  real user query logs are needed to estimate how stable these results are.
- The queries are generated from the graph. They provide a controlled benchmark,
  but do not substitute for real query logs.
- The current RDF2Vec integration walks entity adjacency and leaves predicates
  out of the embedding walks, while answer evaluation is predicate-specific.
- No separate APEX or other summarization baseline was enabled; “no embeddings”
  means SANKOFA with its current PPR-based fallback selection.
- Gensim printed an ignored `our_dot_float` shutdown warning during embedding
  runs. Each process exited successfully and wrote its results.

## Result files

- `baseline_1pct/` and `embeddings_1pct/`
- `baseline_5pct/` and `embeddings_5pct/`
- `baseline_5pct_3000q/` and `embeddings_5pct_3000q/`
- `baseline_2hop_1pct/` and `embeddings_2hop_1pct/`

Each run directory contains `sankofa_results.csv`, including pooled and macro
test F1, plus `sankofa_summary_composition.csv`, `queries.json`, and
`run_metadata.json`.
