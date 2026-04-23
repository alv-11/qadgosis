
import pandas as pd
import time
import os
from src.summary import process_snapshots, analyze_graph_properties, process_summary_sankofa,  process_summary_baselines, prepare_queries_and_splits, evaluate_sankofa_timestep, BaselineEngine
from src.progress import format_seconds, progress, progress_write
from src.sankofa_delta import ensure_sankofa_delta_cache, graph_edge_count, graph_node_count, initialize_streamed_evaluation_graph, load_delta, load_initial_graph, load_triple_delta, apply_triple_delta_to_knowledge_graph
import src.core.sankofa  as walks
import sys
from pathlib import Path
import json
from datetime import datetime, timezone

run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
query_bundle_index = int(sys.argv[2]) if len(sys.argv) > 2 else None

MIN_SUMMARY_SIZE = 4
DEFAULT_SUBGRAPH_DIR = "/mnt/c/Users/kaspe/OneDrive/Skrivebord/Reps/qadgosis/kg_summarization/data/test_750"
DEFAULT_NUM_WALKS = 5
DEFAULT_C = 2
DEFAULT_EPSILON = 0.25
DEFAULT_DECAY_RATE = 0.03
DEFAULT_WINDOW_SIZE = 5
DEFAULT_BUDGET_RATIO = 0.005
DEFAULT_BASELINE_METHODS = ["hhop_random", "hhop_random_outgoing", "ppr_frequency"]


def compute_summary_size(edge_count, budget_ratio, floor=MIN_SUMMARY_SIZE):
    return max(floor, int(edge_count * budget_ratio))


def resolve_subgraph_dir():
    return os.environ.get("QADGOSIS_SUBGRAPH_DIR", DEFAULT_SUBGRAPH_DIR)


def output_path(filename):
    return run_dir / filename


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def live_progress_enabled():
    return os.environ.get("QADGOSIS_LIVE_RUN_PROGRESS") == "1"




def main_():
    dynagraph = {} 
    subgraph_dir ="data/dbp"
 
    print("Loading all subgraph snapshots...")
    graphs, training_data, test_data, queries_ = process_snapshots(subgraph_dir)

    # Analyze graph properties
    for i, graph in graphs.items():
        G = analyze_graph_properties(graph, i)
        dynagraph[i] = G  

    # Convert graphs and queries for processing
    converted_dynagraph = []
    for i, (timestep, graph) in enumerate(dynagraph.items()):
        if i == 0:
            b = 0.01
            summary_size = compute_summary_size(len(graph.edges), b)
            k1 = (len(graph.nodes))**2 / len(graph.edges)
            K = int(k1 / (0.1/b)) 
            print(f"Summary triple size: {summary_size}")
            #print(density := len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1)))

        # Convert the graph to the required format
        converted_graph = {
            int(node): {int(neighbor): float(graph[node][neighbor].get("weight", 0.0001)) 
                        for neighbor in graph.neighbors(node)}
            for node in graph.nodes
        }
        converted_dynagraph.append(converted_graph)


    converted_queries = {
        int(timestep): {int(i): int(query) for i, query in enumerate(query_list)}
        for timestep, query_list in queries_.items()
    }
    
    print("Summarizing for SANKOFA")
    # Generate top-k summaries
    # results = walks.generate_top_k_summary(
    #     num_timesteps=len(converted_dynagraph),
    #     num_walks=5,
    #     dynamic_graph=converted_dynagraph,
    #     queries=converted_queries,
    #     c=2,
    #     epsilon=0.25,
    #     k=K,
    #     decay_rate = 0.8
    # )
    results = None
    total_timesteps = len(converted_dynagraph)

    for t in range(len(converted_dynagraph)):

        current_graph = converted_dynagraph[t]

        if t + 1 < len(converted_dynagraph):
            next_graph = converted_dynagraph[t+1]
            has_next = True
        else:
            next_graph = current_graph
            has_next = False

        results = walks.process_one_timestep(
            num_walks=5,
            current_graph=current_graph,
            next_graph=next_graph,
            queries=converted_queries,
            c=2,
            epsilon=0.25,
            k=K,
            decay_rate=0.8,
            t=t,
            has_next=has_next,
            window_size=5
        )
    # Process summaries and evaluate
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
   
    runtime_per_timestep = results["runtime_per_timestep"]
    # for timestep, runtime in runtime_per_timestep.items():
    #     print(f"Timestep {timestep}: Runtime for walks = {runtime:.4f} seconds")
    non_topk_global_pr =  results["global_pr"]

    runtime_df = pd.DataFrame(
        [{"Timestep": t, "Runtime (walks)": runtime} for t, runtime in runtime_per_timestep.items()]
    ).sort_values(by="Timestep")


    triples_no, sankofa = process_summary_sankofa(
        top_k_cumsum_walks,  graphs, label="Cumsum Walks", weights=weights, global_pr_dict=non_topk_global_pr,
        test_data=test_data, walks_runtime=runtime_df, max_triples=summary_size
    )
    sankofa.to_csv("sankofa_results.csv", index=False)
   


    # sota = process_summary_baselines(
    #     graphs, triples_no, training_data, test_data,
    #     methods=["glimpse", "apex", "isummary", "hhop_random", "hhop_random_outgoing", "ppr_frequency"]
    #     , weights=weights
    # )
    # merged_df = pd.merge(sankofa, sota, on="Timestep")
    # merged_df.to_csv("all_results.csv", index=False)


# def _main_():
#     subgraph_dir = "data/wiki"

#     print("Preparing queries...")
#     training_data, test_data, queries_, snapshot_files = \
#         prepare_queries_and_splits(subgraph_dir)

#     total_timesteps = len(snapshot_files)

#     converted_queries = {
#         int(timestep): {int(i): int(query) for i, query in enumerate(query_list)}
#         for timestep, query_list in queries_.items()
#     }

#     previous_graph = None
#     results = None
#     summary_size = None
#     K = None

#     print("Streaming snapshots into C++...")

#     for t, file_name in enumerate(snapshot_files):


#         print(f"Loading {file_name}")

#         KG = load_single_snapshot(subgraph_dir, file_name)
#         G = analyze_graph_properties(KG, t)

#         # Compute summary parameters once
#         if t == 0:
#             b = 0.05
#             summary_size = int(len(G.edges) * b)
#             k1 = (len(G.nodes))**2 / len(G.edges)
#             K = int(k1 / (0.1/b))

#             print(f"Summary triple size: {summary_size}")
#             # print("density:",
#             #       len(G.edges) / (len(G.nodes) * (len(G.nodes) - 1)))

#         # Convert graph to C++ format
#         current_graph = {
#             int(node): {
#                 int(neighbor): float(G[node][neighbor].get("weight", 0.0001))
#                 for neighbor in G.neighbors(node)
#             }
#             for node in G.nodes
#         }

#         # Free heavy KG immediately
#         del KG
#         del G

#         if previous_graph is None:
#             previous_graph = current_graph
#             continue

#         results = walks.process_one_timestep(
#             num_walks=5,
#             current_graph=previous_graph,
#             next_graph=current_graph,
#             queries=converted_queries,
#             c=2,
#             epsilon=0.25,
#             k=K,
#             decay_rate=0.03,
#             t=t-1,
#             has_next=True,
#             total_timesteps=total_timesteps
#         )

#         del previous_graph
#         previous_graph = current_graph

#     # Final timestep
#     results = walks.process_one_timestep(
#         num_walks=5,
#         current_graph=previous_graph,
#         next_graph=previous_graph,
#         queries=converted_queries,
#         c=2,
#         epsilon=0.25,
#         k=K,
#         decay_rate=0.03,
#         t=total_timesteps-1,
#         has_next=False,
#         total_timesteps=total_timesteps
#     )

#     print("Processing final summaries...")

#     top_k_cumsum_walks = results["top_k_cumsum_walks"]
#     weights = results["all_weights"]
#     runtime_per_timestep = results["runtime_per_timestep"]
#     non_topk_global_pr = results["global_pr"]
#     for timestep, runtime in runtime_per_timestep.items():
#         print(f"Timestep {timestep}: Runtime for walks = {runtime:.4f} seconds")

#     runtime_df = pd.DataFrame(
#         [{"Timestep": t, "Runtime (walks)": runtime}
#          for t, runtime in runtime_per_timestep.items()]
#     ).sort_values(by="Timestep")

#     # triples_no, sankofa = process_summary_sankofa(
#     #     top_k_cumsum_walks,
#     #     None,  # no full graphs stored
#     #     label="Cumsum Walks",
#     #     weights=weights,
#     #     global_pr_dict=non_topk_global_pr,
#     #     test_data=test_data,
#     #     walks_runtime=runtime_df,
#     #     max_triples=summary_size
#     # )

#     triples_no, sankofa = process_summary_sankofa_streaming(
#         top_k_cumsum_walks,
#         snapshot_files,
#         subgraph_dir,
#         label="Cumsum Walks",
#         weights=weights,
#         global_pr_dict=non_topk_global_pr,
#         test_data=test_data,
#         walks_runtime=runtime_df,
#         max_triples=summary_size
#     )

#     sankofa.to_csv("sankofa_results.csv", index=False)

#     print("Done.")

def build_changes_dataframe(changes_over_time, queries_):
    rows = []

    # Convert seeds to sets (per timestep)
    # seeds_per_timestep = {
    #     int(t): set(map(int, nodes))
    #     for t, nodes in queries_.items()
    # }

    seeds_per_timestep = {
    int(t): {int(x) for x in nodes if x is not None}
    for t, nodes in queries_.items()
    }

    for t, changes in changes_over_time.items():
        t = int(t)

        seeds_t = seeds_per_timestep.get(t, set())
        seeds_next = seeds_per_timestep.get(t+1, set())

        seeds = seeds_t.union(seeds_next)
        #seeds = seeds_per_timestep.get(t, set())

        nodes_added = changes.get("nodes_added", [])
        nodes_removed = changes.get("nodes_removed", [])
        edges_added = changes.get("edges_added", [])
        edges_removed = changes.get("edges_removed", [])

        # ------------------------------
        # Overlap computations
        # ------------------------------
        overlap_nodes_added = [n for n in nodes_added if n in seeds]
        overlap_nodes_removed = [n for n in nodes_removed if n in seeds]

        affected_nodes_added = set()
        affected_nodes_removed = set()

        # From edges added
        for (u, v) in edges_added:
            affected_nodes_added.add(u)
            affected_nodes_added.add(v)

        # From edges removed
        for (u, v) in edges_removed:
            affected_nodes_removed.add(u)
            affected_nodes_removed.add(v)

        # Include explicitly added/removed nodes too
        affected_nodes_added.update(nodes_added)
        affected_nodes_removed.update(nodes_removed)

        overlap_nodes_added = [
            n for n in affected_nodes_added if n in seeds
        ]

        overlap_nodes_removed = [
            n for n in affected_nodes_removed if n in seeds
        ]

        overlap_edges_added = [
            (u, v) for (u, v) in edges_added
            if u in seeds or v in seeds
        ]

        overlap_edges_removed = [
            (u, v) for (u, v) in edges_removed
            if u in seeds or v in seeds
        ]

        # ------------------------------
        # Store everything in ONE row
        # ------------------------------
        row = {
            "timestep": t,

            # GLOBAL
            "nodes_added": len(nodes_added),
            "nodes_removed": len(nodes_removed),
            "edges_added": len(edges_added),
            "edges_removed": len(edges_removed),

            # QUERY-AWARE ⭐
            "overlap_nodes_added": len(overlap_nodes_added),
            "overlap_nodes_removed": len(overlap_nodes_removed),
            "overlap_edges_added": len(overlap_edges_added),
            "overlap_edges_removed": len(overlap_edges_removed),
        }

        # ------------------------------
        # Optional ratios (VERY useful)
        # ------------------------------
        row["node_overlap_ratio_added"] = (
            len(overlap_nodes_added) / len(nodes_added)
            if nodes_added else 0
        )

        row["node_overlap_ratio_removed"] = (
            len(overlap_nodes_removed) / len(nodes_removed)
            if nodes_removed else 0
        )

        row["edge_overlap_ratio_added"] = (
            len(overlap_edges_added) / len(edges_added)
            if edges_added else 0
        )

        row["edge_overlap_ratio_removed"] = (
            len(overlap_edges_removed) / len(edges_removed)
            if edges_removed else 0
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="timestep")


def main():
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    subgraph_dir = resolve_subgraph_dir()

    progress_write("Preparing queries...")
    training_data, test_data, queries_, snapshot_files = \
        prepare_queries_and_splits(subgraph_dir, query_bundle_index=query_bundle_index)

    total_timesteps = len(snapshot_files)

    progress_write(f"Preparing Sankofa cache for {subgraph_dir}...")
    cache = ensure_sankofa_delta_cache(subgraph_dir)
    if list(cache.snapshot_files) != list(snapshot_files):
        raise ValueError("Sankofa cache snapshot order does not match query preparation order.")
    progress_write(f"Sankofa cache ready at {cache.cache_dir}.")

    progress_write("Encoding query seeds for C++ streaming...")
    converted_queries = {
        int(timestep): {int(i): int(query) for i, query in enumerate(query_list)}
        for timestep, query_list in queries_.items()
    }
    progress_write(f"Prepared query map for {len(converted_queries)} timesteps.")

    initial_graph_started = time.perf_counter()
    progress_write(f"Loading initial walk graph from {cache.initial_graph_path}...")
    initial_graph = load_initial_graph(cache)
    results = None

    initial_edge_count = graph_edge_count(initial_graph)
    initial_node_count = graph_node_count(initial_graph)
    if initial_edge_count == 0:
        raise ValueError("Initial Sankofa graph has no edges.")
    progress_write(
        f"Initial walk graph ready with {initial_node_count:,} nodes and {initial_edge_count:,} edges "
        f"in {format_seconds(time.perf_counter() - initial_graph_started)}."
    )

    b = DEFAULT_BUDGET_RATIO
    summary_size = compute_summary_size(initial_edge_count, b)
    k1 = (initial_node_count ** 2) / initial_edge_count
    K = int(k1 / (0.1 / b))

    progress_write(f"Summary triple size: {summary_size}")

    progress_write("Initializing C++ Sankofa stream state...")
    walks.reset_stream_state()
    walks.initialize_stream_graph(initial_graph)
    progress_write("C++ Sankofa stream state initialized.")

    sankofa_rows = []
    triples_no = {}
    baseline_methods = list(DEFAULT_BASELINE_METHODS)
    baseline_engine = BaselineEngine(
        methods=baseline_methods  # e.g. ["glimpse", "apex", "isummary", "hhop_random", "hhop_random_outgoing", "ppr_frequency"]
    )
    progress_write("Loading evaluation graph for timestep 0...")
    evaluation_graph = initialize_streamed_evaluation_graph(
        cache,
        progress_label=f"{run_dir.name}: evaluation graph",
    )
    progress_write("Evaluation graph initialization complete.")
    incoming_triple_delta = None

    progress_write("Streaming snapshots into C++ and evaluating each timestep...")

    # ==========================================================
    # PHASE 1 — STREAM INTO C++ (INCREMENTAL SANKOFA ENGINE)
    # ==========================================================

    stream_steps = progress(
        range(total_timesteps),
        desc="Streaming Sankofa timesteps",
        unit="timestep",
    )
    for t in stream_steps:
        has_next = t < total_timesteps - 1
        step_label = f"{run_dir.name}: timestep {t + 1}/{total_timesteps}"

        progress_write(f"{step_label}: running Sankofa stream update...")

        results = walks.process_one_timestep_delta(
            num_walks=DEFAULT_NUM_WALKS,
            raw_delta=load_delta(cache, t) if has_next else None,
            queries=converted_queries,
            c=DEFAULT_C,
            epsilon=DEFAULT_EPSILON,
            k=K,
            decay_rate=DEFAULT_DECAY_RATE,
            t=t,
            has_next=has_next,
            window_size=DEFAULT_WINDOW_SIZE
        )
        step_runtime = results.get("runtime_per_timestep", {}).get(t)
        top_k_nodes = results.get("top_k_cumsum_walks", {}).get(t, {})
        top_k_count = len(top_k_nodes)

        progress_write(
            f"{step_label}: Sankofa walks ready "
            f"topk={top_k_count} walk={format_seconds(step_runtime)}; evaluating Sankofa summary..."
        )

        sankofa_triples, sankofa_df = evaluate_sankofa_timestep(
            t,
            evaluation_graph,
            top_k_nodes,
            test_data,
            weights=results.get("all_weights", {}).get(t, {}),
            global_pr=results.get("global_pr", {}).get(t, {}),
            max_triples=summary_size,
            walk_runtime=step_runtime,
        )

        triples_no[t] = sankofa_triples
        sankofa_rows.append(sankofa_df)

        progress_write(
            f"{step_label}: Sankofa summary complete triples={sankofa_triples}; moving to baselines..."
        )

        baseline_engine.process_timestep(
            KG=evaluation_graph,
            t=t,
            K=sankofa_triples,
            training_data=training_data,
            test_data=test_data,
            weights=results.get("all_weights", {}).get(t, {}),
            graph_delta=incoming_triple_delta,
            progress_prefix=step_label,
        )

        if has_next:
            evaluation_delta_started = time.perf_counter()
            outgoing_triple_delta = load_triple_delta(cache, t)
            removed_count = len(outgoing_triple_delta.get("triples_removed", []))
            added_count = len(outgoing_triple_delta.get("triples_added", []))
            changed = apply_triple_delta_to_knowledge_graph(evaluation_graph, outgoing_triple_delta)
            progress_write(
                f"{run_dir.name}: evaluation graph advanced {t}->{t + 1} "
                f"removed={removed_count} added={added_count} changed={changed} "
                f"triples={evaluation_graph.number_of_triples():,} "
                f"dt={format_seconds(time.perf_counter() - evaluation_delta_started)}"
            )
            incoming_triple_delta = outgoing_triple_delta

        stream_steps.set_postfix(
            step=f"{t}->{t + 1}" if has_next else f"{t} final",
            walk=format_seconds(step_runtime),
            topk=top_k_count,
            triples=sankofa_triples,
        )

        if live_progress_enabled():
            progress_write(
                f"{step_label} "
                f"walk={format_seconds(step_runtime)} topk={top_k_count} triples={sankofa_triples}"
            )

    progress_write("Finished C++ streaming and per-timestep evaluation.")

    # Extract Sankofa outputs
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
    runtime_per_timestep = results["runtime_per_timestep"]
    #print("Walk runtimes per timestep:", runtime_per_timestep)
    non_topk_global_pr = results["global_pr"]
    changes_in_topk = results["changes_over_time"]


    with output_path("queries.json").open("w", encoding="utf-8") as f:
        json.dump(queries_, f, indent=4)
    df_changes = build_changes_dataframe(changes_in_topk, queries_)

    # count_changes = []

    # for timestep, changes in changes_in_topk.items():
    #     count_changes.append({
    #         "timestep": int(timestep),
    #         "nodes_added": len(changes.get("nodes_added", [])),
    #         "nodes_removed": len(changes.get("nodes_removed", [])),
    #         "edges_added": len(changes.get("edges_added", [])),
    #         "edges_removed": len(changes.get("edges_removed", [])),
    #     })



    #df_changes = pd.DataFrame(count_changes).sort_values(by="timestep")

    sankofa = pd.concat(sankofa_rows, ignore_index=True)
    sankofa.to_csv(output_path("sankofa_results.csv"), index=False)
    with output_path("changes_in_topk.json").open("w", encoding="utf-8") as f:
        json.dump(changes_in_topk, f, indent=4)
    df_changes.to_csv(output_path("changes_in_topk.csv"), index=False)

    sota = baseline_engine.finalize()

    if not sota.empty:
        merged_df = pd.merge(sankofa, sota, on="Timestep", how="inner")
        merged_df.to_csv(output_path("all_results.csv"), index=False)
    else:
        progress_write("No baseline results produced.")
        merged_df = sankofa

    with output_path("run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "run_dir": str(run_dir),
            "subgraph_dir": str(subgraph_dir),
            "query_bundle_index": query_bundle_index,
            "snapshot_count": total_timesteps,
            "initial_node_count": initial_node_count,
            "initial_edge_count": initial_edge_count,
            "summary_size": summary_size,
            "settings": {
                "num_walks": DEFAULT_NUM_WALKS,
                "c": DEFAULT_C,
                "epsilon": DEFAULT_EPSILON,
                "k": K,
                "decay_rate": DEFAULT_DECAY_RATE,
                "window_size": DEFAULT_WINDOW_SIZE,
                "budget_ratio": b,
                "baseline_methods": baseline_methods,
            },
            "outputs": {
                "queries": str(output_path("queries.json")),
                "sankofa_results": str(output_path("sankofa_results.csv")),
                "changes_in_topk_json": str(output_path("changes_in_topk.json")),
                "changes_in_topk_csv": str(output_path("changes_in_topk.csv")),
                "all_results": str(output_path("all_results.csv")) if not sota.empty else None,
            },
            "cache_dir": str(cache.cache_dir),
        }, f, indent=2)

    progress_write("Done.")

if __name__ == "__main__":
    main()
