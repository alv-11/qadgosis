
import pandas as pd
from src.summary import process_snapshots, analyze_graph_properties, process_summary_sankofa,  process_summary_baselines, prepare_queries_and_splits, load_single_snapshot, process_summary_sankofa_streaming, BaselineEngine
import src.core.sankofa  as walks
import sys
from pathlib import Path
import json
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")




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
            summary_size = int(len(graph.edges) * b) 
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
    #     methods=["glimpse", "apex", "isummary"]
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

    subgraph_dir = "/home/ama/wiki_new/wiki_50"#/Users/au735195/Pictures/wiki2017/experiments/snapshots"#"/Users/au735195/Desktop/wiki_original/snapshots_all"

    print("Preparing queries...")
    training_data, test_data, queries_, snapshot_files = \
        prepare_queries_and_splits(subgraph_dir)

    total_timesteps = len(snapshot_files)

    converted_queries = {
        int(timestep): {int(i): int(query) for i, query in enumerate(query_list)}
        for timestep, query_list in queries_.items()
    }

    previous_graph = None
    results = None
    summary_size = None
    K = None

    print("Streaming snapshots into C++...")

    # ==========================================================
    # PHASE 1 — STREAM INTO C++ (INCREMENTAL SANKOFA ENGINE)
    # ==========================================================

    for t, file_name in enumerate(snapshot_files):


        print(f"Loading {file_name}")

        KG = load_single_snapshot(subgraph_dir, file_name)
        G = analyze_graph_properties(KG, t)

        if t == 0:
            b = 0.005
            summary_size = int(len(G.edges) * b)
            k1 = (len(G.nodes))**2 / len(G.edges)
            K = int(k1 / (0.1 / b))

            print(f"Summary triple size: {summary_size}")

        current_graph = {
            int(node): {
                int(neighbor): float(G[node][neighbor].get("weight", 0.0001))
                for neighbor in G.neighbors(node)
            }
            for node in G.nodes
        }

        del KG
        del G

        if previous_graph is None:
            previous_graph = current_graph
            continue

        
        print(f"Processing timestep {t} in C++...")

        results = walks.process_one_timestep(
            num_walks=5,
            current_graph=previous_graph,
            next_graph=current_graph,
            queries=converted_queries,
            c=2,
            epsilon=0.25,
            k=K,
            decay_rate=0.03,
            t=t-1,
            has_next=True,
            window_size=5
        )

        del previous_graph
        previous_graph = current_graph

    # Final timestep
    results = walks.process_one_timestep(
        num_walks=5,
        current_graph=previous_graph,
        next_graph=previous_graph,
        queries=converted_queries,
        c=2,
        epsilon=0.25,
        k=K,
        decay_rate=0.03,
        t=total_timesteps - 1,
        has_next=False,
        window_size=5
    )

    print("Finished C++ streaming.")

    # Extract Sankofa outputs
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
    runtime_per_timestep = results["runtime_per_timestep"]
    #print("Walk runtimes per timestep:", runtime_per_timestep)
    non_topk_global_pr = results["global_pr"]
    changes_in_topk = results["changes_over_time"]


    with open("queries.json", "w") as f:
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

    runtime_df = pd.DataFrame(
        [{"Timestep": t, "Runtime (walks)": runtime}
         for t, runtime in runtime_per_timestep.items()]
    ).sort_values(by="Timestep")

    # ==========================================================
    # PHASE 2 — SINGLE STREAMING EVALUATION
    # ==========================================================

    print("Evaluating Sankofa + Baselines...")

    sankofa_rows = []
    triples_no = {}

    baseline_engine = BaselineEngine(
        methods=[]#["glimpse"]#, "apex", "isummary"]
    )

    for t, file_name in enumerate(snapshot_files):

        print(f"Evaluating {file_name}")

        KG = load_single_snapshot(subgraph_dir, file_name)

        if t not in top_k_cumsum_walks:
            del KG
            continue

        # ------------------- SANKOFA -------------------
        triples_no_t, sankofa_df = process_summary_sankofa_streaming(
            top_k_nodes_dict={t: top_k_cumsum_walks[t]},
            snapshot_files=snapshot_files,
            subgraph_dir=subgraph_dir,
            label="Cumsum Walks",
            weights={t: weights[t]},
            global_pr_dict={t: non_topk_global_pr[t]},
            test_data={t: test_data[t]},
            walks_runtime=runtime_df[runtime_df["Timestep"] == t],
            max_triples=summary_size
        )


        triples_no[t] = triples_no_t[t]
        sankofa_rows.append(sankofa_df)

        # ------------------- BASELINES -------------------

        baseline_engine.process_timestep(
            KG=KG,
            t=t,
            K=triples_no[t],
            training_data=training_data,
            test_data=test_data,
            weights=weights[t]
        )

        del KG

    # ==========================================================
    # FINAL MERGE
    # ==========================================================
    
    sankofa = pd.concat(sankofa_rows, ignore_index=True)
    sankofa.to_csv("sankofa_results.csv", index=False)
    with open("changes_in_topk.json", "w") as f:
        json.dump(changes_in_topk, f, indent=4)
    df_changes.to_csv("changes_in_topk.csv", index=False)

    sota = baseline_engine.finalize()

    if not sota.empty:
        merged_df = pd.merge(sankofa, sota, on="Timestep", how="inner")
        merged_df.to_csv("all_results.csv", index=False)
    else:
        print("No baseline results produced.")
        merged_df = sankofa

    print("Done.")

if __name__ == "__main__":
    main()
