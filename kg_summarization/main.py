
import pandas as pd
from src.summary import process_snapshots, analyze_graph_properties, process_summary_sankofa, process_summary_baselines
import src.core.sankofa  as walks
import sys
from pathlib import Path
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")




def main():
    dynagraph = {} 
    subgraph_dir ="data/imdb"
 
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
            print(density := len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1)))

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
            total_timesteps=total_timesteps
        )
    # Process summaries and evaluate
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
   
    runtime_per_timestep = results["runtime_per_timestep"]
    # for timestep, runtime in runtime_per_timestep.items():
    #     print(f"Timestep {timestep}: Runtime for walks = {runtime:.4f} seconds")
    non_topk_global_pr =  results["global_pr"]#
    # changes = results["changes_over_time"] 

    # # Count the number of changes for each timestep
    # for timestep, change in changes.items():
    #     nodes_added_count = len(change.get("nodes_added", []))
    #     nodes_removed_count = len(change.get("nodes_removed", []))
    #     edges_added_count = len(change.get("edges_added", []))
    #     edges_removed_count = len(change.get("edges_removed", []))

    #     print(f"Timestep {timestep}:")
    #     print(f"  Nodes Added: {nodes_added_count}")
    #     print(f"  Nodes Removed: {nodes_removed_count}")
    #     print(f"  Edges Added: {edges_added_count}")
    #     print(f"  Edges Removed: {edges_removed_count}")
    #     print("-" * 40)
 
    # for i, change in enumerate(changes):
       
    #     print(f"Timestep {i}: {change:.4f} change in graph structure compared to previous timestep.")

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

if __name__ == "__main__":
    main()
