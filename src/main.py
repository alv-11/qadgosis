import networkx as nx
import random
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import json
import numpy as np
import dynetx as dn
import copy
import time
from collections import Counter
import logging
from queries import generate_query, synthetic_graph
from walks import generate_random_walks_directed_with_weights, update_random_walks_directed_with_weights,  personalized_complete_pagerank_walk
from sklearn.metrics import ndcg_score
import csv
import os



def graph_and_queries(num_timesteps, n_nodes , m_edges, num_queries):  
    """
    param num_timesteps: int, number of timesteps
    param n_nodes: int, number of nodes
    param m_edges: int, number of edges
    param num_queries: int, number of queries
    """  
    #create dynamic graph or import dynamic graph
    dynamic_graph = synthetic_graph(num_timesteps , n_nodes , m_edges )

    # Generate for query each timestep of the dynamic graph
    queries = dict()
    #fixed queries
    G_t = dynamic_graph[0]
        # Convert NetworkX graph to the expected dictionary format
    graph_dict = {
        node: {nbr: data.get('weight', 0) for nbr, data in G_t[node].items()}
        for node in G_t.nodes()
    }
    timestep_query = dict()
    for i in range(num_queries):
        query = generate_query(graph_dict, path_len=2, qid=i, constraint_index=None)
        timestep_query[i] = query
    for timestep in range(len(dynamic_graph)): 
        queries[timestep] = timestep_query
    # for timestep in range(len(dynamic_graph)): 
    #     G_t = dynamic_graph[timestep]
    #         # Convert NetworkX graph to the expected dictionary format
    #     graph_dict = {
    #     node: {nbr: data.get('weight', 0) for nbr, data in G_t[node].items()}
    #     for node in G_t.nodes()
    #     }

    #     timestep_query = dict()
    #     for i in range(num_queries):
    #         query = generate_query(graph_dict, path_len=2, qid=i, constraint_index=None)
    #         timestep_query[i] = query
    #     queries[timestep] = timestep_query

    return dynamic_graph, queries



def graph_changes_with_dummy_node(dynamic_graph, queries, timestep):
    """
    Add dummy nodes and edges to the dynamic graph to represent queries.
    param dynamic_graph: Dictionary representation of the dynamic graph.
    param queries: Dictionary of queries for each timestep.
    return: Dictionary representation of the dynamic graph with queries, changes_BetweeenTimesteps and the starting node for PPR walk
    """
    start_time = time.time()
    # Get the time slice between two snapshots of the graph
    snapshot_s, w, personalization_nodes = personalized_graph(dynamic_graph, queries, timestep)
    snapshot_t,_ ,_ = personalized_graph(dynamic_graph, queries, timestep+1)
    end_time = time.time()
    #logging.info(f"Runtime for preprocessing of dummynode{timestep}, {end_time - start_time} seconds")


    # Get nodes and edges in each snapshot
    nodes_s = set(snapshot_s.nodes())  # Nodes in first snapshot
    nodes_t = set(snapshot_t.nodes())  # Nodes in next snapshot

    # Extract edge sets with weights
    edges_s = set(snapshot_s.edges())
    edges_t = set(snapshot_t.edges())

    # Compare nodes
    nodes_added = nodes_t - nodes_s  # Nodes added in next snapshot
    nodes_removed = nodes_s - nodes_t  # Nodes removed in snapshot

    # Compare edges
    edges_added = edges_t - edges_s  # Edges added in next snapshot
    edges_removed = edges_s - edges_t  # Edges removed in next snapshot
    
    changes = {
    'nodes_added': nodes_added,
    'nodes_removed': nodes_removed,
    'edges_added': edges_added,
    'edges_removed': edges_removed
    } 
    return snapshot_s, changes, w, personalization_nodes


def personalized_graph(dynamic_graph, queries, timestep):
    snapshot_s = copy.deepcopy(dynamic_graph[timestep])#make a copy of the graph for easy reference

    query_nodes = list(queries[timestep].values())
    w = f"dummy{timestep}" #dummy_node and start node
    equal_weight = 1 / len(query_nodes)  # Equal weights for all edges
    for node in query_nodes:
        snapshot_s.add_edge(w, node, weight=equal_weight)

    personalization = dict()      #personalization vector for the power method
    for node in query_nodes:
        personalization[node] = 1 / len(query_nodes)  # Equal probability for personalization nodes

    # Scale other edges weights to maintain importance
    for u, v, data in snapshot_s.edges(data=True):
        if u != w:
            data['weight'] = data.get('weight', 1) / 10  

    return snapshot_s, w, personalization


def write_to_file(data, prefix = "data", file_extension="json", output_directory="./output/"):
    file_name = f"{output_directory}{prefix}.{file_extension}"
    with open(file_name, 'w') as f:
        json.dump(data, f, indent = 4)
        print(f"Data saved to {file_name}")


def ndcg_metric(top_k_power, top_k_cumsum_walks):
    ndcg_values = dict()
    shared_keys = set(top_k_power.keys()).intersection(set(top_k_cumsum_walks.keys()))
    for key in shared_keys:
        # groundtruth/power iteration ranking
        power_ranking = list(top_k_power[key].keys()) 
        power_ranking = [float(x) for x in power_ranking]

        # monte carlo ranking
        monte_carlo_ranking = list(top_k_cumsum_walks[key].keys()) 
        monte_carlo_ranking = [float(x) for x in monte_carlo_ranking]

        #if len(monte_carlo_ranking)==len(power_ranking):
        if monte_carlo_ranking:
            def compute_relevance(true_ranking, predicted_ranking):
                relevance = []
                for node in predicted_ranking:
                    if node in true_ranking:
                        relevance.append(1)
                    else:
                        relevance.append(0)
                return relevance

            # Compute true relevance based on the groundtruth
            power_ranking = compute_relevance(monte_carlo_ranking, power_ranking)

            dcg = [power_ranking] #actual dcg

            idcg = [1] * len(power_ranking)  # ideal DCG

            ndcg = ndcg_score(dcg, [idcg]) 	# Compute NDCG
            ndcg_values[key] = ndcg

            
    return ndcg_values


def generate_top_k_summary(num_timesteps = 10, n_nodes=1000, m_edges = 5, num_queries = 20, num_walks = 10 ):
    """convert some part of these to functions"""
    # Initialize a dictionary to store the changes for each timestep
    changes_over_time = dict() #dictionary to store the changes
    random_walks = dict() #dictionary to store the random walks for each node 
    edge_to_walks = dict() #dictionary to store the edges to walk_ids 
    node_to_walks = dict() #dictionary to store the nodes to walk_ids
    walks_to_ids = dict() #dictionary to store the walks to ids 
    startnode_to_ids = dict() #dictionary to store the random walk ids for each node
    ppr_walks = dict() #dictionary to store the complete ppr 
    ppr_cumsum_walks = dict() #dictionary to store the cumulative sum of the ppr walks
    top_k_cumsum_walks = dict()#dictionary to store the top k walks 
    top_k_power = dict() #dictionary to store the top k for the power method
    top_k_power_nx = dict() #dictionary to store the top k for the power method using networkx

    counters = dict() #dictionary to store the counters for each node 
    cumulative_counts = Counter()#counter to store the cumulative counts of the ppr walks 

    # Generate the dynamic graph and queries
    dynamic_graph, queries = graph_and_queries(num_timesteps , n_nodes, m_edges, num_queries)

    for timestep in range(len(dynamic_graph) - 1): 
        print("Timestep", timestep)

        #time_start = time.time()
        dyngraph, changes, w, personalization_nodes = graph_changes_with_dummy_node(dynamic_graph, queries, timestep)   
        #time_end = time.time()

        #print("Time taken for preprocessing", time_end - time_start)

        changes_over_time[timestep] = changes
        n = len(dyngraph.nodes())
        k = 100 #fixed summary size
        #k = int(0.1*n) #summary size
        L = (c/epsilon) *k *(n/k)**epsilon #required length of PPR walk
        logging.info(f"Required length of PPR walk {timestep}: {L}") 

        print(L, "Required length of PPR walk") 

        starttime = time.time()
        count_start = time.time()
        if timestep == 0:       
            precomputed_walks, inverted_edges, inverted_nodes, walk_ids, random_walk_ids = generate_random_walks_directed_with_weights(dyngraph, num_walks, walk_length=10)
            random_walks[timestep] = precomputed_walks
            edge_to_walks[timestep] = inverted_edges
            node_to_walks[timestep] = inverted_nodes
            walks_to_ids[timestep] = walk_ids
            startnode_to_ids[timestep] = random_walk_ids
            R = copy.deepcopy(precomputed_walks)

        else:
            previous_edge_walks = edge_to_walks[timestep - 1].copy()  # Get the walks from the previous timestep
            previous_node_walks = node_to_walks[timestep - 1].copy()  # Get the walks from the previous timestep
            previous_walk_ids = walks_to_ids[timestep - 1].copy()  # Get the walks from the previous timestep
            previous_startnode_ids = startnode_to_ids[timestep - 1].copy()  # Get the walks from the previous timestep


            updated_walks, updated_edge_walks, updated_node_walks, updated_walks_to_ids, updated_startnode_ids = update_random_walks_directed_with_weights(dyngraph,  previous_walk_ids, 
                                                                                                            previous_edge_walks,previous_node_walks, previous_startnode_ids, 
                                                                                                            changes_over_time[timestep - 1], num_walks, walk_length=10)
            random_walks[timestep] = updated_walks  # Save updated walks for future reference
            edge_to_walks[timestep] = updated_edge_walks
            node_to_walks[timestep] = updated_node_walks
            walks_to_ids[timestep] = updated_walks_to_ids
            startnode_to_ids[timestep] = updated_startnode_ids
            R = copy.deepcopy(updated_walks)

        count_end = time.time()
        # Generate the directed graph for the graph in the current timestep
        directed_graph = {node: list(dyngraph.neighbors(node)) for node in dyngraph.nodes()}

        # Generate the PPR walks for the graph in the current timeste


        walk = personalized_complete_pagerank_walk(w, L, epsilon, R, directed_graph)
        ppr_walks[timestep] = walk
        logging.info(f"Actual length of PPR walk {timestep}: {len(walk)}") 

        print(len(walk), "Actual length of PPR walk")

        counts = Counter(walk)
        counters[timestep] = counts
        cumulative_counts += counts# Update cumulative counts
        ppr_cumsum_walks[timestep] = cumulative_counts.copy() # Store cumulative counts in summaries
        
        string_counter = Counter({str(key): count for key, count in cumulative_counts.items()})# Convert keys to strings
        filtered_counter = Counter({key: count for key, count in string_counter.items() if not key.startswith('dummy')})#filter out the dummy node
        
        # Normalize counts by dividing each by the sum of all counts
        #total_cumulative_walks = sum(filtered_counter.values())
        #if total_cumulative_walks > 0:
         #   normalized_cumulative_counts = Counter({
         #       key: value  / total_cumulative_walks for key, value in filtered_counter.items()
          #  })
        #else:
        #normalized_cumulative_counts = filtered_counter  # Handle edge case of 0 total walks
        normalized_cumulative_counts =  Counter({
                key: (value *epsilon) / (n*num_walks) for key, value in filtered_counter.items()
            })
      

        # Get the top-k items based on normalized probabilities
        top_k_cumulative = dict(normalized_cumulative_counts.most_common(k))
        top_k_cumsum_walks[timestep] = top_k_cumulative
        endtime = time.time()

        logging.info(f"Runtime for MC method timestep {timestep}: {endtime - starttime} seconds") 

        #Compute library pagerank for the current snapshot
        library_start = time.time()
        library_scores = nx.pagerank(dyngraph, alpha=0.80, personalization=personalization_nodes)
            # Select the top-k nodes
 
        top_k_pagerank = {
            str(key): value  
            for key, value in sorted(library_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            if not str(key).lower().startswith('dummy')  
        }

        top_k_power_nx[timestep]  = top_k_pagerank
        library_end = time.time()
        runtime = library_end - library_start

        logging.info(f"Runtime for networkx power method timestep {timestep}: {runtime} seconds")
        
    return changes_over_time, random_walks, ppr_walks, ppr_cumsum_walks, top_k_cumsum_walks, top_k_power_nx


if __name__=="__main__":

    c = 2 #constant for the PPR walk
    epsilon = 0.2  # Teleportation probability
    nodes = 10000
    num_walks = [50, 100, 200, 300]#,5 10, 50, 100]#,0.01 0.005, 0.001, 0.0005] #[0.1, 0.05, 0.01, 0.05]
    for per in num_walks:
        print(f"Running for {per} walkss")
        output_directory = f"./output/nodes{nodes}_{per}/"
        os.makedirs(output_directory, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(message)s', 
            filename=f"{output_directory}ndcg_{nodes}nodes_{per}runtime.log",
            #filename='./output/runtime_.log', 
            filemode='w'  # 'w' for overwrite, 'a' for append
        )

        changes_over_time, random_walks, ppr_walks, ppr_cumsum_walks, top_k_cumsum_walks, top_k_power_nx = generate_top_k_summary(n_nodes=nodes, num_walks = per)
        ndcg =  ndcg_metric(top_k_power_nx, top_k_cumsum_walks)
        file_name = f"{output_directory}ndcg_{nodes}nodes_{per}walks.csv"
        with open(file_name,'w') as f:
            w = csv.writer(f)
            w.writerows(ndcg.items())


        # Convert sets to lists in changes_over_time dictionary
        for key, value in changes_over_time.items():
            changes_over_time[key] = {
                "nodes_added": list(value["nodes_added"]),
                "nodes_removed": list(value["nodes_removed"]),
                "edges_added": list(value["edges_added"]),
                "edges_removed": list(value["edges_removed"]),
            }
        # Write the dictionaries to files
        #write_to_file(changes_over_time, "changes_over_time")
        #write_to_file(random_walks, "random_walks")    
        #write_to_file(ppr_walks, "ppr_walks")
        write_to_file(ppr_cumsum_walks, "ppr_cumsum_walks", output_directory= output_directory)
        write_to_file(top_k_cumsum_walks, "top_k_cumsum_walks", output_directory= output_directory)
        write_to_file(top_k_power_nx, "top_k_power_nx", output_directory=output_directory)





