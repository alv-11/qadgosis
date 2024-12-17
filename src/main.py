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
from walks import generate_random_walks_directed_with_weights, update_random_walks_directed_with_weights, fetch_for_node, personalized_complete_pagerank_walk



c = 2 #constant for the PPR walk
epsilon = 0.2  # Reset probability

# Define files to store the outputs
#changes_file = "./output/changes_over_time.json"
##random_walks_file = "./output/random_walks.json"
#ppr_walks_file = "./output/ppr_walks.json"
#ppr_walks_cumsum_file = "./output/cumsum.json"
output_directory = "./output/"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s', 
    filename='./output/runtime_.log', 
    filemode='w'  # 'w' for overwrite, 'a' for append
)


def graph_and_queries(num_timesteps = 10, n_nodes = 1000, m_edges = 3, num_queries = 5):  
    """
    param num_timesteps: int, number of timesteps
    param n_nodes: int, number of nodes
    param m_edges: int, number of edges
    param num_queries: int, number of queries
    """  
    #create dynamic graph or import dynamic graph
    dynamic_graph = synthetic_graph(num_timesteps = 10, n_nodes = 1000, m_edges = 3)

    # Generate for query each timestep of the dynamic graph
    queries = dict()
    for timestep in range(len(dynamic_graph)): 
        G_t = dynamic_graph[timestep]
            # Convert NetworkX graph to the expected dictionary format
        graph_dict = {node: {nbr: data['weight'] for nbr, data in G_t[node].items()} for node in G_t.nodes()}

        timestep_query = dict()
        for i in range(num_queries):
            query = generate_query(graph_dict, path_len=2, qid=i, constraint_index=None)
            timestep_query[i] = query
        queries[timestep] = timestep_query

    return dynamic_graph, queries



def dynamic_graph_with_query_dummies(dynamic_graph, queries, timestep):
    """
    Add dummy nodes and edges to the dynamic graph to represent queries.
    param dynamic_graph: Dictionary representation of the dynamic graph.
    param queries: Dictionary of queries for each timestep.
    return: Dictionary representation of the dynamic graph with queries, changes_BetweeenTimesteps and the starting node for PPR walk
    """

    # Iterate over all timesteps except the last one to prevent errors

    #starttime = time.time()

    # Get the time slice between two snapshots of the graph
    snapshot_s = dynamic_graph[timestep]
    try:
        snapshot_t = dynamic_graph[timestep + 1]
    except IndexError:
        snapshot_t = dynamic_graph[timestep]   

    # Get nodes and edges in each snapshot
    nodes_s = set(snapshot_s.nodes())  # Nodes in first snapshot
    nodes_t = set(snapshot_t.nodes())  # Nodes in next snapshot

    # Extract edge sets with weights
    edges_s = {(u, v, snapshot_s[u][v]['weight']) for u, v in snapshot_s.edges()}
    edges_t = {(u, v, snapshot_t[u][v]['weight']) for u, v in snapshot_t.edges()}

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

    dyngraph = copy.deepcopy(dynamic_graph[timestep])#make a copy of the graph for easy reference since the walks may be updated

    for timestamp, values in queries.items():  
        if timestamp == timestep:
            query_nodes = list(values.values())
            #print(query_nodes)    
            w = f"dummy{timestep}" #dummy_node and start node for the ppr walk
            equal_weight = 1 / len(query_nodes)  # Assign equal weights for all edges
        else:
            query_nodes = []

        #dyngraph = dyngraph[timestamp]
        for node in query_nodes:
            dyngraph.add_edge(w, node, weight=equal_weight)

    return dyngraph, changes, w



def write_to_file(data, prefix = "data", file_extension="json", output_directory="./output/"):
    file_name = f"{output_directory}{prefix}.{file_extension}"
    with open(file_name, 'w') as f:
        json.dump(data, f, indent = 4)
        print(f"Data saved to {file_name}")




def generate_top_k_summary(n_nodes=10000):
    # Initialize a dictionary to store the changes for each timestep
    #timestamps = num_timesteps
    changes_over_time = dict() #dictionary to store the changes for each timestep
    random_walks = dict() #dictionary to store the random walks for each node for each timestep
    ppr_walks = dict() #dictionary to store the complete ppr for each timestep
    ppr_cumsum_walks = dict() #dictionary to store the cumulative sum of the ppr walks from each timestep
    top_k_cumsum_walks = dict()#dictionary to store the top k walks for each timestep

    counters = dict() #dictionary to store the counters for each node for each timestep
    cumulative_counts = Counter()#counter to store the cumulative counts of the ppr walks from each timestep

    # Generate the dynamic graph and queries
    print(n_nodes)
    dynamic_graph, queries = graph_and_queries(num_timesteps = 10, n_nodes=n_nodes, m_edges = 3, num_queries = 5)

    for timestep in range(len(dynamic_graph) - 1): 
        print("Timestep", timestep)
        starttime = time.time()
        dyngraph, changes, w = dynamic_graph_with_query_dummies(dynamic_graph, queries, timestep)    
        # Store the changes for the current timestep
        changes_over_time[timestep] = changes

        # Convert NetworkX graph to the expected dictionary format
        graph_dict = {node: {nbr: data['weight'] for nbr, data in dyngraph[node].items()} for node in dyngraph.nodes()}
        

        # Generate random walks only for the first timestep (i == 0)
        if timestep == 0:
            precomputed_walks = generate_random_walks_directed_with_weights(graph_dict,num_walks=3, walk_length=int(1 / epsilon))
            random_walks[timestep] = precomputed_walks
            R = copy.deepcopy(precomputed_walks)
            
        else:#for the subsequent timesteps, run an update on the walks from the previous timestep
            previous_walks = random_walks[timestep - 1].copy()  # Get the walks from the previous timestep
            updated_walks = update_random_walks_directed_with_weights(graph_dict, precomputed_walks, changes_over_time[timestep - 1], num_walks=3, walk_length=int(1/epsilon))
            #print(f"Updated Walks in timestep {i}: {updated_walks}")
            random_walks[timestep] = updated_walks  # Save updated walks for future reference
            R = copy.deepcopy(updated_walks)

        # Generate the directed graph for the graph with the added edges
        directed_graph = {node: list(dyngraph.neighbors(node)) for node in dyngraph.nodes()}

        #determine required length of PPR walk
        n = len(dyngraph.nodes())
        k = int(0.1*n) #summary size
        L = (c/epsilon) *k *(n/k)**epsilon #required length of PPR walk
        print(L, "Required length of PPR walk") 


        walk = personalized_complete_pagerank_walk(w, L, epsilon, R, directed_graph)
        ppr_walks[timestep] = walk
        print(len(walk), "Actual length of PPR walk")

        counts = Counter(walk)
        counters[timestep] = counts
        cumulative_counts += counts# Update cumulative counts
        ppr_cumsum_walks[timestep] = cumulative_counts.copy() # Store cumulative counts in summaries
        
        string_counter = Counter({str(key): count for key, count in cumulative_counts.items()})# Convert keys to strings
        filtered_counter = Counter({key: count for key, count in string_counter.items() if not key.startswith('dummy')})#filter out the dummy node
        
        # Normalize counts by dividing each by the sum of all counts
        total_cumulative_walks = sum(filtered_counter.values())
        if total_cumulative_walks > 0:
            normalized_cumulative_counts = Counter({
                key: value / total_cumulative_walks for key, value in filtered_counter.items()
            })
        else:
            normalized_cumulative_counts = filtered_counter  # Handle edge case of 0 total walks

        # Get the top-k items based on normalized probabilities
        top_k_cumulative = dict(normalized_cumulative_counts.most_common(k))
        top_k_cumsum_walks[timestep] = top_k_cumulative
        endtime = time.time()
        print("duration", endtime - starttime, "seconds")
        logging.info(f"Runtime for MC method timestep {timestep}: {endtime - starttime} seconds")    
        
    return changes_over_time, random_walks, ppr_walks, ppr_cumsum_walks, top_k_cumsum_walks


if __name__=="__main__":
    n_nodes = 100000
    changes_over_time, random_walks, ppr_walks, ppr_cumsum_walks, top_k_cumsum_walks = generate_top_k_summary(n_nodes)


    # Convert sets to lists in changes_over_time dictionary
    for key, value in changes_over_time.items():
        changes_over_time[key] = {
            "nodes_added": list(value["nodes_added"]),
            "nodes_removed": list(value["nodes_removed"]),
            "edges_added": list(value["edges_added"]),
            "edges_removed": list(value["edges_removed"]),
        }
    # Write the dictionaries to files
    write_to_file(changes_over_time, "changes_over_time")
    write_to_file(random_walks, "random_walks")    
    write_to_file(ppr_walks, "ppr_walks")
    write_to_file(ppr_cumsum_walks, "ppr_cumsum_walks")
    write_to_file(top_k_cumsum_walks, "top_k_cumsum_walks")




