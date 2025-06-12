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
#from queries import generate_query, synthetic_graph
from src.qadgosis.walks_original import  personalized_complete_pagerank_walk, generate_random_walks_directed_with_weights, update_random_walks_directed_with_weights 
from sklearn.metrics import ndcg_score, dcg_score, precision_score
import csv
import os
import walks
from collections import defaultdict
import concurrent.futures




def graph_changes_with_dummy_node(dynamic_graph, queries, timestep, par='qadgosis'):
    """
    Add dummy nodes and edges to the dynamic graph to represent queries.
    param dynamic_graph: Dictionary representation of the dynamic graph.
    param queries: Dictionary of queries for each timestep.
    return: Dictionary representation of the dynamic graph with queries, changes_BetweeenTimesteps and the starting node for PPR walk
    """
    start_time = time.time()
    # Get the time slice between two snapshots of the graph
    snapshot_s, w, personalization_nodes = personalized_graph(dynamic_graph, queries, timestep)
    print(len(dynamic_graph))
    if (len(dynamic_graph)-1 == timestep):
           snapshot_t = snapshot_s
    else:
        snapshot_t,_ ,_ = personalized_graph(dynamic_graph, queries, timestep+1)
    #except:
    #    snapshot_t,_ ,_ = personalized_graph(dynamic_graph, queries, timestep)

    end_time = time.time()
    #logging.info(f"Runtime for preprocessing of dummynode{timestep}, {end_time - start_time} seconds")



    # Get nodes and edges in each snapshot
    nodes_s = {int(node) for node in snapshot_s.nodes()}# # Nodes in first snapshot
    nodes_t = {int(node) for node in snapshot_t.nodes()}#set(snapshot_t.nodes())  # Nodes in next snapshot

    # Extract edge sets with weights
    edges_s = {(int(u), int(v)) for u, v in snapshot_s.edges()}#set(snapshot_s.edges())
    edges_t = {(int(u), int(v)) for u, v in snapshot_t.edges()}#set(snapshot_t.edges())

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
    snapshot_s = dynamic_graph[timestep] #copy.deepcopy(dynamic_graph[timestep])#make a copy of the graph for easy reference

    query_nodes = list(queries[timestep])
    #print(query_nodes)
    #w = f"dummy{timestep}" #dummy_node and start node
    w = int('{}{}'.format(123456789, timestep))

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


def initialize_data_structures():
    """Initialize all required data structures."""
    return (
        dict(),  # changes_over_time
        defaultdict(dict),  # random_walks
        defaultdict(dict),  # edge_to_walks
        defaultdict(dict),  # node_to_walks
        defaultdict(dict),  # walks_to_ids
        defaultdict(dict),  # startnode_to_ids
        dict(),  # ppr_walks
        dict(),  # ppr_cumsum_walks
        dict(),  # top_k_cumsum_walks
        Counter(),  # cumulative_counts
        dict(),  # all_weights
    )

def preprocess_graph(dyngraph):
    """Convert the dynamic graph into a weighted graph."""
    return {
        int(node): {int(nbr): float(data.get('weight', 0.1)) for nbr, data in dyngraph[node].items()}
        for node in dyngraph.nodes()
    }

def calculate_required_walk_length(c, epsilon, k, n):
    """Calculate the required length of the PPR walk."""
    return (c / epsilon) * k * (n / k) ** epsilon

def generate_random_walks(weighted_graph, num_walks):
    """Generate random walks for the first timestep."""
    walks_result = walks.generate_random_walks_directed_with_weights(weighted_graph, num_walks, walk_length=10)
    return (
        walks_result.precomputed_walks,
        walks_result.inverted_edges,
        walks_result.inverted_nodes,
        walks_result.walk_ids,
        walks_result.random_walk_ids
    )

def update_random_walks(timestep, weighted_graph, previous_data, changes, num_walks):
    """Update random walks based on changes from the previous timestep."""
    walks_result = walks.update_random_walks_directed_with_weights(
        weighted_graph,
        previous_data["walks_to_ids"],
        previous_data["edge_to_walks"],
        previous_data["node_to_walks"],
        previous_data["startnode_to_ids"],
        changes,
        num_walks,
        walk_length=10
    )
    return walks_result

def process_walk_results(precomputed_walks):
    """Convert walk data to string format."""
    return {str(key): [list(map(str, inner_list)) for inner_list in value] for key, value in precomputed_walks.items()}

def compute_personalized_pagerank(w, L, epsilon, R, directed_graph):
    """Perform the Personalized PageRank walk computation."""
    start_time = time.time()
    walk = personalized_complete_pagerank_walk(w, L, epsilon, R, directed_graph)
    print(f"Time taken for PPR walk: {time.time() - start_time} seconds")
    return walk

def calculate_top_k_cumulative(walk, cumulative_counts, k):
    """Update cumulative counts and calculate the top-k nodes."""
    cumulative_counts.update(walk)
    total_cumulative_walks = sum(cumulative_counts.values())
    if total_cumulative_walks > 0:
        normalized_counts = Counter({
            key: count / total_cumulative_walks for key, count in cumulative_counts.items()
        })
    else:
        normalized_counts = cumulative_counts  # Handle edge case of 0 total walks
    return dict(normalized_counts.most_common(k))

def calculate_top_k_cumulative_with_weights(walk, cumulative_counts, k, timestep, counters, decay_rate=0.38, window_size=5):
    """
    Update cumulative counts and calculate the top-k nodes with weights and exponential decay.

    :param walk: List of nodes visited during the walk.
    :param cumulative_counts: Counter for cumulative counts of nodes.
    :param k: Number of top nodes to select.
    :param timestep: Current timestep.
    :param counters: Dictionary of counters for each timestep.
    :param decay_rate: Exponential decay rate for weights.
    :param window_size: Number of timesteps to consider for decay.
    :return: Dictionary of top-k nodes with normalized scores.
    """
    # Update the current timestep's counts
    counts = Counter(walk)
    counters[timestep] = counts

    # Determine relevant timesteps for exponential decay
    keys = sorted(counters.keys())
    latest_timestep = keys[-1]
    start_index = max(0, len(keys) - window_size)
    relevant_timesteps = keys[start_index:]

    # Apply exponential decay to counts
    decayed_counts = Counter()
    individual_weights = {}
    for j in relevant_timesteps:
        weight = decay_rate ** (latest_timestep - j)
        print(f"Timestep {j} with weight {weight:.4f}")
        individual_weights[j] = weight

        for node, count in counters[j].items():
            decayed_counts[node] += count * weight

    # Update cumulative counts
    cumulative_counts += decayed_counts

    # Normalize counts
    total_cumulative_walks = sum(cumulative_counts.values())
    if total_cumulative_walks > 0:
        normalized_counts = Counter({
            key: count / total_cumulative_walks for key, count in cumulative_counts.items()
        })
    else:
        normalized_counts = cumulative_counts  # Handle edge case of 0 total walks
    # Select top-k nodes
    top_k_cumulative = dict(normalized_counts.most_common(k))
    return top_k_cumulative, individual_weights

def generate_top_k_summary(num_timesteps=5, n_nodes=1000, m_edges=5, num_queries=20, num_walks=100, dynamic_graph=None, queries=None, c=2, epsilon=0.2, k=50):
    """Main function for generating top-k summaries."""
    (changes_over_time, random_walks, edge_to_walks, node_to_walks, walks_to_ids,
     startnode_to_ids, ppr_walks, ppr_cumsum_walks, top_k_cumsum_walks, cumulative_counts, all_weights) = initialize_data_structures()

    counters = {}  # Initialize counters for each timestep

    for timestep in range(len(dynamic_graph)):
        print(f"Timestep {timestep}")
        time_start = time.time()
        dyngraph, changes, w, personalization_nodes = graph_changes_with_dummy_node(dynamic_graph, queries, timestep)
        changes_over_time[timestep] = changes

        n = len(dyngraph.nodes())
        L = calculate_required_walk_length(c, epsilon, k, n)
        logging.info(f"Required length of PPR walk {timestep}: {L}")

        weighted_graph = preprocess_graph(dyngraph)
        time_end = time.time()
        print(f"Time taken for preprocessing graph step: {time_end - time_start} seconds")
        if timestep == 0:
            print("Generating random walks...")
            (precomputed_walks, inverted_edges, inverted_nodes, walk_ids, random_walk_ids) = generate_random_walks(weighted_graph, num_walks)
            random_walks[timestep], edge_to_walks[timestep], node_to_walks[timestep], walks_to_ids[timestep], startnode_to_ids[timestep] = (
                precomputed_walks, inverted_edges, inverted_nodes, walk_ids, random_walk_ids
            )
            R = process_walk_results(precomputed_walks)
        else:
            print("Updating random walks...")
            previous_data = {
                "edge_to_walks": edge_to_walks[timestep - 1],
                "node_to_walks": node_to_walks[timestep - 1],
                "walks_to_ids": walks_to_ids[timestep - 1],
                "startnode_to_ids": startnode_to_ids[timestep - 1]
            }
            updated_walks_result = update_random_walks(timestep, weighted_graph, previous_data, changes, num_walks)
            random_walks[timestep], edge_to_walks[timestep], node_to_walks[timestep], walks_to_ids[timestep], startnode_to_ids[timestep] = (
                updated_walks_result.precomputed_walks,
                updated_walks_result.inverted_edges,
                updated_walks_result.inverted_nodes,
                updated_walks_result.walk_ids,
                updated_walks_result.random_walk_ids
            )
            R = process_walk_results(updated_walks_result.precomputed_walks)

        directed_graph = {node: list(dyngraph.neighbors(node)) for node in dyngraph.nodes()}
        walk = compute_personalized_pagerank(w, L, epsilon, R, directed_graph)
        ppr_walks[timestep] = walk
        time_start = time.time()
        top_k_cumsum_walks[timestep], all_weights[timestep] = calculate_top_k_cumulative_with_weights(walk, cumulative_counts, k, timestep, counters)

        #top_k_cumsum_walks[timestep] = calculate_top_k_cumulative(walk, cumulative_counts, k)
        time_end = time.time()
        print(f"Time taken for top k: {time_end - time_start} seconds")

    return changes_over_time, random_walks, ppr_walks, ppr_cumsum_walks, top_k_cumsum_walks, all_weights







