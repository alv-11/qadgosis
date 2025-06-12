import os
import json
import pdb
import time
import argparse
import logging
import random
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from time import time
from src.base import KnowledgeGraph

from itertools import chain
from src.base import KnowledgeGraph, YAGO, DBPedia, Freebase, MetaQA
from src.path import DBPEDIA_DATA_DIR, YAGO_DATA_DIR, METAQA_DATA_DIR, METAQA_DATA_DIR_ABLATION, ENCODING
from src.user import query_log_by_mids, query_log_by_topics
from src.glimpse import SummaryMethod, GLIMPSE
from src.metrics import total_query_log_metrics, average_query_log_metrics
#import src.qadgosis.main as qadgosis
import src.qadgosis.main  as walks



DETAILED_LOGGING = True
DETAILED_PRINTING = False

METHODS = {
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    #'sankofa': SummaryMethod(SANKOFA, 'SANKOFA')
}



class SummaryGraph(KnowledgeGraph):
    def __init__(self, KG, top_k_nodes, max_triples=None):
        super().__init__()
        self.parent_ = KG
        self.top_k_nodes = set(top_k_nodes)
        self.max_triples = max_triples
        self.triple_count = 0
        self.triples_set = set()
        self.reverse_index = self._build_reverse_index()  # New
        self.build_summary()

    def _build_reverse_index(self):
        """Index for reverse lookup: e1 ∈ neighbors → (e2, r, e1)"""
        reverse_idx = {}
        for e2, rel_dict in self.parent().triples_.items():
            for r, neighbors in rel_dict.items():
                for e1 in neighbors:
                    if e1 not in reverse_idx:
                        reverse_idx[e1] = []
                    reverse_idx[e1].append((e2, r))
        return reverse_idx

    def build_summary(self):
        parent_triples = self.parent().triples_
        entity_ids = self.parent().entity_id_

        for e1 in self.top_k_nodes:
            if self._limit_reached():
                break
            if e1 not in entity_ids:
                continue

            # Outgoing edges
            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in self.top_k_nodes:
                        self._try_add_triple((e1, r, e2))
                        if self._limit_reached():
                            return

            # Incoming edges (faster using pre-built reverse index)
            for e2, r in self.reverse_index.get(e1, []):
                if e2 in self.top_k_nodes:
                    self._try_add_triple((e2, r, e1))
                    if self._limit_reached():
                        return

        # Pad with random related triples if needed
        if self.max_triples and self.triple_count < self.max_triples:
            #print(f"Adding related random triples to reach {self.max_triples}.")
            attempts = 0
            while self.triple_count < self.max_triples and attempts < 10 * self.max_triples:
                self.add_related_random_triple()
                attempts += 1

    def parent(self):
        return self.parent_

    def _limit_reached(self):
        return self.max_triples and self.triple_count >= self.max_triples

    def _try_add_triple(self, triple):
        if triple not in self.triples_set:
            self.add_triple(triple)
            self.triples_set.add(triple)
            self.triple_count += 1

    def add_related_random_triple(self):
        if self._limit_reached():
            return
        e1 = random.choice(list(self.top_k_nodes))
        parent_triples = self.parent().triples_

        # Outgoing
        for _ in range(5):  # Limit attempts
            r_list = list(parent_triples.get(e1, {}).keys())
            if r_list:
                r = random.choice(r_list)
                e2_list = list(parent_triples[e1][r])
                if e2_list:
                    e2 = random.choice(e2_list)
                    self._try_add_triple((e1, r, e2))
                    return

        # Incoming (faster using reverse index)
        if e1 in self.reverse_index:
            e2, r = random.choice(self.reverse_index[e1])
            self._try_add_triple((e2, r, e1))


def flatten_list(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def sliding_window(window_size=2, training_set=None, summary=None, i=None, test_weights=None):
    data = []
    timesteps = sorted(training_set.keys())
    for j in range(len(timesteps) - window_size + 1):
        t1, t2 = timesteps[j:j + window_size]
        if t2 <= i and i != 0:
            t1_weight = test_weights[t1]
            t2_weight = test_weights[t2]
            data_t1 = np.random.choice(training_set[t1], size=int(len(training_set[t1]) * t1_weight), replace=False)
            data_t2 = np.random.choice(training_set[t2], size=int(len(training_set[t2]) * t2_weight), replace=False)
            combined_data = np.concatenate((data_t1, data_t2))
            total_F1, _, _ = total_query_log_metrics(summary, combined_data)
            data.append({"Timestep Window": f"{t1}-{t2}", "Total F1": total_F1})
    return pd.DataFrame(data)


def process_all_timesteps(training_set=None, summary=None, i=None, test_weights=None):
    if training_set is None or test_weights is None:
        print("Error: training_set and test_weights must be provided.")
        return None

    output_data = []
    data = []
    timesteps = sorted(training_set.keys())

    for t in timesteps:
        if t in test_weights:
            sampled_data = np.random.choice(training_set[t], size=int(len(training_set[t]) * test_weights[t]), replace=False)
            sampled_data = [q for q in sampled_data if isinstance(q, dict) and 'Parse' in q]
            if not sampled_data:
                continue
            data.extend(sampled_data)

    data = list(flatten_list(data))
    total_F1, _, _ = total_query_log_metrics(summary, data)
    output_data.append({"Timestep Window": f"{i}", "Total F1": total_F1})
    return pd.DataFrame(output_data)


def process_summary_sankofa(top_k_nodes_dict, graphs, label, weights, test_data, walks_runtime=None):
    """
    Processes the summary graph and computes metrics.

    :param top_k_nodes_dict: Dictionary of top-k nodes for each timestep.
    :param graphs: Dictionary of graphs for each timestep.
    :param label: Label for logging purposes.
    :param weights: Dictionary of weights for test data sampling.
    :param test_data: Dictionary of test data for each timestep.
    :param walks_runtime: DataFrame containing runtime for walks.
    :return: Dictionary containing triples count for each timestep.
    """
    triples_no = {}
    future = []
    weighted_tests = []
    runtime = []
    print("Summarizing for SANKOFA")
    for key in top_k_nodes_dict.keys() & graphs.keys() & weights.keys():
        KG = graphs[key]
        top_k_nodes = top_k_nodes_dict[key]
        top_k_nodes = {str(node): value for node, value in top_k_nodes_dict[key].items()}

        # Create a summary graph
        time_start = time()
        S = SummaryGraph(KG, top_k_nodes.keys(), max_triples=200)
        time_end = time()
        
        runtime.append({"Timestep": key, "Runtime (summary)": time_end - time_start})
        triples_no[key] = S.triple_count
        log = test_data[key]

        # Compute metrics
        total_F1, total_precision, total_recall = total_query_log_metrics(S, log)
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, log)
        future.append({"Timestep": key, "Total F1": total_F1})

        test_weights = weights[key]

        # Process all timesteps
        df = process_all_timesteps(training_set=test_data, summary=S, i=key, test_weights=test_weights)
        weighted_tests.append(df)

    # Convert runtime to DataFrame
    timestep_runtime = pd.DataFrame(runtime)
    F1_future_tests = pd.DataFrame(future).rename(columns={"Total F1": "Future Test F1"})
    F1_future_weighted_tests = pd.concat(weighted_tests, ignore_index=True).rename(columns={"Timestep Window": "Timestep", "Total F1": "Weighted Future Test F1"})

    # Ensure 'Timestep' column has the same type in all DataFrames
    timestep_runtime["Timestep"] = timestep_runtime["Timestep"].astype(int)
    F1_future_tests["Timestep"] = F1_future_tests["Timestep"].astype(int)
    F1_future_weighted_tests["Timestep"] = F1_future_weighted_tests["Timestep"].astype(int)

    # Merge DataFrames sequentially
    merged_df = pd.merge(walks_runtime, timestep_runtime, on="Timestep")
    merged_df = pd.merge(merged_df, F1_future_tests, on="Timestep")
    merged_df = pd.merge(merged_df, F1_future_weighted_tests, on="Timestep")

 
    merged_df["Runtime (seconds)"] = merged_df["Runtime (summary)"] + merged_df["Runtime (walks)"]
    sankofa = merged_df[["Timestep", "Runtime (seconds)", "Future Test F1", "Weighted Future Test F1"]]

    sankofa.to_csv(f"sankofa_results.csv", index=False)
    return triples_no


def process_summary_glimpse(graphs, triples_no_mc, training_data, test_data, method, weights):
    """
    Answers queries in the log and computes metrics.

    :param graphs: Dictionary of graphs for each timestep.
    :param triples_no_mc: Dictionary of triples count for each timestep.
    :param training_data: Dictionary of training data for each timestep.
    :param test_data: Dictionary of test data for each timestep.
    :param method: List of summarization methods to use.
    :param weights: Dictionary of weights for test data sampling.
    :return: DataFrame containing metrics for each timestep.
    """
    summary_methods = [METHODS[name] for name in method]
    future = []
    weighted_tests = []
    runtime = []
    print("Summarizing for GLIMPSE")
    for summary_method in summary_methods:
        for i in graphs.keys() & triples_no_mc.keys() & weights.keys():
            KG = graphs[i]
            K = triples_no_mc[i]

            update_time_list = []
            acc_list = []
            test_weights = weights[i]


            if i == 0:
                #print(f"Summarizing for query: {i}")
                t0 = time()
                S = summary_method(KG, K, training_data[i])
                #runtime = time() - t0
                runtime.append({"Timestep": i, "Runtime (seconds)": time() - t0})


                total_F1, total_precision, total_recall = total_query_log_metrics(S, test_data[i])
                avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, test_data[i])
                #print(total_F1, total_precision, total_recall)
                #print(avg_F1, avg_precision, avg_recall)

                future.append({"Timestep": i, "Total F1": total_F1})
                df = process_all_timesteps(training_set=test_data, summary=S, i=i, test_weights=test_weights)
                weighted_tests.append(df)

            else:
                #print(f"Re-summarizing for query: {i}")
                t0 = time()

                keys = list(training_data.keys())
                target_index = keys.index(i)
                merged_logs = [training_data[key] for key in keys[:target_index + 1]]
                merged_logs = list(flatten_list(merged_logs))

                S = summary_method(KG, K, merged_logs)
                runtime.append({"Timestep": i, "Runtime (seconds)": time() - t0})
                #runtime = time() - t0
                #print("Runtime:", runtime)
                #update_time_list.append(runtime)

                total_F1, total_precision, total_recall = total_query_log_metrics(S, test_data[i])
                avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, test_data[i])
                future.append({"Timestep": i, "Total F1": total_F1})

                df = process_all_timesteps(training_set=test_data, summary=S, i=i, test_weights=test_weights)
                weighted_tests.append(df)
    timestep_runtime = pd.DataFrame(runtime)
    F1_future_tests = pd.DataFrame(future).rename(columns={"Total F1": "Future Test F1"})
    F1_future_weighted_tests = pd.concat(weighted_tests, ignore_index=True).rename(columns={"Timestep Window": "Timestep", "Total F1": "Weighted Future Test F1"})

    # Ensure 'Timestep' column has the same type in all DataFrames
    timestep_runtime["Timestep"] = timestep_runtime["Timestep"].astype(int)
    F1_future_tests["Timestep"] = F1_future_tests["Timestep"].astype(int)
    F1_future_weighted_tests["Timestep"] = F1_future_weighted_tests["Timestep"].astype(int)

    # Merge DataFrames sequentially
    merged_df = pd.merge(timestep_runtime, F1_future_tests, on="Timestep")
    glimpse = pd.merge(merged_df, F1_future_weighted_tests, on="Timestep")

    glimpse.to_csv("glimpse_results.csv", index=False)


if __name__ == "__main__":
    subgraph_dir ='../PKG_EXP/DBpedia/dbpedia_subgraphs_int'
    dynagraph, queries_, graphs, training_data, test_data = {}, {}, {}, {}, {}

    print("Loading all subgraph snapshots...")
    for i, file_name in enumerate(sorted(os.listdir(subgraph_dir))):
        start = time()  # Fixed: Use `time()` directly
        path = os.path.join(subgraph_dir, file_name)

        KG = DBPedia(rdf_gz=path)
        KG.load()
        KG.update_user(1)

        G = nx.DiGraph()
        for s, p, o in KG.triples():
            G.add_edge(s, o, predicate=p, weight=0.01)

        dynagraph[i] = G
        graphs[i] = KG

        topic_mids = random.sample(list(KG.triples_.keys()), k=2)
        query_log = query_log_by_mids(
            KG,
            topic_mids,
            200,
            topic_dist=np.ones(len(topic_mids))/len(topic_mids),
            shuffle=True,
            random_query_prob=0
        )

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]

        queries_[i] = [q['Parse']['TopicEntityMid'] for q in training_data[i]]

        print(f"Processed snapshot {i}: {file_name} in {time() - start:.2f}s")

    # Example: simulate top-k node selection (this should be replaced with actual method)
    top_k_nodes_dict = {}
    for t, G in dynagraph.items():
        node_scores = {n: G.degree(n) for n in G.nodes()}
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_nodes_dict[t] = dict(sorted_nodes[:50])  # top-50 by degree

    # Run evaluation on the summaries
    print("Evaluating summary graphs...")
    K=100
    #changes_over_time, random_walks, _, _, top_k_cumsum_walks, weights= qadgosis.generate_top_k_summary(num_walks = 100, dynamic_graph=dynagraph , queries=queries_, c= 2, epsilon = 0.20, k=K)

    converted_dynagraph = []
    for timestep, graph in dynagraph.items():
        converted_graph = {}
        for node in graph.nodes:
            neighbors = {}
            for neighbor in graph[node]:
                # Ensure neighbors exist and weights are valid
                if graph[node][neighbor]:
                    neighbors[int(neighbor)] = float(graph[node][neighbor].get("weight", 0.1))  # Default weight is 0.5
            converted_graph[int(node)] = neighbors
        converted_dynagraph.append(converted_graph)



    # Convert queries to the expected format
    converted_queries = {}
    for timestep, query_list in queries_.items():
        converted_queries[int(timestep)] = {i: int(query) for i, query in enumerate(query_list)}

    results = walks.generate_top_k_summary(
    num_timesteps=len(converted_dynagraph),
    num_walks=10,  # Reduce number of walks for testing
    dynamic_graph=converted_dynagraph,
    queries=converted_queries,
    c=2,
    epsilon=0.2,
    k=K,
    decay_rate = .9
    )

    # Access top_k_cumsum_walks from the results dictionary
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
    runtime_per_timestep = results["runtime_per_timestep"]
    print(weights)

    # Convert runtime_per_timestep to a DataFrame-friendly format
    runtime_df = pd.DataFrame(
        [{"Timestep": t, "Runtime (walks)": runtime} for t, runtime in runtime_per_timestep.items()]
    ).sort_values(by="Timestep")

    triples_no = process_summary_sankofa(top_k_cumsum_walks, graphs, label="Cumsum Walks", weights=weights, test_data=test_data, walks_runtime = runtime_df)
    process_summary_glimpse(graphs, triples_no, training_data, test_data, method=["glimpse"], weights=weights)