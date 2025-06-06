import itertools
import argparse
import random
import logging
import json
import pdb
import numpy as np

from src.path import DBPEDIA_DATA_DIR, YAGO_DATA_DIR, METAQA_DATA_DIR, METAQA_DATA_DIR_ABLATION, ENCODING
import os
from itertools import chain
import networkx as nx
#import src.qadgosis.main_old  as qadgosis
#import walks
import src.qadgosis.main  as walks
import random
import numpy as np
import scipy.sparse as sp
from src.base import KnowledgeGraph
from time import time

from src.base import YAGO, DBPedia, Freebase, MetaQA
from src.user import query_log_by_mids, query_log_by_topics
from src.glimpse import SummaryMethod, GLIMPSE
from src.metrics import total_query_log_metrics, average_query_log_metrics
DETAILED_LOGGING = True
DETAILED_PRINTING = False




class SummaryGraph(KnowledgeGraph):
    def __init__(self, KG, top_k_nodes):
        """
        Constructs a summary of the knowledge graph based on top-k nodes.

        :param KG: KnowledgeGraph instance (the full KG)
        :param top_k_nodes: List of important nodes (entities) to include in the summary
        """
        super().__init__()
        self.parent_ = KG  
        self.top_k_nodes = set(top_k_nodes)  
        self.build_summary()

    def build_summary(self):
        """
        Builds the summary graph by adding triples that involve top-k nodes.
        """
        for e1 in self.top_k_nodes:
            if e1 not in self.parent().entity_id_:
                #print(f"Node {e1} is not in the parent graph.")  # Log nodes not in the parent graph
                continue  # Skip nodes that are not in the original KG
            #else:
             #   print(f"Node {e1} is in the parent graph.")  # Log nodes not in the parent graph


         #infer forward relations by checking all triples in KG
            if e1 in self.parent().triples_:
                for r, neighbors in self.parent().triples_[e1].items():
                    for e2 in neighbors:
                        if e2 in self.top_k_nodes:  
                            self.add_triple((e1, r, e2))

            # Infer backward relations by checking all triples in KG
            for e2, rel_dict in self.parent().triples_.items():
                for r, neighbors in rel_dict.items():
                    if e1 in neighbors and e2 in self.top_k_nodes: 
                        self.add_triple((e2, r, e1))

    def parent(self):
        return self.parent_

    def add_random_triple(self):
        """ Adds a random triple from the parent KG to the summary """
        e1 = random.choice(list(self.parent().triples_.keys()))
        r = random.choice(list(self.parent().triples_[e1].keys()))
        e2 = random.choice(list(self.parent().triples_[e1][r]))
        self.add_triple((e1, r, e2))




dynagraph = {}
queries_ = dict()
graphs = dict()
logs =dict()
triples_no_mc = dict()
triples_no_pr = dict()
dbpedia_dir = '../PKG_EXP/DBpedia/dbpedia_subgraphs_int'
for i, file_name in enumerate(sorted(os.listdir(dbpedia_dir))):
    time_start = time()
    graph = os.path.join(dbpedia_dir, file_name)            
    KG = DBPedia(rdf_gz=graph)
    #K = int(0.0001 * KG.number_of_triples())

    KG.load()
    directed_graph = nx.DiGraph()
    #print(KG.entity_id_)

    edges = KG.triples()
    for s, p, o in edges:
        directed_graph.add_edge(s, o, predicate=p)

        directed_graph[s][o]['weight'] = 0.1

    dynagraph[i] = directed_graph
    user = 0
    KG.update_user(user)
    graphs[i] = KG

    topic_mids = random.sample(list(KG.triples_.keys()), k=9)
    query_log = query_log_by_mids(
            KG, topic_mids, 200, topic_dist=np.ones(len(topic_mids))/len(topic_mids),
            shuffle=False,
            random_query_prob=0)
    logs[i] = query_log
    query = [item['Parse']['TopicEntityMid'] for item in query_log]

    queries_[i] = query
    time_end = time()
    print(f"Time taken for processing {file_name}: {time_end - time_start} seconds")

K=100


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

#print((converted_dynagraph[0]))

# Convert queries to the expected format
converted_queries = {}
for timestep, query_list in queries_.items():
    converted_queries[int(timestep)] = {i: int(query) for i, query in enumerate(query_list)}


print(len(converted_queries[0]), "queries", len(converted_dynagraph[0]))

results = walks.generate_top_k_summary(
    num_timesteps=len(converted_dynagraph),
    n_nodes=len(converted_dynagraph[0]),
    m_edges=5,  # Adjust as needed
    num_queries=len(converted_queries[0]),
    num_walks=10,  # Reduce number of walks for testing
    dynamic_graph=converted_dynagraph,
    queries=converted_queries,
    c=2,
    epsilon=0.2,
    k=K
)

    # Access top_k_cumsum_walks from the results dictionary
top_k_cumsum_walks = results["top_k_cumsum_walks"]


#if qadgosis.check_summary(S, KG):
def process_summary_graph(top_k_nodes_dict, graphs, label):
    triples_no = {}

    for key in top_k_nodes_dict.keys() & graphs.keys():
        KG = graphs[key]
        top_k_nodes = top_k_nodes_dict[key]
        top_k_nodes = {str(node): value for node, value in top_k_nodes_dict[key].items()}
        #print(key, top_k_nodes.keys())

    # Create a summary graph
        time_start = time()
        S = SummaryGraph(KG, top_k_nodes.keys())
        time_end = time()
        print(f"Time taken for processing summary {key}: {time_end - time_start} seconds")

        print(f"\n[{label}] Processing timestap: {key}")
        print("top_k", len(top_k_nodes.keys()))
        #print("attributes", (S.__dict__))
        print("triples", len(S.__dict__['triples_'].keys()))
        
        
        triples_no[key] = S.__dict__['number_of_triples_']

        # Merge logs
        keys = list(logs.keys())
        target_index = keys.index(key)
        merged_logs = [logs[key] for key in keys[:target_index + 1]]
        merged_logs = list(chain.from_iterable(value if isinstance(value, list) else [value] for value in merged_logs))
        #print(merged_logs)
        log = logs[key]
        #print(len(merged_logs))

        # Compute metrics
        total_F1, total_precision, total_recall = total_query_log_metrics(S, log)
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, log)

        # Logging
        if DETAILED_LOGGING:
            logging.info(f"\t  [{label}] Total F1/precision/recall: {total_F1:.2f}/{total_precision:.2f}/{total_recall:.2f}")
            logging.info(f"\t  [{label}] Average F1/precision/recall: {avg_F1:.2f}/{avg_precision:.2f}/{avg_recall:.2f}")

        print(total_F1, total_precision, total_recall)
        print(avg_F1, avg_precision, avg_recall)

    print(f"\n[{label}] triples_no:", triples_no)
    return triples_no

# Process both top_k_cumsum_walks and top_k_power_nx
for i, j in top_k_cumsum_walks.items():
    print(f"Top 10 nodes for timestep {i}: {len(j.values())}")
triples_no_mc = process_summary_graph(top_k_cumsum_walks, graphs, "Cumsum Walks")

































































