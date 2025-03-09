
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
import src.qadgosis.main  as qadgosis
#import src.qadgosis.walks  as walks
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
                continue  # Skip nodes that are not in the original KG

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
    graph = os.path.join(dbpedia_dir, file_name)            
    KG = DBPedia(rdf_gz=graph)
    #K = int(0.0001 * KG.number_of_triples())

    KG.load()
    directed_graph = nx.DiGraph()
    #print(KG.__dict__)

    edges = KG.triples()
    for s, p, o in edges:
        directed_graph.add_edge(s, o, predicate=p)

        directed_graph[s][o]['weight'] = 0.5

    dynagraph[i] = directed_graph
    user = 0
    KG.update_user(user)
    graphs[i] = KG

    topic_mids = random.sample(list(KG.triples_.keys()), k=5)

    query_log = query_log_by_mids(
            KG, topic_mids, 20, topic_dist=np.ones(len(topic_mids))/len(topic_mids),
            shuffle=False,
            random_query_prob=0)
    logs[i] = query_log
    query = [item['Parse']['TopicEntityMid'] for item in query_log]
    #print(query)

    queries_[i] = query

K=100
_, _, _, _, top_k_cumsum_walks= qadgosis.generate_top_k_summary(num_walks = 100, dynamic_graph=dynagraph , queries=queries_, c= 2, epsilon = 0.20, k=K)
#top_k_power_nx = qadgosis.pagerank(dynamic_graph = dynagraph, queries = queries_, k=K)

#if qadgosis.check_summary(S, KG):
def process_summary_graph(top_k_nodes_dict, graphs, label):
    triples_no = {}

    for key in top_k_nodes_dict.keys() & graphs.keys():
        KG = graphs[key]
        top_k_nodes = top_k_nodes_dict[key]




        S = SummaryGraph(KG, top_k_nodes.keys())
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

        # Compute metrics
        total_F1, total_precision, total_recall = total_query_log_metrics(S, merged_logs)
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, merged_logs)

        # Logging
        if DETAILED_LOGGING:
            logging.info(f"\t  [{label}] Total F1/precision/recall: {total_F1:.2f}/{total_precision:.2f}/{total_recall:.2f}")
            logging.info(f"\t  [{label}] Average F1/precision/recall: {avg_F1:.2f}/{avg_precision:.2f}/{avg_recall:.2f}")

        print(total_F1, total_precision, total_recall)
        print(avg_F1, avg_precision, avg_recall)

    print(f"\n[{label}] triples_no:", triples_no)
    return triples_no

# Process both top_k_cumsum_walks and top_k_power_nx
triples_no_mc = process_summary_graph(top_k_cumsum_walks, graphs, "Cumsum Walks")
#triples_no_pr = process_summary_graph(top_k_power_nx, graphs, "Pagerank")

































































