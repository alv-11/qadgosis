import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from time import time
from src.base import KnowledgeGraph
from collections import Counter
import matplotlib.pyplot as plt
import powerlaw
#from itertools import chain
from src.base import KnowledgeGraph, YAGO, DBPedia, Freebase, MetaQA
from src.path import DBPEDIA_DATA_DIR, YAGO_DATA_DIR, METAQA_DATA_DIR, METAQA_DATA_DIR_ABLATION, ENCODING
from src.user import query_log_by_mids, query_log_by_topics
from src.glimpse import SummaryMethod, GLIMPSE
from src.metrics import total_query_log_metrics, average_query_log_metrics
#import src.qadgosis.main_old as qadgosis
import src.qadgosis.main  as walks
from src.apex import APEX, DETAILED_LOGGING, APEX_
from collections import defaultdict, deque
import os
import pickle
import heapq





DETAILED_LOGGING = True
DETAILED_PRINTING = False


METHODS = { 
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    'glimpse-2': SummaryMethod(GLIMPSE, 'GLIMPSE-2', power=2),
    'apex': SummaryMethod(APEX, 'APEX'),
    #'apex-n': SummaryMethod(APEX_naive, 'APEX-N'),
    #'pagerank': SummaryMethod(PR, 'PageRank')
}





class SummaryGraph_(KnowledgeGraph):
    # def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
    #     super().__init__()
    #     self.parent_ = KG
    #     self.top_k_nodes = set(top_k_nodes)
    #     self.max_triples = max_triples
    #     self.batch_size = batch_size
    #     self.triple_count = 0
    #     self.triples_set = set()
    #     self.reverse_index = self._build_reverse_index()
    #     self.build_summary()
    def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10, query_freq=None,n_priority=None):
        super().__init__()
        self.parent_ = KG
        self.top_k_nodes = (top_k_nodes)
        self.max_triples = max_triples
        self.batch_size = batch_size
        self.triple_count = 0
        self.triples_set = set()
        self.n_priority = n_priority
        self.query_freq = query_freq or {}
        self.reverse_index = self._build_reverse_index()
        #self.build_summary_()
        #
        self._build_summary()
        

    def set_parent(self, parent_kg):
        self._parent = parent_kg

    def parent(self):
        return self.parent_

    def _limit_reached(self):
        return self.max_triples is not None and self.triple_count >= self.max_triples

    def _build_reverse_index(self):
        reverse_idx = {}
        top_k = self.top_k_nodes
        triples = self.parent().triples_
        for e2, rel_dict in triples.items():
            for r, neighbors in rel_dict.items():
                for e1 in neighbors:
                    if e1 in top_k:
                        reverse_idx.setdefault(e1, []).append((e2, r))
        return reverse_idx

    def _try_add_triple(self, triple):
        if triple not in self.triples_set:
            self.add_triple(triple)
            self.triples_set.add(triple)
            self.triple_count += 1




    def build_summary_(self):
        parent_triples = self.parent().triples_
        entity_ids = self.parent().entity_id_
        top_k_all = self.top_k_nodes  

        # Step 1: Select top-k nodes
        top_k = dict(list(top_k_all.items())[:100])  # Select the top 100 nodes based on their scores
        #top_k = dict(sorted(top_k_all.items(), key=lambda x: x[1], reverse=True)[:100])
        

        # Step 2: Collect and score triples
        candidate_triples = []
        remaining_nodes = {node: score for node, score in top_k_all.items() if node not in top_k}  # Remaining nodes

        for e1 in top_k:
            if e1 not in entity_ids:
                continue

            # Outgoing triples
            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in top_k or e2 in remaining_nodes:  
                        score = self.compute_triple_score(e1, e2, top_k, remaining_nodes)
                        candidate_triples.append(((e1, r, e2), score))

            # Incoming triples (using reverse index)
            for e2, r in self.reverse_index.get(e1, []):
                if e2 in top_k or e2 in remaining_nodes:  # Include triples from top-k and remaining nodes
                    score = self.compute_triple_score(e2, e1, top_k, remaining_nodes)
                    candidate_triples.append(((e2, r, e1), score))

        # Step 3: Sort triples by score
        scored_triples = sorted(candidate_triples, key=lambda x: x[1], reverse=True)
        

        # Step 4: Add top triples to the summary
        for triple, _ in scored_triples:
            if self._limit_reached():
                return
            self._try_add_triple(triple)



    def compute_triple_score(self, node1, node2, top_k, remaining_nodes):
        """
        Compute the score of a triple based on the scores of the two nodes.
        """
        score1 = top_k.get(node1, remaining_nodes.get(node1, 0))  # Score of the first node
        score2 = top_k.get(node2, remaining_nodes.get(node2, 0))  # Score of the second node

        return score1 + score2  # Example: Sum of the two node scores


    def build_summary__(self):
        parent_triples = self.parent().triples_
        top_k = self.top_k_nodes

        candidate_triples = []

        for e1 in top_k:
            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in top_k:
                        candidate_triples.append((e1, r, e2))

            for e2, r in self.reverse_index.get(e1, []):
                if e2 in top_k:
                    candidate_triples.append((e2, r, e1))

        # Score and sort
        scored = sorted(candidate_triples, key=lambda t: self.compute_info_gain(t, self.query_freq), reverse=True)
    
        # Add top triples up to the limit
        for triple in scored:
            if self._limit_reached():
                break
            self._try_add_triple(triple)

        # Pad if needed
        print("Padding summary with related triples")
        while not self._limit_reached():
            self.add_related_random_triples_batch(batch_size=self.batch_size)



    def _build_summary(self):
        parent_triples = self.parent().triples_
        entity_ids = self.parent().entity_id_
        top_k = self.top_k_nodes

        for e1 in top_k:
            if e1 not in entity_ids:
                continue

            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in top_k:
                        self._try_add_triple((e1, r, e2))
                        if self._limit_reached():
                            return

            for e2, r in self.reverse_index.get(e1, []):
                if e2 in top_k:
                    self._try_add_triple((e2, r, e1))
                    if self._limit_reached():
                        return

        # Pad with related triples if needed
        #padding summary
        print("Padding summary with related triples")
        while not self._limit_reached():
            self.add_related_random_triples_batch(batch_size=self.batch_size)


    def add_related_random_triples_batch(self, batch_size=100):
        parent_triples = self.parent().triples_
        reverse_idx = self.reverse_index
        top_k = list(self.top_k_nodes)

        remaining = self.max_triples - self.triple_count
        batch_size = min(batch_size, remaining)

        candidates = []
        for _ in range(batch_size):
            e1 = random.choice(top_k)

            # Outgoing
            if e1 in parent_triples:
                r_list = list(parent_triples[e1].keys())
                if r_list:
                    r = random.choice(r_list)
                    e2_list = list(parent_triples[e1][r])
                    if e2_list:
                        e2 = random.choice(e2_list)
                        candidates.append((e1, r, e2))

            # Incoming
            if e1 in reverse_idx:
                e2_r_list = reverse_idx[e1]
                if e2_r_list:
                    e2, r = random.choice(e2_r_list)
                    candidates.append((e2, r, e1))

        for triple in candidates:
            if self._limit_reached():
                break
            self._try_add_triple(triple)

            

    def build_summary(self):
        parent_triples = self.parent().triples_
        entity_ids = self.parent().entity_id_
        top_k_all = self.top_k_nodes
        n_priority = int(len(top_k_all))
        #print(f"Building summary with {len(top_k_all)} top-k nodes, prioritizing {n_priority} nodes")

        # Step 1: sort top-k nodes
        top_k = dict(sorted(top_k_all.items(), key=lambda x: x[1], reverse=True))

        top_k_list = list(top_k.keys())
        # Split top-k into priority and secondary nodes
        priority_nodes = set(top_k_list[:n_priority])
        secondary_nodes = set(top_k_list[n_priority:])

    
        if not hasattr(self, "triple_count"):
            self.triple_count = len(self.triples_)
        else:
            # repair if inconsistent
            if self.triple_count != len(self.triples_):
                print("Repairing triple_count (was {}, now {})".format(self.triple_count, len(self.triples_)))
                self.triple_count = len(self.triples_)


        # def add_triples(nodes_to_consider):
        #     visited = set()  # avoid duplicate triples
        #     for e1 in nodes_to_consider:
        #         if self._limit_reached():
        #             return
        #         if e1 not in entity_ids:
        #             continue
        #         # Outgoing triples
        #         for r, neighbors in parent_triples.get(e1, {}).items():
        #             for e2 in neighbors:
        #                 if e2 in top_k_all:
        #                     triple = (e1, r, e2)
        #                     if triple in visited:
        #                         continue
        #                     visited.add(triple)
        #                     score = self.compute_triple_score(e1, e2, top_k, top_k_all)
        #                     self._try_add_triple(triple)
        #                     if self._limit_reached():
        #                         return
        #         # Incoming triples
        #         for e2, r in self.reverse_index.get(e1, []):
        #             if e2 in top_k_all:
        #                 triple = (e2, r, e1)
        #                 if triple in visited:
        #                     continue
        #                 visited.add(triple)
        #                 score = self.compute_triple_score(e2, e1, top_k, top_k_all)
        #                 self._try_add_triple(triple)
        #                 if self._limit_reached():
        #                     return

        # # Step 2: Priority nodes first
        # add_triples(priority_nodes)
        # # Step 3: Secondary nodes if limit not reached
        # print("Padding summary with secondary nodes")
        # add_triples(secondary_nodes)


        #Helper function to collect triples for a given set of nodes
        def collect_triples_(nodes_to_consider):
            triples = []
            for e1 in nodes_to_consider:
                if e1 not in entity_ids:
                    continue
                # Outgoing triples
                for r, neighbors in parent_triples.get(e1, {}).items():
                    for e2 in neighbors:
                        if e2 in top_k_all:  # Only consider top-k or remaining nodes
                            score = self.compute_triple_score(e1, e2, top_k, top_k_all)
                            triples.append(((e1, r, e2), score))
                # Incoming triples
                for e2, r in self.reverse_index.get(e1, []):
                    if e2 in top_k_all:
                        score = self.compute_triple_score(e2, e1, top_k, top_k_all)
                        triples.append(((e2, r, e1), score))
            return triples
        

        def collect_triples(nodes_to_consider, top_k, top_k_all, n_priority):
            triples = []
            for e1 in nodes_to_consider:
                if e1 not in entity_ids:
                    continue
                # Outgoing
                for r, neighbors in parent_triples.get(e1, {}).items():
                    for e2 in neighbors:
                        if e2 in top_k_all:
                            score = self.compute_triple_score(e1, e2, top_k, top_k_all)
                            triples.append(((e1, r, e2), score))
                # Incoming
                for e2, r in self.reverse_index.get(e1, []):
                    if e2 in top_k_all:
                        score = self.compute_triple_score(e2, e1, top_k, top_k_all)
                        triples.append(((e2, r, e1), score))
            # Keep only top-k triples by score
            return heapq.nlargest(n_priority, triples, key=lambda x: x[1])


        # Step 2: Collect, sort, and add triples for priority nodes first
        priority_triples = collect_triples(priority_nodes,top_k, top_k_all, n_priority)
        print(len(priority_triples), "triples collected from priority nodes")


        for triple, _ in priority_triples:
            if self._limit_reached():
                break
            self._try_add_triple(triple)
            
 
        remaining_limit = self.max_triples - len(self.triples_)
        if remaining_limit > 0:
            print("Padding summary with secondary nodes")
            secondary_triples = collect_triples(secondary_nodes, top_k, top_k_all, remaining_limit)
            for triple, _ in secondary_triples:
                if self._limit_reached():
                    break
                self._try_add_triple(triple)    

        # Step 3: If summary limit not reached, do the same for secondary nodes
        # secondary_triples = sorted(collect_triples(secondary_nodes, top_k, top_k_all, self.max_triples), key=lambda x: x[1], reverse=True)
        # for triple, _ in secondary_triples:
        #     if self._limit_reached():
        #         return
        #     self._try_add_triple(triple)


        # Pad if needed
        print("Padding summary with related triples")
        while not self._limit_reached():
            self.add_related_random_triples_batch(batch_size=self.batch_size)

    # def update_summary(self, new_top_k_nodes, query_freq, bfs_depth=2):
    #     removed_nodes = self.top_k_nodes - new_top_k_nodes
    #     added_nodes = new_top_k_nodes - self.top_k_nodes

    #     if not removed_nodes and not added_nodes:
    #         print("No changes in top-k nodes, skipping update")
    #         self.build_summary()
    #         return
    

    #     self.top_k_nodes = set(new_top_k_nodes)
    #     parent = self._parent
    #     parent_triples = parent.triples_

    #     # 1. Remove triples involving removed nodes
    #     updated_triples = {
    #         t for t in self.triples_set
    #         if t[0] not in removed_nodes and t[2] not in removed_nodes
    #     }
    #     self.triples_set = updated_triples

    #     # Rebuild triples_ and triple_count
    #     self.triples_ = defaultdict(lambda: defaultdict(set))
    #     self.triple_count = 0
    #     for h, r, t in self.triples_set:
    #         self.triples_[h][r].add(t)
    #         self.triple_count += 1

    #     # 2. Add new triples using BFS and score by information gain
    #     candidate_triples = set()
    #     visited = set()

    #     for start_node in added_nodes:
    #         if start_node not in parent.entity_id_:
    #             continue

    #         queue = deque([(start_node, 0)])
    #         visited.add(start_node)

    #         while queue:
    #             current_node, depth = queue.popleft()
    #             if depth >= bfs_depth:
    #                 continue

    #             # Forward edges
    #             for r, neighbors in parent_triples.get(current_node, {}).items():
    #                 for neighbor in neighbors:
    #                     triple = (current_node, r, neighbor)
    #                     if neighbor in self.top_k_nodes:
    #                         candidate_triples.add(triple)
    #                     if neighbor not in visited:
    #                         visited.add(neighbor)
    #                         queue.append((neighbor, depth + 1))

    #             # Reverse edges
    #             if hasattr(parent, "reverse_index") and current_node in parent.reverse_index:
    #                 for neighbor, r in parent.reverse_index[current_node]:
    #                     triple = (neighbor, r, current_node)
    #                     if neighbor in self.top_k_nodes:
    #                         candidate_triples.add(triple)
    #                     if neighbor not in visited:
    #                         visited.add(neighbor)
    #                         queue.append((neighbor, depth + 1))

    #     # Score triples using info gain
    #     scored = sorted(candidate_triples, key=lambda t: self.compute_info_gain(t, self.query_freq), reverse=True)

    #     for triple in scored:
    #         if self._limit_reached():
    #             break
    #         self._try_add_triple(triple)

    #     # 3. Pad if needed
    #     while not self._limit_reached():
    #         self.add_related_random_triples_batch(batch_size=self.batch_size)





    def _update_summary(self, new_top_k_nodes, query_freq):
        removed_nodes = self.top_k_nodes - new_top_k_nodes
        added_nodes = new_top_k_nodes - self.top_k_nodes
    

        if not removed_nodes and not added_nodes:
            print("No changes in top-k nodes, skipping update")
            return  # Nothing to update

        self.top_k_nodes = set(new_top_k_nodes)
        #parent = self.parent()
        parent = self._parent
        parent_triples = parent.triples_

        # 1. Remove triples involving removed nodes only (partial removal)
        updated_triples = {
            t for t in self.triples_set
            if t[0] not in removed_nodes and t[2] not in removed_nodes
        }
        self.triples_set = updated_triples

        # Rebuild triples_ and triple_count after removal
        self.triples_ = defaultdict(lambda: defaultdict(set))
        self.triple_count = 0
        for h, r, t in self.triples_set:
            self.triples_[h][r].add(t)
            self.triple_count += 1

        # 2. Add new triples involving only newly added nodes, scored by info gain
        candidate_triples = []

        for e1 in added_nodes:
            if e1 not in parent.entity_id_:
                continue

            # Forward triples from e1
            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in self.top_k_nodes:
                        candidate_triples.append((e1, r, e2))

            # Reverse triples to e1 (using reverse index if available)
            if hasattr(parent, "reverse_index") and e1 in parent.reverse_index:
                for e2, r in parent.reverse_index[e1]:
                    if e2 in self.top_k_nodes:
                        candidate_triples.append((e2, r, e1))

        # Score triples by info gain and add top scoring ones
        scored = sorted(candidate_triples, key=lambda t: self.compute_info_gain(t, self.query_freq), reverse=True)
       

        for triple in scored:
            if self._limit_reached():
                break
            self._try_add_triple(triple)

        # 3. Pad with related random triples if needed
        print("Padding summary with related triples")
        while not self._limit_reached():
            self.add_related_random_triples_batch(batch_size=self.batch_size)






    def update_summary_(self, new_top_k_nodes):
        removed_nodes = self.top_k_nodes - new_top_k_nodes
        added_nodes = new_top_k_nodes - self.top_k_nodes

        if not removed_nodes and not added_nodes:
            return  # Nothing to update

        self.top_k_nodes = set(new_top_k_nodes)
        parent = self._parent#
        parent_triples = parent.triples_

        # Remove triples with nodes no longer in top-k
        updated_triples = {
            t for t in self.triples_set
            if t[0] not in removed_nodes and t[2] not in removed_nodes
        }

        self.triples_set = updated_triples
        self.triples_ = defaultdict(lambda: defaultdict(set))
        self.triple_count = 0
        for h, r, t in self.triples_set:
            self.triples_[h][r].add(t)
            self.triple_count += 1

        # Add triples involving only new nodes 
        for e1 in added_nodes:
            if e1 not in parent.entity_id_:
                continue

            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in self.top_k_nodes:
                        self._try_add_triple((e1, r, e2))
                        if self._limit_reached():
                            return

            # # Instead of scanning parent again, use reverse index from parent if it exists
            if hasattr(parent, "reverse_index") and e1 in parent.reverse_index:
                for e2, r in parent.reverse_index[e1]:
                    if e2 in self.top_k_nodes:
                        self._try_add_triple((e2, r, e1))
                        if self._limit_reached():
                            return

        #  Pad if needed
     
        while not self._limit_reached():
            self.add_related_random_triples_batch(batch_size=self.batch_size)



    def compute_info_gain(self, triple, query_freq):
        """
        Information gain proxy: sum of query frequencies for head and tail entities.
        """
        h, r, t = triple
        gain_h = query_freq.get(h, 0)
        gain_t = query_freq.get(t, 0)
        return gain_h + gain_t




    def update_summary(self, new_top_k_nodes):
        print(f"Updating summary with {len(new_top_k_nodes)} new top-k nodes")
        removed_nodes = self.top_k_nodes - new_top_k_nodes
        added_nodes = new_top_k_nodes - self.top_k_nodes

        if not removed_nodes and not added_nodes:
            return

        self.top_k_nodes = set(new_top_k_nodes)
        parent = parent = self._parent##self.parent()
        parent_triples = parent.triples_

        # === Step 1: Bulk remove triples
        self.triples_set = {
            t for t in self.triples_set
            if t[0] not in removed_nodes and t[2] not in removed_nodes
        }
        # Rebuild triples_ dict for speed
        self.triples_ = defaultdict(lambda: defaultdict(set))
        self.triple_count = 0
        for h, r, t in self.triples_set:
            self.triples_[h][r].add(t)
            self.triple_count += 1

        # === Step 2: Add new triples involving added nodes (no reverse scan)
        for e1 in added_nodes:
            if e1 not in parent.entity_id_:
                continue

            for r, neighbors in parent_triples.get(e1, {}).items():
                for e2 in neighbors:
                    if e2 in self.top_k_nodes:
                        self._try_add_triple((e1, r, e2))
                        if self._limit_reached():
                            return

            # Instead of scanning parent again, use reverse index from parent if it exists
            if hasattr(parent, "reverse_index") and e1 in parent.reverse_index:
                for e2, r in parent.reverse_index[e1]:
                    if e2 in self.top_k_nodes:
                        self._try_add_triple((e2, r, e1))
                        if self._limit_reached():
                            return

        # === Step 3: Padding if needed
        while not self._limit_reached():
            self.add_related_random_triples_batch(batch_size=self.batch_size)






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


def process_summary_sankofa(top_k_nodes_dict, graphs, label, weights, test_data, walks_runtime=None, max_triples=None, query_freq=None, n_priority=None): 
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
    #S = None
    print("Summarizing for SANKOFA")
    for key in top_k_nodes_dict.keys() & graphs.keys() & weights.keys():
        KG = graphs[key]
        #top_k_nodes = top_k_nodes_dict[key]
        top_k_nodes = {str(node): value for node, value in top_k_nodes_dict[key].items()}
        query_freq_ = query_freq.get(key, {})

        print((len(top_k_nodes)), "top_k nodes for timestep", key)



        # Create a summary graph
        time_start = time()
        S = SummaryGraph_(KG, top_k_nodes, max_triples , batch_size=50, query_freq=query_freq_, n_priority = None)##len(top_k_nodes)/100)##, max_triples=500) 10


        # if key == 0:
        #     S = SummaryGraph_(KG, top_k_nodes.keys(), max_triples , batch_size=50, query_freq=query_freq_, n_priority=None)##len(top_k_nodes)/100)##, max_triples=500) 10
        # else:
        #     #S = SummaryGraph_(KG, top_k_nodes.keys(), max_triples = summary_size, batch_size=50)#, query_freq=query_freq_)##len(top_k_nodes)/100)##, max_triples=500) 10
        #     S.set_parent(KG)

        #     S.update_summary(top_k_nodes.keys())#, query_freq=query_freq_)

        time_end = time()
        
        runtime.append({"Timestep": key, "Runtime (summary)": time_end - time_start})
        print(f"Summary graph created for timestep {key} in {time_end - time_start:.2f} seconds")
        triples_no[key] = (S.triple_count)
        log = test_data[key]

        # Compute metrics
        total_F1, total_precision, total_recall = total_query_log_metrics(S, log)
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, log)
        future.append({"Timestep": key, "Total F1": total_F1})

        
        test_weights = weights[key]
        #print("orignal weights", test_weights)

        total = sum(test_weights.values())
        test_weights = {j: test_weights[j] / total for j in test_weights}   
        
        print(test_weights)

        # Process all timesteps
        df = process_all_timesteps(training_set=test_data, summary=S, i=key, test_weights=test_weights)
        weighted_tests.append(df)
       # sliding = sliding_window(window_size=2, training_set=test_data, summary=S, i=key, test_weights=test_weights)
        #print(sliding)

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
    print(sankofa)
    return triples_no, sankofa


def process_summary_glimpse(graphs, triples_no_mc, training_data, test_data, method , weights):
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
    weighted_tests_apex = []
    apex = []
    

    update_time_list = []
    acc_list = []

    for summary_method in summary_methods:
        if summary_method.name_[0:7] == 'GLIMPSE' and summary_method.name_[0:4] != 'APEX':
            print("Summarizing for GLIMPSE")
            for i in graphs.keys() & triples_no_mc.keys() & weights.keys():
                KG = graphs[i]
                K = triples_no_mc[i]


                #test_weights = weights[i]
                test_weights = weights[i]
                total = sum(test_weights.values())
                test_weights = {j: test_weights[j] / total for j in test_weights}   
        

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
                    
                    N=2
                    keys = sorted(training_data.keys())
                    target_index = keys.index(i)

                    start_index = max(0, target_index - N + 1)
                    selected_keys = keys[start_index:target_index + 1]
                    merged_logs = [training_data[key] for key in selected_keys]

                    #merged_logs = [training_data[key] for key in keys[:target_index + 1]]
                    merged_logs = list(flatten_list(merged_logs))

                    t0 = time()
                    S = summary_method(KG, K, merged_logs)
                    runtime.append({"Timestep": i, "Runtime (seconds)": time() - t0})
                    #runtime = time() - t0
                    #print("Runtime:", runtime)
                    #update_time_list.append(runtime)

                    total_F1, total_precision, total_recall = total_query_log_metrics(S, test_data[i])
                    avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, test_data[i])
                    future.append({"Timestep": i, "Total F1": total_F1})
                    #print(avg_F1)

                    df = process_all_timesteps(training_set=test_data, summary=S, i=i, test_weights=test_weights)
                    weighted_tests.append(df)
                    #sliding = sliding_window(window_size=2, training_set=test_data, summary=S, i=i, test_weights=test_weights)
                    #print(sliding)

        elif summary_method.name_ == 'APEX':
            print("Summarizing for APEX")
                
           
            for i in graphs.keys() & triples_no_mc.keys() & weights.keys():
                KG = graphs[i]
                K = triples_no_mc[i]
                training = training_data[i]
                test = test_data[i]
                test_weights = weights[i]

                update_time_list = []
                acc_list = []

                #acc_list, update_time_list, P  = APEX(KG, K, training, test, t=i)
                # Initial run
                if i ==0:
                    total_F1, run_time, P = APEX_(KG, K, training, test, t=i)
                    ##P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(KG, K, training, test, t=i)
                    apex.append({
                        "Timestep": i,
                        "Future Test F1": total_F1,
                        "Runtime (seconds)": run_time
                    })
                    df = process_all_timesteps(training_set=test_data, summary=P, i=i, test_weights=test_weights)
                    weighted_tests_apex.append(df)
                    #sliding = sliding_window(window_size=2, training_set=test_data, summary=S, i=i, test_weights=test_weights)
                    #print(sliding)

                
                else:
                    
                    try:
                        total_F1, run_time, P = APEX_(KG, K, training, test, t=i)
                        # P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(
                        #     KG, K, training, test, t=i,
                        #     H=H, q=q, e=e, r=r,
                        #     csr_indirect_matrices=csr,
                        #     index_list=index_list
                        #)
                    except:
                        #break
                        pass
                    apex.append({
                        "Timestep": i,
                        "Future Test F1": total_F1,
                        "Runtime (seconds)": run_time, 
                    })
                    df = process_all_timesteps(training_set=test_data, summary=P, i=i, test_weights=test_weights)
                    weighted_tests_apex.append(df)
                    #sliding = sliding_window(window_size=2, training_set=test_data, summary=P, i=i, test_weights=test_weights)
                    #print(sliding)


    
    ########clean this up
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

    print("GLIMPSE")
    print(glimpse)

    


    ########clean this up
    apex_df = pd.DataFrame(apex).drop_duplicates(subset='Timestep')
    F1_apex_weighted_tests = pd.concat(weighted_tests_apex, ignore_index=True).rename(columns={"Timestep Window": "Timestep", "Total F1": "Weighted Future Test F1"}).drop_duplicates(subset='Timestep')

    # Ensure 'Timestep' column has the same type in all DataFrames
    apex_df["Timestep"] = apex_df["Timestep"].astype(int)
    F1_apex_weighted_tests["Timestep"] = F1_apex_weighted_tests["Timestep"].astype(int)

    # Merge DataFrames sequentially
    apex_df = pd.merge(apex_df, F1_apex_weighted_tests, on="Timestep")
    apex_df = apex_df[["Timestep", "Runtime (seconds)", "Future Test F1", "Weighted Future Test F1"]]

    sota  = pd.merge(glimpse, apex_df, on="Timestep")
    
    print("APEX")
    print(apex_df)
    #glimpse.to_csv("glimpse_results.csv", index=False)
    apex_df.to_csv("apex_results.csv", index=False)

    return sota

def compute_query_entity_frequencies_by_timestep(query_log):
    """
    query_log: dict of {timestep: {query_id: entity_id, ...}}
    returns: dict of {timestep: Counter({entity_id: frequency, ...})}
    """
    freq_by_timestep = {}
    for timestep, queries in query_log.items():
        timestep_counter = Counter()
        for qid, entity_id in queries.items():
            timestep_counter[entity_id] += 1
        freq_by_timestep[timestep] = timestep_counter
    return freq_by_timestep


def extract_snapshot_number(filename):
    # Extract the number from 'snapshot_1.nt.gz' → 1
    base = os.path.basename(filename)
    num_part = base.replace("snapshot_", "").replace(".nt.gz", "")
    return int(num_part)




def main() :

    #subgraph_dir ='../PKG_EXP/DBpedia/dbpedia_subgraphs_int10'#dbpedia_subgraphs_int7'#
    #
    #subgraph_dir = '../PKG_EXP/DBpedia/dbpedia_subgraphs_intall'#dbpedia_subgraphs_int8'#
    #subgraph_dir =   "/Users/au735195/Downloads/KG_GDELT" 
    subgraph_dir = "/home/ama/wiki_all"#/Users/au735195/Desktop/wiki/snapshots"

    #subgraph_dir ='../PKG_EXP/DBpedia/KG_snapshots'#dbpedia_subgraphs_int8'#
    #subgraph_dir = "/Users/au735195/Downloads/KG_IMDB-13-3SP"
    #subgraph_dir = "/Users/au735195/Downloads/KG_IMDB30"
    #subgraph_dir = "/Users/au735195/Downloads/KG_synthetic"
    #subgraph_dir =   "/Users/au735195/Downloads/KG_snapshots" 
    dynagraph, queries_, graphs, training_data, test_data, triple_size = {}, {}, {}, {}, {}, []

    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
    [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
    key=extract_snapshot_number
        )

    print("Loading all subgraph snapshots...")

    for i, file_name in enumerate(snapshot_files):
        #file = "snapshot_0.nt.gz"
        path = os.path.join(subgraph_dir, file_name)
        print(f"Processing snapshot {i}: {file_name}")
        
        start = time()  

    # print("Loading all subgraph snapshots...")
    # for i, file_name in enumerate(sorted(os.listdir(subgraph_dir))):
    #     start = time()  # Fixed: Use `time()` directly
    #     #file = "categories_1.nt.gz"
    #     path = os.path.join(subgraph_dir, file_name)

        KG = DBPedia(rdf_gz=path)
        KG.load()
        KG.update_user(1)

        G = nx.DiGraph()
        for s, p, o in KG.triples():
            G.add_edge(s, o, predicate=p, weight=0.01)

        # degrees = [G.degree(n) for n in G.nodes()]
        # # Frequency of each degree
        # degree_counts = Counter(degrees)


        # x, y = zip(*sorted(degree_counts.items()))
        # degrees = [d for _, d in G.degree()]
        # fit = powerlaw.Fit(degrees, verbose=False)

        # print(f"Alpha: {fit.alpha}, xmin: {fit.xmin}")
        # R, p = fit.distribution_compare('power_law', 'lognormal')
        # print(f"R={R}, p={p}")

        # Get degrees
        degrees = [d for _, d in G.degree()]

        # Fit power-law
        #fit = powerlaw.Fit(degrees, verbose=False)
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

        print(f"Alpha: {fit.alpha}, xmin: {fit.xmin}")

        # Compare to lognormal
        R, p = fit.distribution_compare('power_law', 'lognormal')
        print(f"R={R}, p={p}")

        # import matplotlib.pyplot as plt
        # from collections import Counter

        # degrees = [d for _, d in G.degree()]
        # degree_counts = Counter(degrees)
        # x, y = zip(*sorted(degree_counts.items()))

        # plt.loglog(x, y, marker='o', linestyle='None')
        # plt.xlabel('Degree')
        # plt.ylabel('Count')
        # plt.title('Degree Distribution')
        # plt.show()





        dynagraph[i] = G
        graphs[i] = KG
        print(KG.number_of_triples(), "triple size")
        print(KG.number_of_entities(), "entity size")
        #print("top k", (KG.number_of_entities())**2/KG.number_of_triples())
        triple_size.append(KG.number_of_triples())
        if i==0:
            k1= (KG.number_of_entities())**2/KG.number_of_triples()
        print(int(k1))


        topic_mids = random.sample(list(KG.triples_.keys()), k = 5)
        valid_topic_mids = [mid for mid in topic_mids if mid in KG.triples_]
        if not valid_topic_mids:
            print(f"No valid topic_mids found in KG for snapshot {i}")
            continue

        query_log = query_log_by_mids(
            KG,
            valid_topic_mids,
            200,
            topic_dist=np.ones(len(valid_topic_mids)) / len(valid_topic_mids),
            shuffle=True,
            random_query_prob=0
        )



        # Define the path to save the query_log
        #query_log_file = f"query_log_snapshotwiki_5___{i}.pkl"#f"query_log_snapshot_110_{i}.pkl"
        query_log_file = f"query_log_snapshotwiki_5_FIXED1.pkl"#f"query_log_snapshot_110_{i}.pkl"
        if os.path.exists(query_log_file):
            # Load the query_log from the file if it exists
            print(f"Loading query_log from {query_log_file}")
            with open(query_log_file, "rb") as f:
                query_log = pickle.load(f)
        else:
            # Generate the query_log if the file does not exist
            print(f"Generating query_log for snapshot {i}")
            query_log = query_log_by_mids(
                KG,
                valid_topic_mids,
                200,
                topic_dist=np.ones(len(valid_topic_mids)) / len(valid_topic_mids),
                shuffle=True,
                random_query_prob=0
            )
            # Save the query_log to the file
            with open(query_log_file, "wb") as f:
                pickle.dump(query_log, f)

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]

        queries_[i] = [q['Parse']['TopicEntityMid'] for q in training_data[i]]

        print(f"Processed snapshot {i}: {file_name} in {time() - start:.2f}s")

        if i == 0:
            summary_size= int(KG.number_of_triples() * 0.005) #0.001, 0.005, 0.01
            K= int(k1/50)#int(KG.number_of_entities()/10)#int(k1)#100#k1#(KG.number_of_entities())**2/KG.number_of_triples()
            #K = len(KG.triples_.keys())#int(1000/(summary_size*100))#int(summary_size*.002)



    # Run evaluation on the summaries
    print("Evaluating summary graphs...")


    dynagraph = {i: v for i, (_, v) in enumerate(sorted(dynagraph.items()))}
    queries_ = {i: v for i, (_, v) in enumerate(sorted(queries_.items()))}
    graphs = {i: v for i, (_, v) in enumerate(sorted(graphs.items()))}
    test_data = {i: v for i, (_, v) in enumerate(sorted(test_data.items()))}
    training_data = {i: v for i, (_, v) in enumerate(sorted(training_data.items()))}


    converted_dynagraph = []
    for timestep, graph in dynagraph.items():
        converted_graph = {}
        for node in graph.nodes:
            neighbors = {}
            for neighbor in graph[node]:
                # Ensure neighbors exist and weights are valid
                if graph[node][neighbor]:
                    neighbors[int(neighbor)] = float(graph[node][neighbor].get("weight", 0.0001))  # Default weight is 0.5
            converted_graph[int(node)] = neighbors
        converted_dynagraph.append(converted_graph)



    # Convert queries to the expected format
    converted_queries = {}
    for timestep, query_list in queries_.items():
        converted_queries[int(timestep)] = {i: int(query) for i, query in enumerate(query_list)}



    print(dynagraph.keys(), "timesteps in dynagraph")



    results = walks.generate_top_k_summary(
    num_timesteps=len(converted_dynagraph),
    num_walks=20,  # Reduce number of walks for testing
    dynamic_graph=converted_dynagraph,
    queries=converted_queries,
    c=5,
    epsilon=0.25,
    k=K,
    decay_rate = 0.8
    )

    queryfreq_ = compute_query_entity_frequencies_by_timestep(converted_queries)


    # Access top_k_cumsum_walks from the results dictionary
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
    runtime_per_timestep = results["runtime_per_timestep"]


   
    runtime_df = pd.DataFrame(
        [{"Timestep": t, "Runtime (walks)": runtime} for t, runtime in runtime_per_timestep.items()]
    ).sort_values(by="Timestep")

    print(summary_size, K, "summary size and k")
    triples_no, sankofa = process_summary_sankofa(top_k_cumsum_walks, graphs, label="Cumsum Walks", weights=weights, test_data=test_data, walks_runtime = runtime_df, max_triples=summary_size, query_freq=queryfreq_, n_priority=k1)
    print(triples_no)
    # #{0: 22434, 1: 22434, 2: 22434, 3: 22434, 4: 22434}
    sankofa.to_csv(f"all_results.csv", index=False)
    sota = process_summary_glimpse(graphs, triples_no, training_data, test_data, method=["glimpse", "apex"], weights=weights)
    print(triple_size)
    
    
    merged_df = pd.merge(sankofa, sota, on="Timestep")
    merged_df.to_csv(f"all_results.csv", index=False)

    # plt.figure(figsize=(10, 6))
    # for column in merged_df.columns[1:]:
    #     plt.plot(merged_df["Timestep"], merged_df[column],  label=column)

    # plt.xlabel("Timestep")
    # plt.ylabel("Value")
    # plt.title("Performance and Runtime over Timesteps")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
plt.show()
if __name__ == "__main__":
    main()
