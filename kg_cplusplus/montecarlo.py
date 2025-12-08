import os
import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
#from time import time
from src.base import KnowledgeGraph
from collections import Counter
import matplotlib.pyplot as plt
import powerlaw
import math
from src.base import KnowledgeGraph, YAGO, DBPedia, Freebase, MetaQA
from src.path import DBPEDIA_DATA_DIR, YAGO_DATA_DIR, METAQA_DATA_DIR, METAQA_DATA_DIR_ABLATION, ENCODING
from src.user import query_log_by_mids, query_log_by_topics
from src.glimpse import SummaryMethod, GLIMPSE
from src.isummary import ISUMMARY
from src.metrics import total_query_log_metrics, average_query_log_metrics
import src.qadgosis.main  as walks
from src.apex import APEX, DETAILED_LOGGING, APEX_
from collections import defaultdict, deque
import os
import pickle
import heapq
import sys
from pathlib import Path
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")







DETAILED_LOGGING = True
DETAILED_PRINTING = False


METHODS = { 
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    'glimpse-2': SummaryMethod(GLIMPSE, 'GLIMPSE-2', power=2),
    'apex': SummaryMethod(APEX, 'APEX'),
    'isummary': SummaryMethod(ISUMMARY, 'ISUMMARY'),

}


import numpy as np
import random






def topk_nodes_by_score(nodes, scores, k):
    """
    Select top-k nodes by score using ONLY argpartition.
    No sorting at all.
    """
    n = len(nodes)
    if k >= n:
        return nodes

    score_arr = np.array([scores[n] for n in nodes], dtype=float)

    # argpartition gives the indices of top-k highest scores
    idx = np.argpartition(-score_arr, k)[:k]

    # return nodes in arbitrary partition order
    return [nodes[i] for i in idx]



class SummaryGraph(KnowledgeGraph):

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
        super().__init__()

        self.parent_ = KG
        self.top_k_nodes = top_k_nodes       
        self.top_k_set = set(top_k_nodes.keys())
        self.max_triples = max_triples

        self.triple_count = 0
        self.triples_set = set()

        # Build reverse index
        self.reverse_index = self._build_reverse_index_topk_only()

        # Compute internal triples (top-k internal structure)
        self.internal_triples = self._compute_internal_triples()

        # Boundary triples will be generated during summary construction
        self._build_summary()


    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def parent(self):
        return self.parent_

    def _limit_reached(self):
        return self.max_triples is not None and self.triple_count >= self.max_triples

    def _try_add_triple(self, triple):
        if triple not in self.triples_set:
            self.triples_set.add(triple)
            self.triple_count += 1
            self.add_triple(triple)


    # ----------------------------------------------------------------------
    # Reverse index for incoming edges
    # ----------------------------------------------------------------------
    def _build_reverse_index_topk_only(self):
        reverse_idx = {}
        triples = self.parent_.triples_
        top_k = self.top_k_set

        for s, rels in triples.items():
            for r, nbrs in rels.items():
                for o in nbrs:
                    if o in top_k:
                        reverse_idx.setdefault(o, []).append((s, r))

        return reverse_idx


    # ----------------------------------------------------------------------
    # INTERNAL TRIPLES (top-k × top-k)
    # ----------------------------------------------------------------------
    def _compute_internal_triples(self):
        parent = self.parent_.triples_
        rev = self.reverse_index
        scores = self.top_k_nodes
        top_k = self.top_k_set
        budget = self.max_triples

        internal = []

        # Collect all internal triples
        for e1 in top_k:
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        if e2 in top_k:
                            internal.append((e1, r, e2))

            if e1 in rev:
                for e2, r in rev[e1]:
                    if e2 in top_k:
                        internal.append((e2, r, e1))

        # If internal fits budget → no ranking needed
        if len(internal) <= budget:
            return internal

        # Else: rank by score(e1) + score(e2) and partial sort
        weights = np.array([scores[t[0]] + scores[t[2]] for t in internal], dtype=float)
        k = budget

        idx = np.argpartition(-weights, k)[:k]
        return [internal[i] for i in idx]


    # ----------------------------------------------------------------------
    # BOUNDARY TRIPLES (Option 1)
    # Node-first expansion: take all triples of each node in top-k order
    # ----------------------------------------------------------------------
    def _boundary_triples_node_level(self):
        parent = self.parent_.triples_
        rev = self.reverse_index
        scores = self.top_k_nodes
        top_k_list = list(self.top_k_set)

        # Sort top-k nodes by score using partial sort
        k_nodes = len(top_k_list)  # use all, just sort them
        ordered_nodes = topk_nodes_by_score(top_k_list, scores, k_nodes)

        for u in ordered_nodes:

            # Outgoing triples
            if u in parent:
                for r, nbrs in parent[u].items():
                    for v in nbrs:
                        yield (u, r, v)

            # Incoming triples
            if u in rev:
                for v, r in rev[u]:
                    yield (v, r, u)


    # ----------------------------------------------------------------------
    # Build summary
    # ----------------------------------------------------------------------
    def _build_summary(self):

        # 1. Add internal triples first
        for t in self.internal_triples:
            if self._limit_reached():
                return
            self._try_add_triple(t)

        # 2. Expand boundary triples by node importance (Option 1)
        print("Adding boundary triples by node importance...")
        for t in self._boundary_triples_node_level():
            if self._limit_reached():
                return
            self._try_add_triple(t)

        # 3. Fallback: include any parent triple if budget remains
        print("Adding fallback triples from parent...")
        if not self._limit_reached():
            for s, rels in self.parent_.triples_.items():
                for r, nbrs in rels.items():
                    for o in nbrs:
                        if self._limit_reached():
                            return
                        self._try_add_triple((s, r, o))


def weighted_topk(items, weights, k):
    """
    Deterministic top-k using partial sort (largest weights).
    """
    if k >= len(items):
        return items

    w = np.asarray(weights, dtype=float)
    idx = np.argpartition(-w, k)[:k]
    return [items[i] for i in idx]


# class _SummaryGraph__(KnowledgeGraph):

#     def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
#         super().__init__()

#         self.parent_ = KG
#         self.top_k_nodes = top_k_nodes               # dict: node → score
#         self.top_k_set = set(top_k_nodes.keys())
#         self.max_triples = max_triples
#         self.batch_size = batch_size

#         self.triple_count = 0
#         self.triples_set = set()

#         # Reverse index restricted to top-k
#         self.reverse_index = self._build_reverse_index_topk_only()

#         # Internal triples (deterministic top weighted)
#         self._topk_neighbors = self._compute_topk_internal_triples()

#         # Boundary candidates (no sampling)
#         cand, w = self._compute_boundary_candidates()

#         remaining = max_triples - self.triple_count
#         self.random_padding_order = weighted_topk(cand, w, remaining)
#         self.random_padding_index = 0

#         self._build_summary()


#     # ---------------------------
#     # Helpers
#     # ---------------------------

#     def parent(self):
#         return self.parent_

#     def _limit_reached(self):
#         return self.max_triples is not None and self.triple_count >= self.max_triples

#     def _try_add_triple(self, triple):
#         if triple not in self.triples_set:
#             self.add_triple(triple)
#             self.triples_set.add(triple)
#             self.triple_count += 1


#     # ---------------------------
#     # Reverse index
#     # ---------------------------

#     def _build_reverse_index_topk_only(self):
#         reverse_idx = {}
#         parent = self.parent_.triples_
#         top_k = self.top_k_set

#         for s, rel_dict in parent.items():
#             for r, nbrs in rel_dict.items():
#                 for o in nbrs:
#                     if o in top_k:
#                         reverse_idx.setdefault(o, []).append((s, r))

#         return reverse_idx


#     # ---------------------------
#     # INTERNAL TRIPLES (top-k × top-k)
#     # ---------------------------

#     def _compute_topk_internal_triples(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         scores = self.top_k_nodes
#         top_k_nodes = self.top_k_set
#         budget = self.max_triples

#         internal = []

#         # Collect internal triples
#         for e1 in top_k_nodes:
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         if e2 in top_k_nodes:
#                             internal.append((e1, r, e2))

#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     if e2 in top_k_nodes:
#                         internal.append((e2, r, e1))

#         # If small, keep all
#         if len(internal) <= budget:
#             return [(0.0, t) for t in internal]

#         # Otherwise: score + partial select budget elements
#         weights = [scores[t[0]] + scores[t[2]] for t in internal]
#         warr = np.asarray(weights)
#         idx = np.argpartition(-warr, budget)[:budget]

#         selected = [(weights[i], internal[i]) for i in idx]
#         return selected


#     # ---------------------------
#     # BOUNDARY TRIPLES
#     # ---------------------------

#     def _compute_boundary_candidates(self):
#         """
#         All (top-k node) boundary triples with weights.
#         Deterministic partial ranking, no sampling.
#         """
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         scores = self.top_k_nodes
#         top_k = self.top_k_set

#         candidates = []
#         weights = []

#         for e1 in top_k:

#             # Outgoing
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         w = scores[e1] + scores.get(e2, 0.0)
#                         candidates.append((e1, r, e2))
#                         weights.append(w)

#             # Incoming
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     w = scores[e1] + scores.get(e2, 0.0)
#                     candidates.append((e2, r, e1))
#                     weights.append(w)

#         return candidates, weights


#     # ---------------------------
#     # Summary builder
#     # ---------------------------

#     def _build_summary(self):

#         # 1. Add internal triples
#         for score, triple in self._topk_neighbors:
#             if self._limit_reached():
#                 break
#             self._try_add_triple(triple)

#         # 2. Add boundary triples in batches
#         while not self._limit_reached():
#             added = self.add_related_random_triples_batch(self.batch_size)
#             if not added:
#                 break

#         # 3. Fallback: add ANY triple from parent
#         if not self._limit_reached():
#             parent_edges = []
#             triples = self.parent_.triples_
#             for s, rels in triples.items():
#                 for r, nbrs in rels.items():
#                     for o in nbrs:
#                         parent_edges.append((s, r, o))

#             while not self._limit_reached() and parent_edges:
#                 t = random.choice(parent_edges)
#                 self._try_add_triple(t)


#     # ---------------------------
#     # Batched addition
#     # ---------------------------

#     def add_related_random_triples_batch(self, batch_size=100):
#         order = self.random_padding_order
#         idx = self.random_padding_index
#         n = len(order)

#         if idx >= n:
#             return False

#         remaining = self.max_triples - self.triple_count
#         k = min(batch_size, remaining)

#         end = min(idx + k, n)
#         batch = order[idx:end]

#         added = False
#         for triple in batch:
#             if self._limit_reached():
#                 break
#             if triple not in self.triples_set:
#                 self._try_add_triple(triple)
#                 added = True

#         self.random_padding_index = end
#         return added



################


# def weighted_topk(items, weights, k):
#     keys = np.empty(len(weights), dtype=float)

#     for i, w in enumerate(weights):
#         if w <= 0:
#             keys[i] = np.inf       
#         else:
#             u = random.random()
#             keys[i] = -math.log(u) / w

#     # get indices of k smallest keys
#     idx = np.argpartition(keys, k)[:k]

#     return [items[i] for i in idx]





# class _SummaryGraph_(KnowledgeGraph):

#     def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
#         super().__init__()

#         self.parent_ = KG
#         self.top_k_nodes = top_k_nodes
#         self.top_k_set = set(top_k_nodes.keys())
#         self.max_triples = max_triples
#         self.batch_size = batch_size
#         # self.global_pr = global_pr
#         # self.global_pr_set = set(global_pr.keys())

#         self.triple_count = 0
#         self.triples_set = set()

#         # Build indexing
#         self.reverse_index = self._build_reverse_index_topk_only()

#         # Internal triples
#         self._topk_neighbors = self._compute_topk_internal_triples()

#         # # Boundary candidates
#         cand, w = self._compute_random_candidates_weighted()

#         # # NEW: weighted sampling WITHOUT replacement
#         # self.random_padding_order = weighted_permutation(cand, w)
#         # self.random_padding_index = 0

#         remaining = self.max_triples - self.triple_count
#         self.random_padding_order = weighted_topk(cand, w, remaining)
#         self.random_padding_index = 0


#         # cand, w = self._compute_random_candidates_weighted()
#         # self.random_padding_order = cand          # a list of triples
#         # self.random_padding_index = 0


#         self._build_summary()


#     def parent(self):
#         return self.parent_

#     def _limit_reached(self):
#         return self.max_triples is not None and self.triple_count >= self.max_triples


#     def _try_add_triple(self, triple):
#         if triple not in self.triples_set:
#             self.add_triple(triple)
#             self.triples_set.add(triple)
#             self.triple_count += 1


#     def _build_reverse_index_topk_only(self):
#         reverse_idx = {}
#         parent = self.parent_.triples_
#         top_k = self.top_k_set

#         for s, rel_dict in parent.items():
#             for r, nbrs in rel_dict.items():
#                 for o in nbrs:
#                     if o in top_k:
#                         reverse_idx.setdefault(o, []).append((s, r))

#         return reverse_idx


#     def _compute_topk_internal_triples(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         top_k_nodes = self.top_k_set
#         scores = self.top_k_nodes
#         budget = self.max_triples

#         internal = []

#         # Collect
#         for e1 in top_k_nodes:
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         if e2 in top_k_nodes:
#                             internal.append((e1, r, e2))

#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     if e2 in top_k_nodes:
#                         internal.append((e2, r, e1))

#         # If small, return all
#         if len(internal) <= budget:
#             return [(0.0, t) for t in internal]

#         # Else score
#         scored = []
#         for (e1, r, e2) in internal:
#             score = scores[e1] + scores[e2]
#             scored.append((score, (e1, r, e2)))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         return scored



#     def _compute_random_candidates_weighted(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         top_k = self.top_k_set
#         scores = self.top_k_nodes

#         candidates = []
#         weights = []

#         for e1 in top_k:

#             # Outgoing
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         w = scores[e1] + scores.get(e2, 0.0)
#                         candidates.append((e1, r, e2))
#                         weights.append(w)

#             # Incoming
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     w = scores[e1] + scores.get(e2, 0.0)
#                     candidates.append((e2, r, e1))
#                     weights.append(w)

#         return candidates, weights




#     def _build_summary(self):

#         # 1. Internal triples
#         for score, triple in self._topk_neighbors:
#             if self._limit_reached():
#                 break
#             self._try_add_triple(triple)

#         # 2.  C_bnd sampling
#         while not self._limit_reached():
#             added = self.add_related_random_triples_batch(self.batch_size)
#             if not added:
#                 break

#         # 3. Fallback: any triple if it doesn't meet the budget
#         if not self._limit_reached():
#             parent_edges = []
#             triples = self.parent_.triples_
#             for s, rels in triples.items():
#                 for r, nbrs in rels.items():
#                     for o in nbrs:
#                         parent_edges.append((s, r, o))

#             while not self._limit_reached() and parent_edges:
#                 t = random.choice(parent_edges)
#                 self._try_add_triple(t)



#     def add_related_random_triples_batch(self, batch_size=100):
#         order = self.random_padding_order
#         idx = self.random_padding_index
#         n = len(order)

#         if idx >= n:
#             return False

#         remaining = self.max_triples - self.triple_count
#         k = min(batch_size, remaining)

#         end = min(idx + k, n)
#         batch = order[idx:end]

#         added = False
#         for triple in batch:
#             if self._limit_reached():
#                 break
#             if triple not in self.triples_set:
#                 self._try_add_triple(triple)
#                 added = True

#         self.random_padding_index = end
#         return added






####################

# class SummaryGraph_(KnowledgeGraph):
#     def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10, query_freq=None,n_priority=None):
#         super().__init__()
#         self.parent_ = KG
#         self.top_k_nodes = (top_k_nodes)
#         self.max_triples = max_triples
#         self.batch_size = batch_size
#         self.triple_count = 0
#         self.triples_set = set()
#         self.n_priority = n_priority
#         self.query_freq = query_freq or {}
#         self.reverse_index = self._build_reverse_index()
#         self._precompute_topk_neighbors()     # for main summary
#         self._precompute_random_candidates()  # for padding
#         self._build_summary()
        

#     def set_parent(self, parent_kg):
#         self._parent = parent_kg

#     def parent(self):
#         return self.parent_

#     def _limit_reached(self):
#         return self.max_triples is not None and self.triple_count >= self.max_triples

#     def _build_reverse_index(self):
#         reverse_idx = {}
#         top_k = self.top_k_nodes
#         triples = self.parent().triples_
#         for e2, rel_dict in triples.items():
#             for r, neighbors in rel_dict.items():
#                 for e1 in neighbors:
#                     if e1 in top_k:
#                         reverse_idx.setdefault(e1, []).append((e2, r))
#         return reverse_idx

#     def _try_add_triple(self, triple):
#         if triple not in self.triples_set:
#             self.add_triple(triple)
#             self.triples_set.add(triple)
#             self.triple_count += 1




#     def compute_triple_score(self, node1, node2, top_k, remaining_nodes):
#         """
#         Compute the score of a triple based on the scores of the two nodes.
#         """
#         score1 = top_k.get(node1, remaining_nodes.get(node1, 0))  # Score of the first node
#         score2 = top_k.get(node2, remaining_nodes.get(node2, 0))  # Score of the second node

#         return score1 + score2  # Example: Sum of the two node scores
    

#     def _precompute_topk_neighbors(self):
#         parent = self.parent().triples_
#         rev = self.reverse_index
#         top_k = self.top_k_nodes
#         top_k_set = set(top_k)

#         flat = {}

#         for e1 in top_k:
#             lst = []

#             # outgoing
#             neighbors_out = parent.get(e1)
#             if neighbors_out:
#                 for r, nbrs in neighbors_out.items():
#                     for e2 in nbrs:
#                         if e2 in top_k_set:
#                             lst.append((e1, r, e2))

#             # incoming
#             neighbors_in = rev.get(e1)
#             if neighbors_in:
#                 for e2, r in neighbors_in:
#                     if e2 in top_k_set:
#                         lst.append((e2, r, e1))

#             flat[e1] = lst

#         self._topk_neighbors = flat


#     def _build_summary(self):
#         limit_reached = self._limit_reached
#         try_add = self._try_add_triple

#         for e1, triples in self._topk_neighbors.items():
#             for triple in triples:
#                 try_add(triple)
#                 if limit_reached():
#                     return

#         # padding
#         while not limit_reached():
#             self.add_related_random_triples_batch(batch_size=self.batch_size)



#     def _precompute_random_candidates(self):
#         parent = self.parent().triples_
#         reverse_index = self.reverse_index
#         top_k = self.top_k_nodes  # should be a set

#         C = []

#         # outgoing triples from top-k
#         for e1 in top_k:
#             rels = parent.get(e1)
#             if rels:
#                 for r, nbrs in rels.items():
#                     for e2 in nbrs:
#                         C.append((e1, r, e2))

#         # incoming triples into top-k
#         for e1 in top_k:
#             pairs = reverse_index.get(e1)
#             if pairs:
#                 for e2, r in pairs:
#                     C.append((e2, r, e1))

#         self._random_candidates = C



#     def add_related_random_triples_batch(self, batch_size=100):
#         try_add = self._try_add_triple
#         limit_reached = self._limit_reached
#         C = self._random_candidates

#         remaining = self.max_triples - self.triple_count
#         batch_size = min(batch_size, remaining)

#         # Get all e.g., 100 random triples in ONE go
#         samples = random.choices(C, k=batch_size)

#         for t in samples:
#             if limit_reached():
#                 return
#             try_add(t)










    # def _build_summary(self):
    #     parent_triples = self.parent().triples_
    #     entity_ids = self.parent().entity_id_
    #     top_k = self.top_k_nodes

    #     for e1 in top_k:
    #         if e1 not in entity_ids:
    #             continue

    #         for r, neighbors in parent_triples.get(e1, {}).items():
    #             for e2 in neighbors:
    #                 if e2 in top_k:
    #                     self._try_add_triple((e1, r, e2))
    #                     if self._limit_reached():
    #                         return

    #         for e2, r in self.reverse_index.get(e1, []):
    #             if e2 in top_k:
    #                 self._try_add_triple((e2, r, e1))
    #                 if self._limit_reached():
    #                     return

    #     # Pad with related triples if needed
    #     #padding summary
    #     #print("Padding summary with related triples")
    #     while not self._limit_reached():
    #         self.add_related_random_triples_batch(batch_size=self.batch_size)


    # def add_related_random_triples_batch(self, batch_size=100):
    #     parent_triples = self.parent().triples_
    #     reverse_idx = self.reverse_index
    #     top_k = list(self.top_k_nodes)

    #     remaining = self.max_triples - self.triple_count
    #     batch_size = min(batch_size, remaining)

    #     candidates = []
    #     for _ in range(batch_size):
    #         e1 = random.choice(top_k)

    #         # Outgoing
    #         if e1 in parent_triples:
    #             r_list = list(parent_triples[e1].keys())
    #             if r_list:
    #                 r = random.choice(r_list)
    #                 e2_list = list(parent_triples[e1][r])
    #                 if e2_list:
    #                     e2 = random.choice(e2_list)
    #                     candidates.append((e1, r, e2))

    #         # Incoming
    #         if e1 in reverse_idx:
    #             e2_r_list = reverse_idx[e1]
    #             if e2_r_list:
    #                 e2, r = random.choice(e2_r_list)
    #                 candidates.append((e2, r, e1))

    #     for triple in candidates:
    #         if self._limit_reached():
    #             break
    #         self._try_add_triple(triple)




# class SummaryGraph(KnowledgeGraph):
#     def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
#         super().__init__()
#         self.parent_ = KG                    
#         self.top_k_nodes = set(top_k_nodes)  
#         self.max_triples = max_triples
#         self.batch_size = batch_size

#         self.triple_count = 0
#         self.triples_set = set()

#         # Build initial indexing
#         #self.reverse_index = self._build_reverse_index(self.parent_.triples_)
#         self.reverse_index = self._build_reverse_index()
#         self._topk_neighbors = self._compute_topk_neighbors(self.parent_.triples_)
#         self._random_candidates = self._compute_random_candidates()


#         self._build_summary()


#     def parent(self):
#         return self.parent_

#     def _limit_reached(self):
#         return self.max_triples is not None and self.triple_count >= self.max_triples


#     def _try_add_triple(self, triple):
#         if triple not in self.triples_set:
#             self.add_triple(triple)
#             self.triples_set.add(triple)
#             self.triple_count += 1

   
#     # def _build_reverse_index(self, triples):
#     #     reverse_idx = {}
#     #     for s, rels in triples.items():
#     #         for r, nbrs in rels.items():
#     #             for o in nbrs:
#     #                 reverse_idx.setdefault(o, []).append((s, r))
#     #     return reverse_idx
#     def _build_reverse_index(self):
#         reverse_idx = {}
#         top_k = self.top_k_nodes
#         triples = self.parent().triples_
#         for e2, rel_dict in triples.items():
#             for r, neighbors in rel_dict.items():
#                 for e1 in neighbors:
#                     if e1 in top_k:
#                         reverse_idx.setdefault(e1, []).append((e2, r))
#         return reverse_idx

#     def _compute_topk_neighbors(self, triples):
#         top_k = self.top_k_nodes
#         rev = self.reverse_index

#         res = {}
#         for e1 in top_k:
#             lst = []

#             # outgoing
#             if e1 in triples:
#                 for r, nbrs in triples[e1].items():
#                     for e2 in nbrs:
#                         if e2 in top_k:
#                             lst.append((e1, r, e2))

#             # incoming
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     if e2 in top_k:
#                         lst.append((e2, r, e1))

#             res[e1] = lst

#         return res


#     def _compute_random_candidates(self):
#         C = []
#         top_k = self.top_k_nodes
#         parent = self.parent_.triples_
#         rev = self.reverse_index

#         # outgoing from top-k
#         for e1 in top_k:
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         C.append((e1, r, e2))

#         # incoming into top-k
#         for e1 in top_k:
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     C.append((e2, r, e1))

#         return C

#     # def _build_summary(self):
#     #     for e1, lst in self._topk_neighbors.items():
#     #         for triple in lst:
#     #             if self._limit_reached():
#     #                 return
#     #             self._try_add_triple(triple)

#     #     # padding
#     #     while not self._limit_reached():
#     #         self.add_related_random_triples_batch(self.batch_size)



#     # def add_related_random_triples_batch(self, batch_size=100):
#     #     C = self._random_candidates
#     #     remaining = self.max_triples - self.triple_count
#     #     n = min(batch_size, remaining)

#     #     samples = random.choices(C, k=n)
#     #     for t in samples:
#     #         if self._limit_reached():
#     #             break
#     #         self._try_add_triple(t)


#     def _build_summary(self):
#         # 1. Add all internal top-k edges greedily
#         for e1, lst in self._topk_neighbors.items():
#             for triple in lst:
#                 if self._limit_reached():
#                     return
#                 self._try_add_triple(triple)

#         # 2. Padding with top-k related random edges
#         while not self._limit_reached():
#             added = self.add_related_random_triples_batch(self.batch_size)
#             if not added:
#                 break  # candidate pool exhausted

#         # 3. Fallback: allow any edge from the parent graph if budget not reached
#         if not self._limit_reached():
#             parent_edges = []
#             triples = self.parent_.triples_
#             for s, rels in triples.items():
#                 for r, nbrs in rels.items():
#                     for o in nbrs:
#                         parent_edges.append((s, r, o))

#             while not self._limit_reached() and parent_edges:
#                 t = random.choice(parent_edges)
#                 self._try_add_triple(t)


#     def add_related_random_triples_batch(self, batch_size=100):
#         """
#         Sample random edges involving top-k nodes for padding.
#         Returns True if at least one triple was added, False otherwise.
#         """
#         C = self._random_candidates
#         if not C:
#             return False  # nothing to sample

#         remaining = self.max_triples - self.triple_count
#         n = min(batch_size, remaining)

#         added_any = False
#         samples = random.choices(C, k=n)
#         for t in samples:
#             if self._limit_reached():
#                 break
#             if t not in self.triples_set:
#                 self._try_add_triple(t)
#                 added_any = True

#         return added_any


# class SummaryGraph__(KnowledgeGraph):

#     def __init__(self, KG, top_k_nodes, global_pr,max_triples=None, batch_size=10):
#         """
#         top_k_nodes : dict {node: importance_score}
#         """
#         super().__init__()

#         self.parent_ = KG
#         self.top_k_nodes = top_k_nodes                     
#         self.top_k_set = set(top_k_nodes.keys())         
#         self.max_triples = max_triples
#         self.batch_size = batch_size
#         self.global_pr = global_pr
#         self.global_pr_set = set(global_pr.keys())

#         self.triple_count = 0
#         self.triples_set = set()

#         # Build indexing
#         self.reverse_index = self._build_reverse_index_topk_only()
#         self._topk_neighbors = self._compute_topk_internal_triples()
#         self._random_candidates = self._compute_random_candidates_weighted()

#         self._build_summary()


#     def parent(self):
#         return self.parent_

#     def _limit_reached(self):
#         return self.max_triples is not None and self.triple_count >= self.max_triples

#     def _try_add_triple(self, triple):
#         if triple not in self.triples_set:
#             self.add_triple(triple)
#             self.triples_set.add(triple)
#             self.triple_count += 1



#     def _build_reverse_index_topk_only(self):
#         reverse_idx = {}
#         parent = self.parent_.triples_
#         top_k = self.top_k_set

#         for s, rel_dict in parent.items():
#             for r, nbrs in rel_dict.items():
#                 for o in nbrs:
#                     if o in top_k:
#                         reverse_idx.setdefault(o, []).append((s, r))

#         return reverse_idx



#     # def _compute_topk_internal_triples(self):
#     #     parent = self.parent_.triples_
#     #     rev = self.reverse_index
#     #     top_k = self.top_k_set
#     #     scores = self.top_k_nodes     # <-- direct scoring source

#     #     scored_triples = []

#     #     for e1 in top_k:

#     #         # outgoing
#     #         if e1 in parent:
#     #             for r, nbrs in parent[e1].items():
#     #                 for e2 in nbrs:
#     #                     if e2 in top_k:
#     #                         score = scores[e1] + scores[e2]
#     #                         scored_triples.append((score, (e1, r, e2)))

#     #         # incoming
#     #         if e1 in rev:
#     #             for e2, r in rev[e1]:
#     #                 if e2 in top_k:
#     #                     score = scores[e1] + scores[e2]
#     #                     scored_triples.append((score, (e2, r, e1)))

#     #     scored_triples.sort(key=lambda x: x[0], reverse=True)
#     #     return scored_triples
    


#     def _compute_topk_internal_triples(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         top_k_nodes = self.top_k_set
#         scores = self.top_k_nodes        # scoring dictionary
#         budget = self.max_triples               # total number of triples allowed

#         internal_triples = []            # raw list without scoring

#         # ---------------------------------------------------------
#         # 1. COLLECT all internal triples unscored
#         # ---------------------------------------------------------
#         for e1 in top_k_nodes:

#             # outgoing edges
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         if e2 in top_k_nodes:
#                             internal_triples.append((e1, r, e2))

#             # incoming edges
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     if e2 in top_k_nodes:
#                         internal_triples.append((e2, r, e1))

#         # ---------------------------------------------------------
#         # 2. If internal triples < budget → just return them all
#         #    (NO scoring required)
#         # ---------------------------------------------------------
#         if len(internal_triples) <= budget:
#             # wrap them in dummy score format expected downstream:
#             return [(0.0, triple) for triple in internal_triples]

#         # ---------------------------------------------------------
#         # 3. Otherwise: score each triple and return top ones
#         # ---------------------------------------------------------
#         scored_triples = []
#         for (e1, r, e2) in internal_triples:
#             score = scores[e1] + scores[e2]
#             scored_triples.append((score, (e1, r, e2)))

#         scored_triples.sort(key=lambda x: x[0], reverse=True)
#         return scored_triples




#     def _compute_random_candidates_weighted(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         top_k = self.top_k_set
#         scores = self.top_k_nodes
#         global_pr = self.global_pr_set
#         global_score = self.global_pr

#         candidates = []
#         weights = []
#         boundary_triples = []

#         for e1 in top_k:

#             # outgoing
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         w = scores[e1] + scores.get(e2, 0.0)
#                         candidates.append((e1, r, e2))
#                         weights.append(w)
#                         boundary_triples.append((w, (e1, r, e2)))

#             # incoming
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     w = scores[e1] + scores.get(e2, 0.0)
#                     candidates.append((e2, r, e1))
#                     weights.append(w)
#                     boundary_triples.append((w, (e1, r, e2)))

#         return candidates, weights#sorted(boundary_triples, key=lambda x: x[0], reverse=True)#


#     def _build_summary(self):

#         # Greedy internal triples (importance-sorted)
#         for score, triple in self._topk_neighbors:
#             if self._limit_reached():
#                 break
#             self._try_add_triple(triple)

#         # Weighted random padding
#         while not self._limit_reached():
#             added = self.add_related_random_triples_batch(self.batch_size)
#             if not added:
#                 break
#         # for score, triple in self._random_candidates:
#         #     if self._limit_reached():
#         #         break
#         #     self._try_add_triple(triple)

#         # 3. Fallback: allow any edge from the parent graph if budget not reached
#         if not self._limit_reached():
#             parent_edges = []
#             triples = self.parent_.triples_
#             for s, rels in triples.items():
#                 for r, nbrs in rels.items():
#                     for o in nbrs:
#                         parent_edges.append((s, r, o))

#             while not self._limit_reached() and parent_edges:
#                 t = random.choice(parent_edges)
#                 self._try_add_triple(t)


#     def add_related_random_triples_batch(self, batch_size=100):
#         candidates, weights = self._random_candidates

#         if not candidates:
#             return False

#         remaining = self.max_triples - self.triple_count
#         k = min(batch_size, remaining)

#         sampled = random.choices(candidates, weights=weights, k=k)

#         added = False
#         for t in sampled:
#             if self._limit_reached():
#                 break
#             if t not in self.triples_set:
#                 self._try_add_triple(t)
#                 added = True

#         return added



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


##########################


def _normalize_weights(w):
    """Normalize a weight dict to sum to 1; handle zeros safely."""
    if not w:
        return {}
    tot = sum(w.values())
    if tot == 0:
        return {k: 0.0 for k in w}
    return {k: v / tot for k, v in w.items()}


def _safe_concat_df(list_of_dfs, rename_from="Timestep Window", rename_to="Timestep", value_name="Total F1"):
    """Concatenate small dataframes from process_all_timesteps; return empty df if none."""
    if not list_of_dfs:
        return pd.DataFrame(columns=[rename_to, value_name])
    df = pd.concat(list_of_dfs, ignore_index=True)
    if rename_from in df.columns:
        df = df.rename(columns={rename_from: rename_to})
    # Ensure columns exist
    if rename_to not in df.columns:
        df[rename_to] = pd.NA
    if value_name not in df.columns:
        df[value_name] = pd.NA
    # Drop duplicates by timestep
    if rename_to in df.columns:
        df = df.drop_duplicates(subset=rename_to)
    return df




def process_summary_sankofa(top_k_nodes_dict,  graphs, label, weights, test_data,
                            walks_runtime=None, max_triples=None,  n_priority=None):
    """
    Fully restored Sankofa processing:
    - Computes Future Test F1
    - Computes Weighted Future Test F1 (never NA)
    - Computes runtime (walks + summary)
    """

    triples_no = {}
    future_rows = []
    weighted_tests = []
    summary_runtime_rows = []

    print("Summarizing for SANKOFA")

    common_keys = top_k_nodes_dict.keys() & graphs.keys() & weights.keys() & test_data.keys()

    for key in sorted(common_keys):

        KG = graphs[key]
        top_k_nodes = {str(n): v for n, v in top_k_nodes_dict[key].items()}
        #global_pr = {str(n): v for n, v in global_pr_dict[key].items()}

        print(f"{len(top_k_nodes)} top_k nodes for timestep {key}")

        # -------- SUMMARY + RUNTIME --------
        t0 = time.time()

        S =SummaryGraph(KG, top_k_nodes, max_triples,
                              batch_size=50)

        t1 = time.time()

        summary_runtime = t1 - t0
        summary_runtime_rows.append({"Timestep": key, "Runtime (summary)": summary_runtime})
        triples_no[key] = S.triple_count

        print(f"Summary graph for t={key} created in {summary_runtime:.2f}s")

        # PRESENT TEST F1
        log = test_data[key]
        total_F1, total_precision, total_recall = total_query_log_metrics(S, log)

        future_rows.append({"Timestep": key, "Future Test F1": total_F1})

        # HISTORICAL TESTS 
        test_weights = weights[key]
        total_w = sum(test_weights.values())
        test_weights_norm = {j: test_weights[j] / total_w for j in test_weights}

        df_weight = process_all_timesteps(
            training_set=test_data,
            summary=S,
            i=key,
            test_weights=test_weights_norm
        )
        weighted_tests.append(df_weight)

        sliding = sliding_window(window_size=2, training_set=test_data,
                                 summary=S, i=key, test_weights=test_weights_norm)
        print(sliding)

    # BUILD DATAFRAMES 
    df_future = pd.DataFrame(future_rows)
    df_summary_rt = pd.DataFrame(summary_runtime_rows)

    # Weighted Future Test F1
    if len(weighted_tests) > 0:
        df_weighted = pd.concat(weighted_tests, ignore_index=True)
        df_weighted = df_weighted.rename(columns={
            "Timestep Window": "Timestep",
            "Total F1": "Weighted Future Test F1"
        })
    else:
        df_weighted = pd.DataFrame(columns=["Timestep", "Weighted Future Test F1"])

    # FIX dtypes 
    for df in (df_future, df_summary_rt, df_weighted):
        if not df.empty:
            df["Timestep"] = df["Timestep"].astype(int)

    #  WALKS RUNTIME 
    if walks_runtime is None:
        walks_runtime = pd.DataFrame({"Timestep": sorted(common_keys),
                                      "Runtime (walks)": [0.0]*len(common_keys)})

    if "Runtime (walks)" not in walks_runtime.columns:
        walks_runtime["Runtime (walks)"] = 0.0

    # MERGE 
    merged = walks_runtime.merge(df_summary_rt, on="Timestep", how="right")
    merged = merged.merge(df_future, on="Timestep", how="left")
    merged = merged.merge(df_weighted, on="Timestep", how="left")

    # INCASE THERE IS A MISSING HISTORICAL TEST
    merged["Weighted Future Test F1"] = merged["Weighted Future Test F1"].fillna(0.0)

    # TOTAL RUNTIME 
    merged["Runtime (seconds)"] = (
        merged["Runtime (summary)"].fillna(0.0) +
        merged["Runtime (walks)"].fillna(0.0)
    )

    # FINAL FORMAT 
    sankofa = merged[[
        "Timestep",
        "Runtime (seconds)",
        "Future Test F1",
        "Weighted Future Test F1"
    ]].copy()

    sankofa.to_csv("sankofa_results.csv", index=False)
    print(sankofa)

    return triples_no, sankofa



# def process_summary_sankofa_(top_k_nodes_dict, graphs, label, weights, test_data,
#                             walks_runtime=None, max_triples=None, query_freq=None, n_priority=None):
#     triples_no = {}
#     sankofa_rows = []
#     weighted_tests = []
#     query_freq = query_freq or {}
#     common_keys = top_k_nodes_dict.keys() & graphs.keys() & weights.keys() & test_data.keys()

#     print("Summarizing for SANKOFA")

#     for key in sorted(common_keys):
#         KG = graphs[key]
#         top_k_nodes = {str(n): v for n, v in top_k_nodes_dict[key].items()}
#         t0 = time.time()
#         S = SummaryGraph_(KG, top_k_nodes, max_triples, batch_size=50,
#                           query_freq=query_freq.get(key, {}), n_priority=n_priority)
#         summary_runtime = time.time() - t0
#         triples_no[key] = S.triple_count

#         total_F1, _, _ = total_query_log_metrics(S, test_data[key])
#         sankofa_rows.append({"Timestep": key, "Future Test F1": total_F1, "Runtime (summary)": summary_runtime})

#         weighted_tests.append(process_all_timesteps(
#             training_set=test_data, summary=S, i=key,
#             test_weights={j: w / sum(weights[key].values()) for j, w in weights[key].items()}
#         ))

#         print(sliding_window(window_size=2, training_set=test_data, summary=S, i=key,
#                              test_weights={j: w / sum(weights[key].values()) for j, w in weights[key].items()}))

#     df_sankofa_rows = pd.DataFrame(sankofa_rows)
#     df_weighted = pd.concat(weighted_tests, ignore_index=True).rename(
#         columns={"Timestep Window": "Timestep", "Total F1": "Weighted Future Test F1"}
#     )

#     walks_runtime = walks_runtime if walks_runtime is not None else pd.DataFrame({"Timestep": sorted(df_sankofa_rows["Timestep"]), "Runtime (walks)": 0.0})
#     sankofa = walks_runtime.merge(df_sankofa_rows, on="Timestep", how="right").merge(df_weighted, on="Timestep", how="left")
#     sankofa["Weighted Future Test F1"] = sankofa["Weighted Future Test F1"].fillna(0.0)
#     sankofa["Runtime (seconds)"] = sankofa["Runtime (summary)"].fillna(0.0) + sankofa["Runtime (walks)"].fillna(0.0)

#     sankofa.to_csv("sankofa_results.csv", index=False)
#     print(sankofa)
#     return triples_no, sankofa


# --------------------- process_summary_glimpse  ---------------------
# 
def run_glimpse(summary_method, graphs, triples_no_mc, training_data, test_data, weights):
    glimpse_rows = []
    weighted_tests = []

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]
        norm_w = _normalize_weights(weights[t])

        # build merged logs (window N=2 as original)
        if t == 0:
            merged_logs = training_data[t]
        else:
            N = 2
            keys = sorted(training_data.keys())
            idx = keys.index(t)
            start = max(0, idx - N + 1)
            selected = keys[start: idx + 1]
            merged_logs = list(flatten_list([training_data[k] for k in selected]))

        t0 = time.time()
        S = summary_method(KG, K, merged_logs)
        run_time = time.time() - t0

        total_F1, _, _ = total_query_log_metrics(S, test_data[t])
        glimpse_rows.append({"Timestep": t, "Glimpse Present F1": total_F1, "Runtime (seconds)": run_time})

        df_w = process_all_timesteps(training_set=test_data, summary=S, i=t, test_weights=norm_w)
        weighted_tests.append(df_w)

        sliding = sliding_window(window_size=2, training_set=test_data, summary=S, i=t, test_weights=norm_w)
        print(sliding)

    return glimpse_rows, weighted_tests


# --------------------- process_summary_APEX  ---------------------
def run_apex(graphs, triples_no_mc, training_data, test_data, weights):
    apex_rows = []
    weighted_tests = []

    H = q = e = r = csr = index_list = None

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]
        test_w = weights[t]  # APEX part of your code used raw weights for process_all_timesteps

        # if t == 0:
        #     P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(KG, K, training_data[t], test_data[t], t=t)
        # else:
        #     try:
        #         P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(
        #             KG, K, training_data[t], test_data[t],
        #             t=t, H=H, q=q, e=e, r=r,
        #             csr_indirect_matrices=csr, index_list=index_list
        #         )
        #     except Exception:
        #         pass
                # fallback to full call if incremental fails
                #P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(KG, K, training_data[t], test_data[t], t=t)
        total_F1, run_time, P = APEX_(KG, K,  training_data[t], test_data[t], t=t)

        apex_rows.append({"Timestep": t, "Apex Present F1": total_F1, "Runtime (seconds)": run_time})
        df_w = process_all_timesteps(training_set=test_data, summary=P, i=t, test_weights=_normalize_weights(test_w))
        weighted_tests.append(df_w)

        sliding = sliding_window(window_size=2, training_set=test_data, summary=P, i=t, test_weights=_normalize_weights(test_w))
        print(sliding)

    return apex_rows, weighted_tests


# --------------------- process_summary_ISUMMARY  ---------------------
def run_isummary(summary_method, graphs, triples_no_mc, training_data, test_data, weights):
    isummary_rows = []
    weighted_tests = []

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]
        norm_w = _normalize_weights(weights[t])

        if t == 0:
            t0 = time.time()
            S = ISUMMARY(KG, K, training_data[t])
            run_time = time.time() - t0
        else:
            N = 2
            keys = sorted(training_data.keys())
            idx = keys.index(t)
            start = max(0, idx - N + 1)
            selected = keys[start: idx + 1]
            merged_logs = list(flatten_list([training_data[k] for k in selected]))

            t0 = time.time()
            S = summary_method(KG, K, merged_logs)
            run_time = time.time() - t0

        total_F1, _, _ = total_query_log_metrics(S, test_data[t])
        isummary_rows.append({"Timestep": t, "Isummary Present F1": total_F1, "Runtime (seconds)": run_time})

        df_w = process_all_timesteps(training_set=test_data, summary=S, i=t, test_weights=norm_w)
        weighted_tests.append(df_w)

        sliding = sliding_window(window_size=2, training_set=test_data, summary=S, i=t, test_weights=norm_w)
        print(sliding)

    return isummary_rows, weighted_tests


# Merge present and weighted results for each method
def _merge_present_and_weighted(present_rows, weighted_dfs, present_col_name, weighted_col_name, rename_weighted="Total F1"):
    present_df = pd.DataFrame(present_rows).drop_duplicates(subset="Timestep") if present_rows else pd.DataFrame(columns=["Timestep", "Runtime (seconds)", present_col_name])
    weighted_df = _safe_concat_df(weighted_dfs, rename_from="Timestep Window", rename_to="Timestep", value_name=rename_weighted)
    if not weighted_df.empty and rename_weighted in weighted_df.columns:
        weighted_df = weighted_df.rename(columns={rename_weighted: weighted_col_name})
   
    if not present_df.empty:
        present_df["Timestep"] = present_df["Timestep"].astype(int)
    if not weighted_df.empty and "Timestep" in weighted_df.columns:
        weighted_df["Timestep"] = weighted_df["Timestep"].astype(int)

    if not present_df.empty and not weighted_df.empty:
        merged = pd.merge(present_df, weighted_df, on="Timestep")
    else:
        merged = present_df if not present_df.empty else weighted_df
    return merged




def process_summary_baselines(graphs, triples_no_mc, training_data, test_data, methods, weights):
    """
    Orchestrates GLIMPSE / APEX / ISUMMARY runs and merges results.
    Returns: DataFrame (sota) with columns for each method.
    """

    summary_methods = [METHODS[name] for name in methods]

    # storage for raw method outputs
    raw_results = {
        "Glimpse": ([], []),
        "Apex": ([], []),
        "Isummary": ([], []),
    }

    # --- RUN METHODS ---
    for summary_method in summary_methods:
        name = summary_method.name_

        if name.startswith("GLIMPSE") and not name.startswith("APEX"):
            print("Summarizing for GLIMPSE")
            rows, weighted = run_glimpse(summary_method, graphs, triples_no_mc, training_data, test_data, weights)
            raw_results["Glimpse"][0].extend(rows)
            raw_results["Glimpse"][1].extend(weighted)

        elif name == "APEX":
            print("Summarizing for APEX")
            rows, weighted = run_apex(graphs, triples_no_mc, training_data, test_data, weights)
            raw_results["Apex"][0].extend(rows)
            raw_results["Apex"][1].extend(weighted)

        elif name == "ISUMMARY":
            print("Summarizing for ISUMMARY")
            rows, weighted = run_isummary(summary_method, graphs, triples_no_mc, training_data, test_data, weights)
            raw_results["Isummary"][0].extend(rows)
            raw_results["Isummary"][1].extend(weighted)

 
    merged_results = {}
    for method, (present, weighted) in raw_results.items():
        if not present:   
            continue

        merged = _merge_present_and_weighted(
            present,
            weighted,
            f"{method} Present F1",
            f"Weighted {method} F1"
        )

        if merged is None or merged.empty:
            continue


        merged.rename(columns={"Runtime (seconds)": f"{method} Runtime (seconds)"}, inplace=True)

        print(f"{method.upper()}\n", merged)
        merged.to_csv(f"{method.lower()}_results.csv", index=False)

        merged_results[method] = merged

    # --- FINAL MERGE ---
    dfs = list(merged_results.values())
    if not dfs:
        print("No results produced by any summary method.")
        return pd.DataFrame()

    sota = dfs[0]
    for df in dfs[1:]:
        sota = pd.merge(sota, df, on="Timestep", how="inner")

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

def load_snapshots(subgraph_dir):
    """Load all snapshots from the specified directory."""
    graphs = {}
    
    for i, file_name in enumerate(sorted(os.listdir(subgraph_dir))):
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()
        graphs[i] = KG
    return graphs

def analyze_graph_properties(graph, timestep):
    """Analyze and print properties of the graph."""
    G = nx.DiGraph()
    for s, p, o in graph.triples():
        G.add_edge(s, o, predicate=p, weight=0.01)
    
    

    degrees = [d for _, d in G.degree()]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    #print(f"Timestep {timestep}: Alpha={fit.alpha}, xmin={fit.xmin}")
    R, p = fit.distribution_compare('power_law', 'lognormal')
    #print(f"Timestep {timestep}: R={R}, p={p}")
    return G

def generate_query_logs(graph, timestep, num_queries=200):
    """Generate query logs for a given graph."""
    topic_mids = random.sample(list(graph.triples_.keys()), k=5)
    valid_topic_mids = [mid for mid in topic_mids if mid in graph.triples_]
    if not valid_topic_mids:
        print(f"No valid topic_mids found in KG for snapshot {timestep}")
        return None

    query_log = query_log_by_mids(
        graph,
        valid_topic_mids,
        num_queries,
        topic_dist=np.ones(len(valid_topic_mids)) / len(valid_topic_mids),
        shuffle=True,
        random_query_prob=0
    )
    return query_log

def save_or_load_query_log(query_log_file, graph, timestep):
    """Save or load query logs from a file."""
    if os.path.exists(query_log_file):
        print(f"Loading query_log from {query_log_file}")
        with open(query_log_file, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Generating query_log for snapshot {timestep}")
        query_log = generate_query_logs(graph, timestep)
        with open(query_log_file, "wb") as f:
            pickle.dump(query_log, f)
        return query_log

def process_snapshots(subgraph_dir):
    """Process all snapshots and generate graphs, query logs, and training/test data."""
    graphs, training_data, test_data, queries_ = {}, {}, {}, {}

    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
    [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
    key=extract_snapshot_number
        )

    print("Loading all subgraph snapshots...")

    for i, file_name in enumerate(snapshot_files):
    
    #for i, file_name in enumerate(sorted(os.listdir(subgraph_dir))):
        #file = "snapshot_1.nt.gz"
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()
        graphs[i] = KG

        # print(KG.number_of_triples(), "triple size")
        # print(KG.number_of_entities(), "entity size")

        query_log_file = f"query_log_snapshot_{i}.pkl"
        #query_log_file = "query_log_snapshot_100.pkl"
        query_log = save_or_load_query_log(query_log_file, KG, i)
        if not query_log:
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]
        queries_[i] = [q['Parse']['TopicEntityMid'] for q in training_data[i]]

    return graphs, training_data, test_data, queries_

def main():
    # Define the subgraph directory
    dynagraph = {}  # Initialize as a dictionary
    subgraph_dir ="/Users/au735195/Downloads/KG_IMDB30"#"/Users/au735195/Desktop/dbp"##"#../PKG_EXP/DBpedia/dbpedia_subgraphs_intall" # #" "/Users/au735195/Downloads/KG_snapshots"#
 
    # Load and process snapshots
    print("Loading all subgraph snapshots...")
    graphs, training_data, test_data, queries_ = process_snapshots(subgraph_dir)

    # Analyze graph properties
    for i, graph in graphs.items():
        G = analyze_graph_properties(graph, i)

        dynagraph[i] = G  

    # Convert graphs and queries for processing
    converted_dynagraph = []
    for i, (timestep, graph) in enumerate(dynagraph.items()):
        #print(f"Converting graph at timestep {timestep}...")
        if i == 0:
            # Calculate summary size and K for the first iteration
            b = 0.0005
            summary_size = int(len(graph.edges) * b)  # Adjust the multiplier as needed
            k1 = (len(graph.nodes))**2 / len(graph.edges)
            K = int(k1 / (0.1/b)) 
            print(f"Summary size: {summary_size}, K: {K}, k1: {k1}, DENSITY: {len(graph.edges)/ (len(graph.nodes)*(len(graph.nodes)-1))}")

        # Convert the graph to the required format
        converted_graph = {
            int(node): {int(neighbor): float(graph[node][neighbor].get("weight", 0.0001))  # Default weight is 0.0001
                        for neighbor in graph.neighbors(node)}
            for node in graph.nodes
        }
        converted_dynagraph.append(converted_graph)


    converted_queries = {
        int(timestep): {int(i): int(query) for i, query in enumerate(query_list)}
        for timestep, query_list in queries_.items()
    }

    # Run evaluation on the summaries
    #print("Evaluating summary graphs...")

    # Generate top-k summaries
    results = walks.generate_top_k_summary(
        num_timesteps=len(converted_dynagraph),
        num_walks=5,
        dynamic_graph=converted_dynagraph,
        queries=converted_queries,
        c=2,
        epsilon=0.25,
        k=K,
        decay_rate = 0.03
    )

    # Process summaries and evaluate
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
   
    runtime_per_timestep = results["runtime_per_timestep"]
    #global_pr = results["global_pr"]

    runtime_df = pd.DataFrame(
        [{"Timestep": t, "Runtime (walks)": runtime} for t, runtime in runtime_per_timestep.items()]
    ).sort_values(by="Timestep")


    triples_no, sankofa = process_summary_sankofa(
        top_k_cumsum_walks,  graphs, label="Cumsum Walks", weights=weights,
        test_data=test_data, walks_runtime=runtime_df, max_triples=summary_size, n_priority=50
    )
    sankofa.to_csv("sankofa_results.csv", index=False)
    print(triples_no)
    #print(weights)

    # sota = process_summary_baselines(
    #     graphs, triples_no, training_data, test_data,
    #     methods=["glimpse"]#, "apex", "isummary"]
    #     , weights=weights
    # )
    # merged_df = pd.merge(sankofa, sota, on="Timestep")
    # merged_df.to_csv("all_results.csv", index=False)

if __name__ == "__main__":
    main()
