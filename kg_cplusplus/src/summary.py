
import time
import random
import numpy as np
import math
from src.base import KnowledgeGraph


def topk_nodes_by_score(nodes, scores, k):
    """
    Select top-k nodes by score using partial sort.
    Returns nodes sorted by descending score.
    """
    # if k >= len(nodes):
    #     # return fully sorted
    #     return sorted(nodes, key=lambda n: scores[n], reverse=True)


    # arr = np.array([scores[n] for n in nodes], dtype=float)
    # idx = np.argpartition(arr, k)[:k]

    # # sort only the selected nodes
    # idx = idx[np.argsort(-arr[idx])]

    # return [nodes[i] for i in idx]

    n = len(nodes)
    arr = np.array([scores[n] for n in nodes], dtype=float)

    keys = np.empty(n, dtype=float)
    for i, w in enumerate(arr):
        if w <= 0:
            pass
            keys[i] = np.inf
        else:
            u = random.random()
            keys[i] = -math.log(u) / (w)

    idx = np.argpartition(keys, k)[:k]

    return [nodes[i] for i in idx]







class SummaryGraph(KnowledgeGraph):

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def __init__(self, KG, top_k_nodes, max_triples=None):
        super().__init__()

        self.parent_ = KG
        self.top_k_nodes = top_k_nodes        # dict: node → importance score
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
        #print("Ranking internal triples...")
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
        k_nodes = len(top_k_list) -1
        # remaining = self.max_triples - self.triple_count
        # k_nodes = max(1, min(len(top_k_list) - 1, remaining//2))
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
        for t in self._boundary_triples_node_level():
            if self._limit_reached():
                return
            self._try_add_triple(t)

        # 3. Fallback: include any parent triple if budget remains
        #print("Adding fallback triples...")
        if not self._limit_reached():
            for s, rels in self.parent_.triples_.items():
                for r, nbrs in rels.items():
                    for o in nbrs:
                        if self._limit_reached():
                            return
                        self._try_add_triple((s, r, o))



# def weighted_topk(items, weights, k):
#     """
#     Pure argpartition version (no randomness, no ES).
#     Selects top-k items with the *highest* weights.
#     Complexity: O(n).
#     """
#     # n = len(weights)
#     # if k >= n:
#     #     return items[:]  # return all

#     w = np.asarray(weights, dtype=float)

#     # partial select k highest weights
#     idx = np.argpartition(w, k)[:k]

#     return [items[i] for i in idx]


# class SummaryGraphqq(KnowledgeGraph):

#     def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
#         super().__init__()

#         self.parent_ = KG
#         self.top_k_nodes = top_k_nodes
#         self.top_k_set = set(top_k_nodes.keys())
#         self.max_triples = max_triples
#         self.batch_size = batch_size

#         self.triple_count = 0
#         self.triples_set = set()

#         # Build indexing
#         self.reverse_index = self._build_reverse_index_topk_only()

#         # Internal triples
#         self._topk_neighbors = self._compute_topk_internal_triples()

#         # Boundary candidates
#         cand, w = self._compute_random_candidates_weighted()

#         # Replace weighted random sampling with pure argpartition top-k
#         remaining = self.max_triples - self.triple_count
#         self.random_padding_order = weighted_topk(cand, w, remaining)
#         self.random_padding_index = 0

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


#     # -------------------------------
#     # Reverse index for incoming edges
#     # -------------------------------
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


#     # -----------------------------------------------
#     # INTERNAL TRIPLES (top-k × top-k using argpartition)
#     # -----------------------------------------------
#     def _compute_topk_internal_triples(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         top_k_set = self.top_k_set
#         scores = self.top_k_nodes
#         budget = self.max_triples

#         internal = []

#         # Collect top-k internal triples
#         for e1 in top_k_set:
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         if e2 in top_k_set:
#                             internal.append((e1, r, e2))

#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     if e2 in top_k_set:
#                         internal.append((e2, r, e1))


#         # If internal fits, keep all (no need for ranking)
#         if len(internal) <= budget:
#             return [(0.0, t) for t in internal]

#         # Compute weights
#         w = np.array([scores[e1] + scores[e2] for (e1, _, e2) in internal], dtype=float)

#         # partial select k highest
#         k = budget
#         idx = np.argpartition(-w, k)[:k]

#         return [(w[i], internal[i]) for i in idx]


#     # -------------------------------------------------
#     # Compute boundary candidates + weights
#     # -------------------------------------------------
#     def _compute_random_candidates_weighted(self):
#         parent = self.parent_.triples_
#         rev = self.reverse_index
#         top_k = self.top_k_set
#         scores = self.top_k_nodes

#         candidates = []
#         weights = []

#         for e1 in top_k:

#             # outgoing
#             if e1 in parent:
#                 for r, nbrs in parent[e1].items():
#                     for e2 in nbrs:
#                         w = scores[e1] + scores.get(e2, 0.0)
#                         candidates.append((e1, r, e2))
#                         weights.append(w)

#             # incoming
#             if e1 in rev:
#                 for e2, r in rev[e1]:
#                     w = scores[e1] + scores.get(e2, 0.0)
#                     candidates.append((e2, r, e1))
#                     weights.append(w)

#         return candidates, weights


#     # -------------------------------------------------
#     # Summary construction
#     # -------------------------------------------------
#     def _build_summary(self):

#         # 1. Add internal triples
#         for score, triple in self._topk_neighbors:
#             if self._limit_reached():
#                 break
#             self._try_add_triple(triple)

#         # 2. Add boundary triples (from argpartition selection)
#         while not self._limit_reached():
#             added = self.add_related_random_triples_batch(self.batch_size)
#             if not added:
#                 break

#         # 3. Fallback: random parent triples
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


#     # -------------------------------------------------
#     # Add boundary triples in batches
#     # -------------------------------------------------
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





#####################edge based#################

def weighted_topk(items, weights, k):
    """
    Deterministic top-k using partial sort (largest weights).
    """
    # if k >= len(items):
    #     return items

    # w = np.asarray(weights, dtype=float)
    # idx = np.argpartition(-w, k)[:k]
    # return [items[i] for i in idx]
    keys = np.empty(len(weights), dtype=float)

    for i, w in enumerate(weights):
        if w <= 0:
            keys[i] = np.inf       
        else:
            u = random.random()
            keys[i] = -math.log(u) / w

    # get indices of k smallest keys
    idx = np.argpartition(keys, k)[:k]

    return [items[i] for i in idx]


class _SummaryGraph__(KnowledgeGraph):

    def __init__(self, KG, top_k_nodes, non_top_k_nodes, max_triples=None, batch_size=10):
        super().__init__()

        self.parent_ = KG
        self.top_k_nodes = top_k_nodes               # dict: node → score
        self.top_k_set = set(top_k_nodes.keys())
        self.max_triples = max_triples
        self.batch_size = batch_size
        self.non_top_k_nodes = non_top_k_nodes

        self.triple_count = 0
        self.triples_set = set()
        #non_top_k_nodes

        # Reverse index restricted to top-k
        self.reverse_index = self._build_reverse_index_topk_only()

        # Internal triples (deterministic top weighted)
        self._topk_neighbors = self._compute_topk_internal_triples()

        # Boundary candidates (no sampling)
        cand, w = self._compute_boundary_candidates()

        remaining = max_triples - self.triple_count
        self.random_padding_order = weighted_topk(cand, w, remaining)
        self.random_padding_index = 0

        self._build_summary()


    # ---------------------------
    # Helpers
    # ---------------------------

    def parent(self):
        return self.parent_

    def _limit_reached(self):
        return self.max_triples is not None and self.triple_count >= self.max_triples

    def _try_add_triple(self, triple):
        if triple not in self.triples_set:
            self.add_triple(triple)
            self.triples_set.add(triple)
            self.triple_count += 1


    # ---------------------------
    # Reverse index
    # ---------------------------

    def _build_reverse_index_topk_only(self):
        reverse_idx = {}
        parent = self.parent_.triples_
        top_k = self.top_k_set

        for s, rel_dict in parent.items():
            for r, nbrs in rel_dict.items():
                for o in nbrs:
                    if o in top_k:
                        reverse_idx.setdefault(o, []).append((s, r))

        return reverse_idx


    # ---------------------------
    # INTERNAL TRIPLES (top-k × top-k)
    # ---------------------------

    def _compute_topk_internal_triples(self):
        parent = self.parent_.triples_
        rev = self.reverse_index
        scores = self.top_k_nodes
        top_k_nodes = self.top_k_set
        budget = self.max_triples

        internal = set() 

        # Collect internal triples
        for e1 in top_k_nodes:
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        if e2 in top_k_nodes:
                            internal.add((e1, r, e2))

            if e1 in rev:
                for e2, r in rev[e1]:
                    if e2 in top_k_nodes:
                        internal.add((e2, r, e1))

        print(len(internal), budget, len(set(internal)))

        #internal = list(dict.fromkeys(internal))
        internal = list(internal)

        # # If small, keep all
        if len(internal) <= budget:
            return [(0.0, t) for t in internal]

        # Otherwise: score + partial select budget elements
        else:
            #print("Ranking internal triples...")
            weights = [scores[t[0]] + scores[t[2]] for t in internal]
            warr = np.asarray(weights)
            idx = np.argpartition(-warr, budget)[:budget]

            selected = [(weights[i], internal[i]) for i in idx]
            return selected
    
        # If small, return all
        # print("Internal triples:", len(internal))
        # if len(internal) <= budget:
        #     return [(0.0, t) for t in internal]

        # # Else score
        # print("Ranking internal triples...", len(internal))
        # scored = []
        # for (e1, r, e2) in internal:
        #     score = scores[e1] + scores[e2]
        #     scored.append((score, (e1, r, e2)))

        # scored.sort(key=lambda x: x[0], reverse=True)
        # return scored


    # ---------------------------
    # BOUNDARY TRIPLES
    # ---------------------------

    def _compute_boundary_candidates(self):
        """
        All (top-k node) boundary triples with weights.
        Deterministic partial ranking, no sampling.
        """
        parent = self.parent_.triples_
        rev = self.reverse_index
        scores = self.top_k_nodes
        other_scores = self.non_top_k_nodes
        top_k = self.top_k_set

        candidates = []
        weights = []

        for e1 in top_k:

            # Outgoing
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        w = scores[e1] + other_scores.get(e2, 0.0)
                        candidates.append((e1, r, e2))
                        weights.append(w)

            # Incoming
            if e1 in rev:
                for e2, r in rev[e1]:
                    w = scores[e1] + other_scores.get(e2, 0.0)
                    candidates.append((e2, r, e1))
                    weights.append(w)

        #print("Boundary candidates:", len(candidates))
        return candidates, weights
    

    def _compute_fallback_candidates(self):

        parent = self.parent_.triples_
        seen = self.triples_set

        top_k = self.top_k_set
        scores_other = self.non_top_k_nodes

        candidates = []
        weights = []

        for s, rels in parent.items():

            if s in top_k:        # skip top-k nodes entirely
                continue

            for r, nbrs in rels.items():
                for o in nbrs:

                    if o in top_k:   # skip edges entering top-k region
                        continue

                    triple = (s, r, o)

                    if triple in seen:
                        continue

                    # weight based only on non-top-k scores
                    w = scores_other.get(s, 0.0) + scores_other.get(o, 0.0)

                    candidates.append(triple)
                    weights.append(w)
        #print("Fallback candidates:", len(candidates))
        return candidates, weights



    # ---------------------------
    # Summary builder
    # ---------------------------

    def _build_summary(self):

        # 1. Add internal triples
        for score, triple in self._topk_neighbors:
            if self._limit_reached():
                break
            self._try_add_triple(triple)

        #print("Added internal triples:", self.triple_count)

        # 2. Add boundary triples in batches
        while not self._limit_reached():
            #print("Adding boundary triples...")
            added = self.add_related_random_triples_batch(self.batch_size)
            if not added:
                break


        #print("Adding fallback triples...")
        fallback_start= time.time()
        # # 3. Fallback: add ANY triple from parent
        # print("Adding fallback triples...")
        # if not self._limit_reached():
        #     parent_edges = []
        #     triples = self.parent_.triples_
        #     for s, rels in triples.items():
        #         for r, nbrs in rels.items():
        #             for o in nbrs:
        #                 parent_edges.append((s, r, o))

        #     while not self._limit_reached() and parent_edges:
        #         t = random.choice(parent_edges)
        #         self._try_add_triple(t)

        while not self._limit_reached():
        #print("Adding boundary triples...")
            fallback = self._fallback_add()
            if not fallback:
                break
        fallback_end = time.time()
        #print("Fallback time:", fallback_end - fallback_start)

    # ---------------------------
    # Batched addition
    # ---------------------------

    def add_related_random_triples_batch(self, batch_size=100):
        order = self.random_padding_order
        idx = self.random_padding_index
        n = len(order)

        if idx >= n:
            return False

        remaining = self.max_triples -  self.triple_count
        k = min(batch_size, remaining)

        end = min(idx + k, n)
        batch = order[idx:end]

        added = False
        for triple in batch:
            if self._limit_reached():
                break
            if triple not in self.triples_set:
                self._try_add_triple(triple)
                added = True

        self.random_padding_index = end
        return added


    def _fallback_add(self):
        remaining = self.max_triples - self.triple_count
        if remaining <= 0:
            return

        cand, w = self._compute_fallback_candidates()
        if not cand:
            return

        arr = np.asarray(w, dtype=float)
        k = min(remaining, len(arr))

        idx = np.argpartition(-arr, k)[:k]

        for i in idx:
            if self._limit_reached():
                break
            self._try_add_triple(cand[i])



###############


def weighted_topk_(items, weights, k):
    keys = np.empty(len(weights), dtype=float)

    for i, w in enumerate(weights):
        if w <= 0:
            keys[i] = np.inf       
        else:
            u = random.random()
            keys[i] = -math.log(u) / w

    # get indices of k smallest keys
    idx = np.argpartition(keys, k)[:k]

    return [items[i] for i in idx]




class _SummaryGraph_(KnowledgeGraph):

    def __init__(self, KG, top_k_nodes, max_triples=None, batch_size=10):
        super().__init__()

        self.parent_ = KG
        self.top_k_nodes = top_k_nodes
        self.top_k_set = set(top_k_nodes.keys())
        self.max_triples = max_triples
        self.batch_size = batch_size
        #self.global_pr = global_pr
        #self.global_pr_set = set(global_pr.keys())

        self.triple_count = 0
        self.triples_set = set()

        # Build indexing
        self.reverse_index = self._build_reverse_index_topk_only()

        # Internal triples
        self._topk_neighbors = self._compute_topk_internal_triples()

        # # Boundary candidates
        cand, w = self._compute_random_candidates_weighted()

        # eighted sampling WITHOUT replacement


        remaining = self.max_triples - self.triple_count
        self.random_padding_order = weighted_topk(cand, w, remaining)
        self.random_padding_index = 0


        self._build_summary()


    def parent(self):
        return self.parent_

    def _limit_reached(self):
        return self.max_triples is not None and self.triple_count >= self.max_triples


    def _try_add_triple(self, triple):
        if triple not in self.triples_set:
            self.add_triple(triple)
            self.triples_set.add(triple)
            self.triple_count += 1


    def _build_reverse_index_topk_only(self):
        reverse_idx = {}
        parent = self.parent_.triples_
        top_k = self.top_k_set

        for s, rel_dict in parent.items():
            for r, nbrs in rel_dict.items():
                for o in nbrs:
                    if o in top_k:
                        reverse_idx.setdefault(o, []).append((s, r))

        return reverse_idx


    def _compute_topk_internal_triples(self):
        parent = self.parent_.triples_
        rev = self.reverse_index
        top_k_nodes = self.top_k_set
        scores = self.top_k_nodes
        budget = self.max_triples

        internal = []

        # Collect
        for e1 in top_k_nodes:
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        if e2 in top_k_nodes:
                            internal.append((e1, r, e2))

            if e1 in rev:
                for e2, r in rev[e1]:
                    if e2 in top_k_nodes:
                        internal.append((e2, r, e1))

        # If small, return all
        #print("Internal triples:", len(internal))
        if len(internal) <= budget:
            return [(0.0, t) for t in internal]

        # Else score
        # print("Ranking internal triples...")
        scored = []
        for (e1, r, e2) in internal:
            score = scores[e1] + scores[e2]
            scored.append((score, (e1, r, e2)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored






    def _compute_random_candidates_weighted(self):
        parent = self.parent_.triples_
        rev = self.reverse_index
        top_k = self.top_k_set
        scores = self.top_k_nodes

        candidates = []
        weights = []

        for e1 in top_k:

            # Outgoing
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        w = scores[e1] + scores.get(e2, 0.0)
                        candidates.append((e1, r, e2))
                        weights.append(w)

            # Incoming
            if e1 in rev:
                for e2, r in rev[e1]:
                    w = scores[e1] + scores.get(e2, 0.0)
                    candidates.append((e2, r, e1))
                    weights.append(w)

        return candidates, weights




    def _build_summary(self):

        # 1. Internal triples
        for score, triple in self._topk_neighbors:
            if self._limit_reached():
                break
            self._try_add_triple(triple)


        # 2.  C_bnd sampling
        while not self._limit_reached():
            added = self.add_related_random_triples_batch(self.batch_size)
            if not added:
                break

        # 3. Fallback: any triple if it doesn't meet the budget
        if not self._limit_reached():
            parent_edges = []
            triples = self.parent_.triples_
            for s, rels in triples.items():
                for r, nbrs in rels.items():
                    for o in nbrs:
                        parent_edges.append((s, r, o))

            while not self._limit_reached() and parent_edges:
                t = random.choice(parent_edges)
                self._try_add_triple(t)



    def add_related_random_triples_batch(self, batch_size=100):
        order = self.random_padding_order
        idx = self.random_padding_index
        n = len(order)

        if idx >= n:
            return False

        remaining = self.max_triples - self.triple_count
        k = min(batch_size, remaining)

        end = min(idx + k, n)
        batch = order[idx:end]

        added = False
        for triple in batch:
            if self._limit_reached():
                break
            if triple not in self.triples_set:
                self._try_add_triple(triple)
                added = True

        self.random_padding_index = end
        return added






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
