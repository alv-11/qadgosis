import random

from collections import defaultdict

from .algorithms import extract_query_seed_entities
from .base import KnowledgeGraph


class Summary(KnowledgeGraph):

    def __init__(self, KG):
        super().__init__()
        self.parent_ = KG

    def parent(self):
        return self.parent_

    def fill(self, triples, k):
        for triple in triples:
            if self.number_of_triples() >= k:
                return
            self.add_triple(triple)


def _build_reverse_index(KG):
    reverse_index = defaultdict(set)

    for head, relations in KG.triples_.items():
        for tails in relations.values():
            for tail in tails:
                reverse_index[tail].add(head)

    return reverse_index


def _outgoing_neighbors(KG, entity):
    neighbors = set()

    if entity not in KG.triples_:
        return neighbors

    for tails in KG[entity].values():
        neighbors.update(tails)

    return neighbors


def _collect_h_hop_nodes(KG, seeds, h, neighborhood_mode):
    if h < 0:
        raise ValueError("h must be non-negative")

    if neighborhood_mode not in {"both", "outgoing_only"}:
        raise ValueError(f"Unsupported neighborhood_mode: {neighborhood_mode}")

    visited = {seed for seed in seeds if KG.has_entity(seed)}
    if not visited:
        return set()

    reverse_index = _build_reverse_index(KG) if neighborhood_mode == "both" else None
    frontier = set(visited)

    for _ in range(h):
        next_frontier = set()

        for entity in frontier:
            next_frontier.update(_outgoing_neighbors(KG, entity))

            if reverse_index is not None:
                next_frontier.update(reverse_index.get(entity, set()))

        next_frontier -= visited

        if not next_frontier:
            break

        visited.update(next_frontier)
        frontier = next_frontier

    return visited


def _collect_candidate_triples(KG, neighborhood):
    candidates = []

    for head in neighborhood:
        if head not in KG.triples_:
            continue

        for relation, tails in KG[head].items():
            for tail in tails:
                if tail in neighborhood:
                    candidates.append((head, relation, tail))

    candidates.sort()
    return candidates


def HHOP_RANDOM(KG, K, query_log, h=2, neighborhood_mode="both", random_state=None):
    summary = Summary(KG)

    if K <= 0:
        return summary

    seed_entities = []
    for query in query_log:
        seed_entities.extend(
            extract_query_seed_entities(query, seed_mode="answers_fallback_topic")
        )

    neighborhood = _collect_h_hop_nodes(KG, seed_entities, h, neighborhood_mode)
    if not neighborhood:
        return summary

    candidates = _collect_candidate_triples(KG, neighborhood)
    if not candidates:
        return summary

    if len(candidates) > K:
        rng = random.Random(random_state) if random_state is not None else random
        candidates = rng.sample(candidates, K)

    summary.fill(candidates, K)
    return summary