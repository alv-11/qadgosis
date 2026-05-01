"""Hard query generation for knowledge graph summarization benchmarks.

A "hard" query has two properties:
  1. The answer entity is NOT reachable from the topic entity in fewer hops
     than the inferential chain length (no shortest-path shortcut to the answer).
  2. Topic entities shift abruptly between timesteps: all topics are replaced
     each step, sampled from nodes that are at least `min_shift_hops` BFS steps
     away from the previous topic set.

Entry points
------------
generate_hard_query        -- single query (WebQSP-format dict)
generate_hard_query_logs   -- full query log for one timestep
update_topics_abrupt       -- topic-selection strategy with maximal BFS distance
"""

import random
from collections import deque

from .query import get_name


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _bfs_reachable_within(KG, source, max_hops):
    """Return the set of nodes reachable from *source* in <= *max_hops* outgoing
    steps (source itself is excluded from the result)."""
    if max_hops <= 0 or source not in KG.triples_:
        return set()

    visited = {source}
    frontier = {source}
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            if node not in KG.triples_:
                continue
            for tails in KG.triples_[node].values():
                for tail in tails:
                    if tail not in visited:
                        visited.add(tail)
                        next_frontier.add(tail)
        frontier = next_frontier
        if not frontier:
            break

    visited.discard(source)
    return visited


def _bfs_distances_from_sources(KG, sources, max_depth):
    """BFS over outgoing edges from a set of source nodes.

    Returns ``{node: min_distance}`` for all nodes reachable within
    *max_depth* hops.  Nodes in *sources* get distance 0.
    """
    dist = {}
    queue = deque()
    for s in sources:
        if s in KG.triples_ and s not in dist:
            dist[s] = 0
            queue.append(s)

    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= max_depth:
            continue
        if node not in KG.triples_:
            continue
        for tails in KG.triples_[node].values():
            for tail in tails:
                if tail not in dist:
                    dist[tail] = d + 1
                    queue.append(tail)

    return dist


# ---------------------------------------------------------------------------
# Hard query generation
# ---------------------------------------------------------------------------

def generate_hard_query(KG, topic_mid, qid=0, min_chain_len=2, max_chain_len=3,
                        entity_names=None, max_attempts=50, rng=None):
    """Generate one hard query in WebQSP format.

    The inferential chain has length in ``[min_chain_len, max_chain_len]`` and
    the final answer entity is **not** reachable from *topic_mid* in fewer than
    *min_chain_len* hops (no shortcut path exists in the KG).

    Falls back to the best chain found (even with a shortcuttable answer) if no
    strictly hard chain is found within *max_attempts* random walks.  Returns
    ``None`` if *topic_mid* has no outgoing edges or no valid chain can be built.

    Parameters
    ----------
    KG : KnowledgeGraph
    topic_mid : str
        Topic entity ID.
    qid : int
        Unique query identifier.
    min_chain_len : int
        Minimum inferential chain length (default 2).  Also the hop distance
        below which the answer entity must not be reachable from the topic.
    max_chain_len : int
        Maximum inferential chain length (chain length is sampled uniformly
        from ``[min_chain_len, max_chain_len]``).
    entity_names : dict, optional
        ``{entity_id: human_readable_label}`` mapping.
    max_attempts : int
        Number of random walks to attempt before using the best fallback.
    rng : random.Random, optional

    Returns
    -------
    dict or None
        Query dict in WebQSP format, or ``None`` when no valid chain exists.
    """
    if entity_names is None:
        entity_names = {}
    rng = random if rng is None else rng

    if topic_mid not in KG.triples_:
        return None

    # All nodes reachable in < min_chain_len hops are "too close" to count as
    # a hard answer.
    too_close = _bfs_reachable_within(KG, topic_mid, min_chain_len - 1)
    too_close.add(topic_mid)

    best_fallback = None  # any valid chain of required length
    found_hard = False

    for _ in range(max_attempts):
        chain_len = rng.randint(min_chain_len, max_chain_len)
        inferential_chain = []
        entity = topic_mid

        for _hop in range(chain_len):
            if entity not in KG.triples_:
                break
            predicates = list(KG.triples_[entity].keys())
            if not predicates:
                break
            predicate = rng.choice(predicates)
            tails = list(KG.triples_[entity][predicate])
            if not tails:
                break
            entity = rng.choice(tails)
            inferential_chain.append(predicate)

        if len(inferential_chain) < min_chain_len:
            continue  # could not build a long enough chain

        if best_fallback is None:
            best_fallback = list(inferential_chain)

        if entity not in too_close:
            best_fallback = list(inferential_chain)
            found_hard = True
            break

    if best_fallback is None:
        return None

    # Execute the chain as a full fan-out to collect all valid answers.
    result = {topic_mid}
    for predicate in best_fallback:
        candidates = set()
        for ent in result:
            if ent in KG.triples_ and predicate in KG.triples_[ent]:
                candidates.update(KG.triples_[ent][predicate])
        result = candidates

    result.discard(topic_mid)
    if not result:
        return None

    return {
        'QuestionId': str(qid),
        'Parse': {
            'TopicEntityMid': topic_mid,
            'TopicEntityName': get_name(topic_mid, entity_names),
            'InferentialChain': best_fallback,
            'Constraints': [],
            'Answers': [
                {
                    'AnswerType': 'Entity' if KG.is_entity(a) else 'Value',
                    'AnswerArgument': a,
                    'EntityName': get_name(a, entity_names),
                }
                for a in result
            ],
            '_hard': found_hard,  # diagnostic flag, ignored by evaluators
        },
    }


def generate_hard_query_logs(graph, timestep, topic_mids, num_queries=200, *,
                              min_chain_len=2, max_chain_len=3,
                              shuffle=True, random_query_prob=0.0,
                              rng=None, np_rng=None):
    """Generate a full hard query log for one timestep.

    Parameters
    ----------
    graph : KnowledgeGraph
    timestep : int
        Current snapshot index (used only for logging).
    topic_mids : list of str
        Topic entity IDs selected for this timestep.
    num_queries : int
        Total number of queries to generate.
    min_chain_len : int
        Passed to :func:`generate_hard_query`.
    max_chain_len : int
        Passed to :func:`generate_hard_query`.
    shuffle : bool
        Whether to shuffle the returned log.
    random_query_prob : float
        Fraction of queries drawn from a random entity (not a topic entity).
    rng : random.Random, optional
    np_rng : numpy.random.RandomState, optional

    Returns
    -------
    list of dict or None
        Query log, or ``None`` when no valid topic entities exist in the graph.
    """
    import numpy as np

    rng = random if rng is None else rng
    np_rng = np.random if np_rng is None else np_rng

    valid_mids = [mid for mid in topic_mids if mid in graph.triples_]
    if not valid_mids:
        return None

    n = len(valid_mids)
    n_topic_queries = max(0, int(num_queries * (1.0 - random_query_prob)))
    queries_per_mid = np_rng.multinomial(n_topic_queries, np.ones(n) / n)

    qid = 0
    query_log = []

    for mid, n_mid_queries in zip(valid_mids, queries_per_mid):
        for _ in range(int(n_mid_queries)):
            q = generate_hard_query(
                graph, mid,
                qid=qid,
                min_chain_len=min_chain_len,
                max_chain_len=max_chain_len,
                rng=rng,
            )
            if q is not None:
                query_log.append(q)
            qid += 1

    # Random queries: drawn from any node in the graph, not just topic entities.
    n_random = int(num_queries * random_query_prob)
    if n_random > 0:
        all_mids = list(graph.triples_.keys())
        random_mids = rng.choices(all_mids, k=n_random)
        for mid in random_mids:
            q = generate_hard_query(
                graph, mid,
                qid=qid,
                min_chain_len=min_chain_len,
                max_chain_len=max_chain_len,
                rng=rng,
            )
            if q is not None:
                query_log.append(q)
            qid += 1

    if not query_log:
        return None

    if shuffle:
        rng.shuffle(query_log)

    return query_log


# ---------------------------------------------------------------------------
# Abrupt topic-shift strategy
# ---------------------------------------------------------------------------

def update_topics_abrupt(prev_topics, graph, K=5, min_shift_hops=3,
                         max_candidates=5000, rng=None):
    """Select K topic entities that are maximally far from the previous set.

    All topics are replaced at every timestep (100 % churn), and new topics are
    sampled from nodes that are at least *min_shift_hops* BFS steps away from
    any previous topic entity (outgoing edges only).  This creates an abrupt,
    high-magnitude shift in query focus between consecutive timesteps.

    Fallback: if fewer than K nodes satisfy the distance constraint the pool is
    extended with the nearest remaining nodes ranked by descending out-degree,
    so the graph always gets exactly K (or as many as available) topic entities.

    Parameters
    ----------
    prev_topics : list of str or None
        Topic entities from the previous timestep.  ``None`` on the first step.
    graph : KnowledgeGraph
    K : int
        Number of topics to select.
    min_shift_hops : int
        Minimum BFS distance from any previous topic for a node to be eligible.
    max_candidates : int
        Cap on the candidate pool before degree-sorting (performance guard).
    rng : random.Random, optional

    Returns
    -------
    list of str
        Exactly K (or fewer if the graph is very small) topic entity IDs.
    """
    rng = random if rng is None else rng
    all_nodes = list(graph.triples_.keys())

    if not all_nodes:
        return []

    if prev_topics is None:
        return rng.sample(all_nodes, k=min(K, len(all_nodes)))

    # BFS from all previous topics to find their reachability frontier.
    distances = _bfs_distances_from_sources(graph, prev_topics,
                                            max_depth=min_shift_hops)

    # Nodes at distance >= min_shift_hops (or unreachable) from ALL prev topics.
    far_nodes = [
        n for n in all_nodes
        if distances.get(n, min_shift_hops + 1) >= min_shift_hops
    ]

    if len(far_nodes) >= K:
        if len(far_nodes) > max_candidates:
            far_nodes = rng.sample(far_nodes, max_candidates)
        # Prefer high-degree nodes (richer topics yield more answerable queries).
        far_nodes.sort(key=lambda n: -len(graph.triples_.get(n, {})))
        # Sample from the top half to balance diversity with query richness.
        top_pool = far_nodes[:max(K, len(far_nodes) // 2)]
        return rng.sample(top_pool, k=min(K, len(top_pool)))

    # Not enough strictly-far nodes: use all of them and pad with closest extras.
    selected = set(far_nodes)
    remaining = sorted(
        (n for n in all_nodes if n not in selected),
        key=lambda n: distances.get(n, min_shift_hops + 1),
        reverse=True,  # farthest first
    )
    for n in remaining:
        if len(selected) >= K:
            break
        selected.add(n)

    return list(selected)[:K]
