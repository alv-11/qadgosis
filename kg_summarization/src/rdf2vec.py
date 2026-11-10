"""RDF2Vec-style node embeddings trained from graph random walks."""

from __future__ import annotations

import random

import numpy as np

from src.progress import progress_write


def build_rdf2vec_embeddings(
    graph,
    q_nodes,
    embedding_size=100,
    max_depth=4,
    max_walks=3,
    sampled_start_nodes=10_000,
    epochs=10,
    workers=4,
    random_seed=42,
):
    """Train node embeddings from random walks, always including query seeds.

    ``graph`` may be a NetworkX graph or an adjacency mapping such as
    ``{node: {neighbour: weight}}``. ``sampled_start_nodes`` is the number of
    non-query nodes sampled as additional walk starts; it is capped at the
    number of available non-query nodes.

    Returns ``(node_ids, vectors)`` where node IDs are int64 and vectors are
    float32. This follows the supplied project code's node-vector approach;
    predicates are not tokens in this initial implementation.
    """
    from gensim.models import Word2Vec

    rng = random.Random(random_seed)

    progress_write(f"RDF2Vec: preparing graph with {len(graph):,} nodes...")

    def get_neighbors(node):
        if hasattr(graph, "neighbors"):
            return list(graph.neighbors(node))

        neighbours = graph.get(node, {})
        if isinstance(neighbours, dict):
            return list(neighbours.keys())
        return list(neighbours)

    if hasattr(graph, "nodes"):
        graph_node_set = set(graph.nodes())
    else:
        graph_node_set = set(graph.keys())
        for neighbours in graph.values():
            graph_node_set.update(neighbours.keys() if isinstance(neighbours, dict) else neighbours)

    # This repository's graph loader and C++ interface use numeric entity IDs.
    node_by_id = {}
    for node in graph_node_set:
        try:
            node_by_id[int(node)] = node
        except (TypeError, ValueError):
            continue

    valid_query_ids = set()
    for node in q_nodes:
        try:
            node_id = int(node)
        except (TypeError, ValueError):
            continue
        if node_id in node_by_id:
            valid_query_ids.add(node_id)

    remaining_ids = [node_id for node_id in node_by_id if node_id not in valid_query_ids]
    sample_count = min(max(0, int(sampled_start_nodes)), len(remaining_ids))
    sampled_ids = rng.sample(remaining_ids, sample_count) if sample_count else []
    start_ids = list(valid_query_ids) + sampled_ids
    rng.shuffle(start_ids)
    start_nodes = [node_by_id[node_id] for node_id in start_ids]

    progress_write(
        f"RDF2Vec: using {len(valid_query_ids):,} query nodes and "
        f"{len(sampled_ids):,} sampled graph nodes as walk starting points."
    )

    def random_walk(start_node):
        walk = [str(start_node)]
        current = start_node
        for _ in range(max_depth):
            neighbours = get_neighbors(current)
            if not neighbours:
                break
            current = rng.choice(neighbours)
            walk.append(str(current))
        return walk

    walks = []
    for index, node in enumerate(start_nodes, start=1):
        for _ in range(max_walks):
            walks.append(random_walk(node))
        if index % 1000 == 0 or index == len(start_nodes):
            progress_write(
                f"RDF2Vec: generated walks for {index:,}/{len(start_nodes):,} "
                "starting nodes."
            )

    if not walks:
        raise RuntimeError(
            "No RDF2Vec walks were generated. Check the graph and query nodes."
        )

    progress_write(f"RDF2Vec: generated {len(walks):,} walks; training Word2Vec...")
    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_size,
        window=5,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=random_seed,
    )

    node_ids = []
    vectors = []
    for token in model.wv.index_to_key:
        try:
            node_id = int(token)
        except (TypeError, ValueError):
            continue
        node_ids.append(node_id)
        vectors.append(model.wv[token])

    if not vectors:
        raise RuntimeError("Word2Vec produced no node embeddings.")

    node_ids = np.asarray(node_ids, dtype=np.int64)
    vectors = np.asarray(vectors, dtype=np.float32)
    progress_write(
        f"RDF2Vec complete: {len(node_ids):,} node embeddings "
        f"of dimension {vectors.shape[1]}."
    )
    return node_ids, vectors


def adjacency_from_knowledge_graph(knowledge_graph):
    """Convert the current KnowledgeGraph triples to entity adjacency."""
    adjacency = {}
    for subject, relations in knowledge_graph.triples_.items():
        subject_id = int(subject)
        neighbors = adjacency.setdefault(subject_id, {})
        for objects in relations.values():
            for obj in objects:
                object_id = int(obj)
                neighbors[object_id] = 1.0
                adjacency.setdefault(object_id, {})
    return adjacency


def embedding_lookup(node_ids, vectors):
    """Return an ``int entity ID -> vector`` mapping."""
    return {int(node_id): vector for node_id, vector in zip(node_ids, vectors)}

