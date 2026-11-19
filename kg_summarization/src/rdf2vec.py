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

    Returns ``(node_ids, vectors)`` where node IDs retain their graph labels
    and vectors are float32. Predicates are not tokens in this initial
    implementation.
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

    node_by_token = {str(node): node for node in graph_node_set}
    valid_query_tokens = {str(node) for node in q_nodes if str(node) in node_by_token}
    remaining_tokens = [token for token in node_by_token if token not in valid_query_tokens]
    sample_count = min(max(0, int(sampled_start_nodes)), len(remaining_tokens))
    sampled_tokens = rng.sample(remaining_tokens, sample_count) if sample_count else []
    start_tokens = list(valid_query_tokens) + sampled_tokens
    rng.shuffle(start_tokens)
    start_nodes = [node_by_token[token] for token in start_tokens]

    progress_write(
        f"RDF2Vec: using {len(valid_query_tokens):,} query nodes and "
        f"{len(sampled_tokens):,} sampled graph nodes as walk starting points."
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
        if token not in node_by_token:
            continue
        node_ids.append(token)
        vectors.append(model.wv[token])

    if not vectors:
        raise RuntimeError("Word2Vec produced no node embeddings.")

    node_ids = np.asarray(node_ids, dtype=str)
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
        subject_node = str(subject)
        neighbors = adjacency.setdefault(subject_node, {})
        for objects in relations.values():
            for obj in objects:
                object_node = str(obj)
                neighbors[object_node] = 1.0
                adjacency.setdefault(object_node, {})
    return adjacency


def embedding_lookup(node_ids, vectors):
    """Return an ``entity label -> vector`` mapping."""
    return {str(node_id): vector for node_id, vector in zip(node_ids, vectors)}

