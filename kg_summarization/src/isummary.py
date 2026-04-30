from collections import Counter, defaultdict
import networkx as nx

from src.user import query_log_by_mids
from src.metrics import total_query_log_metrics
from src.base import DBPedia


def extract_entities_from_query(q, seed_mode="answers_fallback_topic"):
    """Extract all entity IDs from a query, handling nested structure."""
    entities = set()
    
    try:
        if isinstance(q, dict) and "Parse" in q:
            parse = q["Parse"]
            
            # Topic entity
            topic = parse.get("TopicEntityMid")
            if topic:
                entities.add(str(topic))
            
            # Answers
            if seed_mode != "topic_only":
                for ans in parse.get("Answers", []):
                    arg = ans.get("AnswerArgument")
                    if arg:
                        entities.add(str(arg))
            
            # Constraints
            for constraint in parse.get("Constraints", []):
                if isinstance(constraint, dict):
                    for key in ['s', 'o', 'subject', 'object']:
                        if key in constraint and constraint[key]:
                            entities.add(str(constraint[key]))
                elif isinstance(constraint, (list, tuple)):
                    for item in constraint:
                        if isinstance(item, str) and item.startswith("m."):
                            entities.add(item)
        
        # Fallback: try to extract any string that looks like an entity ID
        elif isinstance(q, (dict, list, tuple)):
            import re
            q_str = str(q)
            entity_pattern = r'm\.[a-z0-9_]+'
            entities.update(re.findall(entity_pattern, q_str))
    
    except Exception as e:
        print(f"[DEBUG] Error extracting entities from query: {e}")
    
    return entities


def filter_queries_by_entities(Q, entities, require_all=False, seed_mode="answers_fallback_topic"):
    """Filter queries that mention any or all of the given entities."""
    filtered = []
    target_entities = set(map(str, entities))

    if not target_entities:
        return filtered
    
    for q in Q:
        q_entities = extract_entities_from_query(q, seed_mode=seed_mode)
        if require_all:
            if target_entities <= q_entities:
                filtered.append(q)
        elif target_entities & q_entities:
            filtered.append(q)
    
    return filtered


def select_top_nodes(Qs, max_nodes=None, seed_mode="answers_fallback_topic"):
    """Select top nodes from queries Qs, optionally limited by max_nodes."""
    node_freq = Counter()
    
    for q in Qs:
        entities = extract_entities_from_query(q, seed_mode=seed_mode)
        for entity in entities:
            node_freq[entity] += 1
    
    if max_nodes:
        return [node for node, freq in node_freq.most_common(max_nodes) if freq > 0]
    else:
        return [node for node, freq in node_freq.most_common() if freq > 0]


def build_predicate_graph(KG):
    """Build a directed graph view of the KG for path-based summarization."""
    G = nx.DiGraph()
    for s_kg, preds in KG.triples_.items():
        s_str = str(s_kg)
        for p, objs in preds.items():
            for o in objs:
                o_str = str(o)
                G.add_edge(s_str, o_str, predicate=p)
    return G


def get_shortest_path_between(G, source, target, allowed_nodes=None):
    """Find the shortest path between explicit endpoints.

    When allowed_nodes is provided, the search is restricted to the induced
    subgraph over those nodes plus the requested endpoints.
    """
    source = str(source)
    target = str(target)

    if source == target:
        return [source] if G.has_node(source) else []

    if allowed_nodes is None:
        if not G.has_node(source) or not G.has_node(target):
            return []
        search_graph = G
    else:
        valid_nodes = {str(node) for node in allowed_nodes if G.has_node(str(node))}
        valid_nodes.update((source, target))

        if source not in valid_nodes or target not in valid_nodes:
            return []

        search_graph = G.subgraph(valid_nodes)

    if not search_graph.has_node(source) or not search_graph.has_node(target):
        return []

    best_path = None

    for start, end in ((source, target), (target, source)):
        try:
            candidate_path = nx.shortest_path(search_graph, source=start, target=end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if (
            best_path is None
            or len(candidate_path) < len(best_path)
            or (
                len(candidate_path) == len(best_path)
                and tuple(candidate_path) < tuple(best_path)
            )
        ):
            best_path = candidate_path

    return best_path or []


def find_most_frequent_path(paths):
    """Find the most frequent path from a list of paths."""
    if not paths:
        return []
    
    path_counts = Counter()
    for path in paths:
        path_tuple = tuple(path)
        path_counts[path_tuple] += 1

    ranked_paths = sorted(
        path_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    return list(ranked_paths[0][0])


def resolve_path_to_triples(G, path):
    """Convert a path of nodes to triples using the graph."""
    triples = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            p = G[u][v]["predicate"]
            triples.add((u, p, v))
    return triples


def build_isummary_with_target_size(KG, query_log, s, target_triples, graph=None, seed_mode="answers_fallback_topic"):
    """
   iSummary that grows summary until reaching target triple count.
    This is a departure from the original k-node approach.
    """    
    S = set()

    if target_triples <= 0:
        return S

    seed = str(s)
    G = graph if graph is not None else build_predicate_graph(KG)

    if not G.has_node(seed):
        return S

    visited = {seed}

    if seed_mode == "topic_only":
        # In topic-only mode each query has exactly one entity, so co-occurrence
        # within a single query never happens.  Instead, collect all topic entities
        # from the full log and find paths between them in the unrestricted graph.
        topic_freq = Counter()
        for q in query_log:
            for entity in extract_entities_from_query(q, seed_mode="topic_only"):
                e_str = str(entity)
                if G.has_node(e_str) and e_str != seed:
                    topic_freq[e_str] += 1

        candidate_nodes = [
            node
            for node, freq in sorted(
                topic_freq.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if freq > 0
        ]

        for x in candidate_nodes:
            if len(S) >= target_triples:
                break

            all_paths = []
            for y in sorted(visited):
                path = get_shortest_path_between(G, x, y)
                if path:
                    all_paths.append(path)

            if not all_paths:
                continue

            selected_path = find_most_frequent_path(all_paths)
            path_triples = resolve_path_to_triples(G, selected_path)

            if not path_triples:
                continue

            visited.add(x)

            if len(S) + len(path_triples) > target_triples:
                remaining = target_triples - len(S)
                if remaining > 0:
                    S.update(set(sorted(path_triples)[:remaining]))
                break

            S.update(path_triples)

        return S

    query_entities_cache = []
    query_indices_by_entity = defaultdict(set)
    node_freq = Counter()

    for q in query_log:
        q_entities = {
            str(entity)
            for entity in extract_entities_from_query(q, seed_mode=seed_mode)
            if G.has_node(str(entity))
        }

        if seed not in q_entities:
            continue

        query_index = len(query_entities_cache)
        query_entities_cache.append(q_entities)

        for entity in q_entities:
            query_indices_by_entity[entity].add(query_index)
            if entity != seed:
                node_freq[entity] += 1

    if not query_entities_cache:
        return S

    candidate_nodes = [
        node
        for node, freq in sorted(
            node_freq.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if freq > 0
    ]

    # Process nodes until we reach target size or run out of candidates
    for x in candidate_nodes:
        if len(S) >= target_triples:
            break

        all_paths = []
        x_query_indices = query_indices_by_entity.get(x)

        if not x_query_indices:
            continue

        for y in sorted(visited):
            y_query_indices = query_indices_by_entity.get(y)

            if not y_query_indices:
                continue

            # For each relevant query, get the shortest path linking x and y.
            for query_index in x_query_indices & y_query_indices:
                q_entities = query_entities_cache[query_index]
                path = get_shortest_path_between(G, x, y, allowed_nodes=q_entities)
                if path:
                    all_paths.append(path)

        # Add path to summary if found
        if not all_paths:
            continue

        selected_path = find_most_frequent_path(all_paths)
        path_triples = resolve_path_to_triples(G, selected_path)

        if not path_triples:
            continue

        visited.add(x)

        # Check if adding these triples would exceed our target
        if len(S) + len(path_triples) > target_triples:
            remaining = target_triples - len(S)
            if remaining > 0:
                triples_to_add = set(sorted(path_triples)[:remaining])
                S.update(triples_to_add)
            break

        S.update(path_triples)
    return S


def build_global_isummary_with_target_size(KG, target_triples, query_log, seed_mode="answers_fallback_topic"):
    """
    Build global summary by combining personalized summaries for each query entity.
    """
    # Extract all unique entities from queries
    all_query_entities = set()
    for q in query_log:
        entities = extract_entities_from_query(q, seed_mode=seed_mode)
        all_query_entities.update(entities)
    
    #print(f"\n[INFO] Found {len(all_query_entities)} unique entities in queries")
    
    # Filter entities that exist anywhere in the KG, including object positions.
    valid_entities = []
    for entity in all_query_entities:
        if KG.has_entity(entity):
            valid_entities.append(entity)
        else:
            pass

    
    if not valid_entities:
        return set()

    G = build_predicate_graph(KG)
    
    # Calculate target size per entity
    entity_freq = Counter()
    for q in query_log:
        entities = extract_entities_from_query(q, seed_mode=seed_mode)
        for entity in entities:
            if entity in valid_entities:
                entity_freq[entity] += 1
    
    total_freq = sum(entity_freq.values())
    all_summary_triples = set()

    
    for i, (entity, freq) in enumerate(entity_freq.most_common()):
        remaining_budget = target_triples - len(all_summary_triples)

        if remaining_budget <= 0:
            break
            
        entity_share = min(
            remaining_budget,
            max(1, int((freq / total_freq) * target_triples)),
        )
        

        try:
            entity_summary = build_isummary_with_target_size(
                KG,
                query_log,
                entity,
                entity_share,
                graph=G,
                seed_mode=seed_mode,
            )
            all_summary_triples.update(entity_summary)
        except Exception as e:
            continue
        
    return all_summary_triples





def build_local_structure(KG, entity, target_triples, graph=None):
    """Build local structure around an entity when queries don't provide enough."""
    S = set()
    entity_str = str(entity)
    
    if not KG.has_entity(entity_str):
        return S

    G = graph if graph is not None else build_predicate_graph(KG)
    
    # Add immediate neighbors
    neighbors = list(G.successors(entity_str)) + list(G.predecessors(entity_str))
    
    for neighbor in neighbors[:target_triples * 2]:  # Get more neighbors than needed
        if len(S) >= target_triples:
            break
            
        # Add triples to/from this neighbor
        if G.has_edge(entity_str, neighbor):
            p = G[entity_str][neighbor]["predicate"]
            S.add((entity_str, p, neighbor))
        
        if G.has_edge(neighbor, entity_str):
            p = G[neighbor][entity_str]["predicate"]
            S.add((neighbor, p, entity_str))
    
    return S


def get_extended_triples(G, path, max_triples=10):
    """Get additional triples around path nodes."""
    triples = set()
    for node in path:
        if len(triples) >= max_triples:
            break
        # Add some outgoing edges
        for succ in G.successors(node)[:2]:  
            p = G[node][succ]["predicate"]
            triples.add((node, p, succ))
        # Add some incoming edges
        for pred in G.predecessors(node)[:2]:  
            p = G[pred][node]["predicate"]
            triples.add((pred, p, node))
    return triples

class SummaryDBPedia:
    """A fully compatible summary KG that mimics the DBPedia interface."""
    
    def __init__(self, triples, parent):
        self.triples_ = defaultdict(lambda: defaultdict(set))
        self.entities_ = set()
        self.relations_ = set()
        self._parent = parent
        
        # Build the triple structure to match DBPedia's format
        for s, p, o in triples:
            self.triples_[s][p].add(o)
            self.entities_.update([s, o])
            self.relations_.add(p)
        
        # Convert back to regular dict for consistency
        self.triples_ = dict(self.triples_)

    def __contains__(self, entity):
        """Check if entity exists in the summary."""
        return entity in self.entities_

    def __getitem__(self, entity):
        """Allow KG[entity] syntax to get predicates."""
        if entity in self.triples_:
            return self.triples_[entity]
        return defaultdict(set)

    def get(self, entity, default=None):
        """Get entity triples with default."""
        return self.triples_.get(entity, default)

    def number_of_triples(self):
        count = 0
        for preds in self.triples_.values():
            for objs in preds.values():
                count += len(objs)
        return count

    def number_of_entities(self):
        return len(self.entities_)

    def number_of_relations(self):
        return len(self.relations_)

    def parent(self):
        return self._parent

    def update_user(self, user_id):
        """Compatibility method - may be needed by some functions."""
        pass

    def load(self):
        """Compatibility method - summary is already loaded."""
        pass

    def triples(self):
        """Return all triples as a list of (s, p, o) tuples."""
        all_triples = []
        for s, preds in self.triples_.values():
            for p, objs in preds.values():
                for o in objs:
                    all_triples.append((s, p, o))
        return all_triples

    def entities(self):
        """Return all entities."""
        return self.entities_

    def relations(self):
        """Return all relations."""
        return self.relations_


def ISUMMARY(KG, target_triples, query_log, seed_mode="answers_fallback_topic"):

    all_summary_triples = build_global_isummary_with_target_size(KG, target_triples, query_log, seed_mode=seed_mode)


    # Create and evaluate summary
    summary = SummaryDBPedia(all_summary_triples, KG)
    
    #print(f"Summary stats: {summary.number_of_triples()} triples, "
    #      f"{summary.number_of_entities()} entities, "
     #     f"{summary.number_of_relations()} relations")


    return summary



 
