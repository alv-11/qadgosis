from collections import Counter, defaultdict
import networkx as nx

from src.user import query_log_by_mids
from src.metrics import total_query_log_metrics
from src.base import DBPedia


def extract_entities_from_query(q):
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


def filter_queries_by_entities(Q, entities):
    """Filter queries that mention any of the given entities."""
    filtered = []
    target_entities = set(map(str, entities))
    
    for q in Q:
        q_entities = extract_entities_from_query(q)
        if target_entities & q_entities:
            filtered.append(q)
    
    return filtered


def select_top_nodes(Qs, max_nodes=None):
    """Select top nodes from queries Qs, optionally limited by max_nodes."""
    node_freq = Counter()
    
    for q in Qs:
        entities = extract_entities_from_query(q)
        for entity in entities:
            node_freq[entity] += 1
    
    if max_nodes:
        return [node for node, freq in node_freq.most_common(max_nodes) if freq > 0]
    else:
        return [node for node, freq in node_freq.most_common() if freq > 0]


def get_shortest_path_between(G, nodes):
    """Find shortest path between any two nodes in the set."""
    nodes_list = list(nodes)
    if len(nodes_list) < 2:
        return []
    
    # Try all pairs to find a path
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            try:
                if nx.has_path(G, nodes_list[i], nodes_list[j]):
                    return nx.shortest_path(G, nodes_list[i], nodes_list[j])
            except nx.NetworkXNoPath:
                continue
    return []


def find_most_frequent_path(paths):
    """Find the most frequent path from a list of paths."""
    if not paths:
        return []
    
    path_counts = Counter()
    for path in paths:
        path_tuple = tuple(path)
        path_counts[path_tuple] += 1
    
    return list(path_counts.most_common(1)[0][0])


def resolve_path_to_triples(G, path):
    """Convert a path of nodes to triples using the graph."""
    triples = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            p = G[u][v]["predicate"]
            triples.add((u, p, v))
    return triples


def build_isummary_with_target_size(KG, query_log, s, target_triples):
    """
   iSummary that grows summary until reaching target triple count.
    This is a departure from the original k-node approach.
    """    
    S = set()
    visited = {str(s)}
    
    # Filter queries mentioning s
    Qs = filter_queries_by_entities(query_log, [s])
    
    if not Qs:
        return S
   
    G = nx.DiGraph()
    for s_kg, preds in KG.triples_.items():
        s_str = str(s_kg)
        for p, objs in preds.items():
            for o in objs:
                o_str = str(o)
                G.add_edge(s_str, o_str, predicate=p)
    
    # Instead of fixed k, we'll process nodes until we reach target size
    candidate_nodes = select_top_nodes(Qs, max_nodes=50)  

    
    # Process nodes until we reach target size or run out of candidates
    for x in candidate_nodes:
        if x in visited:
            continue
            
        if len(S) >= target_triples:
            break
            
        selected_path = []
        all_paths = []
        

        for y in list(visited):
            Qsxy = filter_queries_by_entities(Qs, [x, y])
            
            if not Qsxy:
                continue
                
            # For each query, get shortest path
            paths_for_pair = []
            for q in Qsxy:
                q_entities = extract_entities_from_query(q)
                q_entities.add(x)
                q_entities.add(y)
                
                path = get_shortest_path_between(G, q_entities)
                if path:
                    paths_for_pair.append(path)
            
            # Find most frequent path for this pair
            if paths_for_pair:
                most_freq_path = find_most_frequent_path(paths_for_pair)
                if most_freq_path and (not selected_path or len(most_freq_path) < len(selected_path)):
                    selected_path = most_freq_path
        
        # Add path to summary if found
        if selected_path:
            visited.add(x)
            path_triples = resolve_path_to_triples(G, selected_path)
            
            # Check if adding these triples would exceed our target
            if len(S) + len(path_triples) > target_triples:
                remaining = target_triples - len(S)
                if remaining > 0:
                    # Take first 'remaining' triples from the path
                    triples_to_add = set(list(path_triples)[:remaining])
                    S.update(triples_to_add)
                break
            else:
                S.update(path_triples)
    return S


def build_global_isummary_with_target_size(KG,  target_triples, query_log,):
    """
    Build global summary by combining personalized summaries for each query entity.
    """
    # Extract all unique entities from queries
    all_query_entities = set()
    for q in query_log:
        entities = extract_entities_from_query(q)
        all_query_entities.update(entities)
    
    #print(f"\n[INFO] Found {len(all_query_entities)} unique entities in queries")
    
    # Build graph once to check node existence
    G = nx.DiGraph()
    for s_kg, preds in KG.triples_.items():
        s_str = str(s_kg)
        for p, objs in preds.items():
            for o in objs:
                o_str = str(o)
                G.add_edge(s_str, o_str, predicate=p)
    
    # Filter entities that exist in KG AND in the graph we built
    valid_entities = []
    for entity in all_query_entities:
        if entity in G:
            valid_entities.append(entity)
        else:
            pass

    
    if not valid_entities:
        return set()
    
    # Calculate target size per entity
    entity_freq = Counter()
    for q in query_log:
        entities = extract_entities_from_query(q)
        for entity in entities:
            if entity in valid_entities:
                entity_freq[entity] += 1
    
    total_freq = sum(entity_freq.values())
    all_summary_triples = set()

    
    for i, (entity, freq) in enumerate(entity_freq.most_common()):
        if len(all_summary_triples) >= target_triples:
            break
            
        entity_share = max(1000, int((freq / total_freq) * target_triples))
        

        try:
            entity_summary = build_isummary_with_target_size(KG, query_log, entity, entity_share)
            all_summary_triples.update(entity_summary)
        except Exception as e:
            continue
        
    return all_summary_triples





def build_local_structure(KG, entity, target_triples):
    """Build local structure around an entity when queries don't provide enough."""
    S = set()
    G = nx.DiGraph()
    
    # Build local graph
    for s_kg, preds in KG.triples_.items():
        s_str = str(s_kg)
        for p, objs in preds.items():
            for o in objs:
                o_str = str(o)
                G.add_edge(s_str, o_str, predicate=p)
    
    if entity not in G:
        return S
    
    # Add immediate neighbors
    entity_str = str(entity)
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


def ISUMMARY(KG, target_triples,  query_log):

    all_summary_triples = build_global_isummary_with_target_size(KG, target_triples,query_log)


    if not all_summary_triples:
        return 0, 0, 0

    # Create and evaluate summary
    summary = SummaryDBPedia(all_summary_triples, KG)
    
    #print(f"Summary stats: {summary.number_of_triples()} triples, "
    #      f"{summary.number_of_entities()} entities, "
     #     f"{summary.number_of_relations()} relations")


    return summary



 
