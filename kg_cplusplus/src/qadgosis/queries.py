import networkx as nx
import random

def synthetic_graph(num_timesteps, n_nodes, m_edges):

    """
    Generate a synthetic dynamic graph with changes at each timestep.
    param num_timesteps: int, number of timesteps
    param n_nodes: int, number of nodes
    param m_edges: int, number of edges to attach per new node
    return: dynamic_graph: dict, dictionary of dynamic graphs at each timestep
    """
    dynamic_graph = {}
    base_graph = nx.powerlaw_cluster_graph(n_nodes, m_edges, p=0.05)  # Undirected power-law graph
    
    # Convert to directed graph with random edges
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(base_graph.nodes())
    for u, v in base_graph.edges():
        if random.random() > 0.5:
            directed_graph.add_edge(u, v)
        else:
            directed_graph.add_edge(v, u)
    
    # Assign random weights to edges
    for u, v in directed_graph.edges():
        directed_graph[u][v]['weight'] = random.uniform(0.5, 10)
    
    dynamic_graph[0] = directed_graph
    
    # Incrementally update the graph for subsequent timesteps
    for t in range(1, num_timesteps):
        previous_graph = dynamic_graph[t - 1].copy()
        
        # Randomly modify a small number of edges
        for _ in range(random.randint(1, 10)):  # Modify 1 to 5 edges
            u, v = random.sample(range(n_nodes), 2)
            if previous_graph.has_edge(u, v):
                previous_graph.remove_edge(u, v)  
            else:
                previous_graph.add_edge(u, v)  
        
        # update weights for small subset of modified edges 
        for u, v in random.sample(list(previous_graph.edges()), k=min(len(previous_graph.edges()), 3)):
            previous_graph[u][v]['weight'] = random.uniform(0.5, 10)


        dynamic_graph[t] = previous_graph
    
    return dynamic_graph


    

def generate_query(graph, path_len=2, qid=0, constraint_index=None):
    """
    Generate a random query with weights.
    param graph: Dictionary representation of the graph {node: {neighbor: weight}}.
    param chain_len: Maximum number of edges in the random path
    param qid: Unique identifier for the query.
    param constraint_index: Optional index to specify where the constraint node is placed.
    return: Generated query as a dictionary.
    """
    # Randomly select a starting node
    start_node = random.choice(list(graph.keys()))
    
    random_path = []
    visited_nodes = [start_node]
    current_node = start_node

    # Build the random path
    for i in range(path_len):
        if current_node not in graph or not graph[current_node]:#error handling
            break

        # Get neighbors and weights
        neighbors = list(graph[current_node].keys())
        weights = list(graph[current_node].values())
        
        # Normalize weights to create a probability distribution
        total_weight = sum(weights)
        if total_weight == 0:
            break
        probabilities = [w / total_weight for w in weights]

        # Select the next node probabilistically based on the weights
        next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
        random_path.append([current_node, next_node])
        visited_nodes.append(next_node)
        current_node = next_node

    # Enforce constraint node from the visited nodes if not none
    constraint_node = None
    if constraint_index is not None and 0 <= constraint_index < len(visited_nodes):
        constraint_node = visited_nodes[constraint_index]

#return {
       # "Queryid": str(qid),
      #      "StartNode": start_node,
            #"RandomPath": random_path,
            #"ConstraintNode": constraint_node,
            #"VisitedNodes": visited_nodes,
            #"EndNode": visited_nodes[-1] if visited_nodes else None
        
  #  }
    return start_node