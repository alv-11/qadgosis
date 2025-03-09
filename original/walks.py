
import random
from collections import defaultdict
import string
import uuid

def generate_random_walks_directed_with_weights(dyngraph, num_walks=2, walk_length=3):
    """
    Generate random walks for each node in a directed weighted graph.

    param graph: Dictionary representation of the graph {node: {neighbor: weight}}.
    param nuum_walks: Number of random walks to generate per node.
    param walk_length: Length of each random walk.
    reeturn: dictionary where each key is a node and the value is a list of random walks starting from that node.
    """
    # Convert NetworkX graph to expected dictionary format
    graph = {
        node: {nbr: data.get('weight', 0) for nbr, data in dyngraph[node].items()}
        for node in dyngraph.nodes()
    }

    precomputed_walks = dict()
    inverted_edges = defaultdict(list) #create a dictionary with edges as keys and an empty list as values
    [inverted_edges[key] for key in dyngraph.edges()]#ensure all edges are present in the dictionary

    inverted_nodes = defaultdict(set)
    [inverted_nodes[key] for key in dyngraph.nodes()]

    walk_ids = dict()
    random_walk_ids = dict()

    for node in graph:        
        walks = []
        random_ids= []
        for _ in range(num_walks):
            walk = [node]  # Start the walk at the current node
            current_node = node
            for _ in range(walk_length):
                if graph[current_node]:  # Ensure the current node has outgoing neighbors
                    neighbors = list(graph[current_node].keys())
                    weights = list(graph[current_node].values())
                    # Choose a neighbor based on edge weights
                    next_node = random.choices(neighbors, weights=weights, k=1)[0]
                    walk.append(next_node)

                    # Generate a unique ID for the walk 
                    unique_id = uuid.uuid4()#''.join(random.choices(string.ascii_letters + string.digits, k=8))
                    current_node = next_node
                else:
                    break  
            walk_ids[unique_id] = walk
            # Update the inverted_edges dictionary
            for  u, v in zip(walk[:-1], walk[1:]):
                inverted_edges[(u,v)].append(unique_id)
        
            # Update the inverted_nodes dictionary
            for any_node in walk:
                inverted_nodes[any_node].add(unique_id)
                    
            walks.append(walk)
            random_ids.append(unique_id)
        random_walk_ids[node] = random_ids #nodes to ids

    for node, walks in random_walk_ids.items():
        precomputed_walks[node] = [walk_ids[walk] for walk in walks if walk in walk_ids] #nodes to walks that start from that node

    return precomputed_walks, inverted_edges, inverted_nodes, walk_ids, random_walk_ids


def update_random_walks_directed_with_weights(dyngraph,  ids_to_walks, inverted_edges, inverted_nodes, startnode_walk_ids, changes, num_walks=2, walk_length=3):
    """
    Update precomputed random walks based on changes in the graph, ensuring walks remain directed, weighted, and consistent.
    
    param graph: Dictionary representation of the graph {node: {neighbor: weight}}.
    param precomputed_walks: Dictionary containing precomputed random walks for each node.
    param changes: Dictionary containing changes to the graph.
                      Example: {
                          'nodes_added': [node1, node2, ...],
                          'nodes_removed': [node1, node2, ...],
                          'edges_added': [(u, v, weight), ...],
                          'edges_removed': [(u, v), ...]
                      }
    param num_walks: Number of random walks to generate per node.
    param walk_length: Length of each random walk.
    
    return: Updated precomputed random walks.
    """

    graph = {
        node: {nbr: data.get('weight', 0) for nbr, data in dyngraph[node].items()}
        for node in dyngraph.nodes()
    }
    truncated_walk_ids={}

    # Handle edge removals
    for edge in changes.get('edges_removed', []):
        if edge in inverted_edges:
            affected_walks = inverted_edges.pop(edge)

            u, v = edge       
            # Update precomputed walks
            for walk_id in affected_walks:
                walk = ids_to_walks[walk_id]
                while len(walk) > 1 and (u,v) in zip(walk[:-1], walk[1:]):
                    edge_index = list(zip(walk[:-1], walk[1:])).index((u, v))
                    walk = walk[:edge_index + 1]  # Truncate the walk at the removed edge
                    
                    truncated_walk_ids[walk_id] = walk


    # Handle node removals
    for node in changes.get('nodes_removed', []):
        if node in inverted_nodes:
            affected_walks = inverted_nodes.pop(node)
            startnode_walk_ids.pop(node, None)#remove random walks for removed nodes

            # Update precomputed walks
            for walk_id in affected_walks:
                walk = ids_to_walks[walk_id]
                while len(walk) > 1 and node in walk:
                    walk = walk[:walk.index(node)]  # Truncate the walk at the removed node
                truncated_walk_ids[walk_id] = walk


    # Handle edge additions
    ## Extend the trunctaed walks
    for walk_id, walk in truncated_walk_ids.items():
        while len(walk) < walk_length+1 and len(walk) > 0: #ensure walk is not empty and less than walk length
            last_node = walk[-1] #this will be the last node in the walk, works for new edges starting at those nodes
            if graph[last_node]:  # Ensure valid neighbors
                neighbors = list(graph[last_node].keys())
                weights = list(graph[last_node].values())
                next_node = random.choices(neighbors, weights=weights, k=1)[0]
                walk.append(next_node)
                last_node = next_node
            else:
                break

        truncated_walk_ids[walk_id] = walk
        # For each edge in the walk, associate the unique ID with the edge
        for u, v in zip(walk[:-1], walk[1:]):  
            inverted_edges[(u, v)].append(walk_id)
        for any_node in walk:
            inverted_nodes[any_node].add(walk_id)

    try:
        print("percentage of walks affected", 100*(len(truncated_walk_ids)/len(ids_to_walks)))
    except:
        print("percentage of walks affected", 0)

    #update the walk_ids with the extended truncated walks
    shared_keys = set(ids_to_walks.keys()).intersection(truncated_walk_ids.keys())
    for key in shared_keys:
        ids_to_walks[key] = truncated_walk_ids[key]


    #handle new node additions:
    for node in changes.get('nodes_added', []):
        walks = []
        random_ids = []
        for _ in range(num_walks):
            walk = [node]  # Start the walk at the current node
            current_node = node
            visited = set()  # Keep track of visited nodes
            visited
            for _ in range(walk_length):
                if graph[current_node]:  # Ensure the current node has outgoing neighbors
                    neighbors = list(graph[current_node].keys())
                    weights = list(graph[current_node].values())
                # Choose a neighbor based on edge weights
                    next_node = random.choices(neighbors, weights=weights, k=1)[0]
                    walk.append(next_node)

                    unique_id = uuid.uuid4()#''.join(random.choices(string.ascii_letters + string.digits, k=8))
                    
            ids_to_walks[unique_id] = walk
            # For each edge in the walk, associate the unique ID with the edge
            for u, v in zip(walk[:-1], walk[1:]):  
                inverted_edges[(u, v)].append(unique_id)
            for any_node in walk:
                inverted_nodes[any_node].add(unique_id)
                    
            walks.append(walk)
            random_ids.append(unique_id)

       #precomputed_walks[node] = walks #may not need this eventually
        startnode_walk_ids[node] = random_ids
        
    #reconstruct the precomputed walks
    precomputed_walks = {}
    for node, walk_ids in startnode_walk_ids.items():
        precomputed_walks[node] = [ids_to_walks[walk_id] for walk_id in walk_ids if walk_id in ids_to_walks]

    return precomputed_walks, inverted_edges, inverted_nodes, ids_to_walks, startnode_walk_ids

 


def fetch_for_node(node, walks_storage, graph):
    # Fetch precomputed walks for the node
    walks = walks_storage.get(node, [])
    neighbors = graph.get(node, [])
    if walks:
        print(f"Fetched walks and neighbours for node {node}")
        return walks, neighbors
    else:
        print(f"No precomputed walks found for node {node}")
        return []


def personalized_complete_pagerank_walk(w, L, epsilon, precomputed_walks, graph):
    """
    Creates a personalized PageRank random walk.

    paramm w: Source node (starting node of the walk)
    param L: Desired walk length
    param epsilon: Teleportation (reset) probability
    param precomputed_walks: Dictionary of precomputed random walks for each node
    param graph: Dictionary representing the adjacency list of the graph

    Returns:
    Pw: List representing the personalized PageRank walk
    """
    # Initialize the walk
    Pw = [w]
    u = w  # Current node

    # Initialize current walk length
    walk_length = 1  # Start with the first node already added

    # Continue the walk until the length of Pw reaches at least L
    while walk_length < L:
        # Generate a random number to decide the next step
        beta = random.random()
       # print(f"Random number (beta): {beta}")

        if beta < epsilon:
            # Teleportation step (reset the walk to the source node)
            Pw.append(w)
            u = w
            walk_length += 1
          #  print(f"Teleportation to {u}. Current walk: {Pw}")
            continue

        # Check for unused precomputed walks
        else:
            if u in precomputed_walks and precomputed_walks[u]:
                # Use a precomputed walk segment
                walk_segment = precomputed_walks[u].pop(0)  # Take the first unused segment
                Pw.extend(walk_segment)  # Append the segment to the walk
                Pw.append(w)
                walk_length += len(walk_segment) + 1
                u = w  # Reset to the source node
              #  print(f"Use precomputed walk segment: {walk_segment} and teleportation to {w}; Current node: {u}. Current walk: {Pw}")
                continue

            else:
                # Fallback to next neighbor
                if u in graph and graph[u]:
                    # Randomly choose a neighbor
                    neighbor = random.choice(graph[u])
                    Pw.append(neighbor)
                    walk_length += 1
                    u = neighbor
                 #   print(f"Already fetched, move to neighbor {neighbor}.Current node: {u}. Current walk: {Pw}")
                else:
                    # Fetch operation (if u has no neighbors or precomputed walks are empty)
                    try:
                        walks, neighbor = fetch_for_node(u, precomputed_walks, graph)
                        if neighbor:
                            Pw.append(neighbor)
                            walk_length += 1
                            u = neighbor
                            print(f"Fetched neighbor {neighbor}. Current walk: {Pw}")
                        else:
                            print(f"No neighbors or precomputed walks available for node {u}. Walk ends.")
                            break
                    except:
                        print(f"Error occurred while fetching for node {u}. Walk ends.")
                        break

    return Pw



def compute_personalized_pagerank(G, alpha=0.85, num_iterations=100, personalize_node=None):
    """
    Compute personalized PageRank scores using power iteration method.
    """   
    # Initialize the PageRank scores
    N = len(G.nodes())
    ranks = {node: 1/N for node in G.nodes()}  # uniform initialization
    if personalize_node in G.nodes():
        ranks[personalize_node] = 1  # Start teleportation at the specified node
    
    # Perform power iterations
    for iteration in range(num_iterations):
        new_ranks = {}
        for node in G.nodes():
            # Calculate incoming contributions
            rank_sum = 0
            for neighbor in G.predecessors(node):  # Account for directed edges
                weight = G[neighbor][node].get('weight', 1.0)
                out_degree = sum(G[neighbor][nbr].get('weight', 1.0) for nbr in G.successors(neighbor))
                if out_degree > 0:
                    rank_sum += (ranks[neighbor] * weight) / out_degree
            # Apply damping factor
            new_ranks[node] = (1 - alpha) * (1 if node == personalize_node else 0) + alpha * rank_sum
        
        ranks = new_ranks

    return ranks