
import random

def generate_random_walks_directed_with_weights(graph, num_walks = 2, walk_length = 3):
    """
    Generate random walks for each node in a directed weighted graph.

    param graph: Dictionary representation of the graph {node: {neighbor: weight}}.
    param nuum_walks: Number of random walks to generate per node.
    param walk_length: Length of each random walk.
    reeturn: dictionary where each key is a node and the value is a list of random walks starting from that node.
    """
    precomputed_walks = {}

    for node in graph:
        walks = []
        for _ in range(num_walks):
            walk = [node]  # Start the walk at the current node
            current_node = node

            for _ in range(walk_length):
                if graph[current_node]:  # Ensure the current node has outgoing neighbors
                    neighbors = list(graph[current_node].keys())
                    weights = list(graph[current_node].values())

                    # Choose a neighbor based on edge weights
                    current_node = random.choices(neighbors, weights=weights, k=1)[0]
                    walk.append(current_node)
                else:
                    break  # If no outgoing neighbors, terminate the walk
            walks.append(walk)
        precomputed_walks[node] = walks

    return precomputed_walks


def update_random_walks_directed_with_weights(graph, precomputed_walks, changes, num_walks=2, walk_length=3):
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

    # Handle edge removals
    for u, v, weight in changes.get('edges_removed', []):
        for node, walks in precomputed_walks.items():
            updated_walks = []
            for walk in walks:
                # Check if the removed edge exists in the walk
                if len(walk) > 1 and (u, v) in zip(walk[:-1], walk[1:]):
                    # Find the index of the removed edge in the walk
                    edge_index = list(zip(walk[:-1], walk[1:])).index((u, v))
                    walk = walk[:edge_index + 1]  # Truncate walks at the first occurrence of the removed edge
                updated_walks.append(walk)  # Save the truncated walk
            precomputed_walks[node] = updated_walks

    # Handle node removals
    for node in changes.get('nodes_removed', []):
        # Remove walks starting from the deleted node
        if node in precomputed_walks:
            precomputed_walks.pop(node, None)

        # Update walks passing through the removed node
        for key, walks in precomputed_walks.items():
            updated_walks = []
            for walk in walks:
                if node in walk:
                    walk = walk[:walk.index(node)]  # Truncate walk at the removed node
                updated_walks.append(walk)
            precomputed_walks[key] = updated_walks
        
    
    # Handle node/edge additions: Extend walks that don't meet the required length
    for node, walks in precomputed_walks.items():
        updated_walks = []
        for walk in walks:
            while len(walk) < walk_length:
                last_node = walk[-1]
                if last_node in graph and graph[last_node]:  # Ensure valid directed neighbors with weights
                    neighbors = list(graph[last_node].keys())
                    weights = list(graph[last_node].values())
                    next_node = random.choices(neighbors, weights=weights, k=1)[0]
                    walk.append(next_node)
                else:
                    break
            updated_walks.append(walk[:walk_length])  # Ensure the walk does not exceed max length
        precomputed_walks[node] = updated_walks


    return precomputed_walks


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



