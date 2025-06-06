
import walks

# Example parameters
num_timesteps = 5
n_nodes = 1000
m_edges = 5
num_queries = 20
num_walks = 100
c = 2.0
epsilon = 0.2
k = 50

# Generate synthetic graph and queries
dynamic_graph = [
    {i: {j: 0.01 for j in range(5) if j != i} for i in range(n_nodes)}
    for _ in range(num_timesteps)
]
queries = {t: {i: i for i in range(num_queries)} for t in range(num_timesteps)}

# Call the function
results = walks.generate_top_k_summary(
    num_timesteps=num_timesteps,
    n_nodes=n_nodes,
    m_edges=m_edges,
    num_queries=num_queries,
    num_walks=num_walks,
    dynamic_graph=dynamic_graph,
    queries=queries,
    c=c,
    epsilon=epsilon,
    k=k
)

# Print results (example)
top_k_cumsum_walks = results["top_k_cumsum_walks"]
for timestep, top_k in top_k_cumsum_walks.items():
    print(f"Timestep {timestep} Top-k nodes:")
    for node, score in top_k.items():
        print(f"Node: {node}, Score: {score}")
