#include "walks.cpp"
#include "mainwrap.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;

// Define our graph as an adjacency map: for each node an unordered_map of neighbor->weight.
using Graph_ = unordered_map<int, unordered_map<int, double>>;
// And the directed graph for random walk purposes.
using DirectedGraph = unordered_map<int, vector<int>>;



// Structure to hold the result after adding the dummy (query) node.
struct PersonalizedGraphResult {
    Graph_ snapshot;
    int dummy_node;
    unordered_map<int, double> personalization;
};

// Given a graph (snapshot) and a set of queries (mapping query index -> node id for a given timestep),
// this function makes a deep copy of the snapshot, adds a dummy node with an edge (weighted equally)
// to each query node, and scales all other edge weights.
PersonalizedGraphResult personalized_graph(const Graph_ &graph, 
                                             const unordered_map<int, int> &queries,
                                             int timestep) {
    // Make a copy of the graph.
    Graph_ snapshot = graph;
    // Extract query nodes from the queries mapping.
    vector<int> query_nodes;
    for (const auto &p : queries) {
        query_nodes.push_back(p.second);
    }
    // Define a dummy node. (In Python the dummy was built by concatenating "123456789" with the timestep;
    // here we simply add the timestep to a base number.)
    int dummy_node = 123456789 + timestep;
    double equal_weight = 1.0 / query_nodes.size();
    // Add an edge from the dummy node to each query node.
    for (int node : query_nodes) {
        snapshot[dummy_node][node] = equal_weight;
    }
    // Build the personalization vector.
    unordered_map<int, double> personalization;
    for (int node : query_nodes) {
        personalization[node] = equal_weight;
    }
    // Scale the weights of all edges not coming from the dummy node.
    for (auto &u_pair : snapshot) {
        int u = u_pair.first;
        if(u != dummy_node) {
            for (auto &v_pair : u_pair.second) {
                v_pair.second = v_pair.second / 10.0;
            }
        }
    }
    return {snapshot, dummy_node, personalization};
}

// Structure holding the result of “graph_changes_with_dummy_node”, including the snapshot and a “changes” summary.
struct GraphChangesDummyResult {
    Graph_ snapshot;
    GraphChanges changes;
    int dummy_node;
    unordered_map<int, double> personalization;
};

// Given a dynamic graph (a vector of snapshots) and queries (a mapping per timestep),
// this function computes the dummy–augmented graph for time t and t+1 and then determines
// the changes (nodes/edges added or removed).
GraphChangesDummyResult graph_changes_with_dummy_node(const vector<Graph_> &dynamic_graph, 
                                                       const unordered_map<int, unordered_map<int,int>> &queries, 
                                                       int timestep) {
    auto res1 = personalized_graph(dynamic_graph[timestep], queries.at(timestep), timestep);
    Graph_ snapshot_s = res1.snapshot;
    int dummy_node = res1.dummy_node;
    auto personalization = res1.personalization;
    
    Graph_ snapshot_t;
    // Cast dynamic_graph.size() to int to avoid signed/unsigned comparison warnings
    if(timestep + 1 < static_cast<int>(dynamic_graph.size())){
        snapshot_t = personalized_graph(dynamic_graph[timestep+1], queries.at(timestep+1), timestep+1).snapshot;
    } else {
        snapshot_t = snapshot_s;
    }
    
    // Collect nodes from both snapshots.
    unordered_set<int> nodes_s, nodes_t;
    for (const auto &p : snapshot_s) {
        nodes_s.insert(p.first);
        for (const auto &q : p.second)
            nodes_s.insert(q.first);
    }
    for (const auto &p : snapshot_t) {
        nodes_t.insert(p.first);
        for (const auto &q : p.second)
            nodes_t.insert(q.first);
    }
    
    // Compute added and removed nodes.
    unordered_set<int> nodes_added, nodes_removed;
    for (int node : nodes_t) {
        if (nodes_s.find(node) == nodes_s.end())
            nodes_added.insert(node);
    }
    for (int node : nodes_s) {
        if (nodes_t.find(node) == nodes_t.end())
            nodes_removed.insert(node);
    }
    
    // Compute differences in edges.
    set<pair<int,int>> edges_s, edges_t;
    for (const auto &p : snapshot_s) {
        int u = p.first;
        for (const auto &q : p.second)
            edges_s.insert({u, q.first});
    }
    for (const auto &p : snapshot_t) {
        int u = p.first;
        for (const auto &q : p.second)
            edges_t.insert({u, q.first});
    }
    
    set<pair<int,int>> edges_added, edges_removed;
    for (auto &edge : edges_t) {
        if (edges_s.find(edge) == edges_s.end())
            edges_added.insert(edge);
    }
    for (auto &edge : edges_s) {
        if (edges_t.find(edge) == edges_t.end())
            edges_removed.insert(edge);
    }
    
    GraphChanges changes {nodes_added, nodes_removed, edges_added, edges_removed};
    return {snapshot_s, changes, dummy_node, personalization};
}


// --- Walks, PageRank, and Aggregation ---

// Calculate the required length of the PPR walk: (c/epsilon)*k*(n/k)^epsilon.
int calculate_required_walk_length(double c, double epsilon, int k, int n) {
    double val = (c / epsilon) * k * pow((double)n/k, epsilon);
    return static_cast<int>(val);
}


// Generate random walks on the weighted graph; here we simply simulate num_walks walks of fixed length (10).
RandomWalkResult generate_random_walks(const Graph_ &weighted_graph, int num_walks) {
    RandomWalkResult result;
    int walk_length = 10;
    random_device rd;
    mt19937 gen(rd());
    vector<int> nodes;
    for (const auto &p : weighted_graph)
        nodes.push_back(p.first);
    
    for (int i = 0; i < num_walks; i++){
        vector<int> walk;
        // Pick a random starting node.
        uniform_int_distribution<> dis(0, nodes.size()-1);
        int current = nodes[dis(gen)];
        walk.push_back(current);
        for (int j = 1; j < walk_length; j++){
            if (weighted_graph.find(current) != weighted_graph.end() && !weighted_graph.at(current).empty()){
                vector<int> neighbors;
                for (const auto &nbr : weighted_graph.at(current))
                    neighbors.push_back(nbr.first);
                uniform_int_distribution<> dis2(0, neighbors.size()-1);
                current = neighbors[dis2(gen)];
            } else {
                break;
            }
            walk.push_back(current);
        }
        result.precomputed_walks[i].push_back(walk);
    }
    return result;
}

RandomWalkResult generate_random_walks_directed_weighted(const Graph_& weighted_graph, int num_walks, int walk_length) {
    RandomWalkResult result;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> nodes;
    for (const auto& p : weighted_graph)
        nodes.push_back(p.first);  // Only consider nodes with outgoing edges

    for (int i = 0; i < num_walks; i++) {
        std::vector<int> walk;

        // Pick a random starting node
        std::uniform_int_distribution<> dis(0, nodes.size() - 1);
        int current = nodes[dis(gen)];
        walk.push_back(current);

        for (int j = 1; j < walk_length; j++) {
            const auto it = weighted_graph.find(current);
            if (it == weighted_graph.end() || it->second.empty())
                break;

            const auto& neighbors_map = it->second;

            std::vector<int> neighbors;
            std::vector<double> weights;
            for (const auto& [neighbor, weight] : neighbors_map) {
                neighbors.push_back(neighbor);
                weights.push_back(weight);
            }

            double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
            if (sum_weights > 0.0) {
                for (auto& w : weights)
                    w /= sum_weights;  // Normalize to probabilities
            } else {
                weights.assign(neighbors.size(), 1.0 / neighbors.size()); // Uniform fallback
            }

            std::discrete_distribution<> weight_dist(weights.begin(), weights.end());
            current = neighbors[weight_dist(gen)];
            walk.push_back(current);
        }

        result.precomputed_walks[i].push_back(walk);
    }

    return result;
}


// Dummy function to update random walks. Here we simply re‐generate them.
RandomWalkResult update_random_walks(int timestep, const Graph_ &weighted_graph, 
                                     const RandomWalkResult &previous_data, 
                                     const GraphChanges &changes, int num_walks) {
    return generate_random_walks(weighted_graph, num_walks);
}

RandomWalkResult update_random_walks_directed_with_weights(
    const Graph_& graph,
    std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
    const GraphChanges& changes,
    int num_walks,
    int walk_length
) {
    RandomWalkResult result;

    using RNG = std::mt19937;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RNG rng(seed);

    // Build neighbor and weight lookup tables
    std::unordered_map<int, std::vector<int>> neighbors_dict;
    std::unordered_map<int, std::vector<double>> weights_dict;

    for (const auto& [node, neighbors] : graph) {
        for (const auto& [nbr, weight] : neighbors) {
            neighbors_dict[node].push_back(nbr);
            weights_dict[node].push_back(weight);
        }
    }

    // Handle removed nodes
    for (int node : changes.nodes_removed) {
        precomputed_walks.erase(node);  // Remove entire entry
    }

    // Handle added nodes (generate fresh walks)
    for (int node : changes.nodes_added) {
        std::vector<std::vector<int>> walks;
        for (int i = 0; i < num_walks; ++i) {
            std::vector<int> walk = {node};
            int current = node;

            for (int j = 0; j < walk_length; ++j) {
                if (neighbors_dict.find(current) != neighbors_dict.end() && !neighbors_dict[current].empty()) {
                    auto& neighbors = neighbors_dict[current];
                    auto& weights = weights_dict[current];
                    std::discrete_distribution<int> dist(weights.begin(), weights.end());
                    current = neighbors[dist(rng)];
                    walk.push_back(current);
                } else {
                    break;
                }
            }

            walks.push_back(std::move(walk));
        }
        precomputed_walks[node] = std::move(walks);
    }

    // Handle edge removals — truncate any walks that pass through a removed edge
    for (auto& [node, walks] : precomputed_walks) {
        for (auto& walk : walks) {
            for (size_t i = 0; i + 1 < walk.size(); ++i) {
                if (changes.edges_removed.count({walk[i], walk[i + 1]})) {
                    walk.resize(i + 1);  // Truncate at the edge
                    break;
                }
            }
        }
    }

    // Extend all truncated or too-short walks
    for (auto& [node, walks] : precomputed_walks) {
        for (auto& walk : walks) {
            while (walk.size() < static_cast<size_t>(walk_length + 1)) {
                int last = walk.back();
                if (neighbors_dict.find(last) != neighbors_dict.end() && !neighbors_dict[last].empty()) {
                    auto& neighbors = neighbors_dict[last];
                    auto& weights = weights_dict[last];
                    std::discrete_distribution<int> dist(weights.begin(), weights.end());
                    int next = neighbors[dist(rng)];
                    walk.push_back(next);
                } else {
                    break;
                }
            }
        }
    }

    result.precomputed_walks = precomputed_walks;
    return result;
}


// Process the raw integer–based walks to string form.
unordered_map<int, vector<vector<string>>> process_walk_results(const unordered_map<int, vector<vector<int>>> &precomputed_walks) {
    unordered_map<int, vector<vector<string>>> result;
    for (const auto &p : precomputed_walks) {
        vector<vector<string>> walks_str;
        for (const auto &walk : p.second) {
            vector<string> temp;
            for (int val : walk)
                temp.push_back(to_string(val));
            walks_str.push_back(temp);
        }
        result[p.first] = walks_str;
    }
    return result;
}

// A dummy implementation of a personalized PageRank–like random walk.
// Starting from node w, it performs L steps; with probability epsilon it teleports to w,
// and otherwise it randomly follows an outgoing edge.
vector<int> compute_personalized_pagerank(int w, int L, double epsilon, 
                                          const unordered_map<int, vector<vector<string>>> &R,
                                          const DirectedGraph &directed_graph) {
    vector<int> walk;
    int current = w;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    walk.push_back(current);
    for (int i = 1; i < L; i++) {
        if (dis(gen) < epsilon) {
            current = w;
        } else {
            if (directed_graph.find(current) != directed_graph.end() && !directed_graph.at(current).empty()) {
                const vector<int>& neighbors = directed_graph.at(current);
                uniform_int_distribution<> dis2(0, neighbors.size() - 1);
                current = neighbors[dis2(gen)];
            }
        }
        walk.push_back(current);
    }
    
    // Print the length of the walk
    cout << "Length of walk: " << walk.size() << endl;

    return walk;
}

// Update cumulative counts from a walk and extract the top–k nodes (normalized).
unordered_map<int, double> calculate_top_k_cumulative(const vector<int> &walk, 
                                                        unordered_map<int,int> &cumulative_counts, 
                                                        int k) {
    for (int node : walk)
        cumulative_counts[node]++;
    
    int total = 0;
    for (auto &p : cumulative_counts)
        total += p.second;
    
    vector<pair<int, double>> norm_counts;
    for (auto &p : cumulative_counts) {
        double normalized = (total > 0) ? (double(p.second) / total) : 0.0;
        norm_counts.push_back({p.first, normalized});
    }
    sort(norm_counts.begin(), norm_counts.end(), [](auto &a, auto &b) {
        return a.second > b.second;
    });
    unordered_map<int, double> top_k;
    for (int i = 0; i < min(k, (int)norm_counts.size()); i++)
        top_k[norm_counts[i].first] = norm_counts[i].second;
    
    return top_k;
}

// Function to update cumulative counts and calculate the top-k nodes with weights and exponential decay.
std::pair<std::unordered_map<int, double>, std::unordered_map<int, double>> calculate_top_k_cumulative_with_weights(
    const std::vector<int>& walk,
    std::unordered_map<int, int>& cumulative_counts,
    int k,
    int timestep,
    std::unordered_map<int, std::unordered_map<int, int>>& counters,
    double decay_rate = 0.38,
    int window_size = 5 
) {
    // Update the current timestep's counts
    for (int node : walk) {
        counters[timestep][node]++;
    }

    // Determine relevant timesteps for exponential decay
    std::vector<int> keys;
    for (const auto& entry : counters) {
        keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());
    int latest_timestep = keys.back();
    int start_index = std::max(0, static_cast<int>(keys.size()) - window_size); // Use num_timesteps explicitly
    std::vector<int> relevant_timesteps(keys.begin() + start_index, keys.end());

    // Apply exponential decay to counts
    std::unordered_map<int, int> decayed_counts;
    std::unordered_map<int, double> individual_weights;
    for (int j : relevant_timesteps) {
        double weight = std::pow(decay_rate, latest_timestep - j);
        individual_weights[j] = weight;

        for (const auto& entry : counters[j]) {
            int node = entry.first;
            int count = entry.second;
            decayed_counts[node] += static_cast<int>(count * weight);
        }
    }

    // Update cumulative counts
    for (const auto& entry : decayed_counts) {
        cumulative_counts[entry.first] += entry.second;
    }

    // Normalize counts
    int total_cumulative_walks = std::accumulate(
        cumulative_counts.begin(),
        cumulative_counts.end(),
        0,
        [](int sum, const std::pair<int, int>& p) { return sum + p.second; }
    );

    std::vector<std::pair<int, double>> normalized_counts;
    for (const auto& entry : cumulative_counts) {
        double normalized = (total_cumulative_walks > 0) ? static_cast<double>(entry.second) / total_cumulative_walks : 0.0;
        normalized_counts.emplace_back(entry.first, normalized);
    }

    // Sort normalized counts and select top-k nodes
    std::sort(normalized_counts.begin(), normalized_counts.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::unordered_map<int, double> top_k_cumulative;
    for (int i = 0; i < std::min(k, static_cast<int>(normalized_counts.size())); i++) {
        top_k_cumulative[normalized_counts[i].first] = normalized_counts[i].second;
    }

    return {top_k_cumulative, individual_weights};
}


// Modify the function to return the new structure
TopKSummaryWithWeights generate_top_k_summary(
    int num_timesteps,
    int num_walks,
    std::vector<Graph_>& dynamic_graph,
    std::unordered_map<int, std::unordered_map<int, int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate
) {
    TopKSummary summary;
    std::unordered_map<int, std::unordered_map<int, double>> weights; // Declare weights
    std::unordered_map<int, double> runtime_per_timestep; // Dictionary to store runtimes

    std::unordered_map<int, int> cumCounts;  // cumulative counts for nodes
    std::unordered_map<int, std::unordered_map<int, int>> counters;  // counters for each timestep

    for (int t = 0; t < static_cast<int>(dynamic_graph.size()); t++) {
        cout << "Processing timestep " << t << endl;
        auto total_start = chrono::high_resolution_clock::now();
        auto graph_changes_result = graph_changes_with_dummy_node(dynamic_graph, queries, t);
        summary.changes_over_time[t] = graph_changes_result.changes;

        // Count all nodes in the snapshot
        unordered_set<int> nodes_set;
        for (const auto& p : graph_changes_result.snapshot) {
            nodes_set.insert(p.first);
            for (const auto& q : p.second)
                nodes_set.insert(q.first);
        }
        int n = nodes_set.size();
        int L = calculate_required_walk_length(c, epsilon, k, n);
        int walk_length = static_cast<int>(std::round(1.0 / epsilon));
        cout << "average walk length " << t << ": " << walk_length << endl;
        cout << "Required length of PPR walk " << t << ": " << L << endl;

        // Preprocess (we already have weighted edges)
        Graph_ weighted_graph = graph_changes_result.snapshot;

        RandomWalkResult R;
        if (t == 0) {
            R = generate_random_walks_directed_weighted(weighted_graph, num_walks, walk_length);
            summary.random_walks[t] = R;
        } else {
            R = update_random_walks_directed_with_weights(weighted_graph, summary.random_walks[t - 1].precomputed_walks, summary.changes_over_time[t], num_walks, walk_length);
            summary.random_walks[t] = R;
        }

        // Create a directed graph for the walk process
        DirectedGraph directed_graph;
        for (const auto& p : weighted_graph) {
            int u = p.first;
            for (const auto& q : p.second)
                directed_graph[u].push_back(q.first);
        }

        // Process the random walk results
        auto processed_R = process_walk_results(R.precomputed_walks);
        auto start = chrono::high_resolution_clock::now();
        vector<int> walk = compute_personalized_pagerank(graph_changes_result.dummy_node, L, epsilon, processed_R, directed_graph);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        summary.ppr_walks[t] = walk;
        auto time_start = chrono::high_resolution_clock::now();

        // Calculate top-k cumulative with weights
        auto top_k_result = calculate_top_k_cumulative_with_weights(walk, cumCounts, k, t, counters, decay_rate, num_timesteps);
        auto top_k = top_k_result.first;
        weights[t] = top_k_result.second; // Store weights for the current timestep

        summary.top_k_cumsum_walks[t] = top_k;

        auto time_end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_topk = time_end - time_start;

        auto total_end = chrono::high_resolution_clock::now();
        chrono::duration<double> total_elapsed = total_end - total_start;
        cout << "Total time taken for timestep " << t << ": " << total_elapsed.count() << " seconds" << endl;

        // Store runtime in dictionary
        runtime_per_timestep[t] = total_elapsed.count();
    }

    // Return both summary, weights, and runtime dictionary
    return {summary, weights, runtime_per_timestep}; // Updated return statement
}

// Dummy pagerank: for each snapshot, assign a random score to nodes (ignoring the dummy nodes)
// and output the top–k nodes.
unordered_map<int, unordered_map<int, double>> pagerank(const vector<Graph_> &dynamic_graph,
                                                         const unordered_map<int, unordered_map<int,int>> &queries,
                                                         int k) {
    unordered_map<int, unordered_map<int, double>> top_k_power;
    int timesteps = dynamic_graph.size();
    for (int t = 0; t < timesteps; t++){
        cout << "Pagerank Timestep " << t << endl;
        auto res = personalized_graph(dynamic_graph[t], queries.at(t), t);
        unordered_map<int, double> scores;
        for (const auto &p : res.snapshot) {
            int node = p.first;
            // Exclude dummy node (dummy nodes are in a defined range here).
            if (node >= 123456789 && node < 123456789 + 100)
                continue;
            scores[node] = ((double) rand() / RAND_MAX);
        }
        vector<pair<int, double>> vec(scores.begin(), scores.end());
        sort(vec.begin(), vec.end(), [](auto &a, auto &b){
            return a.second > b.second;
        });
        unordered_map<int, double> top_k_nodes;
        for (int i = 0; i < min(k, (int)vec.size()); i++){
            top_k_nodes[vec[i].first] = vec[i].second;
        }
        top_k_power[t] = top_k_nodes;
    }
    return top_k_power;
}



// Wrap generate_top_k_summary so that it can be called from Python.
py::dict wrap_generate_top_k_summary(int num_timesteps,
    int num_walks,
    vector<Graph_>& dynamic_graph,
    unordered_map<int, unordered_map<int,int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate) {
// Call the original C++ function; internally it calls the C++ walks library.
TopKSummaryWithWeights result = generate_top_k_summary(num_timesteps,  num_walks, dynamic_graph, queries, c, epsilon, k, decay_rate);

// Access the summary, weights, and runtime from the result
TopKSummary summary = result.summary;
std::unordered_map<int, std::unordered_map<int, double>> weights = result.weights;
std::unordered_map<int, double> runtime_per_timestep = result.runtime_per_timestep;

// Prepare a Python dictionary to hold the results.
py::dict result_dict;

// Convert runtime_per_timestep to a Python dict
py::dict runtime_dict;
for (const auto& entry : runtime_per_timestep) {
    runtime_dict[py::int_(entry.first)] = entry.second;
}
result_dict["runtime_per_timestep"] = runtime_dict;

// Convert changes_over_time (unordered_map<int, GraphChanges>) to a Python dict.
py::dict changes_over_time;
for (const auto &entry : summary.changes_over_time) {
py::dict ch;
// Convert unordered_set and std::set to std::vector (pybind11 converts std::vector to Python list automatically).
std::vector<int> nodes_added(entry.second.nodes_added.begin(), entry.second.nodes_added.end());
std::vector<int> nodes_removed(entry.second.nodes_removed.begin(), entry.second.nodes_removed.end());
// For edges, use a vector of pairs.
std::vector<std::pair<int, int>> edges_added(entry.second.edges_added.begin(), entry.second.edges_added.end());
std::vector<std::pair<int, int>> edges_removed(entry.second.edges_removed.begin(), entry.second.edges_removed.end());
ch["nodes_added"] = nodes_added;
ch["nodes_removed"] = nodes_removed;
ch["edges_added"] = edges_added;
ch["edges_removed"] = edges_removed;
changes_over_time[py::int_(entry.first)] = ch;
}
result_dict["changes_over_time"] = changes_over_time;

// Convert ppr_walks (unordered_map<int, vector<int>>)
py::dict ppr_walks;
for (const auto &item : summary.ppr_walks) {
ppr_walks[py::int_(item.first)] = item.second;
}
result_dict["ppr_walks"] = ppr_walks;

// Convert top_k_cumsum_walks (unordered_map<int, unordered_map<int, double>>)
py::dict top_k_cumsum_walks;
for (const auto &entry : summary.top_k_cumsum_walks) {
py::dict inner;
for (const auto &p : entry.second) {
inner[py::int_(p.first)] = p.second;
}
top_k_cumsum_walks[py::int_(entry.first)] = inner;
}
result_dict["top_k_cumsum_walks"] = top_k_cumsum_walks;

// Convert weights (unordered_map<int, unordered_map<int, double>>)
py::dict all_weights;
for (const auto& entry : weights) {
py::dict inner;
for (const auto& p : entry.second) {
inner[py::int_(p.first)] = p.second;
}
all_weights[py::int_(entry.first)] = inner;
}
result_dict["all_weights"] = all_weights;

return result_dict;
}

PYBIND11_MODULE(main, m) {
m.doc() = "Python bindings for generate_top_k_summary function using the C++ walks library";
m.def("generate_top_k_summary", &wrap_generate_top_k_summary,
"Generate a top-k summary from dynamic graph updates.",
py::arg("num_timesteps"), 
py::arg("num_walks"), py::arg("dynamic_graph"), py::arg("queries"), 
py::arg("c"), py::arg("epsilon"), py::arg("k"), py::arg("decay_rate"));
}

