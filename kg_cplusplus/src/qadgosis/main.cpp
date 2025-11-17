#include "walks.cpp"
#include "mainwrap.h"
#include "mc_algo.h"
#include "hybrid.h"

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
    int walk_length = 5;
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
    //std::cout << "Length of walk: " << walk.size() << std::endl;
    

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

std::unordered_map<int, std::unordered_map<int, int>> compute_query_entity_frequencies_by_timestep(
    const std::unordered_map<int, std::unordered_map<int, int>>& query_log) {
    
    // Resulting map: {timestep: {entity_id: frequency}}
    std::unordered_map<int, std::unordered_map<int, int>> freq_by_timestep;

    // Iterate over timesteps and their queries
    for (const auto& [timestep, queries] : query_log) {
        std::unordered_map<int, int> timestep_counter;

        // Count frequencies of entity IDs
        for (const auto& [query_id, entity_id] : queries) {
            timestep_counter[entity_id]++;
        }

        // Store the counter for the current timestep
        freq_by_timestep[timestep] = timestep_counter;
    }

    return freq_by_timestep;
}

std::pair<std::unordered_map<int, double>, std::unordered_map<int, double>> calculate_top_k_cumulative_with_weights_(
    const std::vector<int>& walk,
    std::unordered_map<int, double>& cumulative_counts,  // using double for decayed weights
    int k,
    int timestep,
    std::unordered_map<int, std::unordered_map<int, int>>& counters, // {timestep -> {node -> freq}}
    //const std::unordered_map<int, std::unordered_map<int, int>>& query_freqs, // {timestep -> {query -> freq}}
    double decay_rate = 0.38,
    int window_size = 5,
    int n = 0             // total number of nodes,
) {
    // 1. Count current timestep
    for (int node : walk) {
        counters[timestep][node]++;
    }

    // 2. Get relevant timesteps
    std::vector<int> keys;
    for (const auto& entry : counters) keys.push_back(entry.first);
    std::sort(keys.begin(), keys.end());
    int latest_timestep = keys.back();
    int start_index = std::max(0, static_cast<int>(keys.size()) - window_size);
    std::vector<int> relevant_timesteps(keys.begin() + start_index, keys.end());

    // 3. Apply decay
    std::unordered_map<int, double> decayed_counts;
    std::unordered_map<int, double> individual_weights;
    for (int j : relevant_timesteps) {
        double weight = std::pow(decay_rate, latest_timestep - j); //std::exp(-decay_rate * (latest_timestep - j));
        //double weight = std::pow(decay_rate,  j - latest_timestep); // reverse decay
        individual_weights[j] = weight;

        for (const auto& [node, count] : counters[j]) {
            decayed_counts[node] += count * weight;
        }
    }

    // 4. Update cumulative
    for (const auto& [node, decayed_count] : decayed_counts) {
        cumulative_counts[node] += decayed_count;
    }

    // 5. Normalize
    double total = 0.0;
    for (const auto& [_, value] : cumulative_counts) total += value;


    std::vector<std::pair<int, double>> normalized;
    for (const auto& [node, value] : cumulative_counts)  {
        double norm = (total > 0.0) ? (value / total) : 0.0;
        normalized.emplace_back(node, norm);
    }


    std::sort(normalized.begin(), normalized.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });


    // 6. Build top-k
    std::unordered_map<int, double> top_k;
    std::unordered_set<int> selected;

    for (int i = 0; i < static_cast<int>(normalized.size()) && top_k.size() < static_cast<size_t>(k); ++i) {
        int node = normalized[i].first;
    
        // Skip if the node starts with 1234567
        if (std::to_string(node).rfind("12345", 0) == 0) {
            continue;
        }
    
        top_k[node] = normalized[i].second;
        selected.insert(node);
    }


    // 8. Final padding from cumulative_counts keys if still needed
    for (const auto& [node, _] : cumulative_counts) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = 0.0;
        }
    }

    // 7. Pad with walk nodes if needed
    for (int node : walk) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = 0.0;
        }
    }

    return {top_k, individual_weights};
}


std::pair<std::unordered_map<int, double>, std::unordered_map<int, double>> calculate_top_k_cumulative_with_weights(
    const std::vector<int>& walk,
    std::unordered_map<int, double>& cumulative_counts,  // using double for decayed weights
    int k,
    int timestep,
    std::unordered_map<int, std::unordered_map<int, int>>& counters, // {timestep -> {node -> freq}}
    std::unordered_map<int, double> & pi_v, // global popularity (π(v))
    double decay_rate = 0.38,
    int window_size = 5,
    int n = 0             // total number of nodes,
    

    ) {
    // 1. Count queries for current timestep
    for (int node : walk) {
        counters[timestep][node]++;
    }

    // 2. Collect relevant timesteps (sliding window)
    std::vector<int> keys;
    for (const auto& entry : counters) keys.push_back(entry.first);
    std::sort(keys.begin(), keys.end());
    int latest_timestep = keys.back();
    int start_index = std::max(0, static_cast<int>(keys.size()) - window_size);
    std::vector<int> relevant_timesteps(keys.begin() + start_index, keys.end());

    // 3. Apply decay
    std::unordered_map<int, double> decayed_counts;
    std::unordered_map<int, double> individual_weights;
    for (int j : relevant_timesteps) {
        double weight = std::pow(decay_rate, latest_timestep - j); //std::exp(-decay_rate * (latest_timestep - j));
        //double weight = std::pow(decay_rate,  j - latest_timestep); // reverse decay
        individual_weights[j] = weight;

        for (const auto& [node, count] : counters[j]) {
            decayed_counts[node] += count * weight;
        }
    }

    // 4. Update cumulative counts
    for (const auto& [node, decayed_count] : decayed_counts) {
        cumulative_counts[node] += decayed_count;
    }

    // ------------------------------------------------------------------
    // NEW: Properly compute π(u), π(u,v), π(v), and w(u,v)
    // ------------------------------------------------------------------

    // Step A: π(u) = normalized count of this timestep
    std::unordered_map<int, double> pi_u; // per timestep
    int total_bigwalk = 0;
    for (const auto& [node, count] : counters[timestep]) {
        total_bigwalk += count;
    }

    for (const auto& [node, count] : counters[timestep]) {
        pi_u[node] = (total_bigwalk > 0) ? (static_cast<double>(count) / total_bigwalk) : 0.0;
    }


    // Step B: π(u,v) approximated by cumulative counts normalized
    std::unordered_map<int, double> pi_uv;
    double total_cumulative = 0.0;
    for (const auto& [_, value] : cumulative_counts) total_cumulative += value;
    for (const auto& [node, value] : cumulative_counts) {
        pi_uv[node] = (total_cumulative > 0) ? (value / total_cumulative) : 0.0;
    }

 

    // Step D: Combine into weights
    double fallback = 1e-12;  // negligible, avoids exact zero
    std::vector<std::pair<int, double>> weighted_scores;
    for (const auto& [node, val] : cumulative_counts) {
        //double pu = pi_u[node];       // local importance
        //double puv = pi_uv[node];     // co-occurrence weight
        //double pv = pi_v[node];       // global popularity

        //double pu  = (pi_u.find(node)  != pi_u.end())  ? pi_u.at(node)  : fallback;
        double puv = (pi_uv.find(node) != pi_uv.end()) ? pi_uv.at(node) : fallback;
        double pv  = (pi_v.find(node)  != pi_v.end())  ? pi_v.at(node)  : fallback;

        //double weight = pu * puv * pv;
        //double weight = puv * pv; // omit pu for now
        double alpha = 0.25;
        double weight = (1.0 - alpha) * puv + alpha * pv;

        weighted_scores.emplace_back(node, weight);



    }
    
    // Sort nodes by weight
    std::sort(weighted_scores.begin(), weighted_scores.end(), 
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // 5. Select top-k nodes
    std::unordered_map<int, double> top_k;
    std::unordered_set<int> selected;

    for (int i = 0; i < static_cast<int>(weighted_scores.size()) && top_k.size() < static_cast<size_t>(k); ++i) {
        int node = weighted_scores[i].first;

        // Skip if node ID starts with "12345"
        if (std::to_string(node).rfind("12345", 0) == 0) continue;

        top_k[node] = weighted_scores[i].second;
        selected.insert(node);
    }

    //Pad if fewer than k nodes
    for (const auto& [node, _] : cumulative_counts) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = 0.0;
        }
    }

    double min_score = weighted_scores.empty() ? 0.0 : weighted_scores.back().second;

    for (const auto& [node, _] : cumulative_counts) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = min_score;  // instead of 0.0
        }
    }
    
    for (int node : walk) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = 0.0;
        }
    }

    // Return top-k nodes and the timestep weights (for debugging/analysis)
    return {top_k, individual_weights};
}


std::pair<std::unordered_map<int, double>, std::unordered_map<int, double>> calculate_top_k_cumulative_with_weights_oracle(
    const std::vector<int>& walk,
    std::unordered_map<int, double>& cumulative_counts,  // using double for decayed weights
    int k,
    int timestep,
    std::unordered_map<int, std::unordered_map<int, int>>& counters, // {timestep -> {node -> freq}}
    std::unordered_map<int, double> & pi_u, // global popularity (π(v))
    std::unordered_map<int, double> & pi_v, // global popularity (π(v))
    double decay_rate = 0.38,
    int window_size = 5,
    int n = 0             // total number of nodes,
    

    ) {
    // 1. Count queries for current timestep
    for (int node : walk) {
        counters[timestep][node]++;
    }

    // 2. Collect relevant timesteps (sliding window)
    std::vector<int> keys;
    for (const auto& entry : counters) keys.push_back(entry.first);
    std::sort(keys.begin(), keys.end());
    int latest_timestep = keys.back();
    int start_index = std::max(0, static_cast<int>(keys.size()) - window_size);
    std::vector<int> relevant_timesteps(keys.begin() + start_index, keys.end());

    // 3. Apply decay
    std::unordered_map<int, double> decayed_counts;
    std::unordered_map<int, double> individual_weights;
    for (int j : relevant_timesteps) {
        double weight = std::pow(decay_rate, latest_timestep - j); //std::exp(-decay_rate * (latest_timestep - j));
        //double weight = std::pow(decay_rate,  j - latest_timestep); // reverse decay
        individual_weights[j] = weight;

        for (const auto& [node, count] : counters[j]) {
            decayed_counts[node] += count * weight;
        }
    }

    // 4. Update cumulative counts
    for (const auto& [node, decayed_count] : decayed_counts) {
        cumulative_counts[node] += decayed_count;
    }

    // ------------------------------------------------------------------
    // NEW: Properly compute π(u), π(u,v), π(v), and w(u,v)
    // ------------------------------------------------------------------

    // Step A: π(u) = normalized count of this timestep
    // std::unordered_map<int, double> pi_u; // per timestep
    // int total_bigwalk = 0;
    // for (const auto& [node, count] : counters[timestep]) {
    //     total_bigwalk += count;
    // }

    // for (const auto& [node, count] : counters[timestep]) {
    //     pi_u[node] = (total_bigwalk > 0) ? (static_cast<double>(count) / total_bigwalk) : 0.0;
    // }


    // Step B: π(u,v) approximated by cumulative counts normalized
    std::unordered_map<int, double> pi_uv;
    double total_cumulative = 0.0;
    for (const auto& [_, value] : cumulative_counts) total_cumulative += value;
    for (const auto& [node, value] : cumulative_counts) {
        pi_uv[node] = (total_cumulative > 0) ? (value / total_cumulative) : 0.0;
    }

 

    // Step D: Combine into weights
    double fallback = 1e-12;  // negligible, avoids exact zero
    std::vector<std::pair<int, double>> weighted_scores;
    for (const auto& [node, val] : cumulative_counts) {
  

        //double pu  = (pi_u.find(node)  != pi_u.end())  ? pi_u.at(node)  : fallback;
        double puv = (pi_uv.find(node) != pi_uv.end()) ? pi_uv.at(node) : fallback;
        double pv  = (pi_v.find(node)  != pi_v.end())  ? pi_v.at(node)  : fallback;

        //double weight = pu * puv * pv;
        double weight = puv * pv;
        weighted_scores.emplace_back(node, weight);


        // std::cout << "Node " << node
        // << " | pu=" << pu
        // << " | puv=" << puv
        // << " | pv=" << pv
        // << " | score=" << weight
        // << std::endl;

    }
    
    // Sort nodes by weight
    std::sort(weighted_scores.begin(), weighted_scores.end(), 
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // 5. Select top-k nodes
    std::unordered_map<int, double> top_k;
    std::unordered_set<int> selected;

    for (int i = 0; i < static_cast<int>(weighted_scores.size()) && top_k.size() < static_cast<size_t>(k); ++i) {
        int node = weighted_scores[i].first;

        // Skip if node ID starts with "12345"
        if (std::to_string(node).rfind("12345", 0) == 0) continue;

        top_k[node] = weighted_scores[i].second;
        selected.insert(node);
    }

    //Pad if fewer than k nodes
    for (const auto& [node, _] : cumulative_counts) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = 0.0;
        }
    }

    double min_score = weighted_scores.empty() ? 0.0 : weighted_scores.back().second;

    for (const auto& [node, _] : cumulative_counts) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = min_score;  // instead of 0.0
        }
    }
    
    for (int node : walk) {
        if (top_k.size() >= static_cast<size_t>(k)) break;
        if (selected.insert(node).second) {
            top_k[node] = 0.0;
        }
    }

    // Return top-k nodes and the timestep weights (for debugging/analysis)
    return {top_k, individual_weights};
}








std::unordered_map<int, double> pagerank_power_iteration_(
    const std::unordered_map<int, std::unordered_map<int, double>>& graph, // adjacency with weights
    int n,
    double damping = 0.85,
    double epsilon = 1e-6,
    int max_iter = 100
) {
    std::unordered_map<int, double> pr, new_pr;
    pr.reserve(n);
    new_pr.reserve(n);

    double init_val = 1.0 / n;

    // Initialize all nodes
    for (const auto& [node, _] : graph) {
        pr[node] = init_val;
        new_pr[node] = 0.0;
    }

    // Precompute out-degree sums (weights)
    std::unordered_map<int, double> out_sum;
    out_sum.reserve(graph.size());
    for (const auto& [u, neighbors] : graph) {
        double total = 0.0;
        for (const auto& [v, w] : neighbors) {
            total += w;
        }
        out_sum[u] = total;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // Reset new_pr to teleportation
        double base = (1.0 - damping) / n;
        for (auto& [node, val] : new_pr) {
            val = base;
        }

        // Distribute rank with weights
        for (const auto& [u, neighbors] : graph) {
            if (out_sum[u] == 0.0) continue; // dangling node

            double contrib = damping * pr[u] / out_sum[u];
            for (const auto& [v, w] : neighbors) {
                new_pr[v] += contrib * w;
            }
        }

        // Check convergence
        double diff = 0.0;
        for (const auto& [node, val] : pr) {
            diff += std::fabs(new_pr[node] - val);
        }

        if (diff < epsilon) break;

        std::swap(pr, new_pr); // avoid copying
    }

    return pr;
}


std::unordered_map<int, double> pagerank_power_iteration(
    const std::unordered_map<int, std::unordered_map<int, double>>& graph,
    int n,
    const std::unordered_set<int>& personalization_set = {}, // optional
    double damping = 0.85,
    double epsilon = 1e-6,
    int max_iter = 100
) {
    std::unordered_map<int, double> pr, new_pr;
    pr.reserve(n);
    new_pr.reserve(n);

    // Teleportation base (uniform if personalization empty)
    bool personalized = !personalization_set.empty();
    double base = personalized ? 1.0 / personalization_set.size() : 1.0 / n;

    // Initialize PR values
    for (const auto& [node, _] : graph) {
        if (personalized) {
            pr[node] = (personalization_set.count(node) ? base : 0.0);
        } else {
            pr[node] = base;
        }
        new_pr[node] = 0.0;
    }

    // Precompute out-degree sums
    std::unordered_map<int, double> out_sum;
    out_sum.reserve(graph.size());
    for (const auto& [u, neighbors] : graph) {
        double total = 0.0;
        for (const auto& [v, w] : neighbors) {
            total += w;
        }
        out_sum[u] = total;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // Reset new_pr to teleportation
        for (auto& [node, val] : new_pr) {
            val = 0.0;
        }
        if (personalized) {
            for (int u : personalization_set) {
                new_pr[u] = (1.0 - damping) * base;
            }
        } else {
            for (auto& [node, val] : new_pr) {
                val = (1.0 - damping) / n;
            }
        }

        // Distribute rank
        for (const auto& [u, neighbors] : graph) {
            if (out_sum[u] == 0.0) continue; // dangling node
            double contrib = damping * pr[u] / out_sum[u];
            for (const auto& [v, w] : neighbors) {
                new_pr[v] += contrib * w;
            }
        }

        // Check convergence
        double diff = 0.0;
        for (const auto& [node, val] : pr) {
            diff += std::fabs(new_pr[node] - val);
        }
        if (diff < epsilon) break;

        std::swap(pr, new_pr);
    }

    return pr;
}



std::unordered_map<int, double> approximate_pagerank_from_walks(
    const std::unordered_map<int, std::vector<std::vector<std::string>>> &precomputed_walks,
    int n,
    int R,
    double epsilon
) {
    std::unordered_map<int, double> walks_counts;

    for (const auto& [start_node, walks] : precomputed_walks) {
        for (const auto& walk : walks) { 
            for (const auto& node_str : walk) {
                int node = std::stoi(node_str); // convert string → int
                walks_counts[node] += 1.0;
            }
        }
    }

    // normalize
    std::unordered_map<int, double> pi_v;
    for (const auto& [node, count] : walks_counts) {
        pi_v[node] = count / (n * R / epsilon);
    }
    return pi_v;
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

    //std::unordered_map<int, int> cumCounts;  // cumulative counts for nodes
    std::unordered_map<int, double> cumCounts;

    std::unordered_map<int, std::unordered_map<int, int>> counters;  // counters for each timestep

    auto query_freqs = compute_query_entity_frequencies_by_timestep(queries);
    int total_R = 0;


    // Result per timestep
    std::vector<std::unordered_map<int, double>> pagerank_per_timestep;
    Graph_ accumulated_graph; // initially empty
    std::unordered_set<int> accumulated_personalization_set;
    std::unordered_map<int, std::vector<std::vector<std::string>>> accumulated_walks;
    std::unordered_set<int> accumulated_nodes;
    std::vector<std::unordered_map<int, std::vector<std::vector<std::string>>>> processed_R_per_t;
    std::unordered_map<int,double> node_walk_weight;
    std::unordered_map<int, std::unordered_map<int,double>> pi_MC_per_t;
    std::unordered_map<int, double> prev_pagerank;             // stores PR from last timestep
    std::unordered_map<int, std::unordered_map<int, double>> pagerank_per_t;  // stores PR per timestep




    //std::unordered_map<int, std::unordered_map<int, double>> accumulated_graph;

    for (int t = 0; t < static_cast<int>(dynamic_graph.size()); t++) {
        cout << "Processing timestep " << t << endl;
        auto total_start = chrono::high_resolution_clock::now();
        auto graph_changes_result = graph_changes_with_dummy_node(dynamic_graph, queries, t);
        summary.changes_over_time[t] = graph_changes_result.changes;
        std::unordered_set<int> personalization_set;

        // Count all nodes in the snapshot
        unordered_set<int> nodes_set;
        for (const auto& p : graph_changes_result.snapshot) {
            nodes_set.insert(p.first);
            for (const auto& q : p.second)
                nodes_set.insert(q.first);
        }
        int n = nodes_set.size();
        int L = calculate_required_walk_length(c, epsilon, k, n);
        //int walk_length = static_cast<int>(std::round(1.0 / epsilon));
        //int walk_length = static_cast<int>(std::round(10 * epsilon));


        int walk_length = 5;
        cout << "average walk length " << t << ": " << walk_length << endl;
        cout << "Required length of PPR walk " << t << ": " << L << endl;

        // Preprocess (we already have weighted edges)
        Graph_ weighted_graph = graph_changes_result.snapshot;

        auto start = chrono::high_resolution_clock::now();
        RandomWalkResult R;
        if (t == 0) {
            R = generate_random_walks_directed_weighted(weighted_graph, num_walks, walk_length);
            summary.random_walks[t] = R;
        } else {
            R = update_random_walks_directed_with_weights(weighted_graph, summary.random_walks[t - 1].precomputed_walks, summary.changes_over_time[t], num_walks, walk_length);
            summary.random_walks[t] = R;
        }
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed_update = end - start;
        //cout << "elapsed time walk update for timestep " << t << ": " << elapsed_update.count() << " seconds" << endl;


        // Create a directed graph for the walk process
        DirectedGraph directed_graph;
        for (const auto& p : weighted_graph) {
            int u = p.first;
            for (const auto& q : p.second)
                directed_graph[u].push_back(q.first);
        }


  
        // Process the random walk results
        auto processed_R = process_walk_results(R.precomputed_walks);
        processed_R_per_t.push_back(processed_R);




     /*   for (const auto& [start_node, walks] : processed_R) {
            auto& dest = accumulated_walks[start_node];
            dest.insert(dest.end(), walks.begin(), walks.end());
        }
       



        for (int past_t = 0; past_t <= t; ++past_t) {
            double decay = std::pow(decay_rate, t - past_t); //std::exp(-decay_rate * (latest_timestep - j));
            for (const auto& [u, nbrs] : dynamic_graph[past_t]) {
                for (const auto& [v, w] : nbrs) {
                    accumulated_graph[u][v] += w * decay;
                    accumulated_nodes.insert(v);
                }
            }
        }
        


        double decay_rate = std::exp(-0.2 * t);       // decays over time
 
        double damping_min = 0.6;
        double damping_max = 0.88;                // new upper cap
        double damping_t = damping_min + (damping_max - damping_min) * (1.0 - decay_rate);


         std::cout << "Timestep: " << t
        //   << ", processed_R size: " << processed_R_per_t[t].size()
        //   << ", prev_pagerank size: " << prev_pagerank.size()
           << ", damping_t: " << damping_t << "\n";


*/


        


        std::cout << "Total_R: " << total_R << std::endl;
        // Compute approximate PageRank using all walks up to this timestep
        //auto pr = approximate_pagerank_from_walks(accumulated_walks, accumulated_nodes.size(), total_R, epsilon);
        // auto pr = approximate_pagerank_from_walks(processed_R, n, len_R, epsilon);
        // auto pr1 = MCWalks::MC_EndPoint_RandomStart_FromWalks(processed_R, n, len_R);//
        // auto pr2 = MCWalks::MC_EndPoint_CyclicStart_FromWalks(processed_R, n, len_R);
        // auto pr3 = MCWalks::MC_CompletePath_FromWalks(accumulated_walks, accumulated_nodes.size(), total_R);//(processed_R, n, len_R);//
    
        // auto pr4 = MCWalks::MC_CompletePath_StopAtDangling_FromWalks(accumulated_walks, accumulated_nodes.size(), total_R);//(processed_R, n, len_R);//
        // auto pr5 = MCWalks::MC_CompletePath_RandomStart_FromWalks(processed_R, n, len_R);
        

  //      auto pr_hybrid = HybridMC::hybrid_completepath_poweriteration_from_walks(accumulated_walks, accumulated_graph, damping_t, 1e-6, 20);
        //auto pr_hybrid = HybridMC::hybrid_completepath_poweriteration_from_walks_weighted(decayed_pi, accumulated_graph, damping_t, 1e-6, 20);
        

        

        
            // Now run PageRank on accumulated_graph
        //auto oracle = pagerank_power_iteration(accumulated_graph, accumulated_graph.size());
        
            // Store result per timestep
        //   pagerank_per_timestep.push_back(pr);
        

        // collect all entity_ids from query_log[t]
        // if (queries.find(t) != queries.end()) {
        //     for (const auto& [query_id, entity_id] : queries.at(t)) {
        //         accumulated_personalization_set.insert(entity_id);
        //         //personalization_set.insert(entity_id);
        //     }
        // }
        //auto personalizedoracle = pagerank_power_iteration(weighted_graph, n, personalization_set);
        //auto personalizedoracle = pagerank_power_iteration(accumulated_graph, accumulated_graph.size(), accumulated_personalization_set);
        

        
        




        //auto start = chrono::high_resolution_clock::now();
        vector<int> walk = compute_personalized_pagerank(graph_changes_result.dummy_node, L, epsilon, processed_R, directed_graph);
        //auto end = chrono::high_resolution_clock::now();
        //chrono::duration<double> elapsed = end - start;

        summary.ppr_walks[t] = walk;
        auto time_start = chrono::high_resolution_clock::now();



        // Calculate top-k cumulative with weights
        //auto top_k_result = calculate_top_k_cumulative_with_weights_oracle(walk, cumCounts, k, t, counters, personalizedoracle, oracle, decay_rate, num_timesteps, n);
    //    auto top_k_result = calculate_top_k_cumulative_with_weights(walk, cumCounts, k, t, counters,  pr_hybrid, decay_rate, num_timesteps, n);
        auto top_k_result = calculate_top_k_cumulative_with_weights_(walk, cumCounts, k, t, counters, decay_rate, num_timesteps, n);
        
        
        
        auto top_k = top_k_result.first;
        weights[t] = top_k_result.second; // Store weights for the current timestep

        summary.top_k_cumsum_walks[t] = top_k;

        auto time_end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_topk = time_end - time_start;

        //cout << "elapsed time taken for timestep " << t << ": " << elapsed_topk.count() << " seconds" << endl;

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

