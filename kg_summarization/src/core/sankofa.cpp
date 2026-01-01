
#include "sankofa.h"
#include "mc_algo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;

using Graph_ = unordered_map<int, unordered_map<int, double>>;
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
    // Define a dummy node
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



RandomWalkResult generate_random_walks_directed_weighted(
    const Graph_& weighted_graph,
    int walks_per_node,
    int walk_length
) {
    RandomWalkResult result;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Get all starting nodes
    std::vector<int> nodes;
    for (const auto& p : weighted_graph)
        nodes.push_back(p.first);

    int walk_id = 0;

    // For each node, generate `walks_per_node` walks
    for (int start_node : nodes) {
        for (int w = 0; w < walks_per_node; w++) {

            std::vector<int> walk;
            int current = start_node;
            walk.push_back(current);

            for (int j = 1; j < walk_length; j++) {
                auto it = weighted_graph.find(current);
                if (it == weighted_graph.end() || it->second.empty())
                    break;

                const auto& neighbors_map = it->second;

                std::vector<int> neighbors;
                std::vector<double> weights;
                neighbors.reserve(neighbors_map.size());
                weights.reserve(neighbors_map.size());

                for (const auto& [nbr, wgt] : neighbors_map) {
                    neighbors.push_back(nbr);
                    weights.push_back(wgt);
                }

                double sum_w = std::accumulate(weights.begin(), weights.end(), 0.0);
                if (sum_w > 0) {
                    for (double& w : weights) w /= sum_w;
                } else {
                    weights.assign(neighbors.size(), 1.0 / neighbors.size());
                }

                std::discrete_distribution<> dist(weights.begin(), weights.end());
                current = neighbors[dist(gen)];
                walk.push_back(current);
            }

            result.precomputed_walks[walk_id].push_back(walk);
            walk_id++;
        }
    }

    return result;
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

// An implementation of a personalized PageRank–like random walk.
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




TopKResult calculate_top_k_cumulative_with_weights(
    const std::vector<int>& walk,
    std::unordered_map<int, double>& cumulative_counts,  // using double for decayed weights
    int k,
    int timestep,
    std::unordered_map<int, std::unordered_map<int, int>>& counters, // {timestep -> {node -> freq}}
    std::unordered_map<int, double> & pi_v, // global popularity (π(v))
    double decay_rate = 0.38,
    int window_size = 5,
    int n = 0,             // total number of nodes,
    int c = 2
    

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

    // // 4. Update cumulative counts
    for (const auto& [node, decayed_count] : decayed_counts) {
        cumulative_counts[node] += decayed_count;
    }



//    # Step B: π(u,v) approximated by cumulative counts normalized
    std::unordered_map<int, double> pi_u;
    double total_cumulative = 0.0;
    for (const auto& [_, value] : cumulative_counts) total_cumulative += value;
    for (const auto& [node, value] : cumulative_counts) {
        pi_u[node] = (total_cumulative > 0) ? (value / total_cumulative) : 0.0;
    }



    std::unordered_map<int, double> pi_uv_raw;
    std::unordered_map<int, double> pi_uv_norm;

    // 1. FILTER FIRST — keep only nodes with raw count >= c
    for (const auto& [node, value] : cumulative_counts) {
        if (value >= c) {
            pi_uv_raw[node] = value;  // store raw filtered count
        }
    }

    // 2. COMPUTE NORMALIZATION CONSTANT *only over filtered nodes*
    double total_filtered = 0.0;
    for (const auto& [node, value] : pi_uv_raw) {
        total_filtered += value;
    }

    // 3. NORMALIZE
    for (const auto& [node, value] : pi_uv_raw) {
        pi_uv_norm[node] = (total_filtered > 0) ? (value / total_filtered) : 0.0;
    }

    
    //double fallback = 0;  // negligible, avoids exact zero
    std::vector<std::pair<int, double>> weighted_scores;
    std::vector<std::pair<int, double>> weighted_scores_others;
    for (const auto& [node, val] : cumulative_counts) {

  
        double puv = pi_uv_norm[node];     // ppr weight
        double pv = pi_v[node];       // global weight


        double alpha = 0.25;
        double weight = (1.0 - alpha) * puv + alpha * pv;
        
        
        
        //double weight = pi_uv_norm[node] ;

        weighted_scores.emplace_back(node, weight);
        


    }
    for (const auto& [node, _] : pi_v) {
        double pu = (pi_u.find(node) != pi_u.end()) ? pi_u.at(node) : 0.0;

        double pv = pi_v[node];       // global popularity

        double alpha = 0.25;
        double weight_others = (1.0 - alpha) * pu + alpha * pv;
        //double weight_others = pu;
        weighted_scores_others.emplace_back(node, weight_others);

  
} 

    // 5. Select top-k nodes
    std::unordered_map<int, double> top_k;
    std::unordered_map<int, double> nontop_k;
    std::unordered_set<int> selected;

    for (int i = 0; i < static_cast<int>(weighted_scores.size()); //&& top_k.size() < static_cast<size_t>(k)
     ++i) {
        int node = weighted_scores[i].first;

        // Skip if node ID starts with "12345"
        if (std::to_string(node).rfind("12345", 0) == 0) continue;

        top_k[node] = weighted_scores[i].second;
        selected.insert(node);
    }


    for (int i = 0; i < static_cast<int>(weighted_scores_others.size()); //&& top_k.size() < static_cast<size_t>(k)
     ++i) {
        int node = weighted_scores_others[i].first;


        nontop_k[node] = weighted_scores_others[i].second;
    }


    TopKResult result;

    // Return top-k nodes and the timestep weights (for debugging/analysis)
    result.top_k = std::move(top_k);
    result.individual_weights = std::move(individual_weights);
    result.nontop_k = std::move(nontop_k);
    return result;
    
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
    std::unordered_map<int, double> cumCounts;
    std::unordered_map<int, std::unordered_map<int, int>> counters;  // counters for each timestep


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




    for (int t = 0; t < static_cast<int>(dynamic_graph.size()); t++) {
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


        int walk_length = 5;

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

        for (const auto& [start_node, walks] : processed_R) {
            auto& dest = accumulated_walks[start_node];
            dest.insert(dest.end(), walks.begin(), walks.end());
        }

        vector<int> walk = compute_personalized_pagerank(graph_changes_result.dummy_node, L, epsilon, processed_R, directed_graph);

        summary.ppr_walks[t] = walk;
        auto time_start = chrono::high_resolution_clock::now();

        auto pr3 = MCWalks::MC_CompletePath_FromWalks(processed_R);

        // Calculate top-k cumulative wi
        auto top_k_result = calculate_top_k_cumulative_with_weights(walk, cumCounts, k, t, counters,  pr3, decay_rate, num_timesteps, n,c);
    
        summary.top_k_cumsum_walks[t] = top_k_result.top_k;
        summary.global_pr[t] = top_k_result.nontop_k;
        weights[t] = top_k_result.individual_weights; // Store weights for the current timestep

        auto time_end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_topk = time_end - time_start;


        auto total_end = chrono::high_resolution_clock::now();
        chrono::duration<double> total_elapsed = total_end - total_start;

        // Store runtime in dictionary
        runtime_per_timestep[t] = total_elapsed.count();
    }

    // Return both summary, weights, and runtime dictionary
    return {summary, weights, runtime_per_timestep}; // Updated return statement
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


py::dict global_pr;
for (const auto &entry : summary.global_pr) {
py::dict inner;
for (const auto &p : entry.second) {
inner[py::int_(p.first)] = p.second;
}
global_pr[py::int_(entry.first)] = inner;
}
result_dict["global_pr"] = global_pr;

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

PYBIND11_MODULE(sankofa, m) {
m.doc() = "Python bindings for generate_top_k_summary function using the C++ walks library";
m.def("generate_top_k_summary", &wrap_generate_top_k_summary,
"Generate a top-k summary from dynamic graph updates.",
py::arg("num_timesteps"), 
py::arg("num_walks"), py::arg("dynamic_graph"), py::arg("queries"), 
py::arg("c"), py::arg("epsilon"), py::arg("k"), py::arg("decay_rate"));
}