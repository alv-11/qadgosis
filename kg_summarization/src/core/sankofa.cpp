
#include "sankofa.h"
#include "mc_algo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;

using Graph_ = unordered_map<int, unordered_map<int, double>>;
using DirectedGraph = unordered_map<int, vector<int>>;

struct RawGraphDelta {
    vector<int> nodes_added;
    vector<int> nodes_removed;
    vector<pair<int, int>> edges_added;
    vector<pair<int, int>> edges_removed;
};

namespace {

constexpr int kDummyNodeBase = 123456789;
constexpr double kGraphScaleFactor = 10.0;
constexpr double kRawEdgeWeight = 0.01;
constexpr double kScaledEdgeWeight = kRawEdgeWeight / kGraphScaleFactor;

vector<int> extract_query_nodes(const unordered_map<int, int>& queries) {
    vector<int> query_nodes;
    query_nodes.reserve(queries.size());
    for (const auto& entry : queries)
        query_nodes.push_back(entry.second);
    return query_nodes;
}

int dummy_node_for_timestep(int timestep) {
    return kDummyNodeBase + timestep;
}

void scale_graph_in_place(Graph_& graph) {
    for (auto& [source, neighbors] : graph) {
        for (auto& [target, weight] : neighbors)
            weight /= kGraphScaleFactor;
    }
}

Graph_ scaled_graph_copy(const Graph_& graph) {
    Graph_ scaled_graph = graph;
    scale_graph_in_place(scaled_graph);
    return scaled_graph;
}

DirectedGraph build_directed_graph(const Graph_& graph) {
    DirectedGraph directed_graph;
    for (const auto& [source, neighbors] : graph) {
        auto& adjacency = directed_graph[source];
        adjacency.reserve(neighbors.size());
        for (const auto& [target, _weight] : neighbors)
            adjacency.push_back(target);
    }
    return directed_graph;
}

struct WeightedTransitionLookup {
    unordered_map<int, vector<int>> neighbors;
    unordered_map<int, vector<double>> weights;
};

void refresh_weighted_transitions_for_node(
    WeightedTransitionLookup& lookup,
    const Graph_& graph,
    int node
) {
    auto graph_it = graph.find(node);
    if (graph_it == graph.end() || graph_it->second.empty()) {
        lookup.neighbors.erase(node);
        lookup.weights.erase(node);
        return;
    }

    auto& neighbor_list = lookup.neighbors[node];
    auto& weight_list = lookup.weights[node];
    neighbor_list.clear();
    weight_list.clear();
    neighbor_list.reserve(graph_it->second.size());
    weight_list.reserve(graph_it->second.size());

    for (const auto& [target, weight] : graph_it->second) {
        neighbor_list.push_back(target);
        weight_list.push_back(weight);
    }
}

WeightedTransitionLookup build_weighted_transition_lookup(const Graph_& graph) {
    WeightedTransitionLookup lookup;
    for (const auto& [node, _neighbors] : graph)
        refresh_weighted_transitions_for_node(lookup, graph, node);
    return lookup;
}

template <typename RNG>
bool try_sample_weighted_successor(
    const WeightedTransitionLookup& lookup,
    int current,
    RNG& rng,
    int& next
) {
    auto neighbors_it = lookup.neighbors.find(current);
    if (neighbors_it == lookup.neighbors.end() || neighbors_it->second.empty())
        return false;

    auto weights_it = lookup.weights.find(current);
    if (weights_it == lookup.weights.end() || weights_it->second.size() != neighbors_it->second.size())
        return false;

    std::discrete_distribution<int> dist(weights_it->second.begin(), weights_it->second.end());
    next = neighbors_it->second[dist(rng)];
    return true;
}

unordered_map<int, double> build_personalization(const vector<int>& query_nodes) {
    unordered_map<int, double> personalization;
    double equal_weight = 1.0 / query_nodes.size();
    for (int node : query_nodes)
        personalization[node] = equal_weight;
    return personalization;
}

void add_dummy_node(Graph_& graph, const vector<int>& query_nodes, int dummy_node) {
    double equal_weight = 1.0 / query_nodes.size();
    auto& dummy_neighbors = graph[dummy_node];
    dummy_neighbors.clear();
    for (int node : query_nodes)
        dummy_neighbors[node] = equal_weight;
}

void remove_dummy_node(Graph_& graph, int dummy_node) {
    graph.erase(dummy_node);
}

}  // namespace



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
    vector<int> query_nodes = extract_query_nodes(queries);
    Graph_ snapshot = scaled_graph_copy(graph);
    int dummy_node = dummy_node_for_timestep(timestep);
    add_dummy_node(snapshot, query_nodes, dummy_node);
    unordered_map<int, double> personalization = build_personalization(query_nodes);
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
    const WeightedTransitionLookup& weighted_transitions,
    int walks_per_node,
    double epsilon
) {
    RandomWalkResult result;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> stop_dis(0.0, 1.0);

    // Get all starting nodes
    std::vector<int> nodes;
    for (const auto& p : weighted_graph)
        nodes.push_back(p.first);

    // For each node, generate `walks_per_node` walks
    for (int start_node : nodes) {
        for (int w = 0; w < walks_per_node; w++) {

            std::vector<int> walk;
            int current = start_node;
            walk.push_back(current);

            while (stop_dis(gen) >= epsilon) {
                int next = 0;
                if (!try_sample_weighted_successor(weighted_transitions, current, gen, next))
                    break;

                current = next;
                walk.push_back(current);
            }

            result.precomputed_walks[start_node].push_back(walk);
        }
    }

    return result;
}


RandomWalkResult generate_random_walks_directed_weighted(
    const Graph_& weighted_graph,
    int walks_per_node,
    double epsilon
) {
    WeightedTransitionLookup weighted_transitions = build_weighted_transition_lookup(weighted_graph);
    return generate_random_walks_directed_weighted(
        weighted_graph,
        weighted_transitions,
        walks_per_node,
        epsilon
    );
}




RandomWalkResult update_random_walks_directed_with_weights(
    const WeightedTransitionLookup& weighted_transitions,
    std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
    const GraphChanges& changes,
    int num_walks,
    double epsilon
) {
    RandomWalkResult result;

    using RNG = std::mt19937;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RNG rng(seed);
    std::uniform_real_distribution<> stop_dis(0.0, 1.0);

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

            while (stop_dis(rng) >= epsilon) {
                int next = 0;
                if (!try_sample_weighted_successor(weighted_transitions, current, rng, next))
                    break;

                current = next;
                walk.push_back(current);
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

    // Extend truncated walks until the next restart check fires.
    for (auto& [node, walks] : precomputed_walks) {
        for (auto& walk : walks) {
            while (stop_dis(rng) >= epsilon) {
                int last = walk.back();
                int next = 0;
                if (!try_sample_weighted_successor(weighted_transitions, last, rng, next))
                    break;

                walk.push_back(next);
            }
        }
    }

    result.precomputed_walks = precomputed_walks;
    return result;
}


RandomWalkResult update_random_walks_directed_with_weights(
    const Graph_& graph,
    std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
    const GraphChanges& changes,
    int num_walks,
    double epsilon
) {
    WeightedTransitionLookup weighted_transitions = build_weighted_transition_lookup(graph);
    return update_random_walks_directed_with_weights(
        weighted_transitions,
        precomputed_walks,
        changes,
        num_walks,
        epsilon
    );
}

// An implementation of a personalized PageRank–like random walk.
// Starting from node w, it performs L steps; with probability epsilon it teleports to w,
// and otherwise it randomly follows an outgoing edge.
vector<int> compute_personalized_pagerank(int w, int L, double epsilon,
                                          const unordered_map<int, vector<vector<int>>> &R,
                                          const DirectedGraph &directed_graph) {
    vector<int> walk;
    int current = w;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    unordered_map<int, vector<int>> remaining_segment_indices;

    walk.push_back(current);
    for (int i = 1; i < L; i++) {
        if (dis(gen) < epsilon) {
            current = w;
        } else {
            auto segments_it = R.find(current);
            if (segments_it != R.end() && !segments_it->second.empty()) {
                auto remaining_it = remaining_segment_indices.find(current);
                if (remaining_it == remaining_segment_indices.end()) {
                    vector<int> initial_indices;
                    initial_indices.reserve(segments_it->second.size());
                    for (size_t index = 0; index < segments_it->second.size(); ++index)
                        initial_indices.push_back(static_cast<int>(index));
                    remaining_it = remaining_segment_indices.emplace(current, std::move(initial_indices)).first;
                }

                auto& available_indices = remaining_it->second;
                if (!available_indices.empty()) {
                    uniform_int_distribution<> segment_dist(0, available_indices.size() - 1);
                    size_t available_index = static_cast<size_t>(segment_dist(gen));
                    int segment_index = available_indices[available_index];
                    available_indices[available_index] = available_indices.back();
                    available_indices.pop_back();

                    const vector<int>& segment = segments_it->second[segment_index];
                    size_t segment_start = (!segment.empty() && segment.front() == current) ? 1 : 0;
                    for (size_t j = segment_start; j < segment.size() && walk.size() < static_cast<size_t>(L); ++j) {
                        current = segment[j];
                        walk.push_back(current);
                    }

                    if (walk.size() < static_cast<size_t>(L)) {
                        current = w;
                    }
                } else if (directed_graph.find(current) != directed_graph.end() && !directed_graph.at(current).empty()) {
                    const vector<int>& neighbors = directed_graph.at(current);
                    uniform_int_distribution<> dis2(0, neighbors.size() - 1);
                    current = neighbors[dis2(gen)];
                } else {
                    current = w;
                }
            } else if (directed_graph.find(current) != directed_graph.end() && !directed_graph.at(current).empty()) {
                const vector<int>& neighbors = directed_graph.at(current);
                uniform_int_distribution<> dis2(0, neighbors.size() - 1);
                current = neighbors[dis2(gen)];
            } else {
                current = w;
            }
        }

        if (walk.size() >= static_cast<size_t>(L)) {
            break;
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
    std::unordered_map<int, double>& cumulative_counts,
    int k,
    int timestep,
    std::unordered_map<int, std::unordered_map<int, int>>& counters, // {timestep -> {node -> freq}}
    std::unordered_map<int, double> & pi_v, // global popularity (π(v))
    double decay_rate = 0.38,
    int window_size = 5,
    int n = 0,             // total number of nodes,
    int c = 2,
    double epsilon = 0.25
    

    ) {
    const int effective_window_size = std::max(1, window_size);
    auto& current_counts = counters[timestep];

    // 1. Count visits for the current timestep.
    for (int node : walk)
        current_counts[node]++;

    // 2. Maintain the decayed rolling window in streaming form:
    //    decay previous state, add current timestep, subtract the expired timestep.
    std::vector<int> stale_nodes;
    stale_nodes.reserve(cumulative_counts.size());
    for (auto& [node, value] : cumulative_counts) {
        value *= decay_rate;
        if (std::abs(value) <= 1e-12)
            stale_nodes.push_back(node);
    }
    for (int node : stale_nodes)
        cumulative_counts.erase(node);

    for (const auto& [node, count] : current_counts)
        cumulative_counts[node] += static_cast<double>(count);

    const int expired_timestep = timestep - effective_window_size;
    if (expired_timestep >= 0) {
        auto expired_it = counters.find(expired_timestep);
        if (expired_it != counters.end()) {
            const double expired_weight = std::pow(decay_rate, effective_window_size);
            for (const auto& [node, count] : expired_it->second) {
                auto cumulative_it = cumulative_counts.find(node);
                if (cumulative_it == cumulative_counts.end())
                    continue;

                cumulative_it->second -= static_cast<double>(count) * expired_weight;
                if (std::abs(cumulative_it->second) <= 1e-12)
                    cumulative_counts.erase(cumulative_it);
            }
            counters.erase(expired_it);
        }
    }

    // 3. Expose the current window weights for downstream weighted evaluation.
    std::unordered_map<int, double> individual_weights;
    const int earliest_timestep = std::max(0, timestep - effective_window_size + 1);
    for (int j = earliest_timestep; j <= timestep; ++j) {
        if (counters.find(j) != counters.end())
            individual_weights[j] = std::pow(decay_rate, timestep - j);
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


        double alpha = epsilon;
        //double weight = (1.0 - alpha) * puv + alpha * pv;
        double weight = (alpha) * puv + (1.0-alpha) * pv;
        
        
        //double weight = pi_uv_norm[node] ;

        weighted_scores.emplace_back(node, weight);
        


    }
    for (const auto& [node, _] : pi_v) {
        double pu = (pi_u.find(node) != pi_u.end()) ? pi_u.at(node) : 0.0;

        double pv = pi_v[node];       // global popularity

        double alpha = epsilon;
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
    const int window_size = 5;


    for (int t = 0; t < static_cast<int>(dynamic_graph.size()); t++) {
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


        // Reuse the personalized snapshot directly to avoid another full graph copy.
        const Graph_& weighted_graph = graph_changes_result.snapshot;

        auto start = chrono::high_resolution_clock::now();
        RandomWalkResult R;
        if (t == 0) {
    
            R = generate_random_walks_directed_weighted(weighted_graph, num_walks, epsilon);
            summary.random_walks[0] = R;


        } else {

            R = update_random_walks_directed_with_weights(weighted_graph, summary.random_walks[0].precomputed_walks, summary.changes_over_time[t], num_walks, epsilon);
            summary.random_walks[0] = R;
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


  
        vector<int> walk = compute_personalized_pagerank(graph_changes_result.dummy_node, L, epsilon, R.precomputed_walks, directed_graph);

        summary.ppr_walks[t] = walk;
        auto time_start = chrono::high_resolution_clock::now();

        auto pr3 = MCWalks::MC_CompletePath_FromWalks(R.precomputed_walks);

        // Calculate top-k cumulative wi
        auto top_k_result = calculate_top_k_cumulative_with_weights(walk, cumCounts, k, t, counters,  pr3, decay_rate, window_size, n,c, epsilon);
    
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


//////////////////////////////////////////////////////////////////////////////////////////////////////


GraphChangesDummyResult graph_changes_with_dummy_node_single(
    const Graph_& current_graph,
    const Graph_& next_graph,
    const std::unordered_map<int, std::unordered_map<int,int>>& queries,
    int timestep,
    bool has_next
) {

    // Personalize current snapshot
    auto res1 = personalized_graph(current_graph, queries.at(timestep), timestep);
    Graph_ snapshot_s = res1.snapshot;
    int dummy_node = res1.dummy_node;
    auto personalization = res1.personalization;

    Graph_ snapshot_t;

    if (has_next) {
        snapshot_t = personalized_graph(next_graph, queries.at(timestep + 1), timestep + 1).snapshot;
    } else {
        snapshot_t = snapshot_s;  // last timestep case
    }

    // --- Collect nodes ---
    std::unordered_set<int> nodes_s, nodes_t;

    for (const auto& p : snapshot_s) {
        nodes_s.insert(p.first);
        for (const auto& q : p.second)
            nodes_s.insert(q.first);
    }

    for (const auto& p : snapshot_t) {
        nodes_t.insert(p.first);
        for (const auto& q : p.second)
            nodes_t.insert(q.first);
    }

    // --- Compute node changes ---
    std::unordered_set<int> nodes_added, nodes_removed;

    for (int node : nodes_t)
        if (!nodes_s.count(node))
            nodes_added.insert(node);

    for (int node : nodes_s)
        if (!nodes_t.count(node))
            nodes_removed.insert(node);

    // --- Compute edge changes ---
    std::set<std::pair<int,int>> edges_s, edges_t;

    for (const auto& p : snapshot_s)
        for (const auto& q : p.second)
            edges_s.insert({p.first, q.first});

    for (const auto& p : snapshot_t)
        for (const auto& q : p.second)
            edges_t.insert({p.first, q.first});

    std::set<std::pair<int,int>> edges_added, edges_removed;

    for (const auto& e : edges_t)
        if (!edges_s.count(e))
            edges_added.insert(e);

    for (const auto& e : edges_s)
        if (!edges_t.count(e))
            edges_removed.insert(e);

    GraphChanges changes {nodes_added, nodes_removed, edges_added, edges_removed};

    return {snapshot_s, changes, dummy_node, personalization};
}


GraphChanges graph_changes_with_dummy_node_streamed(
    const RawGraphDelta& raw_delta,
    const std::unordered_map<int, std::unordered_map<int,int>>& queries,
    int timestep,
    bool has_next
) {
    GraphChanges changes;

    if (has_next) {
        changes.nodes_added = std::unordered_set<int>(
            raw_delta.nodes_added.begin(),
            raw_delta.nodes_added.end()
        );
        changes.nodes_removed = std::unordered_set<int>(
            raw_delta.nodes_removed.begin(),
            raw_delta.nodes_removed.end()
        );
        changes.edges_added = std::set<std::pair<int, int>>(
            raw_delta.edges_added.begin(),
            raw_delta.edges_added.end()
        );
        changes.edges_removed = std::set<std::pair<int, int>>(
            raw_delta.edges_removed.begin(),
            raw_delta.edges_removed.end()
        );

        int current_dummy_node = dummy_node_for_timestep(timestep);
        int next_dummy_node = dummy_node_for_timestep(timestep + 1);
        changes.nodes_removed.insert(current_dummy_node);
        changes.nodes_added.insert(next_dummy_node);

        for (int query_node : extract_query_nodes(queries.at(timestep)))
            changes.edges_removed.insert({current_dummy_node, query_node});

        for (int query_node : extract_query_nodes(queries.at(timestep + 1)))
            changes.edges_added.insert({next_dummy_node, query_node});
    }

    return changes;
}


static TopKSummary summary;
static std::unordered_map<int, std::unordered_map<int,double>> weights;
static std::unordered_map<int,double> runtime_per_timestep;
static std::unordered_map<int,double> cumCounts;
static std::unordered_map<int, std::unordered_map<int,int>> counters;
static Graph_ streamed_graph;
static DirectedGraph streamed_directed_graph;
static WeightedTransitionLookup streamed_weighted_transitions;
static int streamed_node_count = 0;
static bool streamed_graph_initialized = false;
//static bool first = true;


void apply_raw_graph_delta(Graph_& graph, const RawGraphDelta& delta) {
    for (const auto& edge : delta.edges_removed) {
        auto it = graph.find(edge.first);
        if (it == graph.end())
            continue;
        it->second.erase(edge.second);
    }

    for (const auto& edge : delta.edges_added)
        graph[edge.first][edge.second] = kScaledEdgeWeight;

    for (int node : delta.nodes_added) {
        if (graph.find(node) == graph.end())
            graph[node] = {};
    }

    for (int node : delta.nodes_removed)
        graph.erase(node);
}


void apply_raw_graph_delta(DirectedGraph& graph, const RawGraphDelta& delta) {
    for (const auto& edge : delta.edges_removed) {
        auto it = graph.find(edge.first);
        if (it == graph.end())
            continue;

        auto& neighbors = it->second;
        neighbors.erase(remove(neighbors.begin(), neighbors.end(), edge.second), neighbors.end());
        if (neighbors.empty())
            graph.erase(it);
    }

    for (const auto& edge : delta.edges_added) {
        auto& neighbors = graph[edge.first];
        if (find(neighbors.begin(), neighbors.end(), edge.second) == neighbors.end())
            neighbors.push_back(edge.second);
    }

    for (int node : delta.nodes_removed)
        graph.erase(node);
}


void apply_raw_graph_delta(
    WeightedTransitionLookup& lookup,
    const Graph_& graph,
    const RawGraphDelta& delta
) {
    unordered_set<int> changed_sources;
    for (const auto& edge : delta.edges_added)
        changed_sources.insert(edge.first);
    for (const auto& edge : delta.edges_removed)
        changed_sources.insert(edge.first);
    for (int node : delta.nodes_removed) {
        lookup.neighbors.erase(node);
        lookup.weights.erase(node);
    }
    for (int node : changed_sources)
        refresh_weighted_transitions_for_node(lookup, graph, node);
}


void reset_stream_state() {
    summary = TopKSummary{};
    weights.clear();
    runtime_per_timestep.clear();
    cumCounts.clear();
    counters.clear();
    streamed_graph.clear();
    streamed_directed_graph.clear();
    streamed_weighted_transitions.neighbors.clear();
    streamed_weighted_transitions.weights.clear();
    streamed_node_count = 0;
    streamed_graph_initialized = false;
}


void initialize_stream_graph(const Graph_& initial_graph) {
    reset_stream_state();
    streamed_graph = scaled_graph_copy(initial_graph);
    streamed_directed_graph = build_directed_graph(streamed_graph);
    streamed_weighted_transitions = build_weighted_transition_lookup(streamed_graph);
    streamed_node_count = static_cast<int>(streamed_graph.size());
    streamed_graph_initialized = true;
}


TopKSummaryWithWeights process_one_timestep_from_graph_changes(
    int num_walks,
    const Graph_& weighted_graph,
    const GraphChanges& changes,
    int dummy_node,
    double c,
    double epsilon,
    int k,
    double decay_rate,
    int t,
    int window_size,
    const DirectedGraph* directed_graph_override = nullptr,
    const WeightedTransitionLookup* weighted_transitions_override = nullptr,
    int node_count_override = -1
) {
    auto total_start = chrono::high_resolution_clock::now();
    summary.changes_over_time[t] = changes;

    int n = node_count_override;
    if (n < 0) {
        unordered_set<int> nodes_set;
        for (const auto& p : weighted_graph) {
            nodes_set.insert(p.first);
            for (const auto& q : p.second)
                nodes_set.insert(q.first);
        }
        n = nodes_set.size();
    }
    int L = calculate_required_walk_length(c, epsilon, k, n);

    auto start = chrono::high_resolution_clock::now();
    RandomWalkResult R;
    WeightedTransitionLookup weighted_transitions_storage;
    const WeightedTransitionLookup* weighted_transitions = weighted_transitions_override;
    if (weighted_transitions == nullptr) {
        weighted_transitions_storage = build_weighted_transition_lookup(weighted_graph);
        weighted_transitions = &weighted_transitions_storage;
    }

    if (false){//(t%20 == 0) {
        R = generate_random_walks_directed_weighted(
            weighted_graph,
            *weighted_transitions,
            num_walks,
            epsilon
        );
        summary.random_walks[0] = R;
        cumCounts.clear();
        counters.clear();
    } else {
        R = update_random_walks_directed_with_weights(
            *weighted_transitions,
            summary.random_walks[0].precomputed_walks,
            summary.changes_over_time[t],
            num_walks,
            epsilon
        );
        summary.random_walks[0] = R;
    }
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed_update = end - start;

    DirectedGraph directed_graph_storage;
    const DirectedGraph* directed_graph = directed_graph_override;
    if (directed_graph == nullptr) {
        directed_graph_storage = build_directed_graph(weighted_graph);
        directed_graph = &directed_graph_storage;
    }

    vector<int> walk = compute_personalized_pagerank(dummy_node, L, epsilon, R.precomputed_walks, *directed_graph);

    summary.ppr_walks[t] = walk;
    auto time_start = chrono::high_resolution_clock::now();

    auto pr3 = MCWalks::MC_CompletePath_FromWalks(R.precomputed_walks);

    auto top_k_result = calculate_top_k_cumulative_with_weights(walk, cumCounts, k, t, counters,  pr3, decay_rate, window_size, n,c, epsilon);

    summary.top_k_cumsum_walks[t] = top_k_result.top_k;
    summary.global_pr[t] = top_k_result.nontop_k;
    weights[t] = top_k_result.individual_weights;

    auto time_end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_topk = time_end - time_start;

    auto total_end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = total_end - total_start;

    runtime_per_timestep[t] = total_elapsed.count();
    return {summary, weights, runtime_per_timestep};
}


TopKSummaryWithWeights process_one_timestep(
    int num_walks,
    Graph_& current_graph,
    Graph_& next_graph,
    std::unordered_map<int, std::unordered_map<int,int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate,
    int t,
    bool has_next,
    int window_size
) {
    auto graph_changes_result = graph_changes_with_dummy_node_single(
        current_graph,
        next_graph,
        queries,
        t,
        has_next
    );
    return process_one_timestep_from_graph_changes(
        num_walks,
        graph_changes_result.snapshot,
        graph_changes_result.changes,
        graph_changes_result.dummy_node,
        c,
        epsilon,
        k,
        decay_rate,
        t,
        window_size
    );
}


TopKSummaryWithWeights process_one_timestep_delta(
    int num_walks,
    const RawGraphDelta& raw_delta,
    std::unordered_map<int, std::unordered_map<int,int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate,
    int t,
    bool has_next,
    int window_size
) {
    if (!streamed_graph_initialized) {
        throw runtime_error("Sankofa streamed graph is not initialized.");
    }

    int dummy_node = dummy_node_for_timestep(t);
    vector<int> query_nodes = extract_query_nodes(queries.at(t));
    add_dummy_node(streamed_graph, query_nodes, dummy_node);
    streamed_directed_graph[dummy_node] = query_nodes;
    refresh_weighted_transitions_for_node(streamed_weighted_transitions, streamed_graph, dummy_node);

    auto changes = graph_changes_with_dummy_node_streamed(
        raw_delta,
        queries,
        t,
        has_next
    );

    auto result = process_one_timestep_from_graph_changes(
        num_walks,
        streamed_graph,
        changes,
        dummy_node,
        c,
        epsilon,
        k,
        decay_rate,
        t,
        window_size,
        &streamed_directed_graph,
        &streamed_weighted_transitions,
        streamed_node_count + 1
    );

    remove_dummy_node(streamed_graph, dummy_node);
    streamed_directed_graph.erase(dummy_node);
    streamed_weighted_transitions.neighbors.erase(dummy_node);
    streamed_weighted_transitions.weights.erase(dummy_node);

    if (has_next) {
        apply_raw_graph_delta(streamed_graph, raw_delta);
        apply_raw_graph_delta(streamed_directed_graph, raw_delta);
        apply_raw_graph_delta(streamed_weighted_transitions, streamed_graph, raw_delta);
        streamed_node_count += static_cast<int>(raw_delta.nodes_added.size());
        streamed_node_count -= static_cast<int>(raw_delta.nodes_removed.size());
    }

    return result;
}


RawGraphDelta raw_graph_delta_from_py(const py::object& raw_delta_obj) {
    if (raw_delta_obj.is_none()) {
        return {};
    }

    py::dict raw_delta = raw_delta_obj.cast<py::dict>();
    RawGraphDelta delta;
    delta.nodes_added = raw_delta["nodes_added"].cast<vector<int>>();
    delta.nodes_removed = raw_delta["nodes_removed"].cast<vector<int>>();
    delta.edges_added = raw_delta["edges_added"].cast<vector<pair<int, int>>>();
    delta.edges_removed = raw_delta["edges_removed"].cast<vector<pair<int, int>>>();
    return delta;
}


py::dict result_to_python_dict(const TopKSummaryWithWeights& result) {
    TopKSummary summary_for_python = result.summary;
    py::dict result_dict;

    py::dict runtime_dict;
    for (const auto& entry : result.runtime_per_timestep)
        runtime_dict[py::int_(entry.first)] = entry.second;
    result_dict["runtime_per_timestep"] = runtime_dict;

    py::dict changes_over_time;
    for (const auto &entry : summary_for_python.changes_over_time) {
        py::dict ch;

        std::vector<int> nodes_added(entry.second.nodes_added.begin(), entry.second.nodes_added.end());
        std::vector<int> nodes_removed(entry.second.nodes_removed.begin(), entry.second.nodes_removed.end());
        std::vector<std::pair<int,int>> edges_added(entry.second.edges_added.begin(), entry.second.edges_added.end());
        std::vector<std::pair<int,int>> edges_removed(entry.second.edges_removed.begin(), entry.second.edges_removed.end());

        ch["nodes_added"] = nodes_added;
        ch["nodes_removed"] = nodes_removed;
        ch["edges_added"] = edges_added;
        ch["edges_removed"] = edges_removed;

        changes_over_time[py::int_(entry.first)] = ch;
    }
    result_dict["changes_over_time"] = changes_over_time;

    py::dict top_k;
    for (const auto &entry : summary_for_python.top_k_cumsum_walks) {
        py::dict inner;
        for (const auto &p : entry.second)
            inner[py::int_(p.first)] = p.second;
        top_k[py::int_(entry.first)] = inner;
    }
    result_dict["top_k_cumsum_walks"] = top_k;

    py::dict global_pr;
    for (const auto &entry : summary_for_python.global_pr) {
        py::dict inner;
        for (const auto &p : entry.second)
            inner[py::int_(p.first)] = p.second;
        global_pr[py::int_(entry.first)] = inner;
    }
    result_dict["global_pr"] = global_pr;

    py::dict all_weights;
    for (const auto& entry : result.weights) {
        py::dict inner;
        for (const auto& p : entry.second)
            inner[py::int_(p.first)] = p.second;
        all_weights[py::int_(entry.first)] = inner;
    }
    result_dict["all_weights"] = all_weights;

    return result_dict;
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


/////////////////////////////////////////////////////
py::dict wrap_process_one_timestep(
    int num_walks,
    Graph_& current_graph,
    Graph_& next_graph,
    std::unordered_map<int, std::unordered_map<int,int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate,
    int t,
    bool has_next,
    int window_size
) {

    TopKSummaryWithWeights result =
        process_one_timestep(
            num_walks,
            current_graph,
            next_graph,
            queries,
            c,
            epsilon,
            k,
            decay_rate,
            t,
            has_next,
            window_size
        );

    return result_to_python_dict(result);
}



py::dict wrap_process_one_timestep_delta(
    int num_walks,
    py::object raw_delta,
    std::unordered_map<int, std::unordered_map<int,int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate,
    int t,
    bool has_next,
    int window_size
) {
    RawGraphDelta parsed_delta = raw_graph_delta_from_py(raw_delta);
    TopKSummaryWithWeights result = process_one_timestep_delta(
        num_walks,
        parsed_delta,
        queries,
        c,
        epsilon,
        k,
        decay_rate,
        t,
        has_next,
        window_size
    );

    return result_to_python_dict(result);
}




// PYBIND11_MODULE(sankofa, m) {
// m.doc() = "Python bindings for generate_top_k_summary function using the C++ walks library";
// m.def("generate_top_k_summary", &wrap_generate_top_k_summary,
// "Generate a top-k summary from dynamic graph updates.",
// py::arg("num_timesteps"), 
// py::arg("num_walks"), py::arg("dynamic_graph"), py::arg("queries"), 
// py::arg("c"), py::arg("epsilon"), py::arg("k"), py::arg("decay_rate"));
// }

PYBIND11_MODULE(sankofa, m) {
    m.doc() = "Python bindings for Sankofa";

    m.def("reset_stream_state",
        &reset_stream_state,
        "Reset Sankofa streaming state.");

    m.def("initialize_stream_graph",
        &initialize_stream_graph,
        py::arg("initial_graph"),
        "Initialize the Sankofa streaming graph.");

    m.def("process_one_timestep",
          &wrap_process_one_timestep,
          py::arg("num_walks"),
          py::arg("current_graph"),
          py::arg("next_graph"),
          py::arg("queries"),
          py::arg("c"),
          py::arg("epsilon"),
          py::arg("k"),
          py::arg("decay_rate"),
          py::arg("t"),
          py::arg("has_next"),
            py::arg("window_size") = 5
    );

        m.def("process_one_timestep_delta",
            &wrap_process_one_timestep_delta,
            py::arg("num_walks"),
            py::arg("raw_delta") = py::none(),
            py::arg("queries"),
            py::arg("c"),
            py::arg("epsilon"),
            py::arg("k"),
            py::arg("decay_rate"),
            py::arg("t"),
            py::arg("has_next"),
            py::arg("window_size") = 5
        );
}
