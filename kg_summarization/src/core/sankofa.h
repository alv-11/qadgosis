#ifndef SANKOFA_H
#define SANKOFA_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>


// Random walk result structure
struct RandomWalkResult {
    std::unordered_map<int, std::vector<std::vector<int>>> precomputed_walks;
};

//Graph changes structure
struct GraphChanges {
    std::unordered_set<int> nodes_added;
    std::unordered_set<int> nodes_removed;
    std::set<std::pair<int, int>> edges_added;
    std::set<std::pair<int, int>> edges_removed;
};

// Structure to hold the top-k summary.
struct TopKSummary {
    std::unordered_map<int, GraphChanges> changes_over_time;
    std::unordered_map<int, RandomWalkResult> random_walks;
    std::unordered_map<int, std::vector<int>> ppr_walks;
    std::unordered_map<int, std::unordered_map<int, double>> top_k_cumsum_walks;
    std::unordered_map<int, std::unordered_map<int, double>> global_pr;
};

// Structure to hold both the summary, weights, and runtime.
struct TopKSummaryWithWeights {
    TopKSummary summary;
    std::unordered_map<int, std::unordered_map<int, double>> weights;
    std::unordered_map<int, double> runtime_per_timestep; // Add runtime_per_timestep
};

// Function declarations.
TopKSummaryWithWeights generate_top_k_summary(
    int num_timesteps,
    int num_walks,
    std::vector<std::unordered_map<int, std::unordered_map<int, double>>>& dynamic_graph,
    std::unordered_map<int, std::unordered_map<int, int>>& queries,
    double c,
    double epsilon,
    int k,
    double decay_rate 
);

std::vector<int> compute_personalized_pagerank(
    int start_node,
    int walk_length,
    double epsilon,
    const std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
    const std::unordered_map<int, std::unordered_map<int, double>>& graph
);



using Map = std::unordered_map<int, double>;

struct TopKResult {
    Map top_k;
    Map individual_weights;
    Map nontop_k;
};

namespace MCWalks {

// Helper to normalize vector
inline void normalize(std::unordered_map<int, double>& pr) {
    double total = 0.0;
    for (auto& [k, v] : pr) total += v;
    if (total > 0) {
        for (auto& [k, v] : pr) v /= total;
    }
}

// ===============================================================
// 3. MC Complete Path
// (count all visits across all nodes in all walks)
// ===============================================================
inline std::unordered_map<int, double>
MC_CompletePath_FromWalks(
    const std::unordered_map<int, std::vector<std::vector<std::string>>>& processed_R)
   // int n, int len_R)
{
    std::unordered_map<int, double> counts;

    for (const auto& [start_node, walks] : processed_R) {
        for (const auto& walk : walks) {
            for (const auto& node_str : walk) {
                int node = std::stoi(node_str);
                counts[node] += 1.0;
            }
        }
    }

    normalize(counts);
    return counts;
}

#endif // SANKOFA_H
