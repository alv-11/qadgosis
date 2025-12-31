#ifndef MAINWRAP_H
#define MAINWRAP_H

#include "shared.h"  // Include shared structures
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

#endif

using Map = std::unordered_map<int, double>;

struct TopKResult {
    Map top_k;
    Map individual_weights;
    Map nontop_k;
};
