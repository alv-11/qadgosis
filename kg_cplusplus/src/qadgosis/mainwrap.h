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
};

// Function declarations.
TopKSummary generate_top_k_summary(
    int num_timesteps,
    int n_nodes,
    int m_edges,
    int num_queries,
    int num_walks,
    std::vector<std::unordered_map<int, std::unordered_map<int, double>>>& dynamic_graph,
    std::unordered_map<int, std::unordered_map<int, int>>& queries,
    double c,
    double epsilon,
    int k
);

std::vector<int> compute_personalized_pagerank(
    int start_node,
    int walk_length,
    double epsilon,
    const std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
    const std::unordered_map<int, std::unordered_map<int, double>>& graph
);

#endif