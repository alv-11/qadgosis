#ifndef WALKS_H
#define WALKS_H

#include "shared.h"  // Include shared structures
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

// Graph structure
struct Graph {
    std::unordered_map<int, std::unordered_map<int, double>> adjacency_list;
};

// Function declarations
std::vector<int> weighted_random_walk(const Graph& graph, int start_node, int walk_length, std::mt19937& rng);
RandomWalkResult generate_random_walks_directed_with_weights(const Graph& graph, int num_walks, int walk_length);
RandomWalkResult update_random_walks_directed_with_weights(
    const Graph& graph,
    std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
    const GraphChanges& changes,
    int num_walks,
    int walk_length
);

#endif