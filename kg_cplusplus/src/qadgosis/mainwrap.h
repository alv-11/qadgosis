#ifndef mainwrap_H
#define mainwrap_H

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



// A dummy structure to hold random walk results.
struct RandomWalkResult {
    // We simulate precomputed_walks as a mapping: for each walk index, a list of walks (each is a vector of ints).
    std::unordered_map<int, std::vector<std::vector<int>>> precomputed_walks;
    // (Other fields such as inverted_edges are omitted for simplicity.)
};


// Structure holding changes between snapshots.
struct GraphChanges {
    std::unordered_set<int> nodes_added;
    std::unordered_set<int> nodes_removed;
    std::set<std::pair<int,int>> edges_added;
    std::set<std::pair<int,int>> edges_removed;
};


struct TopKSummary {
    std::unordered_map<int, GraphChanges> changes_over_time;
    std::unordered_map<int, RandomWalkResult> random_walks;
    std::unordered_map<int, std::vector<int>> ppr_walks;
    std::unordered_map<int, std::unordered_map<int, double>> top_k_cumsum_walks;
};

// The main “generate_top_k_summary” function iterates over timesteps,
// updating random walks, computing a personalized random walk (as a PPR surrogate),
// and aggregating counts to produce top–k summary data.
TopKSummary generate_top_k_summary(int num_timesteps, int n_nodes, int m_edges, int num_queries, int num_walks, 
    std::vector<std::unordered_map<int, std::unordered_map<int, double>>>& dynamic_graph, std::unordered_map<int, std::unordered_map<int,int>>& queries, 
    double c, double epsilon, int k);

#endif