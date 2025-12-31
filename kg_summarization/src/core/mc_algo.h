#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_set>

// ===============================================================
// Monte Carlo PageRank (using precomputed walks)
// Based on: Avrachenkov et al. (2007) "Monte Carlo methods in PageRank computation"
// ===============================================================

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



} // namespace MCWalks
