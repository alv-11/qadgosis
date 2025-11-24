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
//
// processed_R: map of node -> vector of walks, each walk = vector<string> of visited nodes
// n: number of unique nodes
// len_R: total number of walks
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
// 1. MC End-point with random start
// (uses only final node of each walk)
// ===============================================================
inline std::unordered_map<int, double>
MC_EndPoint_RandomStart_FromWalks(
    const std::unordered_map<int, std::vector<std::vector<std::string>>>& processed_R,
    int n, int len_R)
{
    std::unordered_map<int, double> counts;

    for (const auto& [start_node, walks] : processed_R) {
        for (const auto& walk : walks) {
            if (!walk.empty()) {
                int end = std::stoi(walk.back());
                counts[end] += 1.0;
            }
        }
    }

    for (auto& [node, v] : counts) v /= (double)len_R;
    normalize(counts);
    return counts;
}

// ===============================================================
// 2. MC End-point with cyclic start
// (same as above, but assumes equal #walks per node)
// ===============================================================
inline std::unordered_map<int, double>
MC_EndPoint_CyclicStart_FromWalks(
    const std::unordered_map<int, std::vector<std::vector<std::string>>>& processed_R,
    int n, int len_R)
{
    std::unordered_map<int, double> counts;

    for (const auto& [start_node, walks] : processed_R) {
        for (const auto& walk : walks) {
            if (!walk.empty()) {
                int end = std::stoi(walk.back());
                counts[end] += 1.0;
            }
        }
    }

    double total_walks = (double)len_R;
    for (auto& [node, v] : counts) v /= total_walks;
    normalize(counts);
    return counts;
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

// ===============================================================
// 4. MC Complete Path stopping at dangling nodes
// (stop early if dangling node reached — but since walks are precomputed,
// assume walk already stops at a dangling or teleported node)
// ===============================================================
inline std::unordered_map<int, double>
MC_CompletePath_StopAtDangling_FromWalks(
    const std::unordered_map<int, std::vector<std::vector<std::string>>>& processed_R,
    int n, int len_R)
{
    std::unordered_map<int, double> counts;
    double total_visits = 0.0;

    for (const auto& [start_node, walks] : processed_R) {
        for (const auto& walk : walks) {
            for (const auto& node_str : walk) {
                int node = std::stoi(node_str);
                counts[node] += 1.0;
                total_visits += 1.0;
            }
        }
    }

    for (auto& [node, v] : counts) v /= total_visits;
    normalize(counts);
    return counts;
}

// ===============================================================
// 5. MC Complete Path with random start
// (aggregate across all walks, weighted equally by total visits)
// ===============================================================
inline std::unordered_map<int, double>
MC_CompletePath_RandomStart_FromWalks(
    const std::unordered_map<int, std::vector<std::vector<std::string>>>& processed_R,
    int n, int len_R)
{
    std::unordered_map<int, double> counts;
    double total_visits = 0.0;

    for (const auto& [start_node, walks] : processed_R) {
        for (const auto& walk : walks) {
            for (const auto& node_str : walk) {
                int node = std::stoi(node_str);
                counts[node] += 1.0;
                total_visits += 1.0;
            }
        }
    }

    for (auto& [node, v] : counts) v /= total_visits;
    normalize(counts);
    return counts;
}

} // namespace MCWalks
