#include "walks.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream> // Include for std::cout and std::endl
#include <mutex> // Include for std::mutex
namespace py = pybind11;

// Define global mutex
std::mutex global_mutex;

// Helper types for data structures
typedef std::unordered_map<int, int> NestedMap;
typedef std::unordered_map<int, NestedMap> DefaultDict;

// Function to initialize the data structures
std::tuple<
    std::unordered_map<int, int>,   // changes_over_time
    std::unordered_map<int, int>,   // ppr_walks
    std::unordered_map<int, int>,   // ppr_cumsum_walks
    std::unordered_map<int, int>,    // top_k_cumsum_walks
    std::unordered_map<int, int>    // all_weights
> initializeDataStructures() {
    return std::make_tuple(
        std::unordered_map<int, int>(),   // changes_over_time
        std::unordered_map<int, int>(),   // ppr_walks
        std::unordered_map<int, int>(),   // ppr_cumsum_walks
        std::unordered_map<int, int>(),    // top_k_cumsum_walks
        std::unordered_map<int, int>()    // all_weights
    );
}

std::unordered_map<int, std::unordered_map<int, double>> preprocessGraph(const std::unordered_map<int, std::map<int, std::unordered_map<std::string, double>>>& dyngraph) {
    std::unordered_map<int, std::unordered_map<int, double>> weightedGraph;
    for (const auto& node : dyngraph) {
        int nodeId = node.first;
        std::unordered_map<int, double> neighbors;
        for (const auto& neighbor : node.second) {
            int neighborId = neighbor.first;
            double weight = neighbor.second.count("weight") ? neighbor.second.at("weight") : 0.1;
            neighbors[neighborId] = weight;
        }
        weightedGraph[nodeId] = neighbors;
    }
    return weightedGraph;
}

