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
    DefaultDict,                    // random_walks
    DefaultDict,                    // edge_to_walks
    DefaultDict,                    // node_to_walks
    DefaultDict,                    // walks_to_ids
    DefaultDict,                    // startnode_to_ids
    std::unordered_map<int, int>,   // ppr_walks
    std::unordered_map<int, int>,   // ppr_cumsum_walks
    std::unordered_map<int, int>    // top_k_cumsum_walks
> initializeDataStructures() {
    return std::make_tuple(
        std::unordered_map<int, int>(),   // changes_over_time
        DefaultDict(),                    // random_walks
        DefaultDict(),                    // edge_to_walks
        DefaultDict(),                    // node_to_walks
        DefaultDict(),                    // walks_to_ids
        DefaultDict(),                    // startnode_to_ids
        std::unordered_map<int, int>(),   // ppr_walks
        std::unordered_map<int, int>(),   // ppr_cumsum_walks
        std::unordered_map<int, int>()    // top_k_cumsum_walks
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

// Random walk function
// std::vector<int> weighted_random_walk(const Graph& graph, int start_node, int walk_length, std::mt19937& rng) {
//     std::vector<int> walk;
//     walk.push_back(start_node);
//     int current_node = start_node;

//     while (walk.size() < static_cast<size_t>(walk_length + 1)) {
//         if (graph.adjacency_list.find(current_node) == graph.adjacency_list.end() ||
//             graph.adjacency_list.at(current_node).empty()) {
//             break;
//         }

//         std::vector<int> neighbors;
//         std::vector<double> weights;
//         for (const auto& neighbor_pair : graph.adjacency_list.at(current_node)) {
//             neighbors.push_back(neighbor_pair.first);
//             weights.push_back(neighbor_pair.second);
//         }

//         double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
//         if (sum_weights > 0) {
//             std::transform(weights.begin(), weights.end(), weights.begin(),
//                            [sum_weights](double w) { return w / sum_weights; });
//         } else {
//             weights.assign(neighbors.size(), 1.0 / neighbors.size()); // Assign equal probabilities
//         }

//         std::discrete_distribution<int> dist(weights.begin(), weights.end());
//         current_node = neighbors[dist(rng)];
//         walk.push_back(current_node);
//     }

//     return walk;
// }

// // Generate random walks at the first timestep
// RandomWalkResult generate_random_walks_directed_with_weights(const Graph& graph, int num_walks, int walk_length) {
//     RandomWalkResult result;

//     using RNG = std::mt19937;
//     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); // Seed the RNG with the current time
//     RNG rng(seed);

//     for (const auto& node_pair : graph.adjacency_list) {
//         int node = node_pair.first;
//         std::vector<std::vector<int>> local_walks;

//         for (int i = 0; i < num_walks; ++i) {
//             std::vector<int> walk = weighted_random_walk(graph, node, walk_length, rng);
//             local_walks.push_back(walk);
//         }

//         result.precomputed_walks[node] = std::move(local_walks);
//     }

//     return result;
// }

// // Update random walks based on changes
// RandomWalkResult update_random_walks_directed_with_weights(
//     const Graph_& graph,
//     std::unordered_map<int, std::vector<std::vector<int>>>& precomputed_walks,
//     const GraphChanges& changes,
//     int num_walks,
//     int walk_length
// ) {
//     RandomWalkResult result;

//     using RNG = std::mt19937;
//     unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//     RNG rng(seed);

//     // Build neighbor and weight lookup tables
//     std::unordered_map<int, std::vector<int>> neighbors_dict;
//     std::unordered_map<int, std::vector<double>> weights_dict;

//     for (const auto& [node, neighbors] : graph) {
//         for (const auto& [nbr, weight] : neighbors) {
//             neighbors_dict[node].push_back(nbr);
//             weights_dict[node].push_back(weight);
//         }
//     }

//     // Handle removed nodes
//     for (int node : changes.nodes_removed) {
//         precomputed_walks.erase(node);  // Remove entire entry
//     }

//     // Handle added nodes (generate fresh walks)
//     for (int node : changes.nodes_added) {
//         std::vector<std::vector<int>> walks;
//         for (int i = 0; i < num_walks; ++i) {
//             std::vector<int> walk = {node};
//             int current = node;

//             for (int j = 0; j < walk_length; ++j) {
//                 if (neighbors_dict.find(current) != neighbors_dict.end() && !neighbors_dict[current].empty()) {
//                     auto& neighbors = neighbors_dict[current];
//                     auto& weights = weights_dict[current];
//                     std::discrete_distribution<int> dist(weights.begin(), weights.end());
//                     current = neighbors[dist(rng)];
//                     walk.push_back(current);
//                 } else {
//                     break;
//                 }
//             }

//             walks.push_back(std::move(walk));
//         }
//         precomputed_walks[node] = std::move(walks);
//     }

//     result.precomputed_walks = precomputed_walks;
//     return result;
// }

// // Function to free memory
// void free_random_walks_result(RandomWalksResult* result) {
//     if (result) {
//         result->precomputed_walks.clear();
//         result->inverted_edges.clear();
//         result->inverted_nodes.clear();
//         result->walk_ids.clear();
//         result->random_walk_ids.clear();
//         delete result;
//     }
// }

// // Binding the C++ functions to Python using pybind11
// PYBIND11_MODULE(walks, m) {
//     m.doc() = "Pybind11 module for weighted random walks on a directed graph";

//     py::class_<Graph>(m, "Graph")
//         .def(py::init<>())
//         .def(py::init<std::unordered_map<int, std::unordered_map<int, float>>>())  // Constructor to accept Python dictionary
//         .def("add_edge", &Graph::add_edge)
//         .def("get_neighbors", &Graph::get_neighbors)
//         .def_readwrite("adjacency_list", &Graph::adjacency_list);

//     py::class_<RandomWalksResult>(m, "RandomWalksResult")
//         .def(py::init<>())
//         .def_readwrite("precomputed_walks", &RandomWalksResult::precomputed_walks)
//         .def_readwrite("random_walk_ids", &RandomWalksResult::random_walk_ids)
//         .def_readwrite("walk_ids", &RandomWalksResult::walk_ids)
//         .def_readwrite("inverted_edges", &RandomWalksResult::inverted_edges)
//         .def_readwrite("inverted_nodes", &RandomWalksResult::inverted_nodes);

//     m.def("generate_random_walks_directed_with_weights",
//           [](const std::unordered_map<int, std::unordered_map<int, float>>& py_graph, int num_walks, int walk_length) {
//               Graph graph(py_graph);  // Convert Python dict to C++ Graph
//               return generate_random_walks_directed_with_weights(graph, num_walks, walk_length);
//           },
//           "Generate random walks for a directed graph with weights",
//           py::arg("graph"), py::arg("num_walks"), py::arg("walk_length"));

//     m.def("free_random_walks_result", &free_random_walks_result, "Free allocated memory for RandomWalksResult");
// }
