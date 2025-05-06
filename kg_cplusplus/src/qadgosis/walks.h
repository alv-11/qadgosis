#ifndef WALKERS_H
#define WALKERS_H

#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <string>    // For std::string
#include <vector>
#include <random>
#include <ctime>
#include <cassert>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <chrono>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <uuid/uuid.h>
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <set>
#include <sstream>


struct pair_hash {
    template <typename T1, typename T2>
    std::size_t operator ()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1); // Combine the two hash values
    }
};
// struct pair_hash {
//     template <class T1, class T2>
//     std::size_t operator()(const std::pair<T1, T2>& p) const {
//         auto hash1 = std::hash<T1>{}(p.first);
//         auto hash2 = std::hash<T2>{}(p.second);
//         return hash1 ^ (hash2 << 1);  // Combine hashes
//     }
// };
// Graph class to store adjacency list
class Graph {
public:
    std::unordered_map<int, std::unordered_map<int, float>> adjacency_list;

    Graph() {} // Default constructor

    // Constructor to accept a Python dictionary (from Pybind11)
    Graph(const std::unordered_map<int, std::unordered_map<int, float>>& py_graph) {
        adjacency_list = py_graph;
    }

    void add_edge(int u, int v, float weight = 0.1) {
        adjacency_list[u][v] = weight;
    }

    std::unordered_map<int, std::unordered_map<int, float>> get_neighbors() {
        return adjacency_list;
    }



};


struct RandomWalksResult {
     std::unordered_map<int, std::vector<std::vector<int>>> precomputed_walks;  // Node -> Walks
     std::unordered_map<std::pair<int, int>, std::vector<std::string>, pair_hash> inverted_edges;  // (From, To) -> UUIDs
     std::unordered_map<int, std::unordered_set<std::string>> inverted_nodes;  // Node -> UUIDs
     std::unordered_map<std::string, std::vector<int>> walk_ids;  // UUID -> Walk
     std::unordered_map<int, std::vector<std::string>> random_walk_ids;  // Node -> UUIDs of random walks

};

// Define the variant type for changes
using ChangeValue = std::variant<
    std::unordered_set<int>,  // For nodes
    std::unordered_set<std::pair<int, int>, pair_hash>  // For edges
>;

// Define the changes map
using ChangesMap = std::unordered_map<std::string, ChangeValue>;

// Define the variant type for nodes and edges
using NodeType = std::variant<int, std::string>;
using EdgeType = std::pair<NodeType, NodeType>;


// Function to generate random walks on a directed weighted graph
RandomWalksResult* generate_random_walks_directed_with_weights(const Graph& graph, int num_walks, int walk_length);



// Function to update random walks based on changes in the graph
RandomWalksResult* update_random_walks_directed_with_weights(const Graph& graph, 
                                                            std::unordered_map<std::string, std::vector<int>> & ids_to_walks, 
                                                            std::unordered_map<std::pair<int, int>, std::vector<std::string>, pair_hash>& inverted_edges, 
                                                            std::unordered_map<int, std::unordered_set<std::string>>& inverted_nodes, 
                                                            std::unordered_map<int, std::vector<std::string>>& startnode_walk_ids, 
                                                            const ChangesMap& changes,
                                                            //std::unordered_map<std::string, std::unordered_set<int>>& changes, 
                                                            int num_walks, 
                                                            int walk_length);

void free_random_walks_result(RandomWalksResult* result);

// Function to generate a UUID as a string
std::string generate_uuid();

#endif // WALKERS_H