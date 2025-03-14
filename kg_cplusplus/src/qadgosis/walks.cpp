
#include "walks.h"


namespace py = pybind11;

// Function to generate a UUID for walk ID
std::string generate_uuid() {
    uuid_t uuid;
    uuid_generate_random(uuid);
    char uuid_str[37];
    uuid_unparse(uuid, uuid_str);
    return std::string(uuid_str);
}


//random walk function
std::vector<int> weighted_random_walk(const Graph& graph, int start_node, int walk_length, std::mt19937& rng) {
    std::vector<int> walk;
    walk.push_back(start_node);
    int current_node = start_node;

    while (walk.size() < static_cast<size_t>(walk_length + 1)) {
        if (graph.adjacency_list.find(current_node) == graph.adjacency_list.end() ||
            graph.adjacency_list.at(current_node).empty()) {
            break;
        }

        std::vector<int> neighbors;
        std::vector<double> weights;
        for (const auto& neighbor_pair : graph.adjacency_list.at(current_node)) {
            neighbors.push_back(neighbor_pair.first);
            weights.push_back(neighbor_pair.second);
        }

        double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum_weights > 0) {
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [sum_weights](double w) { return w / sum_weights; });
        } else {
            weights.assign(neighbors.size(), 1.0 / neighbors.size()); // Assign equal probabilities
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        current_node = neighbors[dist(rng)];
        walk.push_back(current_node);
    }

    // If the walk is shorter than the required length, continue the walk from the last node
    while (walk.size() < static_cast<size_t>(walk_length + 1)) {
        if (graph.adjacency_list.find(current_node) == graph.adjacency_list.end() ||
            graph.adjacency_list.at(current_node).empty()) {
            break;
        }

        std::vector<int> neighbors;
        std::vector<double> weights;
        for (const auto& neighbor_pair : graph.adjacency_list.at(current_node)) {
            neighbors.push_back(neighbor_pair.first);
            weights.push_back(neighbor_pair.second);
        }

        double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum_weights > 0) {
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [sum_weights](double w) { return w / sum_weights; });
        } else {
            weights.assign(neighbors.size(), 1.0 / neighbors.size()); // Assign equal probabilities
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        current_node = neighbors[dist(rng)];
        walk.push_back(current_node);
    }

    return walk;
}


//genrate random walk at first timestep
RandomWalksResult* generate_random_walks_directed_with_weights(const Graph& graph, int num_walks, int walk_length) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto* result = new RandomWalksResult();

    using RNG = std::mt19937;    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); // Seed the RNG with the current time
    RNG rng(seed);  

    std::mutex mutex;
    std::vector<int> no_walk_nodes;  // Track nodes that fail to generate walks

    auto worker = [&](int node) {
        std::vector<std::vector<int>> local_walks;
        std::vector<std::string> local_walk_ids;
        std::unordered_map<std::pair<int, int>, std::vector<std::string>, pair_hash> local_inverted_edges;
        std::unordered_map<int, std::unordered_set<std::string>> local_inverted_nodes;
        std::unordered_map<std::string, std::vector<int>> local_walk_ids_map;  // New local storage for walk_ids
    
        for (int i = 0; i < num_walks; ++i) {
            std::vector<int> walk = weighted_random_walk(graph, node, walk_length, rng);
            std::string walk_id = generate_uuid();
            
            local_walks.push_back(walk);
            local_walk_ids.push_back(walk_id);
            local_walk_ids_map[walk_id] = walk;  // Store walk in local map
    
            for (size_t j = 1; j < walk.size(); ++j) {
                std::pair<int, int> edge = {walk[j - 1], walk[j]};
                local_inverted_edges[edge].push_back(walk_id);
                local_inverted_nodes[walk[j]].insert(walk_id);
            }
        }
    
        std::lock_guard<std::mutex> lock(mutex);
        if (local_walks.empty()) {
            no_walk_nodes.push_back(node);
        } else {
            result->precomputed_walks[node] = std::move(local_walks);
            result->random_walk_ids[node] = std::move(local_walk_ids);
    
            for (const auto& [edge, ids] : local_inverted_edges) {
                result->inverted_edges[edge].insert(result->inverted_edges[edge].end(), ids.begin(), ids.end());
            }
            for (const auto& [node, ids] : local_inverted_nodes) {
                result->inverted_nodes[node].insert(ids.begin(), ids.end());
            }
            for (const auto& [walk_id, walk] : local_walk_ids_map) {
                result->walk_ids[walk_id] = walk;  
            }
        }
    };
    

    size_t max_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (const auto& node_pair : graph.adjacency_list) {
        int node = node_pair.first;
        if (threads.size() >= max_threads) {
            for (auto& thread : threads) {
                thread.join();
            }
            threads.clear();
        }
        threads.emplace_back(worker, node);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (!no_walk_nodes.empty()) {
        std::cout << "Nodes with no walks: ";
        for (int node : no_walk_nodes) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    
        // Counting nodes in no_walk_nodes
        //std::cout << "Number of nodes with no walks: " << no_walk_nodes.size() << std::endl;
    }
    

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return result;
}

// Function to free memory
void free_random_walks_result(RandomWalksResult* result) {
    if (result) {
        result->precomputed_walks.clear();
        result->inverted_edges.clear();
        result->inverted_nodes.clear();
        result->walk_ids.clear();
        result->random_walk_ids.clear();
        delete result;  // Free allocated memory
    }
}

// Function to update random walks
RandomWalksResult* update_random_walks_directed_with_weights(
    const Graph& graph,
    std::unordered_map<std::string, std::vector<int>>& walk_ids,
    std::unordered_map<std::pair<int, int>, std::vector<std::string>, pair_hash>& inverted_edges,
    std::unordered_map<int, std::unordered_set<std::string>>& inverted_nodes,
    std::unordered_map<int, std::vector<std::string>>& random_walk_ids,
    const ChangesMap& changes,
    int num_walks,
    int walk_length
) {
    auto* result = new RandomWalksResult();
    if (!result) {
        std::cerr << "Error: Failed to allocate memory for RandomWalksResult." << std::endl;
        return nullptr;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    using RNG = std::mt19937;    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RNG rng(seed);  // Alias used here
    std::unordered_map<int, std::vector<int>> neighbors_dict;
    std::unordered_map<int, std::vector<double>> weights_dict;


    // Populate neighbors and weights dictionaries
    if (graph.adjacency_list.empty()) {
        std::cerr << "Error: Graph adjacency list is empty." << std::endl;
        delete result;
        return nullptr;
    }

    for (const auto& node_pair : graph.adjacency_list) {
        int node = node_pair.first;
        neighbors_dict[node] = {};
        weights_dict[node] = {};

        for (const auto& neighbor_pair : node_pair.second) {
            neighbors_dict[node].push_back(neighbor_pair.first);
            weights_dict[node].push_back(neighbor_pair.second);
        }
    }

    std::unordered_map<std::string, std::vector<int>> truncated_walk_ids;

    // Handle edge removals
    auto start_block1 = std::chrono::high_resolution_clock::now();
    if (changes.find("edges_removed") != changes.end()) {
        const auto& edges_removed_variant = changes.at("edges_removed");
        if (std::holds_alternative<std::unordered_set<std::pair<int, int>, pair_hash>>(edges_removed_variant)) {
            const auto& edges_removed = std::get<std::unordered_set<std::pair<int, int>, pair_hash>>(edges_removed_variant);
            
            for (const auto& edge : edges_removed) {
    
                if (inverted_edges.find(edge) != inverted_edges.end()) {
                    auto affected_walks = inverted_edges[edge];
                    
                    inverted_edges.erase(edge);
    
                    for (const auto& walk_id : affected_walks) {
                    
                        if (walk_ids.find(walk_id) != walk_ids.end()) {
                            auto& walk = walk_ids.at(walk_id);
                    
                            // Debugging: Check if edge appears in the walk at all
                            bool edge_found = false;
                            for (size_t i = 0; i < walk.size() - 1; ++i) {
                                if (walk[i] == edge.first && walk[i + 1] == edge.second) {
                                    edge_found = true;
                                    break;
                                }
                            }
                            if (!edge_found) {

                                continue;  // Skip this walk since the edge is not in it
                            }
                    
                            auto it = std::adjacent_find(walk.begin(), walk.end(),
                                [edge](int a, int b) { return a == edge.first && b == edge.second; });
                    
                            if (it != walk.end()) {
                                //std::cout << "Found edge in walk, truncating..." << std::endl;
                                std::vector<int> truncated_walk(walk.begin(), it + 1);
                    
                                if (!truncated_walk.empty()) {
                                    truncated_walk_ids[walk_id] = truncated_walk;
 
                                } 
                            } 
                        } 
                    }                    
                }   
            }
        } 
    } 
    auto end_block1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> block1_duration = end_block1 - start_block1;
    std::cout << "Block 1 (edge removals): " << block1_duration.count() << " seconds" << std::endl;


    // Handle node removals
    auto start_block2 = std::chrono::high_resolution_clock::now();
    if (changes.find("nodes_removed") != changes.end()) {
        const auto& nodes_removed_variant = changes.at("nodes_removed");
        if (std::holds_alternative<std::unordered_set<int>>(nodes_removed_variant)) {
            const auto& nodes_removed = std::get<std::unordered_set<int>>(nodes_removed_variant);
            for (const auto& node : nodes_removed) {
                if (inverted_nodes.find(node) != inverted_nodes.end()) {
                    auto affected_walks = inverted_nodes[node];
                    inverted_nodes.erase(node);  // Remove node from inverted_nodes

                    if (random_walk_ids.find(node) != random_walk_ids.end()) {
                        random_walk_ids.erase(node); // Remove from start nodes
                    }

                    for (const auto& walk_id : affected_walks) {
                        if (walk_ids.find(walk_id) != walk_ids.end()) {
                            auto& walk = walk_ids.at(walk_id);
                            auto it = std::find(walk.begin(), walk.end(), node);

                            if (it != walk.end()) {
                                std::vector<int> truncated_walk(walk.begin(), it);  // Truncate walk at removed node

                                if (!truncated_walk.empty()) {
                                    truncated_walk_ids[walk_id] = truncated_walk;
                                } 
                             }
                        }
                    }
                }
            }
        }
    }
    auto end_block2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> block2_duration = end_block2 - start_block2;
    std::cout << "Block 2 (node removals): " << block2_duration.count() << " seconds" << std::endl;
    

    // Extend truncated walks
    auto start_block3 = std::chrono::high_resolution_clock::now();
    for (auto& [walk_id, walk] : truncated_walk_ids) {
        while (walk.size() < static_cast<size_t>(walk_length + 1)) {
            int last_node = walk.back();
            
            if (neighbors_dict.find(last_node) == neighbors_dict.end() || neighbors_dict[last_node].empty()) {
                // If no neighbors, restart the walk from a random node
                walk.push_back(walk.front()); 
                continue;
            }
    
            auto& neighbors = neighbors_dict[last_node];
            auto& weights = weights_dict[last_node];
    
            if (weights.empty()) {
                // Assign equal probabilities instead of breaking
                weights.assign(neighbors.size(), 1.0 / neighbors.size()); // Assign equal probabilities

            }    
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            int next_node = neighbors[dist(rng)];
            walk.push_back(next_node);
        }

        // Update ids_to_walks and inverted structures
        truncated_walk_ids[walk_id] = walk;
        for (size_t i = 1; i < walk.size(); ++i) {
            inverted_edges[{walk[i - 1], walk[i]}].push_back(walk_id);
            inverted_nodes[walk[i]].insert(walk_id);
        }
    }
    // Merge truncated_walk_ids into ids_to_walks
    for (const auto& [walk_id, walk] : truncated_walk_ids) {
        walk_ids[walk_id] = walk;
    }
    auto end_block3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> block3_duration = end_block3 - start_block3;
    std::cout << "Block 3 (merge truncated walks): " << block3_duration.count() << " seconds" << std::endl;


        // Handle new node additions
    auto start_block4 = std::chrono::high_resolution_clock::now();
    if (changes.find("nodes_added") != changes.end()) {
        const auto& nodes_added_variant = changes.at("nodes_added");
        if (std::holds_alternative<std::unordered_set<int>>(nodes_added_variant)) {
            const auto& nodes_added = std::get<std::unordered_set<int>>(nodes_added_variant);
            
            // Generate random walks for the added nodes
            for (const auto& node : nodes_added) {
                std::vector<std::string> random_ids;  // Collect walk IDs per node
                
                // Generate random walks for the node
                for (int i = 0; i < num_walks; ++i) {
                    std::vector<int> walk = { node };
                    int current_node = node;

                    // Generate the walk
                    for (int j = 0; j < walk_length; ++j) {
                        if (neighbors_dict.find(current_node) != neighbors_dict.end() && !neighbors_dict[current_node].empty()) {
                            auto& neighbors = neighbors_dict[current_node];
                            auto& weights = weights_dict[current_node];
                            if (weights.empty()) {
                                
                                weights.assign(neighbors.size(), 1.0 / neighbors.size()); // Assign equal probabilities
                
                            }
                            std::discrete_distribution<int> dist(weights.begin(), weights.end());
                            int next_node = neighbors[dist(rng)];
                            walk.push_back(next_node);
                            current_node = next_node;
                        } else {
                            break;  // Stop walk if no neighbors
                        }
                    }
                    // Create a unique ID for the walk
                    std::string unique_id = generate_uuid();
                    walk_ids[unique_id] = walk;
                    random_ids.push_back(unique_id);  // Store the ID

                    // Update inverted edges and nodes for each step in the walk
                    for (size_t k = 1; k < walk.size(); ++k) {
                        inverted_edges[{walk[k - 1], walk[k]}].push_back(unique_id);
                        inverted_nodes[walk[k - 1]].insert(unique_id);
                        inverted_nodes[walk[k]].insert(unique_id);
                    }
                } 
                if (!random_ids.empty()) {
                    random_walk_ids[node] = random_ids;
                } else {
                    std::cerr << "No walks generated for node " << node << std::endl;
                }
                

            } // End of for (const auto& node : nodes_added)
        } else {
            std::cerr << "Error: 'nodes_added' does not contain a set of nodes." << std::endl;
        }
    } 
    auto end_block4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> block4_duration = end_block4 - start_block4;
    std::cout << "Block 4 (node additions): " << block4_duration.count() << " seconds" << std::endl;


    // Reconstruct precomputed_walks
    auto start_block5 = std::chrono::high_resolution_clock::now();
    
    std::unordered_map<int, std::vector<std::vector<int>>> precomputed_walks;

    
    for (const auto& [node, walks] : random_walk_ids) {
        std::vector<std::vector<int>> walks_for_node;
        walks_for_node.reserve(walks.size()); // Reserve space 
    
        for (const auto& walk : walks) {
            auto it = walk_ids.find(walk);
            if (it != walk_ids.end()) {
                walks_for_node.emplace_back(it->second); 
            }
        }
    
        if (!walks_for_node.empty()) {
            precomputed_walks.emplace(node, std::move(walks_for_node)); // Use emplace for efficiency
        }
    }
    

    auto end_block5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> block5_duration = end_block5 - start_block5;
    std::cout << "Block 5 (reconstruct precomputed walks): " << block5_duration.count() << " seconds" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    result->inverted_edges = inverted_edges;
    result->inverted_nodes = inverted_nodes;
    result->precomputed_walks = precomputed_walks;
    result->random_walk_ids = random_walk_ids;
    result->walk_ids = walk_ids;

    return result;
}





// Binding the C++ functions to Python using pybind11
PYBIND11_MODULE(walks, m) {
    m.doc() = "Pybind11 module for weighted random walks on a directed graph";


    py::class_<Graph>(m, "Graph")
        .def(py::init<>())  
        .def(py::init<std::unordered_map<int, std::unordered_map<int, float>>>())  // Constructor to accept Python dictionary
        .def("add_edge", &Graph::add_edge)
        .def("get_neighbors", &Graph::get_neighbors)
        .def_readwrite("adjacency_list", &Graph::adjacency_list);

    py::class_<RandomWalksResult>(m, "RandomWalksResult")
        .def(py::init<>())
        .def_readwrite("precomputed_walks", &RandomWalksResult::precomputed_walks)
        .def_readwrite("random_walk_ids", &RandomWalksResult::random_walk_ids)
        .def_readwrite("walk_ids", &RandomWalksResult::walk_ids)
        .def_readwrite("inverted_edges", &RandomWalksResult::inverted_edges)
        .def_readwrite("inverted_nodes", &RandomWalksResult::inverted_nodes);

    m.def("generate_random_walks_directed_with_weights", 
          [](const std::unordered_map<int, std::unordered_map<int, float>>& py_graph, int num_walks, int walk_length) {
              Graph graph(py_graph);  // Convert Python dict to C++ Graph
              return generate_random_walks_directed_with_weights(graph, num_walks, walk_length);
          },
          "Generate random walks for a directed graph with weights",
          py::arg("graph"), py::arg("num_walks"), py::arg("walk_length"));


    
    m.def("update_random_walks_directed_with_weights",
        [](const std::unordered_map<int, std::unordered_map<int, float>>& py_graph,
           std::unordered_map<std::string, std::vector<int>>& walk_ids,
           std::unordered_map<std::pair<int, int>, std::vector<std::string>, pair_hash>& inverted_edges,
           std::unordered_map<int, std::unordered_set<std::string>>& inverted_nodes,
           std::unordered_map<int, std::vector<std::string>>& random_walk_ids,
           const py::dict& changes_dict,
           int num_walks,
           int walk_length) -> RandomWalksResult* {
        
            Graph graph(py_graph);  

            // Convert the Python dictionary to a ChangesMap
            ChangesMap changes;
            for (const auto& item : changes_dict) {
                std::string key = item.first.cast<std::string>();
                py::handle value = item.second;

                if (key == "nodes_added" || key == "nodes_removed") {
                    // Handle sets of integers (nodes)
                    auto py_set = value.cast<py::set>();
                    std::unordered_set<int> cpp_set;
                    for (const auto& elem : py_set) {
                        cpp_set.insert(elem.cast<int>());
                    }
                    changes[key] = ChangeValue{cpp_set}; 
                } else if (key == "edges_added" || key == "edges_removed") {
                    // Handle sets of tuples (edges)
                    auto py_set = value.cast<py::set>();
                    std::unordered_set<std::pair<int, int>, pair_hash> cpp_set;
                    for (const auto& elem : py_set) {
                        auto py_tuple = elem.cast<py::tuple>();
                        if (py_tuple.size() == 2) {
                            cpp_set.insert({py_tuple[0].cast<int>(), py_tuple[1].cast<int>()});
                        }
                    }
                    changes[key] = ChangeValue{cpp_set};  // Explicitly construct the variant
                } else {
                    throw std::runtime_error("Unsupported key in changes dictionary: " + key);
                }
            }
        // Call the C++ function
        return update_random_walks_directed_with_weights(
            py_graph, walk_ids, inverted_edges, inverted_nodes,
            random_walk_ids, changes, num_walks, walk_length);
    },
    "Update random walks in a directed graph with weights",
    py::arg("graph"),
    py::arg("walk_ids") = py::dict(),
    py::arg("inverted_edges") = py::dict(),
    py::arg("inverted_nodes") = py::dict(),
    py::arg("random_walk_ids") = py::dict(),
    py::arg("changes") = py::dict(),  // for a Python dictionary
    py::arg("num_walks") = 10,
    py::arg("walk_length") = 100);



    m.def("free_random_walks_result", &free_random_walks_result, "Free allocated memory for RandomWalksResult");
}
