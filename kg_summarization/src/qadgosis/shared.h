#ifndef SHARED_H
#define SHARED_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>

// Random walk result structure
struct RandomWalkResult {
    std::unordered_map<int, std::vector<std::vector<int>>> precomputed_walks;
};

// Graph changes structure
struct GraphChanges {
    std::unordered_set<int> nodes_added;
    std::unordered_set<int> nodes_removed;
    std::set<std::pair<int, int>> edges_added;
    std::set<std::pair<int, int>> edges_removed;
};

#endif
