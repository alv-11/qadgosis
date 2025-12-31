#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <numeric>
#include <limits>

namespace HybridMC {

// Type aliases for convenience
using Graph = std::unordered_map<int, std::unordered_map<int, double>>;
using WalksMap = std::unordered_map<int, std::vector<std::vector<std::string>>>;

// ------------------------- helpers -------------------------
inline void normalize_map(std::unordered_map<int,double>& v) {
    double s = 0.0;
    for (auto &p : v) s += p.second;
    if (s > 0.0) for (auto &p : v) p.second /= s;
}

inline std::unordered_set<int> collect_nodes_from_graph_and_walks(const Graph &graph,
    const WalksMap &processed_R) {
    std::unordered_set<int> nodes;
    for (const auto &p : graph) {
        nodes.insert(p.first);
        for (const auto &q : p.second) nodes.insert(q.first);
    }
    for (const auto &p : processed_R) {
        try { nodes.insert(p.first); } catch(...) {}
        for (const auto &walk : p.second) {
            for (const auto &ns : walk) {
                if (!ns.empty()) nodes.insert(std::stoi(ns));
            }
        }
    }
    return nodes;
}

// ------------------------- MC complete-path estimator from precomputed walks -------------------------
inline std::unordered_map<int,double>
mc_completepath_from_walks(const WalksMap &processed_R) {
    std::unordered_map<int,double> counts;
    for (const auto & [start, walks] : processed_R) {
        for (const auto & walk : walks) {
            for (const auto & node_str : walk) {
                if (node_str.empty()) continue;
                int node = std::stoi(node_str);
                counts[node] += 1.0;
            }
        }
    }
    normalize_map(counts);
    return counts; // sums to 1 (if there were visits)
}

// ------------------------- Power iteration that accepts an initial vector -------------------------
inline std::unordered_map<int,double>
pagerank_power_iteration_with_init(
    const Graph &graph,
    const std::unordered_set<int> &all_nodes, // set of nodes to include (size = n)
    const std::unordered_map<int,double> &init_pr, // may be sparse; missing entries treated as 0
    double damping = 0.85,
    double epsilon = 1e-6,
    int max_iter = 200,
    bool personalized = false,
    const std::unordered_set<int> &personalization_set = {}
) {
    int n = (int)all_nodes.size();
    // Build pr and new_pr including all nodes in all_nodes
    std::unordered_map<int,double> pr, new_pr;
    pr.reserve(n*2); new_pr.reserve(n*2);
    for (int node : all_nodes) {
        double v = 0.0;
        auto it = init_pr.find(node);
        if (it != init_pr.end()) v = it->second;
        pr[node] = v;
        new_pr[node] = 0.0;
    }
    // If init sums to zero (no visits), fallback to uniform init
    double sum_init = 0.0;
    for (auto &p : pr) sum_init += p.second;
    if (sum_init <= 0.0) {
        double base = 1.0 / n;
        for (auto &p : pr) p.second = base;
    } else {
        // renormalize initial
        for (auto &p : pr) p.second /= sum_init;
    }

    // precompute out-degree sums (weights) for graph nodes (for nodes not in graph -> out_sum 0)
    std::unordered_map<int,double> out_sum;
    out_sum.reserve(graph.size()*2);
    for (const auto & [u, nbrs] : graph) {
        double total = 0.0;
        for (const auto & [v,w] : nbrs) total += w;
        out_sum[u] = total;
    }

    double base_uniform = (personalized && !personalization_set.empty()) ? 0.0 : (1.0 - damping) / n;
    double base_person = 0.0;
    if (personalized && !personalization_set.empty()) base_person = (1.0 - damping) / (double)personalization_set.size();

    for (int iter = 0; iter < max_iter; ++iter) {
        // initialize teleportation
        for (auto & [node, val] : new_pr) val = 0.0;
        if (personalized && !personalization_set.empty()) {
            for (int u : personalization_set) {
                if (new_pr.find(u) != new_pr.end()) new_pr[u] = base_person;
            }
        } else {
            for (auto & [node, val] : new_pr) val = base_uniform;
        }

        // distribute rank through edges
        for (const auto & [u, nbrs] : graph) {
            double os = 0.0;
            auto it = out_sum.find(u);
            if (it != out_sum.end()) os = it->second;
            if (os == 0.0) continue; // dangling handled via teleport part
            double contrib = damping * pr.at(u) / os;
            for (const auto & [v, w] : nbrs) {
                if (new_pr.find(v) != new_pr.end()) new_pr[v] += contrib * w;
            }
        }

        // handle dangling nodes by distributing their rank uniformly (classical approach)
        // sum dangling mass
        double dangling_mass = 0.0;
        for (int node : all_nodes) {
            if (out_sum.find(node) == out_sum.end() || out_sum.at(node) == 0.0) {
                dangling_mass += damping * pr.at(node);
            }
        }
        if (dangling_mass > 0.0) {
            double add = dangling_mass / n;
            for (auto & [node, val] : new_pr) val += add;
        }

        // compute diff
        double diff = 0.0;
        for (auto & [node, val] : pr) {
            diff += std::fabs(new_pr[node] - val);
        }
        pr.swap(new_pr);
        if (diff < epsilon) break;
    }

    // final normalization (safety)
    normalize_map(pr);
    return pr;
}

// ------------------------- Hybrid: MC complete-path from walks + PI refinement -------------------------
inline std::unordered_map<int,double>
hybrid_completepath_poweriteration_from_walks(
    const WalksMap &processed_R,
    const Graph &graph,
    double damping = 0.85,
    double epsilon = 1e-6,
    int max_iter = 200,
    bool personalized = false,
    const std::unordered_set<int> &personalization_set = {}
) {
    // 1) MC complete-path estimate (from precomputed walks)
    auto pi_mc = mc_completepath_from_walks(processed_R);


    // 2) collect node universe (include nodes that might appear only in walks)
    auto all_nodes = collect_nodes_from_graph_and_walks(graph, processed_R);

    // Ensure all nodes are in pi_mc map (missing get zero)
    for (int node : all_nodes) if (pi_mc.find(node) == pi_mc.end()) pi_mc[node] = 0.0;

    // 3) Run power iteration starting from pi_mc
    auto pr_refined = pagerank_power_iteration_with_init(graph, all_nodes, pi_mc,
                                                         damping, epsilon, max_iter,
                                                         personalized, personalization_set);
    return pr_refined;
}



inline std::unordered_map<int,double>
hybrid_progressive_poweriteration_from_walks(
    const WalksMap &processed_R,
    const Graph &graph,
    const std::unordered_map<int,double> &prev_pagerank,  // previous PR vector
    double alpha = 0.6,                                   // blend weight for MC
    double damping = 0.85,
    double epsilon = 1e-6,
    int max_iter = 50
) {
    // 1️⃣ Monte Carlo estimate from walks
    auto pi_mc = mc_completepath_from_walks(processed_R);

    // 2️⃣ Collect all nodes from graph + walks + previous PR
    std::unordered_set<int> all_nodes = collect_nodes_from_graph_and_walks(graph, processed_R);
    for (const auto& [node, _] : prev_pagerank)
        all_nodes.insert(node);
    for (const auto& [node, _] : pi_mc)
        all_nodes.insert(node);

    // 3️⃣ Fill missing nodes with 0
    for (int node : all_nodes) {
        if (pi_mc.find(node) == pi_mc.end()) pi_mc[node] = 0.0;
        if (prev_pagerank.find(node) == prev_pagerank.end()) const_cast<std::unordered_map<int,double>&>(prev_pagerank)[node] = 0.0;
    }

    // 4️⃣ Blend Monte Carlo estimate and previous PR
    std::unordered_map<int,double> init_pr;
    init_pr.reserve(all_nodes.size());
    for (int node : all_nodes) {
        double mc_val = pi_mc[node];
        double prev_val = prev_pagerank.count(node) ? prev_pagerank.at(node) : 0.0;
        init_pr[node] = alpha * mc_val + (1.0 - alpha) * prev_val;
    }
    normalize_map(init_pr);

    // 5️⃣ Refine with power iteration
    auto pr_refined = pagerank_power_iteration_with_init(
        graph, all_nodes, init_pr, damping, epsilon, max_iter
    );

    return pr_refined;
}


inline std::unordered_map<int,double>
hybrid_temporal_poweriteration_from_walks(
    const WalksMap &processed_R,
    const Graph &graph,
    const std::unordered_map<int,double> &prev_pagerank,  // previous timestep PR
    double damping = 0.85,
    double epsilon = 1e-6,
    int max_iter = 20,
    double blend_alpha = 0.3                               // weight for previous PR
) {
    // 1️⃣ Monte Carlo estimate for current timestep
    auto pi_mc = mc_completepath_from_walks(processed_R);

    // 2️⃣ Collect all nodes (from graph, walks, and prev PR)
    std::unordered_set<int> all_nodes;
    for (const auto & [u, nbrs] : graph) {
        all_nodes.insert(u);
        for (const auto & [v, _] : nbrs) all_nodes.insert(v);
    }
    for (const auto & [n, _] : prev_pagerank) all_nodes.insert(n);
    for (const auto & [n, _] : pi_mc) all_nodes.insert(n);

    // Ensure all nodes have entries in both maps
    for (int node : all_nodes) {
        if (pi_mc.find(node) == pi_mc.end()) pi_mc[node] = 0.0;
    }

    // 3️⃣ Blend previous PR and new MC estimate
    std::unordered_map<int,double> init_pr;
    for (int node : all_nodes) {
        double oldv = 0.0, newv = 0.0;
        if (prev_pagerank.find(node) != prev_pagerank.end()) oldv = prev_pagerank.at(node);
        if (pi_mc.find(node) != pi_mc.end()) newv = pi_mc.at(node);
        init_pr[node] = (1.0 - blend_alpha) * newv + blend_alpha * oldv;
    }
    normalize_map(init_pr);

    // 4️⃣ Short power iteration refinement (like hybrid)
    auto pr_refined = pagerank_power_iteration_with_init(
        graph, all_nodes, init_pr, damping, epsilon, max_iter
    );

    return pr_refined;
}



    inline std::unordered_map<int,double>
    hybrid_completepath_poweriteration_from_walks_weighted(
        const std::unordered_map<int,double>& node_walk_weight,
        const Graph& graph,
        double damping = 0.85,
        double epsilon = 1e-6,
        int max_iter = 20)
    {
        // Normalize the weights to sum to 1
        std::unordered_map<int,double> pr = node_walk_weight;
        normalize_map(pr);
    
        // Use these as initialization for power iteration
        std::unordered_map<int,double> new_pr;
        new_pr.reserve(pr.size());
    
        // Precompute out-degree sums
        std::unordered_map<int,double> out_sum;
        out_sum.reserve(graph.size());
        for (const auto& [u, neighbors] : graph) {
            double total = 0.0;
            for (const auto& [v, w] : neighbors)
                total += w;
            out_sum[u] = total;
        }
    
        for (int iter = 0; iter < max_iter; ++iter) {
            // Reset new_pr to teleportation
            for (const auto& [node, _] : pr)
                new_pr[node] = (1.0 - damping) * node_walk_weight.at(node);
    
            // Distribute rank
            for (const auto& [u, neighbors] : graph) {
                if (out_sum[u] == 0.0) continue;
                double contrib = damping * pr[u] / out_sum[u];
                for (const auto& [v, w] : neighbors)
                    new_pr[v] += contrib * w;
            }
    
            // Check convergence
            double diff = 0.0;
            for (const auto& [node, val] : pr)
                diff += std::fabs(new_pr[node] - val);
            if (diff < epsilon) break;
    
            std::swap(pr, new_pr);
        }
    
        return pr;
    }
    
    } // namespace HybridMC
    




