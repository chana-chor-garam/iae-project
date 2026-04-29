#include "ifub.h"
#include <random>
#include <algorithm>
#include <vector>

// --- iFUB (iterative Fringe Upper Bound) Algorithm ---
int iFUB(Graph& g) {
    if (g.V == 0) return 0;

    // Step A: Find a good starting node and a baseline Lower Bound (LB)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, g.V - 1);
    int v = distrib(gen); // pick_random_node()

    auto distances_from_v = g.bfs(v);
    int u = std::distance(distances_from_v.begin(), std::max_element(distances_from_v.begin(), distances_from_v.end())); // furthest_node_from(v)

    // Step B: Get distances from the "center" (u)
    auto distances_from_u = g.bfs(u);
    int LB = *std::max_element(distances_from_u.begin(), distances_from_u.end()); // Lower Bound

    // Step C: Sort nodes from furthest to closest to u
    std::vector<int> nodes(g.V);
    for(int i = 0; i < g.V; ++i) nodes[i] = i;

    std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
        return distances_from_u[a] > distances_from_u[b];
    });

    // Step D: Work inwards from the fringe
    for (int x : nodes) {
        // A tighter upper bound used in the paper is 2 * distance_from_u_to_x
        int UpperBound = distances_from_u[x] * 2;
        
        if (UpperBound <= LB) {
            break; // STOP! No remaining node can beat our current LB!
        }

        auto x_distances = g.bfs(x);
        int max_dist_from_x = *std::max_element(x_distances.begin(), x_distances.end());
        LB = std::max(LB, max_dist_from_x); // Update longest path if we found a better one
    }

    return LB;
}
