#include "takeskosters.h"
#include <limits>
#include <vector>
#include <algorithm>

// --- Takes and Kosters (2015) Algorithm ---
int TakesKosters(Graph& g) {
    if (g.V == 0) return 0;

    int Graph_Diameter_LB = 0;
    std::vector<int> UpperBounds(g.V, std::numeric_limits<int>::max());
    std::vector<bool> visited(g.V, false);
    int visited_count = 0;

    while (visited_count < g.V) {
        // Find unvisited node with the highest UpperBound
        int u = -1;
        int max_ub = -1;
        for (int i = 0; i < g.V; ++i) {
            if (!visited[i] && UpperBounds[i] > max_ub) {
                max_ub = UpperBounds[i];
                u = i;
            }
        }

        if (u == -1 || UpperBounds[u] <= Graph_Diameter_LB) {
            break; // All remaining nodes have an upper bound smaller than our best diameter
        }

        // Run BFS and find its exact longest path (eccentricity)
        auto distances_from_u = g.bfs(u);
        int exact_eccentricity = *std::max_element(distances_from_u.begin(), distances_from_u.end());

        // Update our known diameter
        Graph_Diameter_LB = std::max(Graph_Diameter_LB, exact_eccentricity);
        
        visited[u] = true;
        visited_count++;
        UpperBounds[u] = exact_eccentricity;

        // The Magic Step: Prune other nodes!
        for (int w = 0; w < g.V; ++w) {
            if (!visited[w]) {
                // Triangle inequality
                int potential_max = distances_from_u[w] + exact_eccentricity;
                UpperBounds[w] = std::min(UpperBounds[w], potential_max);
            }
        }
    }

    return Graph_Diameter_LB;
}
