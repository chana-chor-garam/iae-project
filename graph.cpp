#include "graph.h"

Graph::Graph(int vertices) : V(vertices), adj(vertices) {}

void Graph::addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
}

// BFS function to find distances from a starting node
std::vector<int> Graph::bfs(int startNode) {
    std::vector<int> distances(V, -1);
    if (startNode < 0 || startNode >= V) return distances;

    std::queue<int> q;

    distances[startNode] = 0;
    q.push(startNode);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            if (distances[v] == -1) {
                distances[v] = distances[u] + 1;
                q.push(v);
            }
        }
    }
    return distances;
}
