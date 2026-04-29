#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <queue>
#include <algorithm>

// Graph representation using an adjacency list
class Graph {
public:
    int V;
    std::vector<std::vector<int>> adj;

    Graph(int vertices);

    void addEdge(int u, int v);

    // BFS function to find distances from a starting node
    std::vector<int> bfs(int startNode);
};

#endif // GRAPH_H
