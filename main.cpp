#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "graph.h"
#include "ifub.h"
#include "takeskosters.h"

// Function to read a graph from a generic stream (file or stdin)
Graph readGraphFromStream(std::istream& in) {
    int num_vertices = 0, num_edges = 0;
    std::string line;
    
    // Read header, skipping comments
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> num_vertices >> num_edges;
        break;
    }

    Graph g(num_vertices);
    int u, v;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        if (ss >> u >> v) {
            g.addEdge(u, v);
        }
    }
    return g;
}

// Function to read a graph from a file (edge list format)
Graph readGraphFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    return readGraphFromStream(file);
}

int main(int argc, char* argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [optional_graph_file]" << std::endl;
        std::cerr << "If no file is provided, it reads from stdin." << std::endl;
        return 1;
    }

    Graph g(0);
    if (argc == 2) {
        std::string filename = argv[1];
        g = readGraphFromFile(filename);
        std::cout << "Graph loaded from: " << filename << std::endl;
    } else {
        g = readGraphFromStream(std::cin);
        std::cout << "Graph loaded from standard input." << std::endl;
    }

    if (g.V == 0) {
        std::cerr << "Graph is empty or could not be read." << std::endl;
        return 1;
    }

    std::cout << "--- Graph Analysis ---" << std::endl;
    std::cout << "Vertices: " << g.V << std::endl;
    
    std::cout << "\nRunning iFUB algorithm..." << std::endl;
    int ifub_diameter = iFUB(g);
    std::cout << "iFUB estimated diameter: " << ifub_diameter << std::endl;

    std::cout << "\nRunning Takes & Kosters algorithm..." << std::endl;
    int tk_diameter = TakesKosters(g);
    std::cout << "Takes & Kosters estimated diameter: " << tk_diameter << std::endl;

    return 0;
}
