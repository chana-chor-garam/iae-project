# Graph Diameter Approximation Algorithms

This project provides C++ implementations of two efficient algorithms for approximating the diameter of a graph:
1.  **iFUB (iterative Fringe Upper Bound)**
2.  **Takes and Kosters (2015)**

The diameter of a graph is the longest shortest path between any two nodes. Calculating it exactly can be computationally expensive for large graphs, which is why approximation algorithms are used.

## File Structure
- `main.cpp`: The main entry point of the program, which runs example graphs.
- `graph.h` / `graph.cpp`: A simple `Graph` class with BFS implementation.
- `ifub.h` / `ifub.cpp`: Implementation of the iFUB algorithm.
- `takeskosters.h` / `takeskosters.cpp`: Implementation of the Takes and Kosters algorithm.

## Algorithms

### iFUB (Crescenzi et al.)
This algorithm works by starting from a peripheral node, calculating distances, and then iteratively running BFS from the "fringe" nodes (those furthest away). It uses an upper bound calculation to prune the search space, often terminating after a very small number of BFS runs.

### Takes and Kosters (2015)
This algorithm uses the triangle inequality to maintain and progressively tighten an upper bound on the eccentricity (longest shortest path) of each node. As it runs BFS from nodes with high upper bounds, it prunes other nodes whose potential eccentricity can no longer exceed the best-known diameter found so far.

## How to Compile and Run

You can compile the code using a C++ compiler like g++.

```bash
# Compile the C++ code
g++ -std=c++11 -o graph_diameter main.cpp graph.cpp ifub.cpp takeskosters.cpp
```

### Running with a Graph File
You can still run the program with a pre-made graph file.

```bash
# Run the executable with a graph file from the 'graphs' directory
./graph_diameter graphs/sparse.txt
```

### Running with the Graph Generator
A Python script `graph_generator.py` is provided to dynamically create graphs. This is the recommended way to test the algorithms on various inputs.

**Generator Usage:**
`./graph_generator.py [type] -v [num_vertices]`

-   **type**: `sparse`, `dense`, `tree`, `complete`, `bipartite`
-   **-v, --vertices**: Number of vertices (default: 50)

You can pipe the output of the generator directly into the C++ program:

```bash
# Generate a sparse graph with 100 vertices and find its diameter
./graph_generator.py sparse -v 100 | ./graph_diameter

# Generate a complete graph with 30 vertices and find its diameter
./graph_generator.py complete -v 30 | ./graph_diameter
```

### Graph File Format
The program expects a graph file in a simple edge list format:
- The first line contains two integers: `number_of_vertices` and `number_of_edges`.
- Each subsequent line contains two integers `u v`, representing an edge between node `u` and node `v`.
- Lines starting with `#` are treated as comments and ignored.

A `sample_graph.txt` is provided, and a `graphs` directory contains various types of graphs for testing.
