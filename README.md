# Graph Diameter Approximation Algorithms

This project provides Python implementations of two efficient algorithms for approximating the diameter of a graph:
1.  **k-BFS (Shun)**
2.  **Takes and Kosters (2015)**

The diameter of a graph is the longest shortest path between any two nodes. Calculating it exactly can be computationally expensive for large graphs, which is why approximation algorithms are used.

## File Structure
- `main.py`: The main entry point of the program, which runs example graphs.
- `algos/graph.py`: A simple `Graph` class with BFS implementation.
- `algos/kbfs.py`: Implementation of the k-BFS algorithm.
- `algos/takeskosters.py`: Implementation of the Takes and Kosters algorithm.

## Algorithms

### k-BFS (Shun)
This algorithm samples $k$ sources, runs a bit-vector BFS to estimate distances to the sample set, then repeats from the $k$ most distant vertices to improve accuracy.

### Takes and Kosters (2015)
This algorithm uses the triangle inequality to maintain and progressively tighten an upper bound on the eccentricity (longest shortest path) of each node. As it runs BFS from nodes with high upper bounds, it prunes other nodes whose potential eccentricity can no longer exceed the best-known diameter found so far.

## How to Run

### Running with a Graph File
You can run the program with a pre-made graph file.

```bash
# Run the executable with a graph file from the 'graphs' directory
python3 main.py graphs/sparse.txt
```

### Running with the Graph Generator
A Python script `graph_generator.py` is provided to dynamically create graphs. This is the recommended way to test the algorithms on various inputs.

**Generator Usage:**
`./graph_generator.py [type] -v [num_vertices]`

-   **type**: `sparse`, `dense`, `tree`, `complete`, `bipartite`
-   **-v, --vertices**: Number of vertices (default: 50)

You can pipe the output of the generator directly into the program:

```bash
# Generate a sparse graph with 100 vertices and find its diameter
./graph_generator.py sparse -v 100 | python3 main.py

# Generate a complete graph with 30 vertices and find its diameter
./graph_generator.py complete -v 30 | python3 main.py

## Experiments
The experiment runner downloads graphs, generates trees/paths, runs k-BFS and Takes-Kosters, and produces a CSV plus timing plots.

```bash
python3 scripts/run_experiments.py --k 5
```

Outputs:
- results.csv
- k_bfs_time.png
- tk_time.png
```

### Graph File Format
The program expects a graph file in a simple edge list format:
- The first line contains two integers: `number_of_vertices` and `number_of_edges`.
- Each subsequent line contains two integers `u v`, representing an edge between node `u` and node `v`.
- Lines starting with `#` are treated as comments and ignored.

A `sample_graph.txt` is provided, and a `graphs` directory contains various types of graphs for testing.
