import sys
from typing import TextIO

from algos.graph import Graph
from algos.kbfs import kbfs_eccentricity_estimate
from algos.takeskosters import takes_kosters


def read_graph_from_stream(stream: TextIO) -> Graph:
    num_vertices = 0
    num_edges = 0

    it = iter(stream)
    for line in it:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        num_vertices = int(parts[0])
        num_edges = int(parts[1])
        break

    graph = Graph(num_vertices)
    for line in it:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        u = int(parts[0])
        v = int(parts[1])
        graph.add_edge(u, v)

    return graph


def read_graph_from_file(filename: str) -> Graph:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return read_graph_from_stream(file)
    except OSError:
        print(f"Error: Could not open file {filename}")
        sys.exit(1)


def main() -> int:
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [optional_graph_file]")
        print("If no file is provided, it reads from stdin.")
        return 1

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        graph = read_graph_from_file(filename)
        print(f"Graph loaded from: {filename}")
    else:
        graph = read_graph_from_stream(sys.stdin)
        print("Graph loaded from standard input.")

    if graph.V == 0:
        print("Graph is empty or could not be read.")
        return 1

    print("--- Graph Analysis ---")
    print(f"Vertices: {graph.V}")

    k = min(5, graph.V)
    print(f"\nRunning k-BFS algorithm (k={k})...")
    eccentricity_estimates = kbfs_eccentricity_estimate(graph.adj, k)
    kbfs_diameter = max(eccentricity_estimates)
    print(f"k-BFS estimated diameter: {kbfs_diameter}")

    print("\nRunning Takes & Kosters algorithm...")
    tk_diameter = takes_kosters(graph)
    print(f"Takes & Kosters estimated diameter: {tk_diameter}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
