#!/usr/bin/env python3
import sys
import random
import argparse

def generate_edges(vertices, edges):
    """Generates a set of random edges."""
    if edges > vertices * (vertices - 1) / 2:
        edges = vertices * (vertices - 1) / 2
    
    edge_set = set()
    while len(edge_set) < edges:
        u = random.randint(0, vertices - 1)
        v = random.randint(0, vertices - 1)
        if u != v:
            # To ensure (u, v) and (v, u) are treated the same for undirected graphs
            edge_set.add(tuple(sorted((u, v))))
    return list(edge_set)

def print_graph(vertices, edges, edge_list):
    """Prints the graph in the required format."""
    print(f"{vertices} {len(edge_list)}")
    for u, v in edge_list:
        print(f"{u} {v}")

def generate_sparse(vertices, **kwargs):
    """Generates a sparse graph."""
    num_edges = int(vertices * 1.5)
    edges = generate_edges(vertices, num_edges)
    print_graph(vertices, len(edges), edges)

def generate_dense(vertices, **kwargs):
    """Generates a dense graph."""
    max_edges = vertices * (vertices - 1) / 2
    num_edges = int(max_edges * 0.8) # 80% density
    edges = generate_edges(vertices, num_edges)
    print_graph(vertices, len(edges), edges)

def generate_tree(vertices, **kwargs):
    """Generates a random tree."""
    edges = []
    nodes = list(range(vertices))
    random.shuffle(nodes)
    
    visited = {nodes[0]}
    for i in range(1, vertices):
        u = nodes[i]
        v = random.choice(list(visited))
        edges.append(tuple(sorted((u, v))))
        visited.add(u)
    print_graph(vertices, len(edges), edges)

def generate_complete(vertices, **kwargs):
    """Generates a complete graph."""
    edges = []
    for i in range(vertices):
        for j in range(i + 1, vertices):
            edges.append((i, j))
    print_graph(vertices, len(edges), edges)

def generate_bipartite(vertices, **kwargs):
    """Generates a bipartite graph."""
    if vertices < 2:
        print_graph(vertices, 0, [])
        return
        
    left_size = vertices // 2
    right_size = vertices - left_size
    left_nodes = list(range(left_size))
    right_nodes = list(range(left_size, vertices))
    
    edges = set()
    num_edges = int(vertices * 1.5) # Arbitrary number of edges
    
    for _ in range(num_edges):
        u = random.choice(left_nodes)
        v = random.choice(right_nodes)
        edges.add(tuple(sorted((u, v))))
        
    print_graph(vertices, len(edges), list(edges))


def main():
    parser = argparse.ArgumentParser(description="Graph Generator for Diameter Approximation Algorithms")
    parser.add_argument("type", choices=["sparse", "dense", "tree", "complete", "bipartite"], help="Type of graph to generate.")
    parser.add_argument("-v", "--vertices", type=int, default=50, help="Number of vertices.")
    
    args = parser.parse_args()

    generators = {
        "sparse": generate_sparse,
        "dense": generate_dense,
        "tree": generate_tree,
        "complete": generate_complete,
        "bipartite": generate_bipartite,
    }

    generators[args.type](vertices=args.vertices)

if __name__ == "__main__":
    main()
