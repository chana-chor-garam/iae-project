from __future__ import annotations

import math
import random
import time
import tracemalloc
from collections import deque
from pathlib import Path

import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.graph_io import build_graph_from_edges


def generate_erdos_renyi(vertex_count: int, p: float) -> object:
    if vertex_count <= 1:
        return build_graph_from_edges([])
    edges = []
    for u in range(vertex_count):
        for v in range(u + 1, vertex_count):
            if random.random() < p:
                edges.append((u, v))
    return build_graph_from_edges(edges)


def generate_sparse(vertex_count: int, p: float = 0.03) -> object:
    return generate_erdos_renyi(vertex_count, p)


def generate_dense(vertex_count: int, p: float = 0.6) -> object:
    return generate_erdos_renyi(vertex_count, p)


def generate_tree(vertex_count: int) -> object:
    edges = []
    for node in range(1, vertex_count):
        parent = random.randrange(node)
        edges.append((parent, node))
    return build_graph_from_edges(edges)


def generate_complete(vertex_count: int) -> object:
    edges = []
    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            edges.append((i, j))
    return build_graph_from_edges(edges)


def generate_bipartite(vertex_count: int, p: float = 0.1) -> object:
    if vertex_count < 2:
        return build_graph_from_edges([])
    left_size = vertex_count // 2
    right_size = vertex_count - left_size
    left_nodes = list(range(left_size))
    right_nodes = list(range(left_size, vertex_count))
    edges = []
    for u in left_nodes:
        for v in right_nodes:
            if random.random() < p:
                edges.append((u, v))
    return build_graph_from_edges(edges)


def generate_path(vertex_count: int) -> object:
    edges = [(node, node + 1) for node in range(vertex_count - 1)]
    return build_graph_from_edges(edges)


def generate_grid(vertex_count: int, rows: int | None = None, cols: int | None = None) -> object:
    if rows is None or cols is None:
        side = max(1, int(math.sqrt(vertex_count)))
        rows = side
        cols = max(1, vertex_count // rows)
    edges = []

    def node_id(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append((node_id(r, c), node_id(r + 1, c)))
            if c + 1 < cols:
                edges.append((node_id(r, c), node_id(r, c + 1)))
    return build_graph_from_edges(edges)


def generate_small_world(vertex_count: int, k: int = 4, p: float = 0.1) -> object:
    if vertex_count < 3:
        return build_graph_from_edges([])
    k = max(2, k)
    if k >= vertex_count:
        k = vertex_count - 1
    if k % 2 == 1:
        k += 1
    neighbors = {i: set() for i in range(vertex_count)}
    for i in range(vertex_count):
        for j in range(1, k // 2 + 1):
            v = (i + j) % vertex_count
            neighbors[i].add(v)
            neighbors[v].add(i)
    edges = set()
    for u in range(vertex_count):
        for v in neighbors[u]:
            if u < v:
                edges.add((u, v))
    edge_list = list(edges)
    for u, v in edge_list:
        if random.random() < p:
            neighbors[u].discard(v)
            neighbors[v].discard(u)
            candidates = [x for x in range(vertex_count) if x != u and x not in neighbors[u]]
            if candidates:
                new_v = random.choice(candidates)
                neighbors[u].add(new_v)
                neighbors[new_v].add(u)
    edges = []
    for u in range(vertex_count):
        for v in neighbors[u]:
            if u < v:
                edges.append((u, v))
    return build_graph_from_edges(edges)


def generate_scale_free(vertex_count: int, m: int = 2) -> object:
    if vertex_count <= m + 1:
        return generate_complete(vertex_count)
    edges = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges.append((i, j))
    degrees = [0] * vertex_count
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    for new_node in range(m + 1, vertex_count):
        candidates = list(range(new_node))
        weights = [degrees[i] if degrees[i] > 0 else 1 for i in candidates]
        targets = set()
        while len(targets) < m:
            targets.add(random.choices(candidates, weights=weights, k=1)[0])
        for v in targets:
            edges.append((new_node, v))
            degrees[new_node] += 1
            degrees[v] += 1
    return build_graph_from_edges(edges)


GENERATOR_BY_TYPE = {
    "random_sparse": generate_sparse,
    "dense": generate_dense,
    "tree": generate_tree,
    "complete": generate_complete,
    "bipartite": generate_bipartite,
    "path": generate_path,
    "grid": generate_grid,
    "small_world": generate_small_world,
    "scale_free": generate_scale_free,
}


def count_edges(graph) -> int:
    return sum(len(neighbors) for neighbors in graph.adj) // 2


def exact_diameter(graph) -> int:
    if graph.V == 0:
        return 0
    diameter = 0
    for node in range(graph.V):
        distances = graph.bfs(node)
        finite = [d for d in distances if d >= 0]
        if finite:
            diameter = max(diameter, max(finite))
    return diameter


def bfs_with_counts(graph, start_node: int) -> tuple[list[int], int, int]:
    distances = [-1] * graph.V
    if start_node < 0 or start_node >= graph.V:
        return distances, 0, 0
    queue = deque([start_node])
    distances[start_node] = 0
    vertex_visits = 1
    edge_checks = 0
    while queue:
        u = queue.popleft()
        for v in graph.adj[u]:
            edge_checks += 1
            if distances[v] == -1:
                distances[v] = distances[u] + 1
                queue.append(v)
                vertex_visits += 1
    return distances, vertex_visits, edge_checks


def _multi_source_bit_bfs_stats(adj: list[list[int]], sources: list[int]) -> tuple[list[int], int, int]:
    vertex_count = len(adj)
    if vertex_count == 0:
        return [], 0, 0
    sources = list(dict.fromkeys(sources))
    if not sources:
        return [-1] * vertex_count, 0, 0
    visited_bits = [0] * vertex_count
    frontier_bits = [0] * vertex_count
    distances = [-1] * vertex_count
    vertex_visits = 0
    edge_checks = 0
    for idx, s in enumerate(sources):
        if s < 0 or s >= vertex_count:
            continue
        bit = 1 << idx
        if visited_bits[s] & bit:
            continue
        visited_bits[s] |= bit
        frontier_bits[s] |= bit
        distances[s] = 0
        vertex_visits += 1
    frontier = [v for v in range(vertex_count) if frontier_bits[v]]
    round_num = 0
    while frontier:
        round_num += 1
        next_bits = [0] * vertex_count
        for u in frontier:
            bits = frontier_bits[u]
            for v in adj[u]:
                edge_checks += 1
                next_bits[v] |= bits
        for u in frontier:
            frontier_bits[u] = 0
        new_frontier = []
        for v, bits in enumerate(next_bits):
            new_bits = bits & ~visited_bits[v]
            if new_bits:
                visited_bits[v] |= new_bits
                frontier_bits[v] = new_bits
                if round_num > distances[v]:
                    distances[v] = round_num
                new_frontier.append(v)
                vertex_visits += 1
        frontier = new_frontier
    return distances, vertex_visits, edge_checks


def kbfs_estimate_with_stats(adj: list[list[int]], k: int) -> tuple[list[int], dict]:
    vertex_count = len(adj)
    if vertex_count == 0:
        return [], {"bfs_passes": 0, "vertex_visits": 0, "edge_checks": 0, "pre_time": 0.0, "search_time": 0.0}
    if k <= 0:
        return [-1] * vertex_count, {"bfs_passes": 0, "vertex_visits": 0, "edge_checks": 0, "pre_time": 0.0, "search_time": 0.0}
    k = min(k, vertex_count)
    pre_start = time.perf_counter()
    sources_phase1 = random.sample(range(vertex_count), k)
    pre_time = time.perf_counter() - pre_start
    search_start = time.perf_counter()
    dist_phase1, v_visits_1, e_checks_1 = _multi_source_bit_bfs_stats(adj, sources_phase1)
    search_time = time.perf_counter() - search_start
    pre_start = time.perf_counter()
    ranked = sorted(range(vertex_count), key=lambda i: dist_phase1[i], reverse=True)
    sources_phase2 = ranked[:k]
    pre_time += time.perf_counter() - pre_start
    search_start = time.perf_counter()
    dist_phase2, v_visits_2, e_checks_2 = _multi_source_bit_bfs_stats(adj, sources_phase2)
    search_time += time.perf_counter() - search_start
    estimates = [max(dist_phase1[i], dist_phase2[i]) for i in range(vertex_count)]
    stats = {
        "bfs_passes": 2,
        "vertex_visits": v_visits_1 + v_visits_2,
        "edge_checks": e_checks_1 + e_checks_2,
        "pre_time": pre_time,
        "search_time": search_time,
    }
    return estimates, stats


def takes_kosters_with_stats(graph) -> tuple[int, dict]:
    if graph.V == 0:
        return 0, {"bfs_passes": 0, "vertex_visits": 0, "edge_checks": 0, "pre_time": 0.0, "search_time": 0.0, "iterations": 0}
    graph_diameter_lb = 0
    upper_bounds = [math.inf] * graph.V
    visited = [False] * graph.V
    visited_count = 0
    pre_time = 0.0
    search_time = 0.0
    bfs_passes = 0
    vertex_visits = 0
    edge_checks = 0
    iterations = 0
    while visited_count < graph.V:
        select_start = time.perf_counter()
        u = -1
        max_ub = -1
        for i in range(graph.V):
            if not visited[i] and upper_bounds[i] > max_ub:
                max_ub = upper_bounds[i]
                u = i
        pre_time += time.perf_counter() - select_start
        if u == -1 or upper_bounds[u] <= graph_diameter_lb:
            break
        bfs_start = time.perf_counter()
        distances_from_u, v_visits, e_checks = bfs_with_counts(graph, u)
        search_time += time.perf_counter() - bfs_start
        bfs_passes += 1
        iterations += 1
        vertex_visits += v_visits
        edge_checks += e_checks
        exact_eccentricity = max(distances_from_u)
        graph_diameter_lb = max(graph_diameter_lb, exact_eccentricity)
        update_start = time.perf_counter()
        visited[u] = True
        visited_count += 1
        upper_bounds[u] = exact_eccentricity
        for w in range(graph.V):
            if not visited[w]:
                potential_max = distances_from_u[w] + exact_eccentricity
                if potential_max < upper_bounds[w]:
                    upper_bounds[w] = potential_max
        pre_time += time.perf_counter() - update_start
    stats = {
        "bfs_passes": bfs_passes,
        "vertex_visits": vertex_visits,
        "edge_checks": edge_checks,
        "pre_time": pre_time,
        "search_time": search_time,
        "iterations": iterations,
    }
    return graph_diameter_lb, stats


def run_single(graph, k: int) -> dict:
    tracemalloc.start()
    kbfs_start = time.perf_counter()
    kbfs_estimates, kbfs_stats = kbfs_estimate_with_stats(graph.adj, k)
    kbfs_time = time.perf_counter() - kbfs_start
    _, kbfs_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kbfs_diameter = max(kbfs_estimates) if kbfs_estimates else 0

    tracemalloc.start()
    tk_start = time.perf_counter()
    tk_diameter, tk_stats = takes_kosters_with_stats(graph)
    tk_time = time.perf_counter() - tk_start
    _, tk_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exact = exact_diameter(graph)
    return {
        "k_bfs_diameter": kbfs_diameter,
        "k_bfs_time": kbfs_time,
        "k_bfs_pre_time": kbfs_stats["pre_time"],
        "k_bfs_search_time": kbfs_stats["search_time"],
        "k_bfs_bfs_passes": kbfs_stats["bfs_passes"],
        "k_bfs_vertex_visits": kbfs_stats["vertex_visits"],
        "k_bfs_edge_checks": kbfs_stats["edge_checks"],
        "k_bfs_peak_mb": kbfs_peak / (1024 * 1024),
        "tk_diameter": tk_diameter,
        "tk_time": tk_time,
        "tk_pre_time": tk_stats["pre_time"],
        "tk_search_time": tk_stats["search_time"],
        "tk_bfs_passes": tk_stats["bfs_passes"],
        "tk_vertex_visits": tk_stats["vertex_visits"],
        "tk_edge_checks": tk_stats["edge_checks"],
        "tk_iterations": tk_stats["iterations"],
        "tk_peak_mb": tk_peak / (1024 * 1024),
        "exact_diameter": exact,
    }


def density_bucket(density: float, sparse_max: float = 0.05, dense_min: float = 0.2) -> str:
    if density < sparse_max:
        return "sparse"
    if density > dense_min:
        return "dense"
    return "medium"


def _grid_dims(target_nodes: int) -> tuple[int, int]:
    rows = max(2, int(math.sqrt(target_nodes)))
    cols = max(2, int(math.ceil(target_nodes / rows)))
    return rows, cols


def build_suite_specs() -> list[dict]:
    node_counts = [50, 70, 90, 120, 150, 180, 220, 260, 320, 400]
    prob_list = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
    seed_list = [3, 7, 11, 19, 23, 29, 31, 37, 41, 43]
    k_list = [4, 4, 4, 6, 6, 6, 8, 8, 10, 10]
    m_list = [1, 1, 2, 2, 2, 3, 3, 4, 4, 5]

    specs: list[dict] = []

    for idx, n in enumerate(node_counts):
        specs.append({"family": "clique", "graph_type": "complete", "nodes": n, "params": {}, "seed": seed_list[idx]})
        specs.append({"family": "path", "graph_type": "path", "nodes": n, "params": {}, "seed": seed_list[idx]})
        specs.append({"family": "tree", "graph_type": "tree", "nodes": n, "params": {}, "seed": seed_list[idx]})

        rows, cols = _grid_dims(n)
        specs.append({
            "family": "grid",
            "graph_type": "grid",
            "nodes": rows * cols,
            "params": {"rows": rows, "cols": cols},
            "seed": seed_list[idx],
        })

        specs.append({
            "family": "bipartite",
            "graph_type": "bipartite",
            "nodes": n,
            "params": {"p": prob_list[idx]},
            "seed": seed_list[idx],
        })

        specs.append({
            "family": "small_world",
            "graph_type": "small_world",
            "nodes": n,
            "params": {"k": k_list[idx], "p": prob_list[idx]},
            "seed": seed_list[idx],
        })

        m_val = min(m_list[idx], max(1, n - 2))
        specs.append({
            "family": "scale_free",
            "graph_type": "scale_free",
            "nodes": n,
            "params": {"m": m_val},
            "seed": seed_list[idx],
        })

    return specs


def generate_suite(k: int = 4, sparse_max: float = 0.05, dense_min: float = 0.2) -> pd.DataFrame:
    specs = build_suite_specs()
    rows = []

    for spec in specs:
        random.seed(spec["seed"])
        graph = GENERATOR_BY_TYPE[spec["graph_type"]](spec["nodes"], **spec["params"])
        edges = count_edges(graph)
        density = 0.0
        if graph.V > 1:
            density = (2 * edges) / (graph.V * (graph.V - 1))
        results = run_single(graph, k)
        rows.append({
            "graph_family": spec["family"],
            "nodes": graph.V,
            "edges": edges,
            "density": density,
            "density_bucket": density_bucket(density, sparse_max=sparse_max, dense_min=dense_min),
            "seed": spec["seed"],
            **results,
        })

    df = pd.DataFrame(rows)
    df["k_bfs_abs_error"] = (df["k_bfs_diameter"] - df["exact_diameter"]).abs()
    df["tk_abs_error"] = (df["tk_diameter"] - df["exact_diameter"]).abs()
    return df
