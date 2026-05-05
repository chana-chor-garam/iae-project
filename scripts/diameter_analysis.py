from __future__ import annotations

import time

from algos.kbfs import kbfs_eccentricity_estimate
from algos.takeskosters import takes_kosters_with_witness
from scripts.synthetic_benchmarks import exact_diameter


def find_node_pair_for_distance(graph, target_distance: int) -> tuple[int, int] | None:
    if graph is None or graph.V == 0:
        return None
    if target_distance < 0:
        return None
    if target_distance == 0:
        return (0, 0)
    for u in range(graph.V):
        distances = graph.bfs(u)
        for v, d in enumerate(distances):
            if d == target_distance:
                return (u, v)
    return None


def find_kbfs_witness_pair(graph, kbfs_estimates: list[int]) -> tuple[int, int] | None:
    if graph is None or graph.V == 0 or not kbfs_estimates:
        return None
    limit = min(graph.V, len(kbfs_estimates))
    source = max(range(limit), key=lambda node: kbfs_estimates[node])
    distances = graph.bfs(source)
    farthest = max(range(graph.V), key=lambda node: distances[node])
    if distances[farthest] < 0:
        return None
    return (source, farthest)


def compare_diameters(graph, k: int) -> dict:
    start = time.perf_counter()
    kbfs_estimates = kbfs_eccentricity_estimate(graph.adj, k)
    kbfs_time = time.perf_counter() - start
    kbfs_diameter = max(kbfs_estimates) if kbfs_estimates else 0
    kbfs_pair = find_kbfs_witness_pair(graph, kbfs_estimates)

    start = time.perf_counter()
    tk_diameter, tk_pair = takes_kosters_with_witness(graph)
    tk_time = time.perf_counter() - start

    start = time.perf_counter()
    exact = exact_diameter(graph)
    exact_time = time.perf_counter() - start

    if kbfs_pair is None:
        kbfs_pair = find_node_pair_for_distance(graph, kbfs_diameter)
    if tk_pair is None:
        tk_pair = find_node_pair_for_distance(graph, tk_diameter)

    return {
        "exact_diameter": exact,
        "exact_time": exact_time,
        "k_bfs_diameter": kbfs_diameter,
        "k_bfs_time": kbfs_time,
        "k_bfs_pair": kbfs_pair,
        "tk_diameter": tk_diameter,
        "tk_time": tk_time,
        "tk_pair": tk_pair,
    }
