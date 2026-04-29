import random
from typing import Iterable


def _multi_source_bit_bfs(adj: list[list[int]], sources: Iterable[int]) -> list[int]:
    vertex_count = len(adj)
    if vertex_count == 0:
        return []

    sources = list(dict.fromkeys(sources))
    if not sources:
        return [-1] * vertex_count

    visited_bits = [0] * vertex_count
    frontier_bits = [0] * vertex_count
    distances = [-1] * vertex_count

    for idx, s in enumerate(sources):
        if s < 0 or s >= vertex_count:
            continue
        bit = 1 << idx
        if visited_bits[s] & bit:
            continue
        visited_bits[s] |= bit
        frontier_bits[s] |= bit
        distances[s] = 0

    frontier = [v for v in range(vertex_count) if frontier_bits[v]]
    round_num = 0

    while frontier:
        round_num += 1
        next_bits = [0] * vertex_count

        for u in frontier:
            bits = frontier_bits[u]
            for v in adj[u]:
                # NextVisited[v] = Visited[v] | Visited[u] for all k BFS runs.
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

        frontier = new_frontier

    return distances


def kbfs_eccentricity_estimate(adj: list[list[int]], k: int) -> list[int]:
    vertex_count = len(adj)
    if vertex_count == 0:
        return []

    if k <= 0:
        return [-1] * vertex_count

    k = min(k, vertex_count)
    sources_phase1 = random.sample(range(vertex_count), k)
    dist_phase1 = _multi_source_bit_bfs(adj, sources_phase1)

    ranked = sorted(range(vertex_count), key=lambda i: dist_phase1[i], reverse=True)
    sources_phase2 = ranked[:k]
    dist_phase2 = _multi_source_bit_bfs(adj, sources_phase2)

    estimates = [max(dist_phase1[i], dist_phase2[i]) for i in range(vertex_count)]
    return estimates
