import math


def takes_kosters(graph) -> int:
    diameter, _ = takes_kosters_with_witness(graph)
    return diameter


def takes_kosters_with_witness(graph) -> tuple[int, tuple[int, int] | None]:
    if graph.V == 0:
        return 0, None

    graph_diameter_lb = 0
    witness_pair: tuple[int, int] | None = (0, 0)
    upper_bounds = [math.inf] * graph.V
    visited = [False] * graph.V
    visited_count = 0

    while visited_count < graph.V:
        u = -1
        max_ub = -1
        for i in range(graph.V):
            if not visited[i] and upper_bounds[i] > max_ub:
                max_ub = upper_bounds[i]
                u = i

        if u == -1 or upper_bounds[u] <= graph_diameter_lb:
            break

        distances_from_u = graph.bfs(u)
        exact_eccentricity = max(distances_from_u)
        if exact_eccentricity >= graph_diameter_lb:
            farthest = max(range(graph.V), key=lambda node: distances_from_u[node])
            witness_pair = (u, farthest)
            graph_diameter_lb = exact_eccentricity

        visited[u] = True
        visited_count += 1
        upper_bounds[u] = exact_eccentricity

        for w in range(graph.V):
            if not visited[w]:
                potential_max = distances_from_u[w] + exact_eccentricity
                if potential_max < upper_bounds[w]:
                    upper_bounds[w] = potential_max

    return graph_diameter_lb, witness_pair
