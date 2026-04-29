from collections import deque


class Graph:
    def __init__(self, vertices: int) -> None:
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]

    def add_edge(self, u: int, v: int) -> None:
        if u < 0 or v < 0 or u >= self.V or v >= self.V:
            return
        self.adj[u].append(v)
        self.adj[v].append(u)

    def bfs(self, start_node: int) -> list[int]:
        distances = [-1] * self.V
        if start_node < 0 or start_node >= self.V:
            return distances

        queue = deque([start_node])
        distances[start_node] = 0

        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if distances[v] == -1:
                    distances[v] = distances[u] + 1
                    queue.append(v)

        return distances
