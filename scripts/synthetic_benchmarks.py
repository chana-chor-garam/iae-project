from __future__ import annotations

import math
import random
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None

from algos.graph import Graph


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for benchmark tables. Install with: pip install pandas")


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")


@dataclass(frozen=True)
class GraphSpec:
    graph_type: str
    nodes: int
    params: dict
    label: str


def _new_graph(vertex_count: int, edges: list[tuple[int, int]]) -> Graph:
    graph = Graph(max(0, vertex_count))
    for u, v in edges:
        graph.add_edge(u, v)
    return graph


def generate_erdos_renyi(vertex_count: int, p: float) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    edges: list[tuple[int, int]] = []
    for u in range(vertex_count):
        for v in range(u + 1, vertex_count):
            if random.random() < p:
                edges.append((u, v))
    return _new_graph(vertex_count, edges)


def generate_sparse(vertex_count: int, p: float = 0.03) -> Graph:
    return generate_erdos_renyi(vertex_count, p)


def generate_dense(vertex_count: int, p: float = 0.6) -> Graph:
    return generate_erdos_renyi(vertex_count, p)


def generate_tree(vertex_count: int) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    edges: list[tuple[int, int]] = []
    for node in range(1, vertex_count):
        parent = random.randrange(node)
        edges.append((parent, node))
    return _new_graph(vertex_count, edges)


def generate_complete(vertex_count: int) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    edges: list[tuple[int, int]] = []
    for u in range(vertex_count):
        for v in range(u + 1, vertex_count):
            edges.append((u, v))
    return _new_graph(vertex_count, edges)


def generate_bipartite(vertex_count: int, p: float = 0.1) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    if vertex_count == 1:
        return Graph(1)
    left_size = vertex_count // 2
    right_nodes = list(range(left_size, vertex_count))
    edges: list[tuple[int, int]] = []
    for u in range(left_size):
        for v in right_nodes:
            if random.random() < p:
                edges.append((u, v))
    return _new_graph(vertex_count, edges)


def generate_path(vertex_count: int) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    edges = [(node, node + 1) for node in range(vertex_count - 1)]
    return _new_graph(vertex_count, edges)


def generate_grid(vertex_count: int, rows: int | None = None, cols: int | None = None) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    if rows is None or cols is None:
        side = max(1, int(math.sqrt(vertex_count)))
        rows = side
        cols = max(1, math.ceil(vertex_count / max(1, rows)))
    if rows * cols < vertex_count:
        cols = math.ceil(vertex_count / rows)
    edges: list[tuple[int, int]] = []

    def node_id(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)
            if u >= vertex_count:
                continue
            if r + 1 < rows:
                v = node_id(r + 1, c)
                if v < vertex_count:
                    edges.append((u, v))
            if c + 1 < cols:
                v = node_id(r, c + 1)
                if v < vertex_count:
                    edges.append((u, v))
    return _new_graph(vertex_count, edges)


def generate_small_world(vertex_count: int, k: int = 4, p: float = 0.1) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    if vertex_count < 3:
        return generate_complete(vertex_count)
    k = max(2, k)
    if k >= vertex_count:
        k = vertex_count - 1
    if k % 2 == 1:
        k += 1

    neighbors: dict[int, set[int]] = {i: set() for i in range(vertex_count)}
    for i in range(vertex_count):
        for step in range(1, k // 2 + 1):
            v = (i + step) % vertex_count
            neighbors[i].add(v)
            neighbors[v].add(i)

    base_edges = [(u, v) for u in range(vertex_count) for v in neighbors[u] if u < v]
    for u, v in base_edges:
        if random.random() >= p:
            continue
        neighbors[u].discard(v)
        neighbors[v].discard(u)
        candidates = [x for x in range(vertex_count) if x != u and x not in neighbors[u]]
        if not candidates:
            continue
        new_v = random.choice(candidates)
        neighbors[u].add(new_v)
        neighbors[new_v].add(u)

    edges = [(u, v) for u in range(vertex_count) for v in neighbors[u] if u < v]
    return _new_graph(vertex_count, edges)


def generate_scale_free(vertex_count: int, m: int = 2) -> Graph:
    if vertex_count <= 0:
        return Graph(0)
    if vertex_count <= m + 1:
        return generate_complete(vertex_count)
    edges: list[tuple[int, int]] = []
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
        targets: set[int] = set()
        while len(targets) < m:
            targets.add(random.choices(candidates, weights=weights, k=1)[0])
        for v in targets:
            edges.append((new_node, v))
            degrees[new_node] += 1
            degrees[v] += 1

    return _new_graph(vertex_count, edges)


GENERATOR_BY_TYPE: dict[str, Callable[..., Graph]] = {
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


def count_edges(graph: Graph) -> int:
    return sum(len(neighbors) for neighbors in graph.adj) // 2


def exact_diameter(graph: Graph) -> int:
    if graph.V == 0:
        return 0
    diameter = 0
    for node in range(graph.V):
        distances = graph.bfs(node)
        finite = [d for d in distances if d >= 0]
        if finite:
            diameter = max(diameter, max(finite))
    return diameter


def bfs_with_counts(graph: Graph, start_node: int) -> tuple[list[int], int, int]:
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

    for idx, source in enumerate(sources):
        if source < 0 or source >= vertex_count:
            continue
        bit = 1 << idx
        if visited_bits[source] & bit:
            continue
        visited_bits[source] |= bit
        frontier_bits[source] |= bit
        distances[source] = 0
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
            if not new_bits:
                continue
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
    empty_stats = {"bfs_passes": 0, "vertex_visits": 0, "edge_checks": 0, "pre_time": 0.0, "search_time": 0.0}
    if vertex_count == 0:
        return [], empty_stats
    if k <= 0:
        return [-1] * vertex_count, empty_stats

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


def takes_kosters_with_stats(graph: Graph) -> tuple[int, dict]:
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
        for node in range(graph.V):
            if not visited[node] and upper_bounds[node] > max_ub:
                max_ub = upper_bounds[node]
                u = node
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
        for node in range(graph.V):
            if visited[node]:
                continue
            potential_max = distances_from_u[node] + exact_eccentricity
            if potential_max < upper_bounds[node]:
                upper_bounds[node] = potential_max
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


def run_single(graph: Graph, k: int) -> dict:
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


def make_spec(graph_type: str, nodes: int, params: dict | None = None, label: str | None = None) -> GraphSpec:
    return GraphSpec(
        graph_type=graph_type,
        nodes=nodes,
        params=params or {},
        label=label or f"n={nodes}",
    )


def build_graph_from_spec(spec: GraphSpec) -> Graph:
    return GENERATOR_BY_TYPE[spec.graph_type](spec.nodes, **spec.params)


def default_experiments() -> list[tuple[str, list[GraphSpec]]]:
    core_specs: list[GraphSpec] = []
    for graph_type in ["random_sparse", "dense", "small_world", "scale_free", "tree"]:
        params: dict = {}
        if graph_type == "random_sparse":
            params = {"p": 0.03}
        elif graph_type == "dense":
            params = {"p": 0.6}
        elif graph_type == "small_world":
            params = {"k": 4, "p": 0.1}
        elif graph_type == "scale_free":
            params = {"m": 2}
        for n in [50, 100, 150]:
            core_specs.append(make_spec(graph_type, n, params=params, label=f"n={n}"))

    size_sweep_specs = [make_spec("random_sparse", n, params={"p": 0.03}, label=f"n={n}") for n in [50, 100, 150, 200]]

    density_sweep_specs = [make_spec("random_sparse", 120, params={"p": p}, label=f"p={p}") for p in [0.02, 0.05, 0.1, 0.2]]

    capacity_specs = [
        make_spec("random_sparse", 120, params={"p": 0.02}, label="low"),
        make_spec("random_sparse", 120, params={"p": 0.05}, label="medium"),
        make_spec("random_sparse", 120, params={"p": 0.1}, label="high"),
    ]

    bipartite_specs = [
        make_spec("bipartite", 50, params={"p": 0.1}, label="n=50"),
        make_spec("bipartite", 100, params={"p": 0.1}, label="n=100"),
        make_spec("bipartite", 150, params={"p": 0.1}, label="n=150"),
    ]

    grid_specs = [
        make_spec("grid", 100, params={"rows": 10, "cols": 10}, label="10x10"),
        make_spec("grid", 225, params={"rows": 15, "cols": 15}, label="15x15"),
        make_spec("grid", 400, params={"rows": 20, "cols": 20}, label="20x20"),
    ]

    return [
        ("core_types", core_specs),
        ("size_sweep_sparse", size_sweep_specs),
        ("density_sweep", density_sweep_specs),
        ("capacity_distribution", capacity_specs),
        ("bipartite", bipartite_specs),
        ("grid", grid_specs),
    ]


def run_experiment(experiment_name: str, specs: list[GraphSpec], k_values: list[int]) -> pd.DataFrame:
    _require_pandas()
    rows = []
    for spec in specs:
        graph = build_graph_from_spec(spec)
        edge_count = count_edges(graph)
        for k in k_values:
            results = run_single(graph, k)
            rows.append(
                {
                    "experiment": experiment_name,
                    "graph_type": spec.graph_type,
                    "label": spec.label,
                    "nodes": graph.V,
                    "edges": edge_count,
                    "k": int(k),
                    **results,
                }
            )
    return pd.DataFrame(rows)


def finalize_results(df: pd.DataFrame) -> pd.DataFrame:
    _require_pandas()
    out = df.copy()
    out["k_bfs_abs_error"] = (out["k_bfs_diameter"] - out["exact_diameter"]).abs()
    out["tk_abs_error"] = (out["tk_diameter"] - out["exact_diameter"]).abs()
    safe_pairs = out["nodes"].where(out["nodes"] > 1, 2)
    out["density"] = (2 * out["edges"]) / (safe_pairs * (safe_pairs - 1))
    return out


def run_all_experiments(k_values: list[int] | None = None, seed: int = 42) -> pd.DataFrame:
    _require_pandas()
    if k_values is None:
        k_values = [1, 2, 4, 8]
    random.seed(seed)
    frames = []
    for name, specs in default_experiments():
        frames.append(run_experiment(name, specs, k_values))
    return finalize_results(pd.concat(frames, ignore_index=True))


def build_core_table(df: pd.DataFrame, k: int) -> pd.DataFrame:
    _require_pandas()
    core_df = df[(df["experiment"] == "core_types") & (df["k"] == k)].copy()
    return core_df[
        [
            "graph_type",
            "nodes",
            "edges",
            "density",
            "exact_diameter",
            "k_bfs_diameter",
            "tk_diameter",
            "k_bfs_abs_error",
            "tk_abs_error",
            "k_bfs_time",
            "tk_time",
            "k_bfs_peak_mb",
            "tk_peak_mb",
        ]
    ].sort_values("graph_type")


def build_slice_table(df: pd.DataFrame, experiment: str, k: int, order_col: str) -> pd.DataFrame:
    _require_pandas()
    table = df[(df["experiment"] == experiment) & (df["k"] == k)].copy()
    return table[
        [
            "label",
            "nodes",
            "edges",
            "density",
            "exact_diameter",
            "k_bfs_time",
            "tk_time",
            "k_bfs_abs_error",
            "tk_abs_error",
        ]
    ].sort_values(order_col)


def build_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_pandas()
    core_summary = (
        df[df["experiment"] == "core_types"]
        .groupby(["graph_type", "k"], as_index=False)[["k_bfs_time", "tk_time", "k_bfs_abs_error", "tk_abs_error"]]
        .mean()
    )
    experiment_summary = (
        df[df["experiment"] != "core_types"]
        .groupby(["experiment", "graph_type", "label"], as_index=False)[["k_bfs_time", "tk_time", "k_bfs_abs_error", "tk_abs_error"]]
        .mean()
    )
    return core_summary, experiment_summary


def create_notebook_style_plots(df: pd.DataFrame, output_dir: Path, k: int = 4) -> list[Path]:
    _require_pandas()
    _require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    core_table = build_core_table(df, k)
    size_table = build_slice_table(df, "size_sweep_sparse", k, "nodes")
    density_table = build_slice_table(df, "density_sweep", k, "label")
    capacity_table = build_slice_table(df, "capacity_distribution", k, "label")

    if not core_table.empty:
        plot_df = core_table.set_index("graph_type")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_df[["k_bfs_time", "tk_time"]].plot(kind="bar", ax=axes[0])
        axes[0].set_title(f"Runtime by graph type (k={k})")
        axes[0].set_ylabel("Time (s)")
        axes[0].set_xlabel("Graph type")
        axes[0].legend(["k-BFS", "Takes-Kosters"], title="Algorithm")

        plot_df[["exact_diameter", "k_bfs_diameter", "tk_diameter"]].plot(kind="bar", ax=axes[1])
        axes[1].set_title("Diameters by graph type")
        axes[1].set_ylabel("Diameter")
        axes[1].set_xlabel("Graph type")
        axes[1].legend(["Exact", "k-BFS", "Takes-Kosters"], title="Measure")

        plt.tight_layout()
        path = output_dir / f"core_runtime_diameter_k{k}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    if not size_table.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(size_table["nodes"], size_table["k_bfs_time"], marker="o", label="k-BFS")
        ax.plot(size_table["nodes"], size_table["tk_time"], marker="o", label="Takes-Kosters")
        ax.set_title("Size sweep (sparse): runtime vs nodes")
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Time (s)")
        ax.legend()
        plt.tight_layout()
        path = output_dir / f"size_sweep_runtime_k{k}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    if not density_table.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(density_table["density"], density_table["k_bfs_abs_error"], marker="o", label="k-BFS")
        ax.plot(density_table["density"], density_table["tk_abs_error"], marker="o", label="Takes-Kosters")
        ax.set_title("Density sweep: absolute error vs density")
        ax.set_xlabel("Density")
        ax.set_ylabel("Absolute error")
        ax.legend()
        plt.tight_layout()
        path = output_dir / f"density_sweep_abs_error_k{k}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    if not capacity_table.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(capacity_table))
        ax.bar([i - 0.2 for i in x], capacity_table["k_bfs_time"], width=0.4, label="k-BFS", alpha=0.8)
        ax.bar([i + 0.2 for i in x], capacity_table["tk_time"], width=0.4, label="Takes-Kosters", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(capacity_table["label"].tolist())
        ax.set_title("Capacity distribution: runtime by regime")
        ax.set_xlabel("Capacity regime")
        ax.set_ylabel("Time (s)")
        ax.legend()
        plt.tight_layout()
        path = output_dir / f"capacity_runtime_k{k}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def save_report_assets(df: pd.DataFrame, report_dir: Path, k_to_plot: int = 4) -> tuple[Path, Path, list[Path]]:
    _require_pandas()
    _require_matplotlib()
    assets_dir = report_dir / "assets"
    figure_dir = assets_dir / "figures"
    table_dir = assets_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    core_summary, experiment_summary = build_summary_tables(df)
    core_summary.to_csv(table_dir / "core_summary.csv", index=False)
    experiment_summary.to_csv(table_dir / "experiment_summary.csv", index=False)

    core_k = core_summary[core_summary["k"] == k_to_plot].copy()
    core_k.to_csv(table_dir / f"core_summary_k{k_to_plot}.csv", index=False)

    saved_figures: list[Path] = []
    if not core_k.empty:
        plot_df = core_k.set_index("graph_type")

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        plot_df[["k_bfs_time", "tk_time"]].plot(kind="bar", ax=ax1)
        ax1.set_title(f"Mean runtime by graph type (k={k_to_plot})")
        ax1.set_ylabel("Time (s)")
        ax1.set_xlabel("Graph type")
        plt.tight_layout()
        fig1_path = figure_dir / f"runtime_by_type_k{k_to_plot}.png"
        fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
        plt.close(fig1)
        saved_figures.append(fig1_path)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        plot_df[["k_bfs_abs_error", "tk_abs_error"]].plot(kind="bar", ax=ax2)
        ax2.set_title(f"Mean absolute error vs exact diameter (k={k_to_plot})")
        ax2.set_ylabel("Absolute error")
        ax2.set_xlabel("Graph type")
        error_max = plot_df[["k_bfs_abs_error", "tk_abs_error"]].max().max()
        if error_max == 0:
            ax2.set_ylim(0, 1)
        plt.tight_layout()
        fig2_path = figure_dir / f"abs_error_by_type_k{k_to_plot}.png"
        fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        saved_figures.append(fig2_path)

    size_df = df[(df["experiment"] == "size_sweep_sparse") & (df["k"] == k_to_plot)].sort_values("nodes")
    if not size_df.empty:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(size_df["nodes"], size_df["k_bfs_time"], marker="o", label="k-BFS")
        ax3.plot(size_df["nodes"], size_df["tk_time"], marker="o", label="Takes-Kosters")
        ax3.set_title("Size sweep (sparse): runtime vs nodes")
        ax3.set_xlabel("Nodes")
        ax3.set_ylabel("Time (s)")
        ax3.legend()
        plt.tight_layout()
        fig3_path = figure_dir / f"size_sweep_runtime_k{k_to_plot}.png"
        fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        saved_figures.append(fig3_path)

    density_df = df[(df["experiment"] == "density_sweep") & (df["k"] == k_to_plot)].sort_values("label")
    if not density_df.empty:
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.plot(density_df["label"], density_df["k_bfs_abs_error"], marker="o", label="k-BFS")
        ax4.plot(density_df["label"], density_df["tk_abs_error"], marker="o", label="Takes-Kosters")
        ax4.set_title("Density sweep: absolute error vs p")
        ax4.set_xlabel("Density label")
        ax4.set_ylabel("Absolute error")
        ax4.legend()
        plt.tight_layout()
        fig4_path = figure_dir / f"density_sweep_abs_error_k{k_to_plot}.png"
        fig4.savefig(fig4_path, dpi=200, bbox_inches="tight")
        plt.close(fig4)
        saved_figures.append(fig4_path)

    return table_dir, figure_dir, saved_figures
