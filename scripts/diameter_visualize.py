from __future__ import annotations

from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import networkx as nx

    HAS_VIS = True
except ImportError:
    plt = None
    nx = None
    HAS_VIS = False


def visualization_available() -> bool:
    return HAS_VIS


def visualize_graph_with_diameters(
    graph,
    title: str,
    exact_diameter: int,
    kbfs_diameter: int,
    tk_diameter: int,
    kbfs_pair: tuple[int, int] | None = None,
    tk_pair: tuple[int, int] | None = None,
    save_dir: str | Path | None = None,
    file_prefix: str | None = None,
    max_nodes_for_plot: int = 150,
) -> None:
    if not HAS_VIS:
        raise RuntimeError("Visualization requires matplotlib + networkx. Install with: pip install matplotlib networkx")

    node_limit = min(graph.V, max_nodes_for_plot)
    visible_nodes = set(range(node_limit))

    g = nx.Graph()
    g.add_nodes_from(range(node_limit))
    for u in range(node_limit):
        for v in graph.adj[u]:
            if v in visible_nodes and u < v:
                g.add_edge(u, v)

    if g.number_of_nodes() == 0:
        print("[WARN] Empty graph, nothing to plot.")
        return

    # Prefer layouts that do not require scipy so visualization works in minimal envs.
    if g.number_of_nodes() <= 80:
        pos = nx.spring_layout(g, seed=42)
    else:
        try:
            pos = nx.kamada_kawai_layout(g)
        except Exception:
            pos = nx.spring_layout(g, seed=42)

    out_dir = Path(save_dir) if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (file_prefix or "visualisation").strip().replace(" ", "_")
    prefix = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in prefix).strip("_") or "visualisation"

    def draw_graph_panel(
        panel_title: str,
        panel_slug: str,
        pair: tuple[int, int] | None = None,
        diameter_value: int | None = None,
    ) -> None:
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(g, pos, node_size=250, node_color="#A7D7F9")
        nx.draw_networkx_edges(g, pos, alpha=0.6, width=1.0, edge_color="#7C7C7C")
        if g.number_of_nodes() <= 60:
            nx.draw_networkx_labels(g, pos, font_size=8)

        title_lines = [title, panel_title]

        if pair is not None:
            u, v = pair
            if u in visible_nodes and v in visible_nodes:
                try:
                    path = nx.shortest_path(g, u, v)
                    path_edges = list(zip(path[:-1], path[1:]))
                    nx.draw_networkx_edges(g, pos, edgelist=path_edges, width=3.0, edge_color="#FF5A5F")
                    nx.draw_networkx_nodes(g, pos, nodelist=[u, v], node_size=360, node_color="#FFD166")
                    actual_len = len(path) - 1
                    title_lines.append(f"Nodes: ({u}, {v}) | Path length: {actual_len}")
                except nx.NetworkXNoPath:
                    title_lines.append(f"Nodes: ({u}, {v}) | no visible path")
            else:
                title_lines.append(f"Nodes: ({u}, {v}) | outside shown node range")
        elif diameter_value is not None:
            title_lines.append(f"Diameter value: {diameter_value}")

        if graph.V > node_limit:
            title_lines.append(f"(showing first {node_limit} of {graph.V} nodes)")

        plt.title("\n".join(title_lines))
        plt.axis("off")
        plt.tight_layout()
        if out_dir is not None:
            out_path = out_dir / f"{prefix}__{panel_slug}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[OK] Visualisation saved: {out_path}")
        plt.show()

    draw_graph_panel(
        panel_title=f"Graph overview | Exact: {exact_diameter} | k-BFS: {kbfs_diameter} | Takes-Kosters: {tk_diameter}",
        panel_slug="overview",
    )
    draw_graph_panel(
        panel_title=f"k-BFS diameter highlight (value={kbfs_diameter})",
        panel_slug="kbfs",
        pair=kbfs_pair,
        diameter_value=kbfs_diameter,
    )
    draw_graph_panel(
        panel_title=f"Takes-Kosters diameter highlight (value={tk_diameter})",
        panel_slug="takes_kosters",
        pair=tk_pair,
        diameter_value=tk_diameter,
    )
