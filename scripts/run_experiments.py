import argparse
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

from algos.kbfs import kbfs_eccentricity_estimate
from algos.takeskosters import takes_kosters
from scripts.datasets import iter_dataset_specs
from scripts.graph_io import build_graph_from_edges, load_graph_from_path


def download_file(url: str, path: Path) -> None:
    if path.exists():
        return
    print(f"Downloading {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def generate_graph(name: str, vertex_count: int = 100):
    if "tree" in name:
        edges = []
        for node in range(1, vertex_count):
            parent = random.randrange(node)
            edges.append((parent, node))
        return build_graph_from_edges(edges)
    if "path" in name:
        edges = [(node, node + 1) for node in range(vertex_count - 1)]
        return build_graph_from_edges(edges)
    return build_graph_from_edges([])


def count_edges(graph) -> int:
    return sum(len(neighbors) for neighbors in graph.adj) // 2


def run_experiments(base_dir: Path, results_file: Path, k: int) -> pd.DataFrame:
    results = []
    for graph_type, spec in iter_dataset_specs():
        graph_dir = base_dir / graph_type
        graph_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {spec.name} ({graph_type})")
        if spec.source == "GENERATE":
            graph = generate_graph(spec.name)
        else:
            extension = Path(spec.source).suffix
            if spec.source.endswith(".tar.gz"):
                extension = ".tar.gz"
            target = graph_dir / f"{spec.name}{extension}"
            download_file(spec.source, target)
            graph = load_graph_from_path(str(target))

        node_count = graph.V
        edge_count = count_edges(graph)

        start = time.perf_counter()
        eccentricity_estimates = kbfs_eccentricity_estimate(graph.adj, k)
        kbfs_diameter = max(eccentricity_estimates) if eccentricity_estimates else 0
        kbfs_time = time.perf_counter() - start

        start = time.perf_counter()
        tk_diameter = takes_kosters(graph)
        tk_time = time.perf_counter() - start

        results.append(
            {
                "graph": spec.name,
                "type": graph_type,
                "nodes": node_count,
                "edges": edge_count,
                "k": k,
                "k_bfs_diameter": kbfs_diameter,
                "k_bfs_time": kbfs_time,
                "tk_diameter": tk_diameter,
                "tk_time": tk_time,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    return df


def plot_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["k_bfs_time", "tk_time"]
    for metric in metrics:
        ax = df.groupby("type")[metric].mean().plot(kind="bar")
        ax.set_title(metric)
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Graph type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = output_dir / f"{metric}.png"
        plt.savefig(output_path)
        plt.clf()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph diameter experiments.")
    parser.add_argument("--base-dir", default="graphs", help="Directory for graph files")
    parser.add_argument("--results", default="results.csv", help="Output CSV file")
    parser.add_argument("--k", type=int, default=5, help="Number of k-BFS sources")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generated graphs and sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_dir = Path(args.base_dir)
    results_file = Path(args.results)

    df = run_experiments(base_dir, results_file, args.k)
    print(f"Results saved to {results_file}")

    plot_metrics(df, Path("."))
    print("Plots saved.")


if __name__ == "__main__":
    main()
