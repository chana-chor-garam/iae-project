#!/usr/bin/env python3
"""plot_results.py

Clean experimental analysis + plotting pipeline for IAE diameter experiments.

Input: a CSV produced by the benchmark pipeline (see columns in scripts/1.csv).
Output: PNG plots saved into a structured folder tree, by default under ./outputs/plots/.

This module is intentionally independent of notebook code.

Requires:
  - pandas
  - matplotlib

Example:
  python plot_results.py --csv scripts/1.csv --out outputs/plots/analysis/manual

You can also import and call the functions from `iae-project/cli.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class PlotPaths:
    base_dir: Path

    # Backwards-compat / soft stability: earlier versions used `comparisons_dir`.
    # Keep it as an alias to `time_dir`.
    @property
    def comparisons_dir(self) -> Path:  # pragma: no cover
        return self.time_dir

    @property
    def accuracy_dir(self) -> Path:
        return self.base_dir / "accuracy"

    @property
    def time_dir(self) -> Path:
        return self.base_dir / "time"

    @property
    def work_dir(self) -> Path:
        return self.base_dir / "work"

    @property
    def k_analysis_dir(self) -> Path:
        return self.base_dir / "k_analysis"

    @property
    def scaling_dir(self) -> Path:
        return self.base_dir / "scaling"

    @property
    def graph_type_dir(self) -> Path:
        return self.base_dir / "graph_type"

    @property
    def density_dir(self) -> Path:
        return self.base_dir / "density"

    @property
    def memory_dir(self) -> Path:
        return self.base_dir / "memory"

    @property
    def tradeoff_dir(self) -> Path:
        return self.base_dir / "tradeoff"

    def ensure(self) -> None:
        self.accuracy_dir.mkdir(parents=True, exist_ok=True)
        self.time_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.k_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.scaling_dir.mkdir(parents=True, exist_ok=True)
        self.graph_type_dir.mkdir(parents=True, exist_ok=True)
        self.density_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.tradeoff_dir.mkdir(parents=True, exist_ok=True)


# ------------------
# Data handling
# ------------------

def load_data(
    csv_path: str | Path,
    *,
    graph_type: str | None = None,
    nodes: int | None = None,
    k: int | None = None,
    experiment: str | None = None,
) -> pd.DataFrame:
    """Load results CSV and apply optional filters."""

    df = pd.read_csv(csv_path)

    # Basic normalization: ensure expected key columns exist.
    required = {"graph_type", "nodes", "k"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Normalize types.
    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")

    if experiment is not None and "experiment" in df.columns:
        df = df[df["experiment"] == experiment]

    if graph_type is not None:
        df = df[df["graph_type"] == graph_type]

    if nodes is not None:
        df = df[df["nodes"] == nodes]

    if k is not None:
        df = df[df["k"] == k]

    # Drop rows with missing keys after coercion.
    df = df.dropna(subset=["nodes", "k", "graph_type"]).copy()
    df["nodes"] = df["nodes"].astype(int)
    df["k"] = df["k"].astype(int)

    return df


def aggregate_runs(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("experiment", "graph_type", "label", "nodes", "edges", "density", "k"),
) -> pd.DataFrame:
    """Average over multiple runs.

    The benchmark CSV often contains multiple rows per (graph_type, nodes, k) (e.g., different seeds).
    This helper collapses duplicates by taking the mean of numeric metrics.

    Non-numeric columns listed in `group_cols` are kept as group keys.
    """

    # Keep only group columns that exist in the current CSV.
    group_cols = tuple([c for c in group_cols if c in df.columns])
    if not group_cols:
        raise ValueError("No grouping columns exist in the dataframe to aggregate on.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Avoid aggregating group keys as metrics.
    numeric_cols = [c for c in numeric_cols if c not in group_cols]

    if not numeric_cols:
        # Degenerate case: no metrics to average.
        return df.drop_duplicates(subset=list(group_cols)).reset_index(drop=True)

    agg = df.groupby(list(group_cols), as_index=False)[numeric_cols].mean(numeric_only=True)
    return agg


# ------------------
# Plot helpers
# ------------------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc


def _safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s).strip("_")


def _save_line_plot(
    *,
    x: Iterable,
    ys: list[tuple[Iterable, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for y, label in ys:
        ax.plot(list(x), list(y), marker="o", linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_plot(
    *,
    series: list[tuple[Iterable, Iterable, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for x, y, label in series:
        ax.scatter(list(x), list(y), alpha=0.75, s=26, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grouped_bar_plot(
    *,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    left_label: str,
    right_label: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([i - width / 2 for i in x], left_values, width=width, label=left_label)
    ax.bar([i + width / 2 for i in x], right_values, width=width, label=right_label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------
# Plot categories
# ------------------

def _slice_by_k(df: pd.DataFrame, k: int | None) -> pd.DataFrame:
    if k is None:
        return df.copy()
    return df[df["k"] == k].copy()


def plot_time_performance(
    df: pd.DataFrame,
    *,
    time_dir: Path,
    density_dir: Path,
    accuracy_dir: Path,
    k: int | None = None,
    fixed_nodes: int | None = None,
) -> list[Path]:
    """(B) Time performance + pre/search breakdown + density-time."""
    subset = _slice_by_k(df, k)
    if subset.empty:
        return []

    paths: list[Path] = []
    for graph_type, gdf in subset.groupby("graph_type"):
        gdf = gdf.sort_values("nodes")
        x_nodes = gdf["nodes"].tolist()

        if "k_bfs_time" in gdf.columns and "tk_time" in gdf.columns:
            out = time_dir / (f"nodes_vs_time__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=x_nodes,
                ys=[(gdf["k_bfs_time"], "k-BFS time"), (gdf["tk_time"], "TK time")],
                title=f"Nodes vs Time ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="nodes",
                ylabel="time (s)",
                out_path=out,
            )
            paths.append(out)

        if "k_bfs_abs_error" in gdf.columns and "tk_abs_error" in gdf.columns:
            out = accuracy_dir / (f"nodes_vs_error__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=x_nodes,
                ys=[(gdf["k_bfs_abs_error"], "k-BFS error"), (gdf["tk_abs_error"], "TK error")],
                title=f"Nodes vs Error ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="nodes",
                ylabel="absolute error",
                out_path=out,
            )
            paths.append(out)

        if "density" in gdf.columns and "k_bfs_time" in gdf.columns and "tk_time" in gdf.columns:
            ddf = gdf.sort_values("density")
            out = density_dir / (f"density_vs_time__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=ddf["density"].tolist(),
                ys=[(ddf["k_bfs_time"], "k-BFS time"), (ddf["tk_time"], "TK time")],
                title=f"Density vs Time ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="density",
                ylabel="time (s)",
                out_path=out,
            )
            paths.append(out)

        if "density" in gdf.columns and "k_bfs_abs_error" in gdf.columns and "tk_abs_error" in gdf.columns:
            ddf = gdf.sort_values("density")
            out = density_dir / (f"density_vs_error__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=ddf["density"].tolist(),
                ys=[(ddf["k_bfs_abs_error"], "k-BFS error"), (ddf["tk_abs_error"], "TK error")],
                title=f"Density vs Error ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="density",
                ylabel="absolute error",
                out_path=out,
            )
            paths.append(out)

    if fixed_nodes is not None:
        ndf = subset[subset["nodes"] == fixed_nodes].copy()
        if not ndf.empty and "k_bfs_time" in ndf.columns and "tk_time" in ndf.columns:
            ndf = ndf.sort_values("graph_type")
            out = time_dir / (f"graph_type_vs_time__n{fixed_nodes}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_grouped_bar_plot(
                labels=ndf["graph_type"].tolist(),
                left_values=ndf["k_bfs_time"].tolist(),
                right_values=ndf["tk_time"].tolist(),
                left_label="k-BFS time",
                right_label="TK time",
                title=f"Graph Type vs Time (nodes={fixed_nodes}" + (f", k={k})" if k is not None else ")"),
                ylabel="time (s)",
                out_path=out,
            )
            paths.append(out)

    return paths


def plot_work_effort(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    k: int | None = None,
) -> list[Path]:
    """(C) Work/computation effort comparisons."""
    subset = _slice_by_k(df, k)
    if subset.empty:
        return []

    metric_specs = [
        ("edge_checks", "k_bfs_edge_checks", "tk_edge_checks", "edge checks"),
        ("vertex_visits", "k_bfs_vertex_visits", "tk_vertex_visits", "vertex visits"),
        ("bfs_passes", "k_bfs_bfs_passes", "tk_bfs_passes", "BFS passes"),
    ]
    paths: list[Path] = []

    for graph_type, gdf in subset.groupby("graph_type"):
        gdf = gdf.sort_values("nodes")
        x_nodes = gdf["nodes"].tolist()
        for short_name, k_col, tk_col, y_label in metric_specs:
            if k_col not in gdf.columns or tk_col not in gdf.columns:
                continue
            out = out_dir / (f"nodes_vs_{short_name}__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=x_nodes,
                ys=[(gdf[k_col], f"k-BFS {y_label}"), (gdf[tk_col], f"TK {y_label}")],
                title=f"Nodes vs {y_label.title()} ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="nodes",
                ylabel=y_label,
                out_path=out,
            )
            paths.append(out)

    return paths


def plot_k_sensitivity(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    graph_type: str,
    nodes: int,
) -> list[Path]:
    """(D) k-BFS sensitivity at fixed graph_type and nodes."""
    subset = df[(df["graph_type"] == graph_type) & (df["nodes"] == nodes)].copy()
    if subset.empty:
        return []
    subset = subset.sort_values("k")

    x_k = subset["k"].tolist()
    paths: list[Path] = []
    metric_specs = [
        ("k_vs_time", "k_bfs_time", "time (s)", "k-BFS Time"),
        ("k_vs_error", "k_bfs_abs_error", "absolute error", "k-BFS Error"),
        ("k_vs_bfs_passes", "k_bfs_bfs_passes", "BFS passes", "k-BFS BFS Passes"),
        ("k_vs_memory", "k_bfs_peak_mb", "memory (MB)", "k-BFS Memory"),
    ]

    for stem, col, y_label, series_label in metric_specs:
        if col not in subset.columns:
            continue
        out = out_dir / f"{stem}__{_safe_filename(graph_type)}__n{nodes}.png"
        _save_line_plot(
            x=x_k,
            ys=[(subset[col], series_label)],
            title=f"k Sensitivity: {series_label} ({graph_type}, nodes={nodes})",
            xlabel="k",
            ylabel=y_label,
            out_path=out,
        )
        paths.append(out)

    return paths


def plot_scaling_analysis(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    k: int | None = None,
) -> list[Path]:
    """(E) Scaling analysis by graph_type."""
    subset = _slice_by_k(df, k)
    if subset.empty:
        return []

    paths: list[Path] = []
    for graph_type, gdf in subset.groupby("graph_type"):
        gdf = gdf.sort_values("nodes")
        x_nodes = gdf["nodes"].tolist()

        if "k_bfs_time" in gdf.columns and "tk_time" in gdf.columns:
            out = out_dir / (f"nodes_vs_time_growth__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=x_nodes,
                ys=[(gdf["k_bfs_time"], "k-BFS"), (gdf["tk_time"], "TK")],
                title=f"Scaling Trend: Nodes vs Time ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="nodes",
                ylabel="time (s)",
                out_path=out,
            )
            paths.append(out)

        if "k_bfs_edge_checks" in gdf.columns and "tk_edge_checks" in gdf.columns:
            out = out_dir / (f"nodes_vs_edge_checks_growth__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_line_plot(
                x=x_nodes,
                ys=[(gdf["k_bfs_edge_checks"], "k-BFS"), (gdf["tk_edge_checks"], "TK")],
                title=f"Scaling Trend: Nodes vs Edge Checks ({graph_type})" + (f", k={k}" if k is not None else ""),
                xlabel="nodes",
                ylabel="edge checks",
                out_path=out,
            )
            paths.append(out)

    return paths


def plot_graph_structure_sensitivity(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    nodes: int,
    k: int | None = None,
) -> list[Path]:
    """(F) Graph structure sensitivity at fixed nodes."""
    subset = df[df["nodes"] == nodes].copy()
    if k is not None:
        subset = subset[subset["k"] == k]
    if subset.empty:
        return []

    subset = subset.sort_values("graph_type")
    paths: list[Path] = []
    metric_specs = [
        ("time", "k_bfs_time", "tk_time", "time (s)", "Graph Structure Sensitivity: Time"),
        ("error", "k_bfs_abs_error", "tk_abs_error", "absolute error", "Graph Structure Sensitivity: Error"),
        ("edge_checks", "k_bfs_edge_checks", "tk_edge_checks", "edge checks", "Graph Structure Sensitivity: Edge Checks"),
    ]

    for stem, kbfs_col, tk_col, y_label, title in metric_specs:
        if kbfs_col not in subset.columns or tk_col not in subset.columns:
            continue
        out = out_dir / (f"graph_type_vs_{stem}__n{nodes}" + (f"__k{k}" if k is not None else "") + ".png")
        _save_grouped_bar_plot(
            labels=subset["graph_type"].tolist(),
            left_values=subset[kbfs_col].tolist(),
            right_values=subset[tk_col].tolist(),
            left_label="k-BFS",
            right_label="TK",
            title=f"{title} (nodes={nodes}" + (f", k={k})" if k is not None else ")"),
            ylabel=y_label,
            out_path=out,
        )
        paths.append(out)

    return paths


def plot_memory_usage(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    k: int | None = None,
    fixed_nodes: int | None = None,
) -> list[Path]:
    """(H) Memory usage comparison."""
    subset = _slice_by_k(df, k)
    if subset.empty:
        return []

    paths: list[Path] = []

    for graph_type, gdf in subset.groupby("graph_type"):
        if "k_bfs_peak_mb" not in gdf.columns or "tk_peak_mb" not in gdf.columns:
            continue
        gdf = gdf.sort_values("nodes")
        out = out_dir / (f"nodes_vs_memory__{_safe_filename(graph_type)}" + (f"__k{k}" if k is not None else "") + ".png")
        _save_line_plot(
            x=gdf["nodes"].tolist(),
            ys=[(gdf["k_bfs_peak_mb"], "k-BFS memory"), (gdf["tk_peak_mb"], "TK memory")],
            title=f"Nodes vs Memory ({graph_type})" + (f", k={k}" if k is not None else ""),
            xlabel="nodes",
            ylabel="memory (MB)",
            out_path=out,
        )
        paths.append(out)

    if fixed_nodes is not None:
        ndf = subset[subset["nodes"] == fixed_nodes].copy()
        if not ndf.empty and "k_bfs_peak_mb" in ndf.columns and "tk_peak_mb" in ndf.columns:
            ndf = ndf.sort_values("graph_type")
            out = out_dir / (f"graph_type_vs_memory__n{fixed_nodes}" + (f"__k{k}" if k is not None else "") + ".png")
            _save_grouped_bar_plot(
                labels=ndf["graph_type"].tolist(),
                left_values=ndf["k_bfs_peak_mb"].tolist(),
                right_values=ndf["tk_peak_mb"].tolist(),
                left_label="k-BFS memory",
                right_label="TK memory",
                title=f"Graph Type vs Memory (nodes={fixed_nodes}" + (f", k={k})" if k is not None else ")"),
                ylabel="memory (MB)",
                out_path=out,
            )
            paths.append(out)

    return paths


def plot_internal_breakdown(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    k: int | None = None,
) -> list[Path]:
    """(I) Internal breakdown: preprocessing vs search time."""
    subset = _slice_by_k(df, k)
    if subset.empty:
        return []

    need_cols = {"k_bfs_pre_time", "k_bfs_search_time", "tk_pre_time", "tk_search_time"}
    if not need_cols.issubset(set(subset.columns)):
        return []

    out = out_dir / ("pre_time_vs_search_time" + (f"__k{k}" if k is not None else "") + ".png")
    _save_scatter_plot(
        series=[
            (subset["k_bfs_pre_time"], subset["k_bfs_search_time"], "k-BFS"),
            (subset["tk_pre_time"], subset["tk_search_time"], "TK"),
        ],
        title="Preprocessing vs Search Time" + (f" (k={k})" if k is not None else ""),
        xlabel="preprocessing time (s)",
        ylabel="search time (s)",
        out_path=out,
    )
    return [out]


def plot_tradeoff(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    k: int | None = None,
) -> list[Path]:
    """(J) Efficiency vs accuracy tradeoff scatter."""
    subset = _slice_by_k(df, k)
    if subset.empty:
        return []

    need_cols = {"k_bfs_time", "tk_time", "k_bfs_abs_error", "tk_abs_error"}
    if not need_cols.issubset(set(subset.columns)):
        return []

    out = out_dir / ("time_vs_error_tradeoff" + (f"__k{k}" if k is not None else "") + ".png")
    _save_scatter_plot(
        series=[
            (subset["k_bfs_time"], subset["k_bfs_abs_error"], "k-BFS"),
            (subset["tk_time"], subset["tk_abs_error"], "TK"),
        ],
        title="Efficiency vs Accuracy Tradeoff" + (f" (k={k})" if k is not None else ""),
        xlabel="time (s)",
        ylabel="absolute error",
        out_path=out,
    )
    return [out]


def generate_all_plots(
    df: pd.DataFrame,
    *,
    plots_dir: str | Path = "outputs/plots/analysis/default",
    k_for_comparisons: int | None = 4,
    k_for_scaling: int | None = 4,
    nodes_for_k_analysis: int | None = None,
    graph_type_for_k_analysis: str | None = None,
    nodes_for_graph_type_comparison: int | None = None,
) -> list[Path]:
    """Generate requested B..J analysis plots in structured folders."""
    plots_dir = Path(plots_dir)
    paths_helper = PlotPaths(plots_dir)
    paths_helper.ensure()

    out_paths: list[Path] = []

    out_paths += plot_time_performance(
        df,
        time_dir=paths_helper.time_dir,
        density_dir=paths_helper.density_dir,
        accuracy_dir=paths_helper.accuracy_dir,
        k=k_for_comparisons,
        fixed_nodes=nodes_for_graph_type_comparison,
    )
    out_paths += plot_work_effort(df, out_dir=paths_helper.work_dir, k=k_for_comparisons)

    if graph_type_for_k_analysis is not None and nodes_for_k_analysis is not None:
        out_paths += plot_k_sensitivity(
            df,
            out_dir=paths_helper.k_analysis_dir,
            graph_type=graph_type_for_k_analysis,
            nodes=nodes_for_k_analysis,
        )

    out_paths += plot_scaling_analysis(df, out_dir=paths_helper.scaling_dir, k=k_for_scaling)

    if nodes_for_graph_type_comparison is not None:
        out_paths += plot_graph_structure_sensitivity(
            df,
            out_dir=paths_helper.graph_type_dir,
            nodes=nodes_for_graph_type_comparison,
            k=k_for_scaling,
        )

    out_paths += plot_memory_usage(
        df,
        out_dir=paths_helper.memory_dir,
        k=k_for_comparisons,
        fixed_nodes=nodes_for_graph_type_comparison,
    )
    out_paths += plot_internal_breakdown(df, out_dir=paths_helper.time_dir, k=k_for_comparisons)
    out_paths += plot_tradeoff(df, out_dir=paths_helper.tradeoff_dir, k=k_for_comparisons)

    return out_paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot IAE experiment results from CSV")
    p.add_argument("--csv", required=True, help="Path to results CSV")
    p.add_argument("--out", default="outputs/plots/analysis/manual", help="Base output directory for plots")

    p.add_argument("--graph-type", default=None, help="Optional filter graph_type")
    p.add_argument("--nodes", type=int, default=None, help="Optional filter nodes")
    p.add_argument("--k", type=int, default=None, help="Optional filter k")
    p.add_argument("--experiment", default=None, help="Optional filter experiment")

    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Average duplicate runs (groupby experiment/graph_type/label/nodes/edges/density/k)",
    )

    p.add_argument("--k-analysis-graph-type", default=None, help="graph_type for k-analysis plots")
    p.add_argument("--k-analysis-nodes", type=int, default=None, help="nodes for k-analysis plots")
    p.add_argument("--comparison-k", type=int, default=4, help="k slice for nodes-vs-* comparison plots")
    p.add_argument("--scaling-k", type=int, default=4, help="k slice for scaling plots")
    p.add_argument(
        "--graph-type-compare-nodes",
        type=int,
        default=None,
        help="nodes value for graph-type comparison bar charts",
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    df = load_data(args.csv, graph_type=args.graph_type, nodes=args.nodes, k=args.k, experiment=args.experiment)
    if args.aggregate:
        df = aggregate_runs(df)
    if df.empty:
        print("[WARN] No rows found after filtering. Nothing to plot.")
        return 0

    # Heuristic defaults if user doesn't specify.
    if args.k_analysis_graph_type is None and ("graph_type" in df.columns and len(df["graph_type"].unique()) > 0):
        args.k_analysis_graph_type = df["graph_type"].iloc[0]
    if args.k_analysis_nodes is None and ("nodes" in df.columns and len(df["nodes"].unique()) > 0):
        args.k_analysis_nodes = int(sorted(df["nodes"].unique().tolist())[0])
    if args.graph_type_compare_nodes is None and ("nodes" in df.columns and len(df["nodes"].unique()) > 0):
        args.graph_type_compare_nodes = int(sorted(df["nodes"].unique().tolist())[0])

    available_k = sorted(df["k"].dropna().astype(int).unique().tolist()) if "k" in df.columns else []
    if available_k:
        if args.comparison_k not in available_k:
            print(f"[WARN] comparison-k={args.comparison_k} not found. Using k={available_k[0]} instead.")
            args.comparison_k = available_k[0]
        if args.scaling_k not in available_k:
            print(f"[WARN] scaling-k={args.scaling_k} not found. Using k={available_k[0]} instead.")
            args.scaling_k = available_k[0]

    out_paths = generate_all_plots(
        df,
        plots_dir=args.out,
        k_for_comparisons=args.comparison_k,
        k_for_scaling=args.scaling_k,
        graph_type_for_k_analysis=args.k_analysis_graph_type,
        nodes_for_k_analysis=args.k_analysis_nodes,
        nodes_for_graph_type_comparison=args.graph_type_compare_nodes,
    )

    print(f"[OK] Generated {len(out_paths)} plot(s) under: {Path(args.out).resolve()}")
    for p in out_paths:
        print(f"  - {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
