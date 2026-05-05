#!/usr/bin/env python3
"""
Interactive CLI for IAE Project (Diameter Approximation)
=======================================================

Main workflows:
1. Visualize random graph + diameters
2. Visualize real dataset + diameters
3. Run single algorithm on chosen graph
4. Run all algorithms on chosen graph
5. Run benchmark experiments
7. VFOA network explorer
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

from algos.kbfs import kbfs_eccentricity_estimate
from algos.takeskosters import takes_kosters
from main import read_graph_from_file, read_graph_from_stream
from scripts.diameter_analysis import compare_diameters
from scripts.diameter_visualize import visualize_graph_with_diameters, visualization_available
from scripts.output_layout import (
    RESULTS_DIR,
    ensure_output_tree,
    resolve_plot_output_dir,
    resolve_report_output_dir,
    resolve_results_input,
    resolve_results_output,
)
from scripts.real_datasets import (
    ensure_real_datasets_copied,
    ensure_vfoa_index_copied,
    list_real_dataset_paths,
    list_vfoa_networks,
    load_real_dataset_graph,
    load_vfoa_graph,
)
from scripts.synthetic_benchmarks import (
    GraphSpec,
    build_graph_from_spec,
    count_edges,
    create_notebook_style_plots,
    exact_diameter,
    run_all_experiments,
    save_report_assets,
)


def clear_screen() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def print_header(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def get_int_input(prompt: str, min_val: int | None = None, max_val: int | None = None, default: int | None = None) -> int:
    while True:
        try:
            raw = input(prompt).strip()
            if not raw and default is not None:
                return default
            value = int(raw)
            if min_val is not None and value < min_val:
                print(f"[ERROR] Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"[ERROR] Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("[ERROR] Invalid input. Please enter an integer.")


def get_float_input(prompt: str, min_val: float | None = None, max_val: float | None = None, default: float | None = None) -> float:
    while True:
        try:
            raw = input(prompt).strip()
            if not raw and default is not None:
                return default
            value = float(raw)
            if min_val is not None and value < min_val:
                print(f"[ERROR] Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"[ERROR] Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")


def parse_k_values(raw: str) -> list[int]:
    if not raw.strip():
        return [1, 2, 4, 8]
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("k values must be positive integers")
        vals.append(value)
    return sorted(set(vals)) if vals else [1, 2, 4, 8]


def choose_graph_spec() -> GraphSpec:
    print("\nGraph type:")
    print("  1. random_sparse (Erdos-Renyi)")
    print("  2. dense (Erdos-Renyi)")
    print("  3. tree")
    print("  4. complete")
    print("  5. bipartite")
    print("  6. path")
    print("  7. grid")
    print("  8. small_world")
    print("  9. scale_free")
    choice = input("\nYour choice: ").strip()

    nodes = get_int_input("Number of nodes: ", min_val=1, default=120)
    params: dict = {}
    graph_type = "random_sparse"
    label = f"n={nodes}"

    if choice == "1":
        graph_type = "random_sparse"
        params["p"] = get_float_input("Edge probability p (default: 0.03): ", min_val=0.0, max_val=1.0, default=0.03)
    elif choice == "2":
        graph_type = "dense"
        params["p"] = get_float_input("Edge probability p (default: 0.6): ", min_val=0.0, max_val=1.0, default=0.6)
    elif choice == "3":
        graph_type = "tree"
    elif choice == "4":
        graph_type = "complete"
    elif choice == "5":
        graph_type = "bipartite"
        params["p"] = get_float_input("Cross-part edge probability p (default: 0.1): ", min_val=0.0, max_val=1.0, default=0.1)
    elif choice == "6":
        graph_type = "path"
    elif choice == "7":
        graph_type = "grid"
        rows = get_int_input("Rows (default: floor(sqrt(n))): ", min_val=1, default=max(1, int(nodes**0.5)))
        cols = get_int_input("Cols (default: ceil(n/rows)): ", min_val=1, default=max(1, (nodes + rows - 1) // rows))
        params["rows"] = rows
        params["cols"] = cols
        label = f"{rows}x{cols}"
    elif choice == "8":
        graph_type = "small_world"
        params["k"] = get_int_input("Neighbor ring size k (default: 4): ", min_val=2, default=4)
        params["p"] = get_float_input("Rewire probability p (default: 0.1): ", min_val=0.0, max_val=1.0, default=0.1)
    elif choice == "9":
        graph_type = "scale_free"
        params["m"] = get_int_input("Edges per new node m (default: 2): ", min_val=1, default=2)
    else:
        print("[WARN] Invalid choice, using random_sparse.")
        params["p"] = 0.03

    return GraphSpec(graph_type=graph_type, nodes=nodes, params=params, label=label)


def show_comparison_result(result: dict) -> None:
    print("\nDiameter Results:")
    print(f"  Exact diameter:         {result['exact_diameter']} ({result['exact_time']:.6f}s)")
    print(f"  k-BFS estimated diam:   {result['k_bfs_diameter']} ({result['k_bfs_time']:.6f}s)")
    print(f"  Takes-Kosters estimate: {result['tk_diameter']} ({result['tk_time']:.6f}s)")
    print(f"  k-BFS abs error:        {abs(result['k_bfs_diameter'] - result['exact_diameter'])}")
    print(f"  TK abs error:           {abs(result['tk_diameter'] - result['exact_diameter'])}")
    kbfs_pair = result.get("k_bfs_pair")
    tk_pair = result.get("tk_pair")
    print(f"  k-BFS diameter nodes:   {kbfs_pair if kbfs_pair is not None else 'not found'}")
    print(f"  TK diameter nodes:      {tk_pair if tk_pair is not None else 'not found'}")


def visualize_random_graph() -> None:
    print_header("VISUALIZE RANDOM GRAPH")
    spec = choose_graph_spec()
    k = get_int_input("k for k-BFS (default: 4): ", min_val=1, default=4)
    seed = get_int_input("Random seed (default: 42): ", min_val=0, default=42)

    random.seed(seed)
    graph = build_graph_from_spec(spec)
    result = compare_diameters(graph, k)

    print(f"\nGraph: {spec.graph_type} ({spec.label})")
    print(f"Nodes: {graph.V}, Edges: {count_edges(graph)}, k: {k}")
    show_comparison_result(result)

    if visualization_available():
        visualize_graph_with_diameters(
            graph=graph,
            title=f"Random graph: {spec.graph_type} ({spec.label})",
            exact_diameter=result["exact_diameter"],
            kbfs_diameter=result["k_bfs_diameter"],
            tk_diameter=result["tk_diameter"],
            kbfs_pair=result.get("k_bfs_pair"),
            tk_pair=result.get("tk_pair"),
        )
    else:
        print("\n[WARN] Visualization skipped. Install matplotlib + networkx to enable plotting.")


def _choose_real_dataset_path() -> Path | None:
    copied = ensure_real_datasets_copied()
    if copied:
        print("\nCopied datasets from AAD project:")
        for p in copied:
            print(f"  - {p}")

    paths = list_real_dataset_paths()
    if not paths:
        print("[ERROR] No real datasets found. Expected AAD data source is missing.")
        return None

    print("\nAvailable real datasets:")
    for idx, path in enumerate(paths, 1):
        print(f"  {idx}. {path.name}")
    choice = get_int_input("Select dataset: ", min_val=1, max_val=len(paths), default=1)
    return paths[choice - 1]


def visualize_real_dataset() -> None:
    print_header("VISUALIZE REAL DATASET")
    dataset_path = _choose_real_dataset_path()
    if dataset_path is None:
        return

    graph = load_real_dataset_graph(dataset_path)
    if graph.V == 0:
        print("[ERROR] Loaded graph is empty.")
        return

    k = get_int_input(f"k for k-BFS (default: {min(5, graph.V)}): ", min_val=1, max_val=graph.V, default=min(5, graph.V))
    result = compare_diameters(graph, k)

    print(f"\nDataset: {dataset_path.name}")
    print(f"Nodes: {graph.V}, Edges: {count_edges(graph)}, k: {k}")
    show_comparison_result(result)

    if visualization_available():
        visualize_graph_with_diameters(
            graph=graph,
            title=f"Real dataset: {dataset_path.name}",
            exact_diameter=result["exact_diameter"],
            kbfs_diameter=result["k_bfs_diameter"],
            tk_diameter=result["tk_diameter"],
            kbfs_pair=result.get("k_bfs_pair"),
            tk_pair=result.get("tk_pair"),
        )
    else:
        print("\n[WARN] Visualization skipped. Install matplotlib + networkx to enable plotting.")


def _build_chosen_graph() -> tuple[object | None, str]:
    print("\nGraph source:")
    print("  1. Random synthetic graph")
    print("  2. Real dataset")
    print("  3. Graph file / stdin")
    source = input("Your choice (default: 1): ").strip() or "1"

    if source == "1":
        spec = choose_graph_spec()
        seed = get_int_input("Random seed (default: 42): ", min_val=0, default=42)
        random.seed(seed)
        return build_graph_from_spec(spec), f"random:{spec.graph_type}:{spec.label}"

    if source == "2":
        path = _choose_real_dataset_path()
        if path is None:
            return None, "real-dataset"
        return load_real_dataset_graph(path), f"dataset:{path.name}"

    if source == "3":
        print("\nInput source:")
        print("  1. Graph file path")
        print("  2. Read from stdin")
        s = input("Your choice (default: 1): ").strip() or "1"
        if s == "2":
            print("\nPaste graph content in expected format, then Ctrl+D (Linux/macOS).")
            return read_graph_from_stream(sys.stdin), "stdin"
        path = input("Graph file path (default: sample_graph.txt): ").strip() or "sample_graph.txt"
        return read_graph_from_file(path), path

    print("[WARN] Invalid source. Falling back to random_sparse.")
    random.seed(42)
    spec = GraphSpec(graph_type="random_sparse", nodes=120, params={"p": 0.03}, label="n=120")
    return build_graph_from_spec(spec), "random:random_sparse:n=120"


def run_single_algorithm() -> None:
    print_header("RUN SINGLE ALGORITHM ON CHOSEN GRAPH")
    graph, desc = _build_chosen_graph()
    if graph is None or graph.V == 0:
        print("[ERROR] Graph is empty.")
        return

    print("\nAlgorithm:")
    print("  1. k-BFS")
    print("  2. Takes-Kosters")
    algo = input("Your choice: ").strip()

    k_default = min(5, graph.V)
    k = get_int_input(f"k for k-BFS (default: {k_default}): ", min_val=1, max_val=graph.V, default=k_default)

    exact = exact_diameter(graph)
    if algo == "1":
        start = time.perf_counter()
        estimates = kbfs_eccentricity_estimate(graph.adj, k)
        elapsed = time.perf_counter() - start
        value = max(estimates) if estimates else 0
        print(f"\nGraph: {desc}")
        print(f"Nodes: {graph.V}, Edges: {count_edges(graph)}")
        print(f"Algorithm: k-BFS (k={k})")
        print(f"Estimated diameter: {value}")
        print(f"Exact diameter:     {exact}")
        print(f"Absolute error:     {abs(value - exact)}")
        print(f"Runtime (s):        {elapsed:.6f}")
    elif algo == "2":
        start = time.perf_counter()
        value = takes_kosters(graph)
        elapsed = time.perf_counter() - start
        print(f"\nGraph: {desc}")
        print(f"Nodes: {graph.V}, Edges: {count_edges(graph)}")
        print("Algorithm: Takes-Kosters")
        print(f"Estimated diameter: {value}")
        print(f"Exact diameter:     {exact}")
        print(f"Absolute error:     {abs(value - exact)}")
        print(f"Runtime (s):        {elapsed:.6f}")
    else:
        print("[ERROR] Invalid algorithm choice.")


def run_all_algorithms() -> None:
    print_header("RUN ALL ALGORITHMS ON CHOSEN GRAPH")
    graph, desc = _build_chosen_graph()
    if graph is None or graph.V == 0:
        print("[ERROR] Graph is empty.")
        return

    k_default = min(5, graph.V)
    k = get_int_input(f"k for k-BFS (default: {k_default}): ", min_val=1, max_val=graph.V, default=k_default)
    result = compare_diameters(graph, k)

    print(f"\nGraph: {desc}")
    print(f"Nodes: {graph.V}, Edges: {count_edges(graph)}, k: {k}")
    show_comparison_result(result)


def run_benchmark_suite() -> None:
    print_header("RUN NOTEBOOK-STYLE BENCHMARK SUITE")
    raw_k = input("k values (comma-separated, default: 1,2,4,8): ").strip()
    try:
        k_values = parse_k_values(raw_k)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    seed = get_int_input("Random seed (default: 42): ", min_val=0, default=42)
    output_name = input("Results file name (default: benchmark_results.csv): ").strip()
    output_path = resolve_results_output(output_name, default_name="benchmark_results.csv")

    print(f"\nRunning benchmark suite for k values: {k_values}")
    df = run_all_experiments(k_values=k_values, seed=seed)
    df.to_csv(output_path, index=False)
    print(f"[OK] Results saved: {output_path}")
    print(f"[OK] Rows: {len(df)}")
    print("\nSample:")
    print(df.head(8).to_string(index=False))


def _pick_results_file(default_name: str = "benchmark_results.csv") -> Path | None:
    name = input(f"Results file name (default: {default_name}): ").strip()
    file_path = resolve_results_input(name, default_name=default_name)
    if file_path is None:
        print(f"[ERROR] File not found. Checked `{RESULTS_DIR}` and local fallback paths.")
        return None
    return file_path


def plot_from_results() -> None:
    print_header("PLOT NOTEBOOK-STYLE FIGURES")
    results_file = _pick_results_file(default_name="benchmark_results.csv")
    if results_file is None:
        return

    import pandas as pd

    df = pd.read_csv(results_file)
    k = get_int_input("k slice for plotting (default: 4): ", min_val=1, default=4)
    run_name = input("Plot run name (default: notebook): ").strip() or "notebook"
    out_dir = resolve_plot_output_dir("notebook", run_name)
    paths = create_notebook_style_plots(df, output_dir=out_dir, k=k)
    if not paths:
        print("[WARN] No plots were generated.")
        return
    print("[OK] Plots generated:")
    for p in paths:
        print(f"  - {p}")


def export_report_from_results() -> None:
    print_header("EXPORT REPORT TABLES + FIGURES")
    results_file = _pick_results_file(default_name="benchmark_results.csv")
    if results_file is None:
        return

    import pandas as pd

    df = pd.read_csv(results_file)
    run_name = input("Report bundle name (default: report): ").strip() or "report"
    report_dir = resolve_report_output_dir(run_name)
    k = get_int_input("k slice for report plots (default: 4): ", min_val=1, default=4)
    table_dir, figure_dir, figure_paths = save_report_assets(df, report_dir=report_dir, k_to_plot=k)
    print(f"[OK] Tables saved to:  {table_dir}")
    print(f"[OK] Figures saved to: {figure_dir}")
    for p in figure_paths:
        print(f"  - {p.name}")


def plot_analysis_pipeline() -> None:
    print_header("PLOT EXPERIMENT ANALYSIS PIPELINE")
    results_file = _pick_results_file(default_name="1.csv")
    if results_file is None:
        return

    # Lazy import so the CLI still works even if plotting deps aren't installed.
    from plot_results import aggregate_runs, generate_all_plots, load_data

    graph_type = input("Filter graph_type (blank = no filter): ").strip() or None
    nodes_raw = input("Filter nodes (blank = no filter): ").strip()
    k_raw = input("Filter k (blank = no filter): ").strip()
    experiment = input("Filter experiment (blank = no filter): ").strip() or None

    nodes = int(nodes_raw) if nodes_raw else None
    k = int(k_raw) if k_raw else None

    aggregate = (input("Average duplicate runs? (y/n, default: y): ").strip().lower() or "y") == "y"

    run_name = input("Analysis run name (default: analysis): ").strip() or "analysis"
    out_dir = resolve_plot_output_dir("analysis", run_name)

    # Optional slice controls.
    comp_k_raw = input("k for comparison/scaling plots (default: 4): ").strip() or "4"
    comp_k = int(comp_k_raw)
    k_analysis_gt = input("k-analysis graph_type (blank = auto): ").strip() or None
    k_analysis_nodes_raw = input("k-analysis nodes (blank = auto): ").strip()
    k_analysis_nodes = int(k_analysis_nodes_raw) if k_analysis_nodes_raw else None
    type_comp_nodes_raw = input("graph-type comparison nodes (blank = auto): ").strip()
    type_comp_nodes = int(type_comp_nodes_raw) if type_comp_nodes_raw else None

    df = load_data(
        results_file,
        graph_type=graph_type,
        nodes=nodes,
        k=k,
        experiment=experiment,
    )
    if df.empty:
        print("[WARN] No rows after filtering. Nothing to plot.")
        return

    if aggregate:
        df = aggregate_runs(df)

    available_k = sorted(df["k"].dropna().astype(int).unique().tolist()) if "k" in df.columns else []
    if available_k and comp_k not in available_k:
        print(f"[WARN] Requested k={comp_k} not found in CSV. Using k={available_k[0]} for comparisons/scaling.")
        comp_k = available_k[0]

    paths = generate_all_plots(
        df,
        plots_dir=out_dir,
        k_for_comparisons=comp_k,
        k_for_scaling=comp_k,
        graph_type_for_k_analysis=k_analysis_gt,
        nodes_for_k_analysis=k_analysis_nodes,
        nodes_for_graph_type_comparison=type_comp_nodes,
    )
    if not paths:
        print("[WARN] No plots were generated (missing columns or empty slices).")
        return
    print(f"[OK] Generated {len(paths)} plot(s) under: {out_dir}")
    for p in paths:
        print(f"  - {p}")


def vfoa_explorer() -> None:
    print_header("VFOA NETWORK EXPLORER")
    if not ensure_vfoa_index_copied():
        print("[ERROR] Could not find VFOA source in AAD project.")
        return

    while True:
        print("\nVFOA Menu:")
        print("  1. List available VFOA networks")
        print("  2. Visualize one VFOA network + diameters")
        print("  3. Run diameter batch on first N VFOA networks")
        print("  0. Back")

        choice = input("\nYour choice: ").strip()
        if choice == "0":
            return
        if choice == "1":
            rows = list_vfoa_networks()
            if not rows:
                print("[WARN] No VFOA metadata found.")
                continue
            print(f"\nFound {len(rows)} networks:")
            for row in rows:
                print(f"  network{row['network']}: participants={row['participants']}")
        elif choice == "2":
            nid = get_int_input("Network id (e.g., 0): ", min_val=0, default=0)
            weighted = (input("Use weighted file? (y/n, default: y): ").strip().lower() or "y") == "y"
            threshold = get_float_input("Attention threshold for undirected edge (default: 0.1): ", min_val=0.0, default=0.1)
            graph, meta = load_vfoa_graph(network_id=nid, weighted=weighted, min_attention=threshold)
            if graph.V == 0:
                print("[ERROR] Empty graph after loading.")
                continue
            k = get_int_input(f"k for k-BFS (default: {min(4, graph.V)}): ", min_val=1, max_val=graph.V, default=min(4, graph.V))
            result = compare_diameters(graph, k)
            print(f"\nVFOA network{nid} (weighted={weighted}, threshold={threshold})")
            print(f"Participants: {meta['participants']}, Timesteps: {meta['timesteps']}, Edges: {meta['edges']}")
            show_comparison_result(result)
            if visualization_available():
                visualize_graph_with_diameters(
                    graph=graph,
                    title=f"VFOA network{nid}",
                    exact_diameter=result["exact_diameter"],
                    kbfs_diameter=result["k_bfs_diameter"],
                    tk_diameter=result["tk_diameter"],
                    kbfs_pair=result.get("k_bfs_pair"),
                    tk_pair=result.get("tk_pair"),
                )
            else:
                print("[WARN] Visualization skipped (missing matplotlib/networkx).")
        elif choice == "3":
            rows = list_vfoa_networks()
            if not rows:
                print("[WARN] No VFOA metadata found.")
                continue
            n = get_int_input(f"How many networks from start? (max {len(rows)}): ", min_val=1, max_val=len(rows), default=min(10, len(rows)))
            weighted = (input("Use weighted files? (y/n, default: y): ").strip().lower() or "y") == "y"
            threshold = get_float_input("Attention threshold (default: 0.1): ", min_val=0.0, default=0.1)
            k = get_int_input("k for k-BFS (default: 4): ", min_val=1, default=4)
            out_name = input("Batch output file name (default: vfoa_results.csv): ").strip()
            out_csv = resolve_results_output(out_name, default_name="vfoa_results.csv")

            results: list[dict] = []
            for row in rows[:n]:
                nid = row["network"]
                try:
                    graph, meta = load_vfoa_graph(network_id=nid, weighted=weighted, min_attention=threshold)
                    if graph.V == 0:
                        continue
                    res = compare_diameters(graph, k)
                    results.append(
                        {
                            "network": nid,
                            "participants": meta["participants"],
                            "timesteps": meta["timesteps"],
                            "edges": meta["edges"],
                            "k": k,
                            "exact_diameter": res["exact_diameter"],
                            "k_bfs_diameter": res["k_bfs_diameter"],
                            "tk_diameter": res["tk_diameter"],
                            "k_bfs_time": res["k_bfs_time"],
                            "tk_time": res["tk_time"],
                        }
                    )
                    print(f"[OK] network{nid} processed.")
                except Exception as exc:
                    print(f"[WARN] network{nid} failed: {exc}")
                    continue

            if not results:
                print("[WARN] No batch results to save.")
                continue
            import pandas as pd

            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            print(f"[OK] Batch results saved: {out_csv}")
            print(df.head(10).to_string(index=False))
        else:
            print("[ERROR] Invalid choice.")


def main_menu() -> None:
    while True:
        clear_screen()
        print_header("IAE PROJECT - DIAMETER APPROXIMATION CLI")
        print(f"\nOutputs are grouped under: {RESULTS_DIR.parent}")
        print("\nMain Menu:")
        print("  1. Visualize Random Graph (graph + diameters)")
        print("  2. Visualize Real Dataset (copied from AAD)")
        print("  3. Run Single Algorithm on Chosen Graph")
        print("  4. Run All Algorithms on Chosen Graph")
        print("  5. Run Benchmark Experiments")
        print("  6. Plot / Export Tools")
        print("  7. VFOA Network Explorer (copied from AAD)")
        print("\n  0. Exit")

        try:
            choice = input("\nYour choice: ").strip()
        except EOFError:
            # When stdin is closed (e.g., non-interactive runs / piping),
            # exit gracefully instead of crashing.
            print("\n[WARN] Input stream closed. Exiting.")
            return
        try:
            if choice == "1":
                visualize_random_graph()
            elif choice == "2":
                visualize_real_dataset()
            elif choice == "3":
                run_single_algorithm()
            elif choice == "4":
                run_all_algorithms()
            elif choice == "5":
                run_benchmark_suite()
            elif choice == "6":
                print("\nPlot / Export Menu:")
                print("  1. Plot notebook-style figures from results CSV")
                print("  2. Export report tables + figures from results CSV")
                print("  3. Plot full experimental analysis pipeline (B-J)")
                print("  0. Back")
                sub = input("\nYour choice: ").strip()
                if sub == "1":
                    plot_from_results()
                elif sub == "2":
                    export_report_from_results()
                elif sub == "3":
                    plot_analysis_pipeline()
            elif choice == "7":
                vfoa_explorer()
            elif choice == "0":
                print("\nGoodbye!")
                return
            else:
                print("\n[ERROR] Invalid choice.")
        except KeyboardInterrupt:
            print("\n\n[WARN] Operation cancelled.")
        except Exception as exc:
            print(f"\n[ERROR] {exc}")
            import traceback

            traceback.print_exc()
        try:
            input("\nPress ENTER to continue...")
        except EOFError:
            # Non-interactive mode: nothing to wait for.
            return


def main() -> int:
    ensure_output_tree()
    main_menu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
