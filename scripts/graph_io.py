import bz2
import gzip
import os
import tarfile
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Iterable

from algos.graph import Graph


@dataclass(frozen=True)
class EdgeList:
    edges: list[tuple[int, int]]


def read_edges_from_lines(lines: Iterable[str]) -> EdgeList:
    data_lines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("c"):
            continue
        if line.startswith("p "):
            continue
        data_lines.append(line)

    if not data_lines:
        return EdgeList([])

    first = data_lines[0].split()
    header_is_counts = False
    if len(first) >= 2 and all(token.lstrip("-").isdigit() for token in first[:2]):
        try:
            declared_edges = int(first[1])
            header_is_counts = len(data_lines) - 1 == declared_edges
        except ValueError:
            header_is_counts = False

    start_idx = 1 if header_is_counts else 0
    edges: list[tuple[int, int]] = []
    for line in data_lines[start_idx:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        if not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
            continue
        u = int(parts[0])
        v = int(parts[1])
        if u == v:
            continue
        edges.append((u, v))

    return EdgeList(edges)


def relabel_edges(edges: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
    labels = sorted({u for edge in edges for u in edge})
    label_map = {label: idx for idx, label in enumerate(labels)}
    relabeled = [(label_map[u], label_map[v]) for u, v in edges]
    return relabeled, len(labels)


def build_graph_from_edges(edges: list[tuple[int, int]]) -> Graph:
    if not edges:
        return Graph(0)

    relabeled, vertex_count = relabel_edges(edges)
    graph = Graph(vertex_count)
    for u, v in relabeled:
        graph.add_edge(u, v)
    return graph


def load_graph_from_path(path: str) -> Graph:
    if path.endswith(".tar.gz"):
        return _load_graph_from_tar_gz(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as file:
            edges = read_edges_from_lines(file).edges
            return build_graph_from_edges(edges)
    if path.endswith(".bz2"):
        with bz2.open(path, "rt") as file:
            edges = read_edges_from_lines(file).edges
            return build_graph_from_edges(edges)
    if path.endswith(".clq"):
        edges = _load_edges_from_clq(path)
        return build_graph_from_edges(edges)

    with open(path, "r", encoding="utf-8") as file:
        edges = read_edges_from_lines(file).edges
        return build_graph_from_edges(edges)


def _load_graph_from_tar_gz(path: str) -> Graph:
    with TemporaryDirectory() as tmp:
        with tarfile.open(path) as tar:
            tar.extractall(tmp)
        for root, _, files in os.walk(tmp):
            for filename in files:
                extracted_path = os.path.join(root, filename)
                return load_graph_from_path(extracted_path)
    return Graph(0)


def _load_edges_from_clq(path: str) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("e "):
                _, u, v = line.split()
                edges.append((int(u), int(v)))
    return edges
