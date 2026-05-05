from __future__ import annotations

import csv
import math
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from algos.graph import Graph
from scripts.datasets import DatasetSpec, iter_real_world_dataset_specs
from scripts.graph_io import build_graph_from_edges, load_graph_from_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
AAD_DATA_DIR = PROJECT_ROOT.parent / "AAD_Error404" / "maxflow-project" / "data"
VFOA_REL = Path("comm-f2f-Resistance-network") / "comm-f2f-Resistance"
CATALOG_DATA_DIR = LOCAL_DATA_DIR / "catalog"


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def ensure_real_datasets_copied() -> list[Path]:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ["email-Eu-core.txt", "road_traffic_network.csv"]:
        src = AAD_DATA_DIR / name
        dst = LOCAL_DATA_DIR / name
        if not dst.exists() and _safe_copy(src, dst):
            copied.append(dst)
    return copied


def ensure_vfoa_index_copied() -> bool:
    src_base = AAD_DATA_DIR / VFOA_REL
    dst_base = LOCAL_DATA_DIR / VFOA_REL
    ok_meta = _safe_copy(src_base / "network_list.csv", dst_base / "network_list.csv")
    _safe_copy(src_base / "README.md", dst_base / "README.md")
    return ok_meta


def ensure_vfoa_network_copied(network_id: int, weighted: bool = True) -> Path:
    ensure_vfoa_index_copied()
    src_base = AAD_DATA_DIR / VFOA_REL / "network"
    dst_base = LOCAL_DATA_DIR / VFOA_REL / "network"
    dst_base.mkdir(parents=True, exist_ok=True)
    suffix = "_weighted.csv" if weighted else ".csv"
    filename = f"network{network_id}{suffix}"
    src = src_base / filename
    dst = dst_base / filename
    if not dst.exists():
        if not src.exists():
            raise FileNotFoundError(f"VFOA source file missing in AAD project: {src}")
        shutil.copy2(src, dst)
    return dst


def list_real_dataset_paths() -> list[Path]:
    ensure_real_datasets_copied()
    paths = []
    for name in ["email-Eu-core.txt", "road_traffic_network.csv"]:
        path = LOCAL_DATA_DIR / name
        if path.exists():
            paths.append(path)
    return paths


def list_catalog_real_dataset_specs() -> list[tuple[str, DatasetSpec]]:
    return list(iter_real_world_dataset_specs())


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    dst.write_bytes(data)


def _infer_suffix_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".tar.gz"):
        return ".tar.gz"
    for suffix in (".txt.gz", ".graph.bz2", ".bz2", ".gz", ".clq", ".txt", ".csv"):
        if path.endswith(suffix):
            return suffix
    return ".dat"


def ensure_catalog_dataset_downloaded(spec: DatasetSpec) -> Path:
    CATALOG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _infer_suffix_from_url(spec.source)
    dst = CATALOG_DATA_DIR / f"{spec.name}{suffix}"
    if dst.exists():
        return dst
    _download_file(spec.source, dst)
    return dst


def list_vfoa_networks() -> list[dict]:
    if not ensure_vfoa_index_copied():
        return []
    meta_path = LOCAL_DATA_DIR / VFOA_REL / "network_list.csv"
    rows: list[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                network = int(row.get("NETWORK", "").strip())
                participants = int(row.get("NUMBER_OF_PARTICIPANTS", "").strip())
            except ValueError:
                continue
            rows.append({"network": network, "participants": participants})
    return rows


def _load_edge_csv(path: Path) -> Graph:
    edges: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                u = int(row[0].strip())
                v = int(row[1].strip())
            except ValueError:
                continue
            if u != v:
                edges.append((u, v))
    return build_graph_from_edges(edges)


def load_real_dataset_graph(path: Path) -> Graph:
    if path.suffix.lower() == ".csv":
        return _load_edge_csv(path)
    return load_graph_from_path(str(path))


def load_vfoa_graph(network_id: int, weighted: bool = True, min_attention: float = 0.1) -> tuple[Graph, dict]:
    vfoa_path = ensure_vfoa_network_copied(network_id=network_id, weighted=weighted)

    rows = list_vfoa_networks()
    participants = next((r["participants"] for r in rows if r["network"] == network_id), None)

    with open(vfoa_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return Graph(0), {"network": network_id, "participants": 0, "timesteps": 0, "weighted": weighted}

        if participants is None:
            # 1 + n*(n+1) columns
            participants = int((math.sqrt(1 + 4 * (len(header) - 1)) - 1) / 2)

        matrix = [[0.0 for _ in range(participants)] for _ in range(participants)]
        timesteps = 0

        for row in reader:
            if len(row) < len(header):
                continue
            timesteps += 1
            for i in range(participants):
                start_col = 1 + i * (participants + 1)
                for j in range(participants):
                    if i == j:
                        continue
                    col_idx = start_col + 1 + j
                    if col_idx >= len(row):
                        continue
                    try:
                        value = float(row[col_idx])
                    except ValueError:
                        value = 0.0
                    matrix[i][j] += value

    if timesteps > 0:
        for i in range(participants):
            for j in range(participants):
                matrix[i][j] /= timesteps

    edges: list[tuple[int, int]] = []
    for i in range(participants):
        for j in range(i + 1, participants):
            weight = (matrix[i][j] + matrix[j][i]) / 2.0
            if weight >= min_attention:
                edges.append((i, j))

    graph = Graph(participants)
    for u, v in edges:
        graph.add_edge(u, v)

    metadata = {
        "network": network_id,
        "participants": participants,
        "timesteps": timesteps,
        "weighted": weighted,
        "threshold": min_attention,
        "edges": len(edges),
    }
    return graph, metadata
