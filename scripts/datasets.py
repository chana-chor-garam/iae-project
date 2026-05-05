from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str


def get_datasets_by_type() -> dict[str, list[DatasetSpec]]:
    return {
        "tree": [
            DatasetSpec("tree_100", "GENERATE"),
        ],
        "path": [
            DatasetSpec("path_100", "GENERATE"),
        ],
        "random_sparse": [
            DatasetSpec(
                "small100",
                "https://graphchallenge.s3.amazonaws.com/synthetic/partitionchallenge/static/"
                "simulated_blockmodel_graph_100_nodes.tar.gz",
            ),
            DatasetSpec(
                "large500",
                "https://graphchallenge.s3.amazonaws.com/synthetic/partitionchallenge/static/"
                "simulated_blockmodel_graph_500_nodes.tar.gz",
            ),
            DatasetSpec(
                "holy",
                "https://graphchallenge.s3.amazonaws.com/synthetic/partitionchallenge/static/"
                "simulated_blockmodel_graph_50_nodes.tar.gz",
            ),
            DatasetSpec(
                "adjnoun_graph",
                "https://sites.cc.gatech.edu/dimacs10/archive/data/clustering/adjnoun.graph.bz2",
            ),
        ],
        "dense": [
            DatasetSpec(
                "C125.9",
                "https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/"
                "master/DIMACS/weighted/C125.9.clq",
            ),
            DatasetSpec(
                "DSJC500_5",
                "https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/"
                "master/DIMACS/weighted/DSJC500_5.clq",
            ),
            DatasetSpec(
                "brock200_2",
                "https://iridia.ulb.ac.be/~fmascia/files/DIMACS/brock200_2.clq",
            ),
            DatasetSpec(
                "MANN_a9",
                "https://raw.githubusercontent.com/jamestrimble/max-weight-clique-instances/"
                "master/DIMACS/weighted/MANN_a9.clq",
            ),
        ],
        "small_world": [
            DatasetSpec(
                "karate",
                "https://sites.cc.gatech.edu/dimacs10/archive/data/clustering/karate.graph.bz2",
            ),
            DatasetSpec(
                "dolphin",
                "https://sites.cc.gatech.edu/dimacs10/archive/data/clustering/dolphins.graph.bz2",
            ),
            DatasetSpec(
                "facebook_combined",
                "https://snap.stanford.edu/data/facebook_combined.txt.gz",
            ),
        ],
        "scale_free": [
            DatasetSpec(
                "facebook_combined",
                "https://snap.stanford.edu/data/facebook_combined.txt.gz",
            ),
        ],
    }


def iter_dataset_specs() -> Iterable[tuple[str, DatasetSpec]]:
    for graph_type, specs in get_datasets_by_type().items():
        for spec in specs:
            yield graph_type, spec


def iter_real_world_dataset_specs() -> Iterable[tuple[str, DatasetSpec]]:
    """Return all downloadable datasets from the legacy catalog."""
    seen: set[str] = set()
    for graph_type, spec in iter_dataset_specs():
        if spec.source == "GENERATE":
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        yield graph_type, spec
