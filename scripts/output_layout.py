from __future__ import annotations

from pathlib import Path

# Output layout (simple and fixed):
# - CSVs: outputs/results
# - plots: outputs/plots/<subfolders>
# - visualisations: outputs/visualisations/<subfolders>
OUTPUTS_DIR = Path("outputs")
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"
VIS_OUTPUT_ROOT = OUTPUTS_DIR / "visualisations"


def ensure_output_tree() -> None:
    for path in (OUTPUTS_DIR, RESULTS_DIR, PLOTS_DIR, VIS_OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def normalize_csv_name(raw_name: str, *, default_name: str) -> str:
    name = (raw_name or "").strip() or default_name
    if "/" in name or "\\" in name:
        name = Path(name).name
    if not name.lower().endswith(".csv"):
        name = f"{name}.csv"
    return name


def resolve_results_output(raw_name: str, *, default_name: str) -> Path:
    ensure_output_tree()
    return RESULTS_DIR / normalize_csv_name(raw_name, default_name=default_name)


def resolve_results_input(raw_name: str, *, default_name: str) -> Path | None:
    ensure_output_tree()
    name = normalize_csv_name(raw_name, default_name=default_name)
    candidates = [
        RESULTS_DIR / name,
        Path(name),
        Path("scripts") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_plot_output_dir(group: str) -> Path:
    ensure_output_tree()
    safe_group = (group or "general").strip().replace("/", "_").replace("\\", "_")
    out = PLOTS_DIR / safe_group
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_visualisation_output_dir(kind: str) -> Path:
    """Return outputs/visualisations/<kind>."""
    ensure_output_tree()
    safe_kind = (kind or "general").strip().replace("/", "_").replace("\\", "_").lower()
    out = VIS_OUTPUT_ROOT / safe_kind
    out.mkdir(parents=True, exist_ok=True)
    return out
