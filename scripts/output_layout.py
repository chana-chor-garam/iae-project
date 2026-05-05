from __future__ import annotations

from pathlib import Path

OUTPUT_ROOT = Path("outputs")
RESULTS_DIR = OUTPUT_ROOT / "results"
PLOTS_DIR = OUTPUT_ROOT / "plots"
PLOTS_NOTEBOOK_DIR = PLOTS_DIR / "notebook"
PLOTS_ANALYSIS_DIR = PLOTS_DIR / "analysis"
REPORT_ASSETS_DIR = OUTPUT_ROOT / "report_assets"


def ensure_output_tree() -> None:
    for path in (
        OUTPUT_ROOT,
        RESULTS_DIR,
        PLOTS_DIR,
        PLOTS_NOTEBOOK_DIR,
        PLOTS_ANALYSIS_DIR,
        REPORT_ASSETS_DIR,
    ):
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


def resolve_plot_output_dir(group: str, run_name: str | None = None) -> Path:
    ensure_output_tree()
    safe_group = (group or "general").strip().replace("/", "_").replace("\\", "_")
    safe_run = (run_name or "default").strip().replace("/", "_").replace("\\", "_")
    out = PLOTS_DIR / safe_group / safe_run
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_report_output_dir(run_name: str | None = None) -> Path:
    ensure_output_tree()
    safe_run = (run_name or "default").strip().replace("/", "_").replace("\\", "_")
    out = REPORT_ASSETS_DIR / safe_run
    out.mkdir(parents=True, exist_ok=True)
    return out
