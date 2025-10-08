"""Generate plots from goal drift gym runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt

from goal_drift_gym.eval.plots import (
    load_metrics,
    load_run_config,
    load_step_log,
    plot_alignment_timeseries,
    plot_panel_traces,
    plot_run_summary_bar,
    plot_tool_usage,
    save_fig,
)


DEFAULT_PANELS = ("pressure", "emission_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot goal drift gym runs")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Path(s) to run directories (runs/<scenario>/<timestamp>-<seed>)",
    )
    parser.add_argument(
        "--panels",
        nargs="*",
        default=None,
        help="Panel names to include in the panel trace plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to each run's artifacts folder.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="If provided with multiple runs, generate a summary bar chart across runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_paths = [Path(run).resolve() for run in args.runs]
    panels = args.panels if args.panels is not None else DEFAULT_PANELS

    for run_path in run_paths:
        if not run_path.exists():
            raise FileNotFoundError(run_path)
        steps = load_step_log(run_path)
        config = load_run_config(run_path)
        metrics = load_metrics(run_path)
        threshold = config.get("run_config", {}).get("metrics", {}).get("alignment", {}).get("threshold")

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_alignment_timeseries(steps, threshold=threshold, ax=ax, title=f"Alignment — {run_path.name}")
        save_fig(fig, output_path(args.output, run_path, "alignment_timeseries.png"))

        fig, ax = plt.subplots(figsize=(8, 3))
        plot_tool_usage(steps, ax=ax, title=f"Tool usage — {run_path.name}")
        save_fig(fig, output_path(args.output, run_path, "tool_usage.png"))

        if panels:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_panel_traces(steps, panels, ax=ax, title=f"Panels — {run_path.name}")
            save_fig(fig, output_path(args.output, run_path, "panels.png"))

    if args.summary and len(run_paths) > 1:
        metrics_map = {path: load_metrics(path) for path in run_paths}
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_run_summary_bar(metrics_map, field="mean_alignment_score", ax=ax, title="Mean alignment score")
        save_fig(fig, output_path(args.output, run_paths[0], "summary_mean_alignment.png"))

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_run_summary_bar(metrics_map, field="alignment_slope", ax=ax, title="Alignment slope")
        save_fig(fig, output_path(args.output, run_paths[0], "summary_alignment_slope.png"))


def output_path(explicit_root: Path | None, run_path: Path, filename: str) -> Path:
    if explicit_root is not None:
        explicit_root.mkdir(parents=True, exist_ok=True)
        return explicit_root / filename
    parts = run_path.parts
    try:
        runs_idx = parts.index("runs")
    except ValueError as exc:
        raise ValueError(f"Run path does not include a 'runs' directory: {run_path}") from exc
    root = run_path.parent
    while root.name != "" and root.name != "runs" and root.parent != root:
        root = root.parent
    if root.name != "runs":
        raise ValueError(f"Run path does not appear inside a 'runs' directory: {run_path}")
    root_prefix = root.parent
    after_runs = parts[runs_idx + 1 :]
    if len(after_runs) < 2:
        raise ValueError(f"Unexpected run path layout: {run_path}")
    scenario = after_runs[0]
    if len(after_runs) >= 3:
        run_label = after_runs[1]
        run_id = after_runs[2]
        artifacts_dir = root_prefix / "artifacts" / scenario / run_label / run_id
    else:
        run_id = after_runs[1]
        artifacts_dir = root_prefix / "artifacts" / scenario / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir / filename


if __name__ == "__main__":
    main()
