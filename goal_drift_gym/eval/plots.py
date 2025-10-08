"""Plotting utilities for goal_drift_gym runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt

RunDict = Dict[str, object]


def load_step_log(run_dir: Path) -> List[Dict[str, object]]:
    """Return the list of step entries (as dictionaries) for a run."""

    step_path = run_dir / "step_log.jsonl"
    steps: List[Dict[str, object]] = []
    with step_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                steps.append(json.loads(line))
    return steps


def load_metrics(run_dir: Path) -> Dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


def load_run_config(run_dir: Path) -> Dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def plot_alignment_timeseries(
    steps: Sequence[Dict[str, object]],
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot alignment score vs. step."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    xs = [entry["observation"]["step"] for entry in steps]
    ys = [entry["outcome"]["alignment_score"] for entry in steps]

    ax.plot(xs, ys, marker="o", label="Alignment score")
    ax.set_xlabel("Step")
    ax.set_ylabel("Alignment score")
    ax.set_ylim(-0.05, 1.05)
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="red", alpha=0.6, label="Threshold")
    if title:
        ax.set_title(title)
    if threshold is not None:
        ax.legend(loc="best")
    return ax


def plot_panel_traces(
    steps: Sequence[Dict[str, object]],
    panels: Iterable[str],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot selected panel values over time."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    xs = [entry["observation"]["step"] for entry in steps]
    for panel in panels:
        ys = [entry["observation"].get("panels", {}).get(panel) for entry in steps]
        if any(value is not None for value in ys):
            ax.plot(xs, ys, marker="o", label=panel)
    ax.set_xlabel("Step")
    ax.set_ylabel("Panel value")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_tool_usage(
    steps: Sequence[Dict[str, object]],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Visualize which tools were selected at each step."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    xs = [entry["observation"]["step"] for entry in steps]
    tools = [entry["action"].get("tool") or "noop" for entry in steps]
    unique_tools = list(dict.fromkeys(tools))
    y_positions = {tool: idx for idx, tool in enumerate(unique_tools)}
    ys = [y_positions[tool] for tool in tools]

    ax.scatter(xs, ys, s=80, c="tab:blue")
    ax.set_yticks(list(y_positions.values()), list(y_positions.keys()))
    ax.set_xlabel("Step")
    ax.set_ylabel("Tool")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return ax


def plot_run_summary_bar(
    metrics_map: Dict[Path, Dict[str, object]],
    *,
    field: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Create a bar chart for a metrics field across multiple runs."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    run_labels = [run_path.name for run_path in metrics_map]
    values = [metrics_map[run_path].get(field) for run_path in metrics_map]
    ax.bar(run_labels, values, color="tab:orange")
    ax.set_ylabel(field.replace("_", " "))
    ax.set_xticklabels(run_labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    return ax


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = [
    "load_step_log",
    "load_metrics",
    "load_run_config",
    "plot_alignment_timeseries",
    "plot_panel_traces",
    "plot_tool_usage",
    "plot_run_summary_bar",
    "save_fig",
]
