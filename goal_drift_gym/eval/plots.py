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
    agent_tools = [entry["action"].get("tool") or "noop" for entry in steps]
    optimal_tools = [
        entry.get("action", {}).get("optimal_action", entry.get("optimal_action"))
        for entry in steps
    ]

    all_tools = agent_tools
    if any(optimal_tools):
        all_tools += [t for t in optimal_tools if t]

    unique_tools = sorted(list(dict.fromkeys(all_tools)))
    y_positions = {tool: idx for idx, tool in enumerate(unique_tools)}

    agent_xs: List[int] = []
    agent_ys: List[int] = []
    for x, tool in zip(xs, agent_tools):
        mapped_tool = tool or "noop"
        if mapped_tool not in y_positions:
            continue
        agent_xs.append(x)
        agent_ys.append(y_positions[mapped_tool])
    ax.scatter(agent_xs, agent_ys, s=120, facecolors="none", edgecolors="tab:blue", label="Agent Action")

    if any(optimal_tools):
        optimal_ys = [y_positions.get(tool) for tool in optimal_tools]
        # Filter out steps where optimal action was not logged
        plot_xs = [x for x, y in zip(xs, optimal_ys) if y is not None]
        plot_ys = [y for y in optimal_ys if y is not None]
        ax.scatter(plot_xs, plot_ys, s=60, marker="x", c="tab:red", label="Optimal Action")
        ax.legend(loc="best")

    ax.set_yticks(list(y_positions.values()), list(y_positions.keys()))
    ax.set_xlabel("Step")
    ax.set_ylabel("Tool")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return ax


def plot_overlay_timeseries(
    steps: Sequence[Dict[str, object]],
    *,
    ax: Optional[plt.Axes] = None,
    threshold: Optional[float] = None,
    pressure_panel: str = "pressure",
    title: Optional[str] = None,
) -> plt.Axes:
    """Overlay alignment, pressure, and action traces on a shared timeline."""

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    xs = [entry["observation"]["step"] for entry in steps]

    alignment_scores = [
        entry.get("outcome", {}).get("alignment_score") for entry in steps
    ]
    ax.plot(xs, alignment_scores, color="tab:green", linewidth=2, label="Alignment score")
    ax.set_xlabel("Step")
    ax.set_ylabel("Alignment score")
    ax.set_ylim(-0.05, 1.05)

    if threshold is not None:
        ax.axhline(
            threshold,
            linestyle="--",
            color="tab:red",
            alpha=0.6,
            label="Threshold",
        )

    pressure_values = [
        entry["observation"]
        .get("panels", {})
        .get(pressure_panel)
        for entry in steps
    ]
    pressure_ax: Optional[plt.Axes] = None
    if any(value is not None for value in pressure_values):
        pressure_ax = ax.twinx()
        pressure_ax.plot(
            xs,
            pressure_values,
            color="tab:orange",
            linewidth=1.5,
            marker="o",
            label="Pressure",
        )
        pressure_ax.set_ylabel("Pressure", color="tab:orange")
        pressure_ax.tick_params(axis="y", labelcolor="tab:orange")

    agent_actions = [
        (entry.get("action") or {}).get("tool") or "noop" for entry in steps
    ]
    optimal_actions = [
        (entry.get("action") or {}).get("optimal_action")
        or entry.get("optimal_action")
        for entry in steps
    ]
    all_actions = [
        action for action in agent_actions + optimal_actions if action
    ]
    action_ax: Optional[plt.Axes] = None
    if all_actions:
        unique_actions = list(dict.fromkeys(all_actions))
        action_positions = {name: idx for idx, name in enumerate(unique_actions)}

        action_ax = ax.twinx()
        action_ax.spines["right"].set_position(("axes", 1.12))
        action_ax.set_frame_on(True)
        action_ax.patch.set_visible(False)

        agent_ys = [action_positions.get(action) for action in agent_actions]
        action_ax.scatter(
            xs,
            agent_ys,
            s=120,
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.5,
            label="Agent action",
        )

        optimal_plot_xs = [x for x, action in zip(xs, optimal_actions) if action]
        optimal_plot_ys = [
            action_positions[action] for action in optimal_actions if action
        ]
        if optimal_plot_xs:
            action_ax.scatter(
                optimal_plot_xs,
                optimal_plot_ys,
                s=80,
                marker="x",
                color="tab:purple",
                label="Optimal action",
            )

        action_ax.set_yticks(list(action_positions.values()))
        action_ax.set_yticklabels(list(action_positions.keys()))
        action_ax.set_ylim(-0.5, len(unique_actions) - 0.5)
        action_ax.set_ylabel("Action")

    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if pressure_ax is not None:
        p_handles, p_labels = pressure_ax.get_legend_handles_labels()
        handles += p_handles
        labels += p_labels
    if action_ax is not None:
        a_handles, a_labels = action_ax.get_legend_handles_labels()
        handles += a_handles
        labels += a_labels
    if handles:
        ax.legend(handles, labels, loc="best")

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
    "plot_overlay_timeseries",
    "plot_run_summary_bar",
    "save_fig",
]
