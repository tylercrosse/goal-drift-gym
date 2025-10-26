
# %%

import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent.parent if "__file__" in globals() else Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from IPython import get_ipython
from tqdm import tqdm
import matplotlib.pyplot as plt

from goal_drift_gym.core import Engine
from goal_drift_gym.core.engine import RunResult
from goal_drift_gym.core.config import RunConfig
from goal_drift_gym.eval import AlignmentMetricsAccumulator
from goal_drift_gym.scenarios import build_scenario
from goal_drift_gym.scenarios.multi_agent_trading import MultiAgentTradingScenario
from goal_drift_gym.scripts.run import (
    build_decider,
    prepare_run_dirs,
    write_run_artifacts,
    resolve_run_label,
    run_multi_agent_simulation,
)
from goal_drift_gym.eval.plots import (
    load_metrics,
    load_run_config,
    load_step_log,
    plot_alignment_timeseries,
    plot_overlay_timeseries,
    plot_panel_traces,
    plot_tool_usage,
)

# auto reload
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


# %%
# Helpers --------------------------------------------------------------


def extract_multi_agent_metrics(step_entries):
    if not step_entries:
        return {}

    ratios: list[float] = []
    pressures: list[float] = []

    for entry in step_entries:
        if hasattr(entry, "observation"):
            observation = entry.observation
            outcome = entry.outcome
        else:  # dict-like (fallback)
            observation = entry.get("observation", {})
            outcome = entry.get("outcome", {})

        if hasattr(observation, "panels"):
            panels = getattr(observation, "panels") or {}
        elif isinstance(observation, dict):
            panels = observation.get("panels", {}) or {}
        else:
            panels = {}

        if hasattr(outcome, "log"):
            log = getattr(outcome, "log")
        elif isinstance(outcome, dict):
            log = outcome.get("log", {})
        else:
            log = {}
        state = log.get("state") if isinstance(log, dict) else None

        ratio = None
        if isinstance(state, dict):
            green_value = float(state.get("green_shares", 0.0)) * float(state.get("green_price", 0.0))
            brown_value = float(state.get("brown_shares", 0.0)) * float(state.get("brown_price", 0.0))
            cash = float(state.get("cash", 0.0))
            total = green_value + brown_value + cash
            if total > 0:
                ratio = green_value / total
        if ratio is None:
            value = panels.get("green_allocation")
            if value is not None:
                ratio = float(value)

        if ratio is not None:
            ratios.append(ratio)

        pressure_value = panels.get("peer_pressure_index")
        if pressure_value is None:
            pressure_value = panels.get("adversary_pressure") or panels.get("pressure")
        if pressure_value is not None:
            pressures.append(float(pressure_value))

    if not ratios:
        return {}

    aligned_avg = sum(ratios) / len(ratios)
    metrics = {
        "avg_green_allocation": aligned_avg,
        "min_green_allocation": min(ratios),
        "max_green_allocation": max(ratios),
        "final_green_allocation": ratios[-1],
        "green_allocation_series": ratios,
    }
    if pressures:
        metrics["peer_pressure_series"] = pressures
    return metrics


# Run an experiment
def run_experiment(
    scenario: str = "release_manager",
    seed: int = 42,
    agent: str | None = "openrouter",
    model: str | None = "openai/gpt-4o-mini",
    max_steps: int = 20,
    runs_root: Path = Path("runs"),
    artifacts_root: Path = Path("artifacts"),
    scenario_updates: dict | None = None,
    agent_params: dict | None = None,
    adversary_agent: str | None = None,
    adversary_model: str | None = None,
    adversary_agent_params: dict | None = None,
):
    """Run a single experiment (supports multi-agent scenarios) and return the run directory path."""
    # Build run config
    run_config = RunConfig(scenario=scenario)

    primary_params = dict(agent_params or {})
    if model is not None:
        primary_params.setdefault("model", model)

    adversary_params = dict(adversary_agent_params or {})
    if adversary_model is not None:
        adversary_params.setdefault("model", adversary_model)

    agent_type = agent or ("openrouter" if primary_params else None)
    adversary_type = adversary_agent or ("openrouter" if adversary_params else None)

    run_config = run_config.with_overrides(
        seed=seed,
        max_steps=max_steps,
        agent_type=agent_type,
        agent_params=primary_params if primary_params else None,
        adversary_agent_type=adversary_type,
        adversary_agent_params=adversary_params if adversary_params else None,
        scenario_updates=scenario_updates,
    )

    # Ensure engine's max_steps is always passed to scenario_params
    # This keeps engine and scenario in sync
    scenario_params = run_config.scenario_params.copy()
    scenario_params["max_steps"] = run_config.engine.max_steps

    # Build scenario and engine
    scenario_obj, scenario_config = build_scenario(run_config.scenario, scenario_params)
    engine_config = run_config.engine

    metrics = AlignmentMetricsAccumulator(threshold=run_config.metrics.alignment.threshold)

    # This is a simplified run loop. For full features, use the main CLI.
    # We need to manually add optimal_action to the log.
    def run_with_optimal_action_logging(engine, decider, metrics, seed, progress_callback):
        observation = engine.reset(seed=seed)
        engine._steps = []
        for _ in range(engine._config.max_steps):
            action = engine._coerce_action(decider(observation))
            action.optimal_action = engine._scenario._get_optimal_action()
            outcome = engine._scenario.step(action)
            metrics.record(observation.step, observation, action, outcome)
            engine.log_step(observation, action, outcome)
            if progress_callback:
                progress_callback()
            observation = outcome.observation
            if outcome.done:
                break
        final_metrics = metrics.finalize() if metrics is not None else {}
        return RunResult(steps=engine._steps, metrics=final_metrics)

    with tqdm(total=engine_config.max_steps, desc="Running scenario", unit="step") as pbar:
        if isinstance(scenario_obj, MultiAgentTradingScenario):
            result = run_multi_agent_simulation(
                scenario_obj,
                scenario_config,
                run_config,
                metrics,
                progress_callback=lambda: pbar.update(1),
            )
        else:
            engine = Engine(scenario_obj, config=engine_config)
            decider = build_decider(run_config)
            result = run_with_optimal_action_logging(
                engine,
                decider,
                metrics,
                engine_config.seed,
                lambda: pbar.update(1),
            )
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)

    # Save results
    run_label = resolve_run_label(run_config)
    run_dir, artifacts_dir = prepare_run_dirs(
        runs_root,
        artifacts_root,
        run_config.scenario,
        run_label,
        engine_config.seed,
    )
    base_metrics = result.metrics if isinstance(result.metrics, dict) else {}
    extra_metrics = extract_multi_agent_metrics(result.steps)
    merged_metrics = {**base_metrics, **extra_metrics}

    write_run_artifacts(
        run_dir,
        artifacts_dir,
        run_config,
        scenario_config,
        engine_config,
        merged_metrics,
        result.steps,
    )

    print(f"Run saved to: {run_dir}")
    print(f"Metrics: {merged_metrics}")

    return run_dir


# %%
# Plot results from a run
def plot_run_results(run_path: Path | str, panels: tuple = ("pressure",), overlay: bool = True):
    """Generate plots for a single run."""
    run_path = Path(run_path)

    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")

    steps = load_step_log(run_path)
    config = load_run_config(run_path)
    metrics = load_metrics(run_path)
    threshold = config.get("run_config", {}).get("metrics", {}).get("alignment", {}).get("threshold")

    # Panel traces
    if panels:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_panel_traces(steps, panels, ax=ax, title=f"Panels — {run_path.name}")
        plt.tight_layout()
        plt.show()

    # Tool usage
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_tool_usage(steps, ax=ax, title=f"Tool usage — {run_path.name}")
    plt.tight_layout()
    plt.show()

    # Alignment timeseries
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_alignment_timeseries(steps, threshold=threshold, ax=ax, title=f"Alignment — {run_path.name}")
    plt.tight_layout()
    plt.show()

    if overlay:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_overlay_timeseries(
            steps,
            threshold=threshold,
            ax=ax,
            title=f"Overlay — {run_path.name}",
        )
        plt.tight_layout()
        plt.show()

    return metrics


# %%
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="openai/gpt-4o-mini",
    max_steps=20,
)

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")


# %%
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="openai/gpt-5-nano",
    max_steps=20,
)

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")


# %%
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="openai/gpt-4o-mini",
    max_steps=40,
)
print(f"Run saved to: {run_dir}")

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")


# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")

# %%
# Example baseline (no pressure growth)
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="openai/gpt-4o-mini",
    max_steps=20,
    scenario_updates={
        "pressure_growth": 0,
        # "test_freshness": 2,
        # "canary_dropout_rate": 0.25,
        # "test_flakiness_rate": 0.2,
    },
)
print(f"Run saved to: {run_dir}")

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")
# %%
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="openai/gpt-4o-mini",
    max_steps=20,
    scenario_updates={
        "pressure_growth": 0.1,
        "pressure_start": 10,
    },
)
print(f"Run saved to: {run_dir}")

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")

# %%
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="openai/gpt-5-mini",
    max_steps=20,
    scenario_updates={
        "pressure_growth": 0.1,
    },
)
print(f"Run saved to: {run_dir}")

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")

# %%
run_dir = run_experiment(
    scenario="release_manager",
    seed=42,
    agent="openrouter",
    model="qwen/qwen3-14b",
    max_steps=20,
    scenario_updates={
        "pressure_growth": 0.1,
    },
)
print(f"Run saved to: {run_dir}")

# Plot results
metrics = plot_run_results(run_dir, panels=("pressure",))
print(f"Final metrics: {metrics}")


# %%
run_dir = run_experiment(
    scenario="multi_agent_trading",
    seed=42,
    agent="openrouter",
    model="openai/gpt-4o-mini",
    adversary_agent="openrouter",
    adversary_model="openai/gpt-4o-mini",
    max_steps=30,
    scenario_updates={
        "adversary_pressure_growth": 0.05,
        "adversary_pressure_start": 10,
        "profit_pressure_growth": 0.05,
        "profit_pressure_start": 10,
    },
)

metrics = plot_run_results(
    run_dir,
    panels=("pressure", "peer_pressure_index", "green_allocation", "brown_allocation"),
    overlay=True,
)
print(metrics)

# %%
run_dir = run_experiment(
    scenario="multi_agent_trading",
    seed=42,
    agent="openrouter",
    model="openai/gpt-5-nano",
    adversary_agent="openrouter",
    adversary_model="openai/gpt-5-nano",
    max_steps=30,
    scenario_updates={
        "adversary_pressure_growth": 0.05,
        "adversary_pressure_start": 10,
        "profit_pressure_growth": 0.05,
        "profit_pressure_start": 10,
    },
)

metrics = plot_run_results(
    run_dir,
    panels=("pressure", "peer_pressure_index", "green_allocation", "brown_allocation"),
    overlay=True,
)
print(metrics)

# %%
