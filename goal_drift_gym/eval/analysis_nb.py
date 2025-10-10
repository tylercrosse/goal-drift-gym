
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
from goal_drift_gym.core.config import RunConfig, AgentConfig
from goal_drift_gym.eval import AlignmentMetricsAccumulator
from goal_drift_gym.scenarios import build_scenario
from goal_drift_gym.scripts.run import (
    build_decider,
    prepare_run_dirs,
    write_run_artifacts,
    resolve_run_label,
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
# Run an experiment
def run_experiment(
    scenario: str = "release_manager",
    seed: int = 42,
    agent: str = "openrouter",
    model: str = "openai/gpt-4o-mini",
    max_steps: int = 20,
    runs_root: Path = Path("runs"),
    artifacts_root: Path = Path("artifacts"),
    scenario_updates: dict | None = None,
):
    """Run a single experiment and return the run directory path."""
    # Build run config
    run_config = RunConfig(scenario=scenario)
    run_config = run_config.with_overrides(
        seed=seed,
        max_steps=max_steps,
        agent_type=agent,
        agent_params={"model": model},
        scenario_updates=scenario_updates,
    )

    # Ensure engine's max_steps is always passed to scenario_params
    # This keeps engine and scenario in sync
    scenario_params = run_config.scenario_params.copy()
    scenario_params["max_steps"] = run_config.engine.max_steps

    # Build scenario and engine
    scenario_obj, scenario_config = build_scenario(run_config.scenario, scenario_params)
    engine_config = run_config.engine

    # Run the experiment
    engine = Engine(scenario_obj, config=engine_config)
    metrics = AlignmentMetricsAccumulator(threshold=run_config.metrics.alignment.threshold)
    decider = build_decider(run_config)

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
        result = run_with_optimal_action_logging(engine, decider, metrics, engine_config.seed, lambda: pbar.update(1))
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
    write_run_artifacts(
        run_dir,
        artifacts_dir,
        run_config,
        scenario_config,
        engine_config,
        result.metrics,
        result.steps,
    )

    print(f"Run saved to: {run_dir}")
    print(f"Metrics: {result.metrics}")

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
