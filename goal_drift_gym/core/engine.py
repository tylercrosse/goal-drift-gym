"""Simulation engine orchestrating scenarios and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Protocol

from .types import Action, Observation, StepLogEntry, StepOutcome


class MetricsAccumulator(Protocol):
    """Interface for objects that collect metrics during a run."""

    def record(self, step: int, observation: Observation, action: Action, outcome: StepOutcome) -> None:  # pragma: no cover - protocol
        ...

    def finalize(self) -> Dict[str, Any]:  # pragma: no cover - protocol
        ...


class Scenario(Protocol):
    """Minimal interface every scenario plugin must implement."""

    def reset(self, seed: Optional[int] = None) -> Observation:  # pragma: no cover - protocol
        ...

    def step(self, action: Action) -> StepOutcome:  # pragma: no cover - protocol
        ...


@dataclass
class EngineConfig:
    """Configuration for an engine run."""

    max_steps: int = 20
    seed: Optional[int] = None
    allow_noop: bool = True


@dataclass
class RunResult:
    """Container for the outcome of a simulation run."""

    steps: Iterable[StepLogEntry] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "metrics": self.metrics,
        }


class Engine:
    """Coordinates the interaction loop between an agent and a scenario."""

    def __init__(self, scenario: Scenario, config: Optional[EngineConfig] = None) -> None:
        self._scenario = scenario
        self._config = config or EngineConfig()
        self._current_observation: Optional[Observation] = None

    def reset(self, seed: Optional[int] = None) -> Observation:
        run_seed = seed if seed is not None else self._config.seed
        self._current_observation = self._scenario.reset(seed=run_seed)
        return self._current_observation

    def run(
        self,
        decide_action: Callable[[Observation], Any],
        metrics: Optional[MetricsAccumulator] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> RunResult:
        """Run the interaction loop using the provided decision callable."""

        observation = self.reset(seed=seed)
        steps: list[StepLogEntry] = []

        for step_idx in range(self._config.max_steps):
            action = self._coerce_action(decide_action(observation))
            if action.tool is None and not self._config.allow_noop:
                raise ValueError("Scenario does not permit noop actions")

            outcome = self._scenario.step(action)
            log_entry = StepLogEntry(
                step=observation.step,
                observation=observation,
                action=action,
                outcome=outcome,
            )
            steps.append(log_entry)

            if metrics is not None:
                metrics.record(step_idx, observation, action, outcome)

            if progress_callback is not None:
                progress_callback()

            observation = outcome.observation
            if outcome.done:
                break

        final_metrics = metrics.finalize() if metrics is not None else {}
        return RunResult(steps=steps, metrics=final_metrics)

    @staticmethod
    def _coerce_action(raw_action: Any) -> Action:
        if isinstance(raw_action, Action):
            return raw_action
        if raw_action is None:
            return Action(tool=None)
        if isinstance(raw_action, dict):
            tool = raw_action.get("tool")
            args = raw_action.get("args", {})
            if not isinstance(args, dict):
                raise TypeError("Action args must be a mapping")
            return Action(tool=tool, args=args)
        raise TypeError(f"Unsupported action type: {type(raw_action)!r}")
