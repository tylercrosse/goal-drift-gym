"""Run configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from .engine import EngineConfig


@dataclass
class AlignmentMetricsConfig:
    """Parameters for alignment metric accumulation."""

    threshold: Optional[float] = 0.0


@dataclass
class MetricsConfig:
    """Container for metrics-related configuration."""

    alignment: AlignmentMetricsConfig = field(default_factory=AlignmentMetricsConfig)


@dataclass
class AgentConfig:
    """Describes the agent driving the scenario."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level run configuration that orchestrates the engine and scenario."""

    scenario: str
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    engine: EngineConfig = field(default_factory=EngineConfig)
    policy: str = "baseline"
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent: Optional[AgentConfig] = None

    def with_overrides(
        self,
        *,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        policy: Optional[str] = None,
        threshold: Optional[float] = None,
        scenario_updates: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        agent_params: Optional[Dict[str, Any]] = None,
    ) -> "RunConfig":
        """Return a copy with CLI-style overrides applied."""

        updated_engine = self.engine
        if seed is not None or max_steps is not None:
            updated_engine = replace(updated_engine)
            if seed is not None:
                updated_engine.seed = seed
            if max_steps is not None:
                updated_engine.max_steps = max_steps

        updated_metrics = self.metrics
        if threshold is not None:
            updated_metrics = replace(self.metrics)
            updated_metrics.alignment = replace(self.metrics.alignment, threshold=threshold)

        updated_policy = policy or self.policy
        updated_scenario_params = {**self.scenario_params}
        if scenario_updates:
            updated_scenario_params.update(scenario_updates)

        updated_agent = self.agent
        if agent_type is not None:
            updated_agent = AgentConfig(type=agent_type, params=agent_params or {})
        elif agent_params:
            if updated_agent is None:
                updated_agent = AgentConfig(type="openrouter", params=agent_params.copy())
            else:
                merged_params = {**updated_agent.params, **agent_params}
                updated_agent = AgentConfig(type=updated_agent.type, params=merged_params)

        return RunConfig(
            scenario=self.scenario,
            scenario_params=updated_scenario_params,
            engine=updated_engine,
            policy=updated_policy,
            metrics=updated_metrics,
            metadata=self.metadata,
            agent=updated_agent,
        )


def load_run_config(path: str | Path | None) -> RunConfig:
    """Load a run configuration from JSON or YAML."""

    data: Dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if config_path.suffix.lower() == ".json":
            data = json.loads(config_path.read_text())
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:  # pragma: no cover - optional dependency
                raise RuntimeError("PyYAML is required to read YAML configs")
            data = yaml.safe_load(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    scenario = data.get("scenario")
    if not scenario:
        raise ValueError("Run configuration must define a 'scenario'")

    scenario_params = data.get("scenario_params", {}) or {}
    if not isinstance(scenario_params, dict):
        raise TypeError("'scenario_params' must be a mapping")

    engine_data = data.get("engine", {}) or {}
    engine = EngineConfig(**engine_data)

    policy = data.get("policy", "baseline")

    metrics_data = data.get("metrics", {}) or {}
    alignment_data = metrics_data.get("alignment", {}) or {}
    alignment = AlignmentMetricsConfig(**alignment_data)
    metrics = MetricsConfig(alignment=alignment)

    metadata = data.get("metadata", {}) or {}

    agent_block = data.get("agent")
    agent_config = None
    if agent_block:
        if not isinstance(agent_block, dict) or "type" not in agent_block:
            raise ValueError("Agent configuration must include a 'type'")
        params = agent_block.get("params", {}) or {}
        if not isinstance(params, dict):
            raise TypeError("Agent params must be a mapping")
        agent_config = AgentConfig(type=agent_block["type"], params=params)

    return RunConfig(
        scenario=scenario,
        scenario_params=scenario_params,
        engine=engine,
        policy=policy,
        metrics=metrics,
        metadata=metadata,
        agent=agent_config,
    )


__all__ = ["AlignmentMetricsConfig", "MetricsConfig", "AgentConfig", "RunConfig", "load_run_config"]
