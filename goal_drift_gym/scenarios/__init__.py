"""Scenario plugins and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

from goal_drift_gym.core.engine import Scenario as ScenarioProtocol

from .release_manager import ReleaseManagerConfig, ReleaseManagerScenario
from .trading_goal_vs_profit import TradingGoalVsProfitConfig, TradingGoalVsProfitScenario
from .multi_agent_trading import MultiAgentTradingConfig, MultiAgentTradingScenario


@dataclass(frozen=True)
class ScenarioSpec:
    """Entry in the scenario registry."""

    name: str
    config_cls: Type[Any]
    builder: Callable[[Any], ScenarioProtocol]


_SCENARIOS: Dict[str, ScenarioSpec] = {}


def register_scenario(name: str, config_cls: Type[Any], builder: Callable[[Any], ScenarioProtocol]) -> None:
    if name in _SCENARIOS:
        raise ValueError(f"Scenario '{name}' already registered")
    _SCENARIOS[name] = ScenarioSpec(name=name, config_cls=config_cls, builder=builder)


register_scenario("release_manager", ReleaseManagerConfig, lambda cfg: ReleaseManagerScenario(cfg))
register_scenario("trading_goal_vs_profit", TradingGoalVsProfitConfig, lambda cfg: TradingGoalVsProfitScenario(cfg))
register_scenario("multi_agent_trading", MultiAgentTradingConfig, lambda cfg: MultiAgentTradingScenario(cfg))


def available_scenarios() -> Tuple[str, ...]:
    return tuple(_SCENARIOS.keys())


def build_scenario(name: str, params: Dict[str, Any]) -> Tuple[ScenarioProtocol, Any]:
    if name not in _SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'")
    spec = _SCENARIOS[name]
    config = spec.config_cls(**params)
    scenario = spec.builder(config)
    return scenario, config


def get_scenario_spec(name: str) -> ScenarioSpec:
    return _SCENARIOS[name]


__all__ = [
    "ScenarioSpec",
    "available_scenarios",
    "build_scenario",
    "get_scenario_spec",
    "register_scenario",
    "ReleaseManagerConfig",
    "ReleaseManagerScenario",
    "TradingGoalVsProfitConfig",
    "TradingGoalVsProfitScenario",
    "MultiAgentTradingConfig",
    "MultiAgentTradingScenario",
]
