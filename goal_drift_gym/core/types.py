"""Core datatypes used across the goal drift gym."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ToolSpec:
    """Describes an available tool for the agent."""

    name: str
    description: str
    args_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """What the agent observes at each step."""

    step: int
    message: str
    panels: Dict[str, Any] = field(default_factory=dict)
    tools: List[ToolSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tools"] = [tool.to_dict() for tool in self.tools]
        return data


@dataclass
class Action:
    """Tool call requested by the agent."""

    tool: Optional[str]
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepOutcome:
    """Result returned by the scenario after applying an action."""

    observation: Observation
    alignment_score: float
    done: bool
    log: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "observation": self.observation.to_dict(),
            "alignment_score": self.alignment_score,
            "done": self.done,
            "log": self.log,
        }
        return data


@dataclass
class StepLogEntry:
    """A single step within a run for JSON logging."""

    step: int
    observation: Observation
    action: Action
    outcome: StepOutcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "outcome": self.outcome.to_dict(),
        }
